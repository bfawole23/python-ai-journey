from google import genai
import os
import time
import json
import logging
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets.db")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["20 per hour"])

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_question TEXT,
            decision TEXT,
            status TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_client():
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def call_with_retry(contents, max_retries=4):
    client = get_client()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents
            )
            logger.info(f"Successful API call on attempt {attempt + 1}")
            return response
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
    logger.error("All retry attempts failed")
    raise Exception("All retry attempts failed")

def get_embedding(text):
    client = get_client()
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

knowledge_base = [
    "VPN connection issues: Restart the VPN client, confirm you're on a stable internet connection, and verify your credentials haven't expired.",
    "Password reset: Go to the self-service portal at portal.company.com/reset, enter your employee ID, and follow the email verification steps.",
    "Printer not responding: Check the printer is powered on and connected to the network. Restart the print spooler service on your machine.",
    "Slow computer performance: Restart the machine, check for pending updates, and close unused background applications."
]

_kb_embeddings = None

def get_kb_embeddings():
    global _kb_embeddings
    if _kb_embeddings is None:
        _kb_embeddings = [get_embedding(doc) for doc in knowledge_base]
    return _kb_embeddings

def search_knowledge_base(query):
    """Searches knowledge base using vector embeddings with keyword fallback."""
    if os.environ.get("GEMINI_API_KEY"):
        try:
            query_emb = get_embedding(query)
            kb_embeddings = get_kb_embeddings()
            scores = [cosine_similarity(query_emb, doc_emb) for doc_emb in kb_embeddings]
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            logger.info(f"RAG search best match score: {best_score:.4f}")
            if best_score >= 0.4:
                return knowledge_base[best_idx]
        except Exception as e:
            logger.warning(f"Embedding search failed: {e}. Falling back to keyword search.")

    # Fallback keyword matching (useful for testing or when API key is missing)
    query_words = [w.lower() for w in query.replace("'", "").replace('"', "").split() if len(w) > 2]
    best_doc = None
    max_matches = 0
    for doc in knowledge_base:
        doc_lower = doc.lower()
        matches = sum(1 for w in query_words if w in doc_lower)
        if matches > max_matches:
            max_matches = matches
            best_doc = doc

    return best_doc if max_matches > 0 else knowledge_base[0]

def provide_fix(issue_summary):
    relevant_doc = search_knowledge_base(issue_summary)
    if relevant_doc:
        return f"Here's a suggested fix for '{issue_summary}': {relevant_doc}"
    return f"Here's a suggested fix: {issue_summary}"

def escalate_to_technician(issue_description):
    return f"This issue has been logged and escalated to a technician: '{issue_description}'"

def save_ticket(employee_question, decision):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tickets (employee_question, decision, status, created_at) VALUES (?, ?, ?, ?)",
        (employee_question, decision, "open", datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

@app.route("/")
def home():
    return "IT Support AI Agent is running!"

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "it-support-agent"})

@app.route("/tickets", methods=["GET"])
def tickets():
    status = request.args.get("status")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if status:
        cursor.execute(
            "SELECT id, employee_question, decision, status, created_at FROM tickets WHERE status = ? ORDER BY id DESC",
            (status,)
        )
    else:
        cursor.execute("SELECT id, employee_question, decision, status, created_at FROM tickets ORDER BY id DESC")
    rows = cursor.fetchall()
    tickets_list = [dict(row) for row in rows]
    conn.close()
    return jsonify({"tickets": tickets_list, "count": len(tickets_list)})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True)

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in request body"}), 400

    issue = data.get("question", "").strip()

    if not issue:
        return jsonify({"error": "Question cannot be empty"}), 400

    if len(issue) > 500:
        return jsonify({"error": "Question too long (max 500 characters)"}), 400

    prompt = f"""An employee reports: "{issue}"

Decide the action:
- Use "provide_fix" ONLY for common, self-resolvable issues (VPN hiccups, printer issues, password resets, slow performance) with no major time pressure.
- Use "escalate_to_technician" if the issue involves: hardware failure, complete inability to work, urgent business deadlines/client-facing situations, or anything a quick fix likely won't solve in time.

Return ONLY valid JSON in this exact format, no other text:
{{
  "action": "provide_fix" or "escalate_to_technician",
  "reason": "why this action was chosen"
}}"""

    response = call_with_retry(prompt)
    decision = json.loads(response.text)

    if decision["action"] == "provide_fix":
        result = provide_fix(issue)
    else:
        save_ticket(issue, decision["action"])
        result = escalate_to_technician(issue)

    return jsonify({"decision": decision, "result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
