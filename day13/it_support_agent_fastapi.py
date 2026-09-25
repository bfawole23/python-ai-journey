import os
import time
import json
import logging
import sqlite3
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from google import genai

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickets.db")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI(title="IT Support AI Agent (FastAPI)")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

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

def call_with_retry(contents, max_retries=1):
    client = get_client()
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )
            logger.info(f"Successful API call on attempt {attempt + 1}")
            return response
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
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
    "Slow computer performance: Restart the machine, check for pending updates, and close unused background applications.",
    "Wi-Fi connection issues: Toggle Wi-Fi off and on, forget and rejoin the network, or flush DNS cache with 'ipconfig /flushdns'.",
    "Outlook password loop: Clear cached Office credentials in Windows Credential Manager, restart Outlook in safe mode, or re-authenticate modern auth.",
    "External monitor not detected: Re-seat HDMI/DisplayPort cable, press Win+P to select 'Extend', and check GPU display drivers.",
    "Headset microphone not working: Check system privacy permissions for microphone access, set headset as default recording device in sound settings.",
    "Disk storage full: Run Windows Disk Cleanup, empty Recycle Bin, and clear temporary files in %TEMP%."
]

def search_knowledge_base(query):
    query_words = [w.lower() for w in query.replace("'", "").replace('"', "").replace(":", "").replace("-", " ").split() if len(w) > 2]
    best_doc = None
    max_matches = 0
    for doc in knowledge_base:
        doc_lower = doc.lower()
        matches = sum(1 for w in query_words if w in doc_lower)
        if matches > max_matches:
            max_matches = matches
            best_doc = doc
    return best_doc if max_matches > 0 else knowledge_base[0]

def local_triage_fallback(issue):
    issue_lower = issue.lower()
    escalate_keywords = [
        "urgent", "emergency", "deadline", "meeting", "client",
        "won't turn on", "dead", "black screen", "smoke", "spill",
        "coffee", "hardware", "burned", "hazard", "10 min", "blocked"
    ]
    should_escalate = any(kw in issue_lower for kw in escalate_keywords)

    if should_escalate:
        return {
            "action": "escalate_to_technician",
            "reason": "High business urgency, physical hardware fault, or critical deadline detected by triage engine."
        }
    else:
        return {
            "action": "provide_fix",
            "reason": "Common self-resolvable issue detected; matched against IT knowledge base."
        }

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

class QuestionPayload(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
@app.get("/portal", response_class=HTMLResponse)
@app.get("/support", response_class=HTMLResponse)
async def serve_ui(request: Request):
    tpl = "portal.html" if os.path.exists(os.path.join(TEMPLATES_DIR, "portal.html")) else "support.html"
    return templates.TemplateResponse(request=request, name=tpl)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "it-support-agent-fastapi"}

@app.get("/tickets")
async def get_tickets(status: str = None):
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
    return {"tickets": tickets_list, "count": len(tickets_list)}

@app.post("/ask")
async def ask(payload: dict):
    if not payload or "question" not in payload:
        return JSONResponse(status_code=400, content={"error": "Missing 'question' field in request body"})

    issue = str(payload.get("question", "")).strip()

    if not issue:
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty"})

    if len(issue) > 500:
        return JSONResponse(status_code=400, content={"error": "Question too long (max 500 characters)"})

    prompt = f"""An employee reports: "{issue}"

Decide the action:
- Use "provide_fix" ONLY for common, self-resolvable issues (VPN hiccups, printer issues, password resets, slow performance) with no major time pressure.
- Use "escalate_to_technician" if the issue involves: hardware failure, complete inability to work, urgent business deadlines/client-facing situations, or anything a quick fix likely won't solve in time.

Return ONLY valid JSON in this exact format, no other text:
{{
  "action": "provide_fix" or "escalate_to_technician",
  "reason": "why this action was chosen"
}}"""

    decision = None
    try:
        response = call_with_retry(prompt, max_retries=1)
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        decision = json.loads(raw_text.strip())
    except Exception as e:
        logger.warning(f"AI API call unavailable ({e}). Using instant local triage fallback.")
        decision = local_triage_fallback(issue)

    if decision["action"] == "provide_fix":
        result = provide_fix(issue)
    else:
        save_ticket(issue, decision["action"])
        result = escalate_to_technician(issue)

    return {"decision": decision, "result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("it_support_agent_fastapi:app", host="0.0.0.0", port=8000, reload=True)
