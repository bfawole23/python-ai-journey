from google import genai
import os
import time
import json
import logging
import sqlite3
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tickets.db")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


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


def call_with_retry(contents, max_retries=2):
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
    logger.error("All retry attempts failed")
    raise Exception("All retry attempts failed")


def parse_json_response(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


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
    "Password reset or locked account: Go to the self-service portal at portal.company.com/reset, enter your employee ID, and follow the email verification steps.",
    "Printer not responding: Check the printer is powered on and connected to the network. Restart the print spooler service on your machine.",
    "Slow computer performance: Restart the machine, check for pending updates, and close unused background applications.",
    "Full disk or low storage: Empty the Recycle Bin, run Disk Cleanup or Storage Sense, uninstall unused applications, and move large files to OneDrive or an external drive.",
    "Wi-Fi connected but no internet: Forget and rejoin the Wi-Fi network, restart the router or move closer to the access point, run the network troubleshooter, and flush DNS with ipconfig /flushdns.",
    "MFA code not arriving: Check phone signal and Do Not Disturb settings, confirm the authenticator app time is synced, request a new code after 60 seconds, or use a backup verification method.",
    "External monitor not detected: Check the cable and input source, press Windows+P and choose Extend, update the graphics driver, and try another port or cable.",
    "Headset microphone not working in meetings: Select the headset as the input device in Teams or Zoom settings, make sure it is not muted, allow microphone access in Windows privacy settings, and reconnect the headset.",
    "Outlook asking for password or not syncing: Sign out and back in, clear cached credentials in Credential Manager, make sure Offline mode is off, and recreate the Outlook profile if it continues.",
    "Teams crashing during screen share: Fully quit and reopen Teams, clear the Teams cache folder, update Teams and Windows, and share a single window instead of the whole screen.",
    "Browser certificate warning: Check the device date and time are correct, clear the browser cache and SSL state, try another browser, and report it to IT if it appears on internal corporate sites.",
    "Docking station or USB hub not working: Unplug the dock and power cycle it, connect the cable directly to the laptop, update the dock firmware and drivers, and test a different port.",
]

STOP_WORDS = {
    "is", "not", "the", "a", "an", "my", "our", "to", "in", "on", "for", 
    "of", "and", "or", "it", "with", "working", "issue", "problem", "broken",
    "having", "wont", "cant", "doesnt", "getting", "type", "typed", "help"
}

DOMAIN_KEYWORDS = {
    "vpn": "vpn", "cisco": "vpn", "tunnel": "vpn",
    "wifi": "wi-fi", "wi-fi": "wi-fi", "internet": "wi-fi", "network": "wi-fi", "dns": "wi-fi",
    "headset": "headset", "microphone": "headset", "mic": "headset", "audio": "headset", "sound": "headset",
    "printer": "printer", "print": "printer", "printing": "printer", "spooler": "printer",
    "password": "password", "reset": "password", "locked": "password", "login": "password", "account": "password",
    "monitor": "external monitor", "screen": "external monitor", "display": "external monitor", "hdmi": "external monitor",
    "teams": "teams", "outlook": "outlook", "email": "outlook",
    "disk": "disk", "storage": "disk", "drive": "disk", "full": "disk",
    "dock": "docking", "docking": "docking", "usb": "docking", "hub": "docking",
    "mfa": "mfa", "2fa": "mfa", "code": "mfa", "authenticator": "mfa",
    "browser": "browser", "certificate": "browser", "ssl": "browser"
}

_kb_embeddings = None


def fallback_keyword_search(query):
    query_clean = query.lower().replace("'", "").replace('"', "").replace(":", " ").replace("-", " ")
    words = [w for w in query_clean.split() if len(w) > 1 and w not in STOP_WORDS]
    best_doc = knowledge_base[0]
    best_score = -1
    for doc in knowledge_base:
        doc_lower = doc.lower()
        title = doc_lower.split(":")[0] if ":" in doc_lower else ""
        body = doc_lower.split(":", 1)[1] if ":" in doc_lower else doc_lower
        score = 0
        for w in words:
            domain = DOMAIN_KEYWORDS.get(w)
            if domain and domain in title:
                score += 150
            elif w in title:
                score += 80
            elif w in body:
                score += 20
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc, 0.85 if best_score > 0 else 0.5


def search_knowledge_base_scored(query):
    global _kb_embeddings
    try:
        if _kb_embeddings is None:
            _kb_embeddings = [get_embedding(doc) for doc in knowledge_base]
        query_embedding = get_embedding(query)
        scores = [cosine_similarity(query_embedding, e) for e in _kb_embeddings]
        best_index = int(np.argmax(scores))
        return knowledge_base[best_index], float(scores[best_index])
    except Exception as e:
        logger.warning(f"Embedding search unavailable ({e}). Using keyword search.")
        return fallback_keyword_search(query)


def search_knowledge_base(query):
    return search_knowledge_base_scored(query)[0]


def local_triage_fallback(issue, action_hint=None):
    issue_lower = issue.lower()
    hazard_keywords = [
        "smoke", "spill", "coffee", "burned", "burning", "hazard", "fire",
        "won't turn on", "dead screen", "black screen", "shattered", "liquid",
        "charger emitted smoke", "client meeting in 10", "demo in 10 min", "meeting in 10 min"
    ]
    has_hazard = any(kw in issue_lower for kw in hazard_keywords)

    if action_hint == "fix" and not has_hazard:
        should_escalate = False
    elif action_hint == "esc":
        should_escalate = True
    else:
        should_escalate = has_hazard

    if should_escalate:
        return {
            "action": "escalate_to_technician",
            "reason": "Physical hardware fault, safety hazard, or critical deadline detected by triage engine.",
            "steps": []
        }
    else:
        doc, _score = search_knowledge_base_scored(issue)
        steps = [s.strip() for s in doc.split(":", 1)[1].split(",") if s.strip()] if ":" in doc else [doc]
        return {
            "action": "provide_fix",
            "reason": "Common self-resolvable issue detected; matched against IT knowledge base.",
            "steps": steps
        }


def provide_fix(issue_summary):
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
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return ticket_id


def get_all_tickets():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, employee_question, decision, status, created_at FROM tickets ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "employee_question": r[1], "decision": r[2], "status": r[3], "created_at": r[4]}
        for r in rows
    ]


class AskRequest(BaseModel):
    question: str | None = None
    urgency: str | None = "normal"
    action_hint: str | None = None


@app.get("/", response_class=HTMLResponse)
@app.get("/portal", response_class=HTMLResponse)
@app.get("/support", response_class=HTMLResponse)
def portal(request: Request):
    return templates.TemplateResponse(request, "portal.html")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "it-support-agent"}


@app.get("/tickets")
def tickets():
    return {"tickets": get_all_tickets()}


@app.post("/ask")
@limiter.limit("20/hour")
def ask(request: Request, payload: AskRequest):
    if payload.question is None:
        return JSONResponse({"error": "Missing 'question' field in request body"}, status_code=400)

    issue = payload.question.strip()

    if not issue:
        return JSONResponse({"error": "Question cannot be empty"}, status_code=400)

    if len(issue) > 500:
        return JSONResponse({"error": "Question too long (max 500 characters)"}, status_code=400)

    reference, _score = search_knowledge_base_scored(issue)

    prompt = f"""You are an expert IT Service Desk triage assistant.
An employee reports the following IT issue:
"{issue}"

Possibly relevant knowledge base reference:
"{reference}"

TRIAGE RULES:
1. Return "provide_fix" for:
   - VPN issues (dropped connection, authentication retry, DNS)
   - Wi-Fi and network connectivity issues
   - Password reset, locked account, and MFA troubleshooting
   - Printers (offline, queue stuck, print spooler)
   - External monitors, display detection, cable settings
   - Headsets, microphone, sound, audio issues
   - Email, Outlook credential prompt, sync issues
   - Teams or application freezing, crashing, or screen share issues
   - Slow computer performance, high CPU, disk cleanup, low storage
   - Browser errors, SSL certificate warnings, clearing cache
   -> For all these issues, provide actionable self-service steps so the employee can unblock themselves!

2. Return "escalate_to_technician" ONLY for:
   - Physical hardware damage (cracked screen, liquid/coffee spill, broken ports, physical drops)
   - Safety hazards (smoke, burning smell, sparking, swollen battery)
   - Completely dead hardware that will not power on at all (no lights, no fan, dead machine)
   - Urgent client deadlines within 10 minutes where immediate human technician dispatch is required

If the action is "provide_fix", give 3 to 5 short, clear, numbered steps specifically resolving this issue.
If the action is "escalate_to_technician", steps must be an empty list [].

Return ONLY valid JSON in this exact format:
{{
  "action": "provide_fix" or "escalate_to_technician",
  "reason": "one concise sentence explaining why",
  "steps": ["Step 1", "Step 2", "Step 3"]
}}"""

    try:
        response = call_with_retry(prompt)
        decision = parse_json_response(response.text)
    except Exception as e:
        logger.warning(f"AI request unavailable ({e}). Using local triage fallback.")
        decision = local_triage_fallback(issue, payload.action_hint)

    # Honor self-fix if no severe physical hazard exists
    if payload.action_hint == "fix" and decision.get("action") == "escalate_to_technician":
        hazard_keywords = ["smoke", "spill", "coffee", "burned", "burning", "hazard", "fire", "won't turn on", "dead screen", "black screen", "shattered", "liquid"]
        if not any(kw in issue.lower() for kw in hazard_keywords):
            logger.info("Overriding incorrect escalation for self-resolvable issue.")
            fallback = local_triage_fallback(issue, "fix")
            decision["action"] = "provide_fix"
            decision["reason"] = "Self-resolvable issue: self-service troubleshooting plan provided."
            decision["steps"] = fallback["steps"]

    ticket_id = None
    if decision.get("action") == "provide_fix":
        steps = [str(s) for s in decision.get("steps", []) if s]
        if not steps:
            steps = [reference]
        result = provide_fix("; ".join(steps))
    else:
        decision["action"] = "escalate_to_technician"
        steps = []
        ticket_id = save_ticket(issue, decision["action"])
        result = escalate_to_technician(issue)

    return {"decision": decision, "result": result, "steps": steps, "ticket_id": ticket_id}
