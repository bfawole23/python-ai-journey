import os
import sys
import re
import json
import time
import sqlite3
import logging
from datetime import datetime
import numpy as np
from google import genai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AI IT Support & Escalation Studio")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

DB_PATH = os.path.join(BASE_DIR, "tickets.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_question TEXT NOT NULL,
            decision TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            resolution_notes TEXT
        )
    """)
    cursor.execute("PRAGMA table_info(tickets)")
    columns = [col[1] for col in cursor.fetchall()]
    if "resolved_at" not in columns:
        cursor.execute("ALTER TABLE tickets ADD COLUMN resolved_at TEXT")
    if "resolution_notes" not in columns:
        cursor.execute("ALTER TABLE tickets ADD COLUMN resolution_notes TEXT")
    conn.commit()
    conn.close()


init_db()

PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODEL = "gemini-3.8-flash"


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment")
    return genai.Client(api_key=api_key)


def call_with_retry(contents, max_retries=2):
    client = get_client()
    candidate_models = [PRIMARY_MODEL]
    if FALLBACK_MODEL not in candidate_models:
        candidate_models.append(FALLBACK_MODEL)

    for model_name in candidate_models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                logger.info(f"Successful API call using model '{model_name}' on attempt {attempt + 1}")
                return response
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} with model '{model_name}' failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
    logger.error("All retry attempts across all candidate models failed")
    raise Exception("All retry attempts failed")


def parse_json_response(text):
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
    return json.loads(cleaned)


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
    "Keyboard and typing issues: Check physical cable or wireless USB dongle, test another USB port, replace wireless batteries, check Device Manager drivers, toggle Num Lock / Fn Lock, and test with the On-Screen Keyboard.",
    "Mouse or trackpad not responding: Reconnect the USB or Bluetooth receiver, clean the optical sensor, check the physical trackpad toggle key, replace or recharge batteries, and update pointing device drivers in Device Manager.",
    "Audio sound or speaker not working: Verify system volume and audio output device in sound settings, restart the Windows Audio service, run the audio troubleshooter, and update your audio device drivers.",
    "Battery draining fast or not charging: Test with known-working power outlet and AC adapter, run Windows Battery Report via powercfg /batteryreport, update battery drivers in Device Manager, and inspect charging port for debris.",
    "Computer overheating or loud fan: Ensure ventilation vents are clear of dust, close high-CPU background tasks in Task Manager, place laptop on a hard flat surface, and update BIOS and thermal management drivers.",
    "Blue screen BSOD or system crash: Reboot into Safe Mode, run sfc /scannow in admin command prompt, check Windows Event Viewer for recent stop codes, uninstall recently installed updates or drivers, and run memory diagnostics.",
    "Bluetooth device pairing failed: Toggle Bluetooth off and back on, remove existing paired device profile and re-pair, ensure device is in pairing mode with fresh battery, and restart Bluetooth Support Service.",
    "Webcam or camera not working: Check physical privacy shutter, verify camera permissions in Windows Privacy settings, close background apps using camera, and reinstall camera drivers in Device Manager.",
    "Ethernet wired connection failure: Check Ethernet cable clip and link LED lights on the port, test cable in alternate router/switch port, ensure Ethernet adapter is enabled in Network Connections, and reinstall Ethernet NIC driver.",
    "Application installation or update error: Run setup executable as Administrator, ensure sufficient free disk space, temporarily disable conflicting antivirus scanners, and verify Windows Installer service is running.",
    "Developer tools or terminal issues: Verify environment PATH variables, ensure Docker Desktop service is active, verify file and directory permissions, test inside an isolated virtual environment, and check log output.",
]

STOP_WORDS = {
    "is", "not", "the", "a", "an", "my", "our", "to", "in", "on", "for", 
    "of", "and", "or", "it", "with", "working", "issue", "problem", "broken",
    "having", "wont", "cant", "doesnt", "getting", "type", "typed", "help"
}

DOMAIN_KEYWORDS = {
    "vpn": "vpn", "cisco": "vpn", "tunnel": "vpn",
    "wifi": "wi-fi", "wi-fi": "wi-fi", "internet": "wi-fi", "network": "wi-fi", "dns": "wi-fi",
    "headset": "headset", "microphone": "headset", "mic": "headset", "audio": "audio", "sound": "audio", "speaker": "audio", "speakers": "audio", "volume": "audio",
    "printer": "printer", "print": "printer", "printing": "printer", "spooler": "printer",
    "password": "password", "reset": "password", "locked": "password", "login": "password", "account": "password",
    "monitor": "external monitor", "screen": "external monitor", "display": "external monitor", "hdmi": "external monitor", "vga": "external monitor", "projector": "external monitor",
    "teams": "teams", "zoom": "teams", "slack": "teams", "outlook": "outlook", "email": "outlook", "mail": "outlook",
    "disk": "disk", "storage": "disk", "drive": "disk", "full": "disk", "ssd": "disk", "hdd": "disk",
    "dock": "docking", "docking": "docking", "usb": "docking", "hub": "docking", "port": "docking",
    "mfa": "mfa", "2fa": "mfa", "code": "mfa", "authenticator": "mfa", "token": "mfa",
    "browser": "browser", "certificate": "browser", "ssl": "browser", "chrome": "browser", "edge": "browser", "firefox": "browser",
    "keyboard": "keyboard", "keys": "keyboard", "typing": "keyboard", "keypad": "keyboard", "spacebar": "keyboard", "capslock": "keyboard",
    "mouse": "mouse", "trackpad": "mouse", "touchpad": "mouse", "cursor": "mouse", "pointer": "mouse",
    "battery": "battery", "charge": "battery", "charger": "battery", "charging": "battery", "power": "battery",
    "fan": "overheating", "overheating": "overheating", "heat": "overheating", "hot": "overheating", "thermal": "overheating",
    "bsod": "blue screen", "crash": "blue screen", "crashed": "blue screen", "freeze": "blue screen", "frozen": "blue screen", "blue": "blue screen",
    "bluetooth": "bluetooth", "pair": "bluetooth", "pairing": "bluetooth",
    "webcam": "webcam", "camera": "webcam", "video": "webcam",
    "ethernet": "ethernet", "lan": "ethernet", "cable": "ethernet",
    "install": "installation", "update": "installation", "setup": "installation", "installer": "installation",
    "git": "developer", "docker": "developer", "terminal": "developer", "python": "developer", "pip": "developer", "bash": "developer"
}

_kb_embeddings = None


def get_kb_embeddings():
    global _kb_embeddings
    if _kb_embeddings is None:
        client = get_client()
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=knowledge_base
        )
        _kb_embeddings = [np.array(e.values) for e in result.embeddings]
    return _kb_embeddings


def fallback_keyword_search(query):
    query_clean = query.lower().replace("'", "").replace('"', "").replace(":", " ").replace("-", " ")
    words = [w for w in query_clean.split() if len(w) > 1 and w not in STOP_WORDS]
    best_doc = None
    best_score = 0
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
    if best_score > 0 and best_doc is not None:
        return best_doc, 0.85
    return "General device troubleshooting: Restart the affected device or application, verify all physical cable connections, check for pending updates, and reconnect the peripheral.", 0.30


def search_knowledge_base_scored(query):
    try:
        kb_embeds = get_kb_embeddings()
        query_embedding = get_embedding(query)
        scores = [cosine_similarity(query_embedding, e) for e in kb_embeds]
        best_index = int(np.argmax(scores))
        return knowledge_base[best_index], float(scores[best_index])
    except Exception as e:
        logger.warning(f"Embedding search unavailable ({e}). Using keyword search.")
        return fallback_keyword_search(query)


def search_knowledge_base(query):
    return search_knowledge_base_scored(query)[0]


IT_DOMAIN_TERMS = {
    "computer", "laptop", "pc", "desktop", "mac", "macbook", "windows", "linux", "machine", "workstation",
    "tablet", "phone", "iphone", "android", "device", "hardware", "software", "system",
    "keyboard", "keys", "typing", "keypad", "spacebar", "mouse", "trackpad", "touchpad", "cursor", "pointer", "monitor",
    "screen", "display", "hdmi", "vga", "usb", "port", "dock", "docking", "hub", "cable", "cord", "wire",
    "printer", "print", "printing", "spooler", "scanner", "headset", "headphones", "mic", "microphone",
    "audio", "sound", "speaker", "speakers", "webcam", "camera", "video", "battery", "charger", "charging", "power",
    "plug", "fan", "vent", "disk", "drive", "storage", "ssd", "hdd", "ram", "memory", "cpu", "processor",
    "motherboard", "bluetooth",
    "app", "application", "program", "browser", "chrome", "edge", "safari", "firefox", "outlook", "email",
    "mail", "teams", "zoom", "slack", "office", "excel", "word", "powerpoint", "os", "driver", "drivers",
    "update", "upgrade", "install", "installation", "uninstall", "reinstall", "download", "sync", "syncing",
    "file", "folder", "cache", "boot", "reboot", "restart", "shutdown", "bios",
    "wifi", "wi-fi", "internet", "vpn", "network", "ethernet", "router", "modem", "dns", "ip", "intranet",
    "lan", "offline", "online", "connection", "connect", "disconnect", "ping", "bandwidth",
    "password", "passcode", "credentials", "login", "log in", "sign in", "signin", "account", "locked",
    "unlock", "mfa", "2fa", "authenticator", "otp", "token", "permission", "permissions", "access",
    "ssl", "certificate", "virus", "malware", "firewall", "phishing",
    "error", "failed", "failure", "crash", "crashed", "crashing", "freeze", "frozen", "freezing",
    "slow", "lag", "hang", "glitch", "bug", "stuck", "unresponsive", "broken", "repair", "repairs",
    "fix", "troubleshoot", "smoke", "spill", "coffee", "burned", "burning", "hazard", "fire",
    "dead", "overheating", "overheat", "shut down", "shuts down"
}


def is_it_support_related(text: str) -> bool:
    text_lower = text.lower()
    words = re.findall(r"[a-z0-9_#\-\+\.]+", text_lower)
    word_set = set(words)
    if any(term in word_set for term in IT_DOMAIN_TERMS):
        return True
    multi_word_terms = [
        "turn on", "power on", "black screen", "blue screen", "slow computer",
        "hard drive", "usb port", "hdmi cable", "wi-fi", "log in", "sign in", "won't work", "wont work"
    ]
    return any(mw in text_lower for mw in multi_word_terms)


def local_triage_fallback(issue, action_hint=None):
    if action_hint is None and not is_it_support_related(issue):
        return {
            "action": "out_of_scope",
            "reason": "This inquiry is outside the scope of IT technical support and computer hardware/software repairs. Please ask a technical question about your computer, network, software, or peripherals.",
            "steps": []
        }

    issue_lower = issue.lower()
    hazard_keywords = [
        "smoke", "spill", "coffee", "burned", "burning", "hazard", "fire",
        "won't turn on", "wont turn on", "shattered", "liquid", "cracked screen", "broken screen",
        "swollen battery", "sparking",
        "charger emitted smoke", "client meeting in 10", "demo in 10 min", "meeting in 10 min"
    ]
    has_hazard = any(kw in issue_lower for kw in hazard_keywords)

    selffix_failed_keywords = [
        "selffix is not working", "self-fix is not working", "selffix not working", "self fix is not working",
        "self fix not working", "self-fix didn't work", "selffix didn't work", "self fix didn't work",
        "fix is not working", "fix didn't work", "fix did not work", "steps didn't work", "steps did not work",
        "tried the fix", "tried the steps", "already tried", "still not working after",
        "still broken after", "troubleshooting didn't work", "troubleshooting failed",
        "escalate to technician", "escalate to engineer", "need an engineer", "need a technician",
        "connect me to engineer", "talk to human", "speak to engineer", "hands on help",
        "send a technician", "dispatch engineer", "not working after restart", "still not working"
    ]
    has_failed_fix = any(kw in issue_lower for kw in selffix_failed_keywords)

    if action_hint == "fix" and not has_hazard:
        should_escalate = False
    elif action_hint == "esc" or has_failed_fix:
        should_escalate = True
    else:
        should_escalate = has_hazard

    if should_escalate:
        if has_failed_fix:
            reason = "Self-service troubleshooting was attempted but did not resolve the issue. Escalated to IT engineer for hands-on technical investigation."
        elif has_hazard:
            reason = "Physical hardware fault, safety hazard, or critical deadline detected by triage engine."
        else:
            reason = "Issue escalated to IT engineer for hands-on technical investigation."

        return {
            "action": "escalate_to_technician",
            "reason": reason,
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
    return f"This issue has been logged and escalated to an IT engineer: '{issue_description}'"


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


def get_all_tickets(status: str | None = None, search: str | None = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = "SELECT id, employee_question, decision, status, created_at, resolved_at, resolution_notes FROM tickets WHERE 1=1"
    params = []
    if status and status.lower() != "all":
        query += " AND LOWER(status) = LOWER(?)"
        params.append(status.strip())
    if search and search.strip():
        term = f"%{search.strip()}%"
        query += " AND (employee_question LIKE ? OR resolution_notes LIKE ? OR CAST(id AS TEXT) LIKE ?)"
        params.extend([term, term, term])
    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "employee_question": r[1],
            "decision": r[2],
            "status": r[3],
            "created_at": r[4],
            "resolved_at": r[5] if len(r) > 5 else None,
            "resolution_notes": r[6] if len(r) > 6 else None
        }
        for r in rows
    ]


def get_ticket_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*),
            SUM(CASE WHEN LOWER(status) = 'open' THEN 1 ELSE 0 END),
            SUM(CASE WHEN LOWER(status) = 'resolved' THEN 1 ELSE 0 END)
        FROM tickets
    """)
    row = cursor.fetchone()
    conn.close()
    total = row[0] or 0
    open_count = row[1] or 0
    resolved_count = row[2] or 0
    rate = round((resolved_count / total * 100), 1) if total > 0 else 0.0
    return {
        "total": total,
        "open_count": open_count,
        "resolved_count": resolved_count,
        "resolution_rate": rate
    }


def update_ticket_status(ticket_id: int, status: str = "resolved", notes: str | None = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, status, resolution_notes FROM tickets WHERE id = ?", (ticket_id,))
    ticket = cursor.fetchone()
    if not ticket:
        conn.close()
        return None

    norm_status = status.strip().lower()
    resolved_at = datetime.now().isoformat() if norm_status == "resolved" else None
    
    if notes is not None and notes.strip():
        resolution_notes = notes.strip()
    elif norm_status == "resolved":
        resolution_notes = ticket[2] or "Resolved by engineer"
    else:
        resolution_notes = None

    cursor.execute(
        "UPDATE tickets SET status = ?, resolved_at = ?, resolution_notes = ? WHERE id = ?",
        (norm_status, resolved_at, resolution_notes, ticket_id)
    )
    conn.commit()

    cursor.execute(
        "SELECT id, employee_question, decision, status, created_at, resolved_at, resolution_notes FROM tickets WHERE id = ?",
        (ticket_id,)
    )
    r = cursor.fetchone()
    conn.close()
    return {
        "id": r[0],
        "employee_question": r[1],
        "decision": r[2],
        "status": r[3],
        "created_at": r[4],
        "resolved_at": r[5],
        "resolution_notes": r[6]
    }


class TicketUpdateRequest(BaseModel):
    status: str | None = "resolved"
    notes: str | None = None


class AskRequest(BaseModel):
    question: str | None = None
    urgency: str | None = "normal"
    action_hint: str | None = None


@app.get("/", response_class=HTMLResponse)
@app.get("/portal", response_class=HTMLResponse)
@app.get("/support", response_class=HTMLResponse)
def portal(request: Request):
    return templates.TemplateResponse(request, "portal.html")


@app.get("/engineer", response_class=HTMLResponse)
@app.get("/engineer/", response_class=HTMLResponse)
@app.get("/engineering", response_class=HTMLResponse)
@app.get("/engineering/", response_class=HTMLResponse)
@app.get("/technician", response_class=HTMLResponse)
@app.get("/technician/", response_class=HTMLResponse)
@app.get("/technical", response_class=HTMLResponse)
@app.get("/technical/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard/", response_class=HTMLResponse)
@app.get("/tech", response_class=HTMLResponse)
@app.get("/tech/", response_class=HTMLResponse)
def engineer_dashboard(request: Request):
    return templates.TemplateResponse(request, "engineer.html")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "it-support-agent"}


@app.get("/tickets")
def tickets(status: str | None = None, search: str | None = None):
    ticket_list = get_all_tickets(status=status, search=search)
    stats = get_ticket_stats()
    return {
        "tickets": ticket_list,
        "stats": stats,
        "total": stats["total"],
        "open_count": stats["open_count"],
        "resolved_count": stats["resolved_count"]
    }


@app.post("/tickets/{ticket_id}/resolve")
def resolve_ticket_route(ticket_id: int, payload: TicketUpdateRequest | None = None):
    notes = payload.notes if payload else None
    updated = update_ticket_status(ticket_id, status="resolved", notes=notes)
    if not updated:
        return JSONResponse({"error": f"Ticket #{ticket_id} not found"}, status_code=404)
    return {"message": f"Ticket #{ticket_id} marked as resolved", "ticket": updated}


@app.post("/tickets/{ticket_id}/reopen")
def reopen_ticket_route(ticket_id: int):
    updated = update_ticket_status(ticket_id, status="open", notes=None)
    if not updated:
        return JSONResponse({"error": f"Ticket #{ticket_id} not found"}, status_code=404)
    return {"message": f"Ticket #{ticket_id} reopened", "ticket": updated}


@app.patch("/tickets/{ticket_id}")
def patch_ticket_route(ticket_id: int, payload: TicketUpdateRequest):
    status = payload.status or "resolved"
    updated = update_ticket_status(ticket_id, status=status, notes=payload.notes)
    if not updated:
        return JSONResponse({"error": f"Ticket #{ticket_id} not found"}, status_code=404)
    return {"message": f"Ticket #{ticket_id} updated to {status}", "ticket": updated}


class EscalateRequest(BaseModel):
    question: str | None = None
    reason: str | None = None


@app.post("/escalate")
def escalate_route(payload: EscalateRequest):
    raw_question = (payload.question or "").strip()
    issue = raw_question if raw_question else "Unresolved IT problem after self-fix attempt"
    reason = payload.reason or "Self-service troubleshooting was attempted but did not resolve the issue. Escalated to IT engineer."
    ticket_id = save_ticket(issue, "escalate_to_technician")
    result = escalate_to_technician(issue)
    return {
        "status": "escalated",
        "decision": {
            "action": "escalate_to_technician",
            "reason": reason,
            "steps": []
        },
        "result": result,
        "steps": [],
        "ticket_id": ticket_id
    }


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

    # Immediate escalation if action_hint is 'esc'
    if payload.action_hint == "esc":
        reason = "Self-service troubleshooting was attempted but did not resolve the issue. Escalated to IT engineer."
        ticket_id = save_ticket(issue, "escalate_to_technician")
        result = escalate_to_technician(issue)
        return {
            "decision": {
                "action": "escalate_to_technician",
                "reason": reason,
                "steps": []
            },
            "result": result,
            "steps": [],
            "ticket_id": ticket_id
        }

    # Immediate scope check for custom text queries
    if payload.action_hint is None and not is_it_support_related(issue):
        decision = {
            "action": "out_of_scope",
            "reason": "This inquiry is outside the scope of IT technical support and computer hardware/software repairs. Please ask a technical question about your computer, network, software, or peripherals.",
            "steps": []
        }
        return {
            "decision": decision,
            "result": "I am an IT Support and Systems Repair AI Agent. I can only assist with computer hardware, software, network, and technical system troubleshooting.",
            "steps": [],
            "ticket_id": None
        }

    reference, _score = search_knowledge_base_scored(issue)

    prompt = f"""You are an expert IT Systems Support & Computer Hardware/Software Repair AI Specialist.
Your purpose is to analyze, diagnose, and troubleshoot ANY and ALL IT problems (hardware, software, operating systems, networking, peripherals, accounts, cybersecurity, and developer tools).

Employee Reported Issue:
"{issue}"

Relevant IT Knowledge Base Reference:
"{reference}"

TRIAGE & ANALYSIS PROTOCOL:

1. DOMAIN SCOPE CHECK:
   - If the user query is completely UNRELATED to IT support, computers, systems, hardware, or software (e.g. food recipes, celebrity gossip, weather, sports, medical advice):
     Return action "out_of_scope" with a polite notice explaining that you only handle IT support and system repairs. Do not provide fix steps.

2. IMMEDIATE ESCALATION ("escalate_to_technician"):
   Return "escalate_to_technician" if ANY of the following are true:
   - FAILED SELF-FIX: The employee states that a suggested fix, restart, or troubleshooting steps did NOT work, or that the issue persists after attempting to resolve it.
   - Physical hardware destruction (shattered glass screen, liquid/coffee spill on electronics, broken hinges, physically broken ports)
   - Safety emergencies (smoke, burning odor, sparks, swollen/bulging battery)
   - Completely dead hardware that will not power on at all (no lights, no fan, dead motherboard/power circuit)
   - Urgent executive/client deadline within 10 minutes where immediate hands-on technician dispatch is required
   -> For "escalate_to_technician", steps must be an empty list [].

3. AUTOMATED ANALYSIS & RESOLUTION ("provide_fix"):
   For ALL other IT support issues across all IT domains (input devices, keyboard, mouse, monitors, audio/sound, battery/charging, overheating, blue screens/crashes, slow performance, disk storage, Wi-Fi, Ethernet, VPN, passwords, MFA, printers, USB/docks, email/Outlook, Teams/Zoom, developer tools):
   - Analyze the root cause of the specific IT issue reported.
   - Provide 3 to 5 clear, actionable, numbered diagnostic and troubleshooting steps tailored directly to resolving this specific problem.
   - Never give unrelated advice (e.g., never suggest VPN steps for a keyboard, mouse, audio, or printer issue).

Return ONLY valid JSON in this exact format:
{{
  "action": "provide_fix" or "escalate_to_technician" or "out_of_scope",
  "reason": "Clear explanation of the technical diagnosis or scope boundary",
  "steps": ["Step 1", "Step 2", "Step 3"]
}}"""

    try:
        response = call_with_retry(prompt)
        decision = parse_json_response(response.text)
    except Exception as e:
        logger.warning(f"AI request unavailable ({e}). Using local triage fallback.")
        decision = local_triage_fallback(issue, payload.action_hint)

    if decision.get("action") == "out_of_scope":
        return {
            "decision": decision,
            "result": "I am an IT Support and Systems Repair AI Agent. I can only assist with computer hardware, software, network, and technical system troubleshooting.",
            "steps": [],
            "ticket_id": None
        }

    # Honor self-fix if no severe physical hazard exists
    if payload.action_hint == "fix" and decision.get("action") == "escalate_to_technician":
        hazard_keywords = ["smoke", "spill", "coffee", "burned", "burning", "hazard", "fire", "won't turn on", "wont turn on", "shattered", "liquid"]
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting IT Support and Escalation Studio on {host}:{port}")
    uvicorn.run("it_support_agent:app", host=host, port=port, reload=True)
