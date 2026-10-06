import os
import tempfile
import pytest

# 1. Create a dedicated temporary throwaway SQLite database for the test run
_test_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_test_db_path = _test_db_file.name
_test_db_file.close()

# 2. Tell the environment and app to use the throwaway database before importing
os.environ["TICKETS_DB_PATH"] = _test_db_path

from fastapi.testclient import TestClient
from it_support_agent import (
    app,
    provide_fix,
    escalate_to_technician,
    knowledge_base,
    search_knowledge_base,
    save_ticket,
    get_all_tickets,
    get_db_path,
    set_db_path,
    init_db,
    DEFAULT_DB_PATH
)

# Explicitly initialize the throwaway database schema
set_db_path(_test_db_path)
init_db(_test_db_path)

client = TestClient(app)

needs_key = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY")


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_test_db():
    """Session fixture that deletes the throwaway test database file when all tests finish."""
    yield
    if os.path.exists(_test_db_path):
        try:
            os.remove(_test_db_path)
        except OSError:
            pass


def test_database_isolation_does_not_touch_production():
    """Verify that tests execute against the temporary throwaway database, protecting production."""
    current_path = get_db_path()
    assert current_path == _test_db_path
    assert current_path != DEFAULT_DB_PATH
    assert os.path.exists(_test_db_path)


def test_provide_fix_format():
    result = provide_fix("VPN issue")
    assert isinstance(result, str)
    assert "VPN issue" in result
    assert "VPN" in result


def test_escalate_format():
    result = escalate_to_technician("Laptop won't turn on")
    assert isinstance(result, str)
    assert "escalated" in result.lower()


def test_knowledge_base_not_empty():
    assert len(knowledge_base) > 0


@needs_key
def test_search_knowledge_base_matching():
    match = search_knowledge_base("password reset help")
    assert "password" in match.lower()


@needs_key
def test_search_knowledge_base_printer():
    match = search_knowledge_base("printer offline not printing")
    assert "printer" in match.lower()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "it-support-agent"


def test_portal_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "IT Support" in response.text or "Support" in response.text


def test_tickets_endpoint():
    response = client.get("/tickets")
    assert response.status_code == 200
    data = response.json()
    assert "tickets" in data
    assert isinstance(data["tickets"], list)


def test_save_ticket_persistence():
    save_ticket("Test monitor broken", "escalate_to_technician")
    response = client.get("/tickets")
    assert response.status_code == 200
    tickets = response.json()["tickets"]
    assert any(t["employee_question"] == "Test monitor broken" for t in tickets)


def test_ask_validation_missing_question():
    response = client.post("/ask", json={})
    assert response.status_code == 400
    assert "Missing 'question'" in response.json()["error"]


def test_ask_validation_empty_question():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["error"]


def test_ask_validation_too_long():
    response = client.post("/ask", json={"question": "x" * 501})
    assert response.status_code == 400
    assert "too long" in response.json()["error"]


def test_ask_endpoint_fix():
    response = client.post("/ask", json={"question": "VPN connection drops repeatedly"})
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"]["action"] in ["provide_fix", "escalate_to_technician"]
    assert "result" in data


def test_ask_endpoint_escalate():
    response = client.post("/ask", json={"question": "Laptop charger emitted smoke and burned the battery"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["action"] == "escalate_to_technician"
    assert data["ticket_id"] is not None


def test_engineer_endpoint():
    response = client.get("/engineer")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Engineer Command Center" in response.text


def test_technician_endpoint_alias():
    response = client.get("/technician")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Engineer Command Center" in response.text


def test_technical_endpoint_alias():
    response = client.get("/technical")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Engineer Command Center" in response.text


def test_dashboard_alias_endpoint():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Engineer Command Center" in response.text


def test_resolve_ticket_endpoint():
    ticket_id = save_ticket("Keyboard liquid spill emergency", "escalate_to_technician")
    response = client.post(f"/tickets/{ticket_id}/resolve", json={"notes": "Cleaned logic board with isopropyl alcohol"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticket"]["status"] == "resolved"
    assert data["ticket"]["resolved_at"] is not None
    assert "Cleaned logic board" in data["ticket"]["resolution_notes"]


def test_resolve_nonexistent_ticket():
    response = client.post("/tickets/999999/resolve", json={})
    assert response.status_code == 404
    assert "not found" in response.json()["error"].lower()


def test_reopen_ticket_endpoint():
    ticket_id = save_ticket("Docking station failure", "escalate_to_technician")
    client.post(f"/tickets/{ticket_id}/resolve", json={"notes": "Reset dock"})
    response = client.post(f"/tickets/{ticket_id}/reopen")
    assert response.status_code == 200
    data = response.json()
    assert data["ticket"]["status"] == "open"
    assert data["ticket"]["resolved_at"] is None


def test_patch_ticket_endpoint():
    ticket_id = save_ticket("Projector lamp blown", "escalate_to_technician")
    response = client.patch(f"/tickets/{ticket_id}", json={"status": "resolved", "notes": "Replaced bulb"})
    assert response.status_code == 200
    data = response.json()
    assert data["ticket"]["status"] == "resolved"
    assert "Replaced bulb" in data["ticket"]["resolution_notes"]


def test_tickets_filtering_and_search():
    ticket_id = save_ticket("Unique keyword X99812 testing search", "escalate_to_technician")
    response = client.get("/tickets?search=X99812")
    assert response.status_code == 200
    results = response.json()["tickets"]
    assert any("X99812" in t["employee_question"] for t in results)
    
    res_open = client.get("/tickets?status=open")
    assert res_open.status_code == 200
    assert all(t["status"] == "open" for t in res_open.json()["tickets"])

    client.post(f"/tickets/{ticket_id}/resolve", json={"notes": "Verified"})
    res_resolved = client.get("/tickets?status=resolved")
    assert res_resolved.status_code == 200
    assert all(t["status"] == "resolved" for t in res_resolved.json()["tickets"])


def test_search_knowledge_base_keyboard():
    match = search_knowledge_base("my keyboard is not working")
    assert "keyboard" in match.lower()
    assert "vpn" not in match.lower()


def test_keyboard_diagnosis_relevance():
    response = client.post("/ask", json={"question": "my keyboard is not working"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["action"] == "provide_fix"
    result_lower = data["result"].lower()
    assert "vpn" not in result_lower
    assert ("usb" in result_lower or "keyboard" in result_lower or "batter" in result_lower or "driver" in result_lower)


def test_mouse_diagnosis_relevance():
    response = client.post("/ask", json={"question": "my mouse and trackpad are not responding"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["action"] == "provide_fix"
    result_lower = data["result"].lower()
    assert "vpn" not in result_lower
    assert ("mouse" in result_lower or "trackpad" in result_lower or "sensor" in result_lower or "usb" in result_lower)


def test_bsod_diagnosis_relevance():
    response = client.post("/ask", json={"question": "computer got a blue screen BSOD crash"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["action"] == "provide_fix"
    result_lower = data["result"].lower()
    assert ("safe mode" in result_lower or "sfc" in result_lower or "stop code" in result_lower or "crash" in result_lower)


def test_battery_diagnosis_relevance():
    response = client.post("/ask", json={"question": "laptop battery not charging when plugged in"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["action"] == "provide_fix"
    result_lower = data["result"].lower()
    assert ("battery" in result_lower or "adapter" in result_lower or "power" in result_lower or "outlet" in result_lower)


def test_out_of_scope_guardrail():
    for non_it in ["how to bake a chocolate cake", "what is the capital of France", "who won the soccer match"]:
        response = client.post("/ask", json={"question": non_it})
        assert response.status_code == 200
        data = response.json()
        assert data["decision"]["action"] == "out_of_scope"
        assert data["ticket_id"] is None
        assert "IT Support" in data["result"] or "IT" in data["decision"]["reason"]


def test_selffix_failed_escalation_via_text():
    response = client.post("/ask", json={"question": "the selffix is not working for my keyboard"})
    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["action"] == "escalate_to_technician"
    assert data["ticket_id"] is not None
    assert "escalated" in data["result"].lower()


def test_escalate_endpoint():
    response = client.post("/escalate", json={
        "question": "Wi-Fi connection still dropping after network reset",
        "reason": "Employee tried self-service fix but connection still drops"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "escalated"
    assert data["decision"]["action"] == "escalate_to_technician"
    assert data["ticket_id"] is not None
    assert "escalated" in data["result"].lower()


def test_action_hint_esc_escalates():
    response = client.post("/ask", json={
        "question": "Sound distorted on laptop speakers",
        "action_hint": "esc"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["decision"]["action"] == "escalate_to_technician"
    assert data["ticket_id"] is not None


# --- Employee Authentication & Registration Tests ---

def test_login_page_renders():
    response = client.get("/login")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Employee" in response.text
    assert "AI IT Support Studio" in response.text


def test_register_page_renders():
    response = client.get("/register")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Create Account" in response.text or "Register" in response.text


def test_employee_registration_success():
    payload = {"username": "sarah_it", "password": "password123"}
    response = client.post("/api/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"
    assert data["user"]["username"] == "sarah_it"
    assert "session_token" in response.cookies


def test_employee_registration_duplicate_rejected():
    payload = {"username": "sarah_it", "password": "anotherpassword"}
    response = client.post("/api/register", json=payload)
    assert response.status_code == 400
    assert "already taken" in response.json()["error"]


def test_employee_registration_validation():
    # Empty username
    r1 = client.post("/api/register", json={"username": "", "password": "password123"})
    assert r1.status_code == 400

    # Short password
    r2 = client.post("/api/register", json={"username": "valid_user", "password": "123"})
    assert r2.status_code == 400
    assert "at least 6" in r2.json()["error"]

    # Invalid characters in username
    r3 = client.post("/api/register", json={"username": "bad user!", "password": "password123"})
    assert r3.status_code == 400


def test_employee_login_success():
    # Register first
    client.post("/api/register", json={"username": "john_dev", "password": "mysecretpassword"})
    
    # Login
    response = client.post("/api/login", json={"username": "john_dev", "password": "mysecretpassword"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["user"]["username"] == "john_dev"
    assert "session_token" in response.cookies


def test_employee_login_invalid_password():
    response = client.post("/api/login", json={"username": "john_dev", "password": "wrongpassword"})
    assert response.status_code == 401
    assert "Invalid username or password" in response.json()["error"]


def test_employee_session_me_endpoint():
    fresh_client = TestClient(app)
    # 1. Unauthenticated request has authenticated=False
    anon_resp = fresh_client.get("/api/me")
    assert anon_resp.status_code == 200
    assert anon_resp.json()["authenticated"] is False
    assert anon_resp.json()["user"] is None

    # 2. Register and log in as alex_support
    reg_resp = fresh_client.post("/api/register", json={"username": "alex_support", "password": "pass4alex!"})
    assert reg_resp.status_code == 201

    # 3. Request with session cookie has authenticated=True
    me_resp = fresh_client.get("/api/me")
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["authenticated"] is True
    assert me_data["user"]["username"] == "alex_support"


def test_employee_logout():
    user_client = TestClient(app)
    reg_resp = user_client.post("/api/register", json={"username": "logout_tester", "password": "testpassword1"})
    assert reg_resp.status_code == 201

    # Logout
    logout_resp = user_client.post("/api/logout")
    assert logout_resp.status_code == 200
    assert "logged out" in logout_resp.json()["message"].lower()

    # Subsequent check should be unauthenticated
    after_resp = user_client.get("/api/me")
    assert after_resp.json()["authenticated"] is False


def test_ticket_associates_with_logged_in_employee():
    emp_client = TestClient(app)
    reg_resp = emp_client.post("/api/register", json={"username": "emily_eng", "password": "securepass99"})
    assert reg_resp.status_code == 201

    # Escalate ticket with logged-in session client
    esc_resp = emp_client.post(
        "/escalate",
        json={"question": "Docking station triple-monitor display is flickering rapidly"}
    )
    assert esc_resp.status_code == 200
    ticket_id = esc_resp.json()["ticket_id"]

    # Verify ticket in database has employee_username
    tickets_resp = emp_client.get("/tickets")
    assert tickets_resp.status_code == 200
    tickets = tickets_resp.json()["tickets"]
    target = next((t for t in tickets if t["id"] == ticket_id), None)
    assert target is not None
    assert target["employee_username"] == "emily_eng"

