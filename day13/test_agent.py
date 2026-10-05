import os
import pytest
from fastapi.testclient import TestClient
from it_support_agent import (
    app,
    provide_fix,
    escalate_to_technician,
    knowledge_base,
    search_knowledge_base,
    save_ticket
)

client = TestClient(app)

needs_key = pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY")


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

