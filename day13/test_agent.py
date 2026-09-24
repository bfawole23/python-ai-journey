import pytest
from it_support_agent import (
    app,
    provide_fix,
    escalate_to_technician,
    knowledge_base,
    search_knowledge_base,
    save_ticket
)

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

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

def test_search_knowledge_base_matching():
    match = search_knowledge_base("password reset help")
    assert "password" in match.lower()

def test_search_knowledge_base_printer():
    match = search_knowledge_base("printer offline not printing")
    assert "printer" in match.lower()

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "it-support-agent"

def test_tickets_endpoint(client):
    response = client.get("/tickets")
    assert response.status_code == 200
    data = response.get_json()
    assert "tickets" in data
    assert isinstance(data["tickets"], list)

def test_save_ticket_persistence(client):
    save_ticket("Test monitor broken", "escalate_to_technician")
    response = client.get("/tickets")
    assert response.status_code == 200
    tickets = response.get_json()["tickets"]
    assert any(t["employee_question"] == "Test monitor broken" for t in tickets)

def test_ask_validation_missing_question(client):
    response = client.post("/ask", json={})
    assert response.status_code == 400
    assert "Missing 'question'" in response.get_json()["error"]

def test_ask_validation_empty_question(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.get_json()["error"]

def test_ask_validation_too_long(client):
    response = client.post("/ask", json={"question": "x" * 501})
    assert response.status_code == 400
    assert "too long" in response.get_json()["error"]

