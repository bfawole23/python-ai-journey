def test_provide_fix_format():
    from it_support_agent import provide_fix
    result = provide_fix("VPN issue")
    assert isinstance(result, str)
    assert "VPN issue" in result

def test_escalate_format():
    from it_support_agent import escalate_to_technician
    result = escalate_to_technician("Laptop won't turn on")
    assert isinstance(result, str)
    assert "escalated" in result.lower()

def test_knowledge_base_not_empty():
    from it_support_agent import knowledge_base
    assert len(knowledge_base) > 0

