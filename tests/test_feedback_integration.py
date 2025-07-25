"""
test_feedback_integration.py
Automated tests for community feedback and continuous improvement.
"""
from brain_mapping.feedback.collector import FeedbackCollector
import requests

API_BASE = "http://localhost:8000/api"


def test_multiple_feedback():
    collector = FeedbackCollector()
    collector.submit_feedback("user1", "Feature request: add VR support.")
    collector.submit_feedback("user2", "Bug: GUI crashes on load.")
    feedback = collector.get_all_feedback()
    assert len(feedback) == 2
    assert feedback[0]["user"] == "user1"
    assert feedback[1]["user"] == "user2"
    assert "Feature request" in feedback[0]["message"]
    assert "Bug" in feedback[1]["message"]


def test_submit_feedback():
    payload = {"user": "test", "feedback": "Great tool!"}
    resp = requests.post(f"{API_BASE}/feedback", json=payload, timeout=5)
    assert resp.status_code == 200
    assert resp.json().get("status") == "received"


def test_get_feedback():
    resp = requests.get(f"{API_BASE}/feedback", timeout=5)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_invalid_feedback():
    payload = {"user": "", "feedback": ""}
    resp = requests.post(f"{API_BASE}/feedback", json=payload, timeout=5)
    assert resp.status_code == 400
