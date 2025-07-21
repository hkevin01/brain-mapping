"""
test_feedback_integration.py
Automated tests for community feedback and continuous improvement.
"""
from brain_mapping.feedback.collector import FeedbackCollector


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
