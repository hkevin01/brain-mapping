"""
test_feedback_collector.py
Automated tests for community feedback collection.
"""
from brain_mapping.feedback.collector import FeedbackCollector


def test_feedback_submission():
    collector = FeedbackCollector()
    collector.submit_feedback("user1", "Great toolkit!")
    feedback = collector.get_all_feedback()
    assert len(feedback) == 1
    assert feedback[0]["user"] == "user1"
    assert feedback[0]["message"] == "Great toolkit!"
