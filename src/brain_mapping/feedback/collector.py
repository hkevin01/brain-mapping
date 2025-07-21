"""
Community feedback collection utilities.
"""
import logging


class FeedbackCollector:
    def __init__(self):
        self.feedback = []

    def submit_feedback(self, user: str, message: str):
        entry = {"user": user, "message": message}
        self.feedback.append(entry)
        logging.info("Feedback submitted: %s", entry)

    def get_all_feedback(self):
        return self.feedback
