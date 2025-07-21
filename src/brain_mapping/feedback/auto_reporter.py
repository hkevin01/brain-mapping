"""
Automated feedback integration and reporting module.
"""
import json
from datetime import datetime

class AutoReporter:
    def __init__(self, log_path="logs/feedback_report.json"):
        self.log_path = log_path
        self.entries = []

    def collect_feedback(self, feedback: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "feedback": feedback
        }
        self.entries.append(entry)
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.entries, f, indent=2)

    def generate_report(self):
        return {
            "total_feedback": len(self.entries),
            "entries": self.entries
        }
