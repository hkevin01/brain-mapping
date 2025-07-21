"""
API monitoring and logging utilities.
"""
import logging
import time


class APIMonitor:
    """Monitor API usage and log requests/responses."""

    def __init__(self):
        self.logs = []

    def log_request(self, endpoint: str, status: int, response_time: float):
        entry = {
            "endpoint": endpoint,
            "status": status,
            "response_time": response_time,
            "timestamp": time.time()
        }
        self.logs.append(entry)
        logging.info("API request: %s", entry)

    def get_logs(self):
        return self.logs
