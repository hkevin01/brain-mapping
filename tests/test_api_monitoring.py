"""
test_api_monitoring.py
Automated tests for API monitoring and logging.
"""
from brain_mapping.api.monitoring import APIMonitor
import time


def test_api_monitoring():
    monitor = APIMonitor()
    start = time.time()
    # Simulate API request
    time.sleep(0.1)
    monitor.log_request("/run-workflow", 200, time.time() - start)
    logs = monitor.get_logs()
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "/run-workflow"
    assert logs[0]["status"] == 200
