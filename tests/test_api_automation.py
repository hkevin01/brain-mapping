"""
test_api_automation.py
Automated tests for REST API endpoints and monitoring integration.
"""
import requests
from brain_mapping.api.monitoring import APIMonitor
import time


def test_api_health_monitoring():
    monitor = APIMonitor()
    start = time.time()
    response = requests.get("http://localhost:8000/health", timeout=10)
    monitor.log_request(
        "/health", response.status_code, time.time() - start
    )
    logs = monitor.get_logs()
    assert response.status_code == 200
    assert logs[-1]["endpoint"] == "/health"
    assert logs[-1]["status"] == 200


def test_api_workflow_monitoring():
    monitor = APIMonitor()
    payload = {"regions": 2, "samples": 10}
    start = time.time()
    response = requests.post(
        "http://localhost:8000/run-workflow", json=payload, timeout=10
    )
    monitor.log_request(
        "/run-workflow", response.status_code, time.time() - start
    )
    logs = monitor.get_logs()
    assert response.status_code == 200
    assert logs[-1]["endpoint"] == "/run-workflow"
    assert logs[-1]["status"] == 200
