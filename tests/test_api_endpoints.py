"""
test_api_endpoints.py
Automated tests for REST API endpoints with output logging.
"""
import requests
import os


def test_health_endpoint(tmp_path):
    response = requests.get("http://localhost:8000/health", timeout=10)
    assert response.status_code == 200
    log_path = tmp_path / "api_health_test.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Health endpoint status: {response.status_code}\n")
    assert os.path.exists(log_path)


def test_run_workflow_endpoint(tmp_path):
    payload = {"regions": 2, "samples": 10}
    response = requests.post(
        "http://localhost:8000/run-workflow", json=payload, timeout=10
    )
    assert response.status_code == 200
    log_path = tmp_path / "api_workflow_test.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Workflow endpoint status: {response.status_code}\n")
        f.write(f"Response: {response.json()}\n")
    assert os.path.exists(log_path)
