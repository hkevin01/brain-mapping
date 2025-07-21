"""
Test suite for API coverage and reporting.
Covers all REST endpoints, edge cases, and error scenarios.
"""
import requests
from hypothesis import given, strategies as st

API_BASE = "http://localhost:8000/api"


def test_get_status():
    resp = requests.get(f"{API_BASE}/status", timeout=5)
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_invalid_endpoint():
    resp = requests.get(f"{API_BASE}/nonexistent", timeout=5)
    assert resp.status_code == 404


def test_post_analysis():
    payload = {"data": [1, 2, 3]}
    resp = requests.post(f"{API_BASE}/analyze", json=payload, timeout=5)
    assert resp.status_code == 200
    assert "result" in resp.json()


@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_property_based_analysis(data):
    resp = requests.post(f"{API_BASE}/analyze", json={"data": data}, timeout=5)
    assert resp.status_code == 200
    assert "result" in resp.json()
