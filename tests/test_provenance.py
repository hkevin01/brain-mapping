"""
Unit tests for ProvenanceTracker (data provenance and audit logging)
"""
import os
import shutil
import pytest
from src.brain_mapping.core.provenance import ProvenanceTracker


def setup_module(_):
    # Clean up logs/provenance before tests
    if os.path.exists("logs/provenance"):
        shutil.rmtree("logs/provenance")


def test_log_event_and_get_events():
    tracker = ProvenanceTracker()
    tracker.log_event(
        "data_loaded",
        {"source": "bids", "file": "sub-01_T1w.nii.gz"}
    )
    tracker.log_event(
        "preprocessing",
        {"step": "bias_correction", "params": {"method": "N4"}}
    )
    events = tracker.get_events()
    assert len(events) == 2
    assert events[0]["event_type"] == "data_loaded"
    assert events[1]["event_type"] == "preprocessing"
    assert events[0]["details"]["file"] == "sub-01_T1w.nii.gz"
    assert events[1]["details"]["step"] == "bias_correction"


def teardown_module(_):
    # Clean up logs/provenance after tests
    if os.path.exists("logs/provenance"):
        shutil.rmtree("logs/provenance")
