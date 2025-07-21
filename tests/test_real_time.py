from src.brain_mapping.core.provenance import ProvenanceTracker

def test_real_time_provenance():
    tracker = ProvenanceTracker()
    tracker.log_event(
        "real_time_analysis",
        {"algorithm": "TestAlgo", "params": {"window": 128}}
    )
    events = tracker.get_events()
    assert any(e["event_type"] == "real_time_analysis" for e in events)