from src.brain_mapping.core.provenance import ProvenanceTracker

def test_multi_modal_provenance():
    tracker = ProvenanceTracker()
    tracker.log_event(
        "multi_modal_integration",
        {"modalities": ["EEG", "fMRI"], "params": {"sync": True}}
    )
    events = tracker.get_events()
    assert any(e["event_type"] == "multi_modal_integration" for e in events)