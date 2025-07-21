"""
test_region_stats.py
Property-based and integration tests for RegionStats.
"""
import numpy as np
import pytest
from brain_mapping.visualization.region_stats import RegionStats
from hypothesis import given, strategies as st
import os


def test_compute_stats_basic():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    labels = {0: "A", 1: "B"}
    rs = RegionStats(data, labels)
    stats = rs.compute_stats()
    assert "A" in stats and "B" in stats
    assert stats["A"]["mean"] == 2.0
    assert stats["B"]["max"] == 6.0


@given(
    st.lists(
        st.lists(
            st.floats(min_value=0, max_value=100),
            min_size=3, max_size=10
        ),
        min_size=2, max_size=5
    )
)
def test_compute_stats_hypothesis(region_data):
    data = np.array(region_data)
    labels = {i: f"Region_{i}" for i in range(data.shape[0])}
    rs = RegionStats(data, labels)
    stats = rs.compute_stats()
    for label in labels.values():
        assert "mean" in stats[label]
        assert "std" in stats[label]


@pytest.mark.integration
def test_plot_stats(tmp_path):
    data = np.array([[1, 2, 3], [4, 5, 6]])
    labels = {0: "A", 1: "B"}
    rs = RegionStats(data, labels)
    stats = rs.compute_stats()
    out_path = tmp_path / "region_stats.png"
    rs.plot_stats(stats, save_path=str(out_path))
    assert os.path.exists(out_path)


def test_log_output(tmp_path):
    data = np.array([[1, 2, 3], [4, 5, 6]])
    labels = {0: "A", 1: "B"}
    rs = RegionStats(data, labels)
    stats = rs.compute_stats()
    log_path = tmp_path / "region_stats_test.log"
    with open(log_path, "w") as f:
        f.write(str(stats))
    assert os.path.exists(log_path)
