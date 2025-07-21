"""
test_advanced_analytics.py
Tests for AdvancedAnalytics module with output logging.
"""
import numpy as np
from brain_mapping.analytics.advanced_analytics import AdvancedAnalytics
import os


def test_run_pca(tmp_path):
    data = np.random.rand(10, 5)
    analytics = AdvancedAnalytics()
    transformed, variance = analytics.run_pca(data)
    assert transformed.shape[1] == 2
    log_path = tmp_path / "advanced_analytics_test.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Variance ratio: {variance}\n")
        f.write(f"Transformed shape: {transformed.shape}\n")
    assert os.path.exists(log_path)
