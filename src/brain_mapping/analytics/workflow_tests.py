"""
End-to-end workflow tests for advanced analytics and cloud integration.
"""
import pytest
from brain_mapping.analytics.advanced_analytics import AdvancedAnalytics
from brain_mapping.cloud.cloud_processor import CloudProcessor


@pytest.mark.parametrize("n_components", [2, 5])
def test_pca_workflow(n_components):
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    aa = AdvancedAnalytics()
    result = aa.run_pca(data, n_components=n_components)
    assert result is not None


def test_cloud_upload():
    cp = CloudProcessor(provider="aws")
    cp.upload_dataset("/tmp/testfile.nii", "test/testfile.nii")
    # Add assertions or mock cloud upload for real tests


def test_end_to_end_workflow():
    aa = AdvancedAnalytics()
    cp = CloudProcessor(provider="aws")
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    result = aa.run_pca(data, n_components=2)
    assert result is not None
    cp.upload_dataset("/tmp/testfile.nii", "test/testfile.nii")
    assert cp.test_cloud_connection() is True
