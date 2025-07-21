"""
Integrates advanced visualization and QC modules
into the main brain mapping workflow.
"""
from brain_mapping.visualization.interactive_tools import (
    InteractiveRegionSelector
)
from brain_mapping.visualization.atlas_overlay import AtlasOverlay
from brain_mapping.visualization.region_stats import RegionStats
from brain_mapping.qc.reporting import QCReporter
from brain_mapping.qc.metrics import QCMetrics
from brain_mapping.analytics.advanced_analytics import AdvancedAnalytics
import numpy as np


class BrainMappingWorkflow:
    """Main workflow integrating visualization and QC features."""
    def __init__(self, data: np.ndarray, region_labels: dict):
        self.data = data
        self.region_labels = region_labels
        self.selector = InteractiveRegionSelector()
        self.atlas_overlay = AtlasOverlay()
        self.region_stats = RegionStats(data, region_labels)
        self.qc_reporter = QCReporter()
        self.qc_metrics = QCMetrics()

    def run(self):
        # Interactive region selection
        selected = self.selector.select_regions(self.data)
        # Atlas overlay visualization
        overlay = self.atlas_overlay.overlay(self.data, selected)
        # Region statistics
        stats = self.region_stats.compute_stats()
        self.region_stats.plot_stats(stats, save_path="logs/region_stats.png")
        # QC metrics and reporting
        qc_metrics = self.qc_metrics.compute_metrics(self.data)
        self.qc_reporter.generate_report(
            qc_metrics,
            output_path="logs/qc_report.txt"
        )
        return {
            "selected_regions": selected,
            "overlay": overlay,
            "stats": stats,
            "qc_metrics": qc_metrics
        }

    def run_advanced_analytics(self):
        analytics = AdvancedAnalytics()
        pca_result, variance = analytics.run_pca(self.data)
        return pca_result, variance

    def save_results(self, results: dict, output_path: str):
        import json
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def run_cloud_upload(self, file_path: str, config_path: str):
        from brain_mapping.cloud.integration_utils import CloudUploader
        uploader = CloudUploader()
        config = uploader.load_config(config_path)
        # Example: upload to AWS S3
        aws_cfg = config.get('aws', {})
        uploader.upload_to_s3(
            file_path,
            aws_cfg.get('bucket', ''),
            'uploaded_file',
            aws_cfg.get('access_key', ''),
            aws_cfg.get('secret_key', '')
        )
