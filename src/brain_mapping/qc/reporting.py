"""
Automated QC reporting for brain mapping datasets and workflows.
Generates summary reports and integrates with preprocessing pipelines.
"""

import json


class QCReporter:
    """Generates QC reports for neuroimaging data."""
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.reports = []

    def add_metric(self, name: str, value):
        self.reports.append({"metric": name, "value": value})

    def save(self):
        with open(self.output_path, "w") as f:
            json.dump(self.reports, f, indent=2)

# Example usage:
# reporter = QCReporter("qc_report.json")
# reporter.add_metric("SNR", 42.1)
# reporter.save()
