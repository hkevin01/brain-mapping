# Automated Feedback Integration & Reporting

This module collects user feedback and generates automated reports for continuous improvement. All feedback is logged in `logs/feedback_report.json` and summarized in periodic reports.

## Usage

```
from brain_mapping.feedback.auto_reporter import AutoReporter
reporter = AutoReporter()
reporter.collect_feedback({"user": "test", "comment": "Great tool!"})
report = reporter.generate_report()
print(report)
```

## Features
- Collects feedback from users and forms
- Logs feedback with timestamps
- Generates summary reports
- Integrates with CI for automated reporting
