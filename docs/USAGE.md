# Usage Guide: Brain-Mapping Toolkit

## Installation
See README.md for installation instructions and dependency options.

## Core Modules
- Data integration: `src/brain_mapping/core/data_loader.py`
- Visualization: `src/brain_mapping/visualization/`
- QC and reporting: `src/brain_mapping/qc/`
- Cloud integration: `src/brain_mapping/cloud/`
- Analytics: `src/brain_mapping/analytics/`

## Example Workflow
```python
from src.brain_mapping.visualization.interactive_atlas import InteractiveAtlas
atlas = InteractiveAtlas()
atlas.select_region("region_1")
atlas.visualize_selected()
```

## Advanced Features
- Real-time analysis: `src/brain_mapping/analytics/real_time_analyzer.py`
- Cloud upload: `src/brain_mapping/cloud/integration_utils.py`
- API monitoring: `src/brain_mapping/api/monitoring.py`
- Feedback integration: `src/brain_mapping/feedback/integration.py`

## Documentation
- [Project Plan](project-plan.md)
- [Test Plan](test-plan.md)
- [Cloud Deployment](cloud_deployment.md)
- [Advanced Analytics](advanced_analytics.md)
