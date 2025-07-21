# Brain-Mapping Toolkit

[![CI](https://github.com/hkevin01/brain-mapping/actions/workflows/ci.yml/badge.svg)](https://github.com/hkevin01/brain-mapping/actions/workflows/ci.yml)
[![Lint](https://github.com/hkevin01/brain-mapping/actions/workflows/lint.yml/badge.svg)](https://github.com/hkevin01/brain-mapping/actions/workflows/lint.yml)
[![Test Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](logs/)
[![Docs](https://img.shields.io/badge/docs-up%20to%20date-blue)](docs/)

## Overview
Open-source toolkit for large-scale brain imaging data integration, visualization, and analysis. Features GPU acceleration, cloud integration, advanced analytics, real-time workflows, and modular plugin architecture.

## Key Features
- Interactive and modular visualization (multi-planar, glass brain, atlas)
- Automated QC metrics and reporting
- REST API endpoints and CLI for workflow execution
- Cloud deployment (AWS, GCP) and scalability
- Advanced analytics (PCA, ML workflows)
- Real-time data streaming and visualization
- Community feedback and continuous improvement
- Comprehensive test suite and coverage

## Quick Start
```bash
# Install core dependencies
pip install -r requirements.txt
# For GPU support
pip install -r requirements-gpu.txt
# For visualization
pip install -r requirements-viz.txt
```

## Usage Example
```python
from src.brain_mapping.visualization.interactive_atlas import InteractiveAtlas
atlas = InteractiveAtlas()
atlas.select_region("region_1")
atlas.visualize_selected()
```

## Documentation
- [Project Plan](docs/project-plan.md)
- [Test Plan](docs/test-plan.md)
- [Cloud Deployment](docs/cloud_deployment.md)
- [Advanced Analytics](docs/advanced_analytics.md)

## Contributing
See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License
MIT
