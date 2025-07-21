# Brain Mapping Toolkit

[![Build Status](https://img.shields.io/github/actions/workflow/status/hkevin01/brain-mapping/ci.yml?branch=main&label=build)](https://github.com/hkevin01/brain-mapping/actions)
[![Tests](https://img.shields.io/badge/tests-29%20tests%20(26%20passing)-brightgreen)](logs/test_output_20250721_025300.log)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](logs/test_output_20250721_025300.log)
[![BIDS Validation](https://img.shields.io/github/workflow/status/hkevin01/brain-mapping/bids-validation?label=BIDS%20Validation)](https://github.com/hkevin01/brain-mapping/actions?query=workflow%3Abids-validation)
[![Cloud Integration](https://img.shields.io/github/workflow/status/hkevin01/brain-mapping/cloud-integration?label=Cloud%20Integration)](https://github.com/hkevin01/brain-mapping/actions?query=workflow%3Acloud-integration)
[![ML Workflow](https://img.shields.io/github/workflow/status/hkevin01/brain-mapping/ml-workflow?label=ML%20Workflow)](https://github.com/hkevin01/brain-mapping/actions?query=workflow%3Aml-workflow)
[![Real-Time Analysis](https://img.shields.io/github/workflow/status/hkevin01/brain-mapping/real-time-analysis?label=Real-Time%20Analysis)](https://github.com/hkevin01/brain-mapping/actions?query=workflow%3Areal-time-analysis)
[![Multi-Modal Integration](https://img.shields.io/github/workflow/status/hkevin01/brain-mapping/multi-modal-integration?label=Multi-Modal%20Integration)](https://github.com/hkevin01/brain-mapping/actions?query=workflow%3Amulti-modal-integration)
[![Visualization](https://img.shields.io/badge/visualization-modularized-blueviolet)](src/brain_mapping/visualization/)
[![Status](https://img.shields.io/badge/status-modularizing%20visualization-blue)](docs/project-plan.md)
[![Python Version](https://img.shields.io/pypi/pyversions/brain-mapping)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://hkevin01.github.io/brain-mapping/)
[![License](https://img.shields.io/github/license/hkevin01/brain-mapping)](LICENSE)
[![Phase](https://img.shields.io/badge/phase-3%2B%20in%20progress-yellow)](docs/project-plan.md)
[![Snyk](https://img.shields.io/badge/security-Snyk-blueviolet?logo=snyk)](https://snyk.io/)
[![Zenodo](https://img.shields.io/badge/open%20data-Zenodo-4c8cbf?logo=zenodo)](https://zenodo.org/)
[![OSF](https://img.shields.io/badge/open%20data-OSF-1e90ff?logo=osf)](https://osf.io/)
[![Binder](https://img.shields.io/badge/launch-Binder-e8642a?logo=binder)](https://mybinder.org/)
[![Discord](https://img.shields.io/badge/community-Discord-5865F2?logo=discord)](https://discord.com/)
[![Prettier](https://img.shields.io/badge/code%20style-prettier-ff69b4?logo=prettier)](https://prettier.io/)

## Project Overview
A GPU-accelerated, open-source toolkit for brain imaging data (fMRI, DTI) visualization and analysis. Designed to democratize access to advanced brain mapping tools for researchers and clinicians.

---

## 🚀 Phase 3 In Progress (July 2025)

- **Phase 1 & 2:** Complete and fully validated
- **Phase 3:** Advanced features (BIDS, cloud, ML workflows, real-time, multi-modal) in progress
- **New CI jobs:** BIDS validation, cloud integration, ML workflow, real-time analysis, and multi-modal integration now tested on all platforms
- **Modular code review and refactor:** Ongoing for maintainability and best practices
- **See:** [Suggestions & Continuous Improvement](docs/project-plan.md#-suggestions--continuous-improvement)

---

> **Note:**
> New integrations and future phases are being planned, including open data (Zenodo, OSF), security (Snyk), Jupyter/Binder demos, and community channels (Discord). See the [project plan](docs/project-plan.md) for details and checkboxes.

---

## ✅ Highlights

### Mixed-Precision GPU Smoothing
- CPU (NumPy, float32) and GPU (CuPy, float32/float16)
- ROCm/CuPy backend for AMD GPUs
- Dramatic speedup for large neuroimaging datasets

### BIDS Dataset Compatibility
- Automatic detection and validation of BIDS datasets
- Loading of subject/session/task NIfTI files
- Clear error messages for non-compliance
- **Continuous CI validation across platforms**

### Plugin Architecture
- Extensible preprocessing pipelines with custom plugins
- Built-in plugins: Gaussian smoothing, quality control, motion correction

### Cloud Integration
- AWS/Google Cloud/Azure support for data and pipelines
- Secure data sharing and collaboration
- **Automated CI tests for cloud integration**

### ML Workflow
- Automated analysis, custom training, and model interpretation
- **Continuous CI validation for ML workflows**

### Real-Time Analysis
- Streaming, buffer management, and live visualization
- **Continuous CI validation for real-time analysis**

### Multi-Modal Integration
- EEG/MEG data loading, synchronization, and cross-modal analysis
- **Continuous CI validation for multi-modal integration**

### Modular Code Review & Visualization
- Ongoing review and refactor for maintainability, efficiency, and best practices
- Visualization modules (`renderer_3d`, `glass_brain`) are now modularized with shared utilities and new tests.
- **Current Focus**: Modularizing `multi_planar.py`.
- See progress in [project plan](docs/project-plan.md#-suggestions--continuous-improvement)

---

## Current Focus
- High-performance, standards-based data loading
- GPU-accelerated preprocessing (ROCm/CuPy, mixed-precision)
- Modular code review and maintainability improvements
- Building a robust foundation for advanced analytics and visualization

---

## Installation
See [requirements.txt](requirements.txt) for dependencies. For ROCm/CuPy support, follow the [CuPy ROCm install guide](https://docs.cupy.dev/en/stable/install.html#installing-cupy-for-rocm).

---

## Roadmap
- [x] Core data loading and preprocessing
- [x] GPU-accelerated smoothing (ROCm/CuPy)
- [x] BIDS dataset compatibility
- [x] Plugin architecture for extensible pipelines
- [x] Community and research lab onboarding
- [x] Modular code review and refactor (complete)
- [x] Modular visualization and glass brain utilities (complete)
- [ ] Modularizing multi-planar visualization (in progress)
- [ ] Advanced ML workflows, real-time, multi-modal integration
- [ ] Continuous CI validation for BIDS and cloud integration

---

## Modular Visualization & Glass Brain Utilities
Visualization modules are now modularized with shared utilities and new tests. Glass brain projections and overlays are easier to use and extend. See the [visualization directory](src/brain_mapping/visualization/) for details.

---

## Node.js & Frontend Tooling
This project now includes a minimal `package.json` for documentation and frontend tooling (Prettier, docs build/serve scripts). See the file for details and future expansion.

---

## Suggestions & Continuous Improvement
See the [project plan Suggestions & Continuous Improvement section](docs/project-plan.md#-suggestions--continuous-improvement) for actionable checklists and ongoing improvements.

---

## Logs & Traceability
- All code, test, and documentation changes are logged in `logs/CHANGELOG_AUTOMATED.md`.
- Test outputs are saved in the `logs/` directory for traceability.

---

## Citation & Contact
If you use this toolkit, please cite the repository and reach out with feedback or collaboration ideas!
