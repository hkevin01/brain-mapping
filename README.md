# Brain Mapping Toolkit

## Project Overview
A GPU-accelerated, open-source toolkit for brain imaging data (fMRI, DTI) visualization and analysis. Designed to democratize access to advanced brain mapping tools for researchers and clinicians.

---

## 🚀 Phase 2 Progress (July 2025)

### ✅ Mixed-Precision GPU Smoothing
- **New:** Spatial smoothing (Gaussian filter) now supports:
  - CPU (NumPy, float32)
  - GPU (CuPy, float32 and float16 for mixed-precision)
  - ROCm/CuPy backend for AMD GPUs (with proper CuPy install)
- **Benefit:** Dramatic speedup for large neuroimaging datasets, with user-selectable precision for performance/quality tradeoff.

#### Usage Example
```python
from brain_mapping.core.data_loader import DataLoader
from brain_mapping.core.preprocessor import Preprocessor

loader = DataLoader()
img = loader.load("path/to/your/image.nii.gz")

# GPU + mixed-precision (float16)
preproc = Preprocessor(gpu_enabled=True, precision='float16')
smoothed_img = preproc.run_pipeline(img, pipeline='advanced')

# Save result
import nibabel as nib
nib.save(smoothed_img, "smoothed_gpu_float16.nii.gz")
```

### 🟡 BIDS Dataset Compatibility (In Progress)
- **Goal:** Robust support for BIDS-compliant datasets (the neuroimaging community standard)
- **Features:**
  - Automatic detection and validation of BIDS datasets
  - Loading of subject/session/task NIfTI files
  - Clear error messages for non-compliance
- **Status:** Implementation in progress. See [docs/project-plan.md](docs/project-plan.md) for roadmap.

---

## Current Focus
- High-performance, standards-based data loading
- GPU-accelerated preprocessing (ROCm/CuPy, mixed-precision)
- Building a robust foundation for advanced analytics and visualization

---

## Installation
See [requirements.txt](requirements.txt) for dependencies. For ROCm/CuPy support, follow the [CuPy ROCm install guide](https://docs.cupy.dev/en/stable/install.html#installing-cupy-for-rocm).

---

## Roadmap
- [x] Core data loading and preprocessing
- [x] GPU-accelerated smoothing (ROCm/CuPy)
- [ ] BIDS dataset compatibility
- [ ] Plugin architecture for extensible pipelines
- [ ] Community and research lab onboarding

---

## Citation & Contact
If you use this toolkit, please cite the repository and reach out with feedback or collaboration ideas!
