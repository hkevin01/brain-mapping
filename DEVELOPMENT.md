# Development Guide

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/your-org/brain-mapping-toolkit.git
cd brain-mapping-toolkit
make dev-install  # or pip install -e ".[dev]"
```

### 2. Run Tests
```bash
make test
```

### 3. Start Development
```bash
# Format code
make format

# Run linting
make lint

# Start GUI
brain-gui

# Use CLI
brain-mapper --help
```

## Installation Options

### Basic Installation
```bash
pip install brain-mapping-toolkit
```

### Development Installation
```bash
git clone https://github.com/your-org/brain-mapping-toolkit.git
cd brain-mapping-toolkit
pip install -e ".[dev]"
```

### GPU Support
```bash
pip install brain-mapping-toolkit[gpu]
```

### Full Installation (all features)
```bash
pip install brain-mapping-toolkit[all]
```

## Docker Usage

### Build Image
```bash
make docker
```

### Run Container
```bash
# Run CLI
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  brain-mapping-toolkit brain-mapper load --input /app/data/brain.nii.gz

# Run with GUI (requires X11 forwarding)
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  brain-mapping-toolkit brain-gui
```

## Architecture Overview

### Core Components

1. **Data Loading** (`src/brain_mapping/core/`)
   - `data_loader.py`: NIfTI, DICOM, and custom format support
   - `preprocessor.py`: Normalization, smoothing, registration

2. **Analysis** (`src/brain_mapping/analysis/`)
   - `statistics.py`: Statistical analysis and connectivity
   - `machine_learning.py`: ML models for classification/regression

3. **Visualization** (`src/brain_mapping/visualization/`)
   - `renderer_3d.py`: 3D brain rendering with VTK/Mayavi
   - `plots.py`: 2D statistical plots and charts

4. **GUI** (`src/brain_mapping/gui.py`)
   - PyQt6-based interface for interactive analysis

5. **CLI** (`src/brain_mapping/cli.py`)
   - Command-line interface for batch processing

### Design Patterns

- **Plugin Architecture**: Extensible analysis methods
- **Observer Pattern**: Real-time processing updates
- **Factory Pattern**: Data loader selection based on format
- **Strategy Pattern**: Configurable visualization backends

## Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-analysis-method

# Make changes
# ... edit files ...

# Test changes
make test
make lint

# Commit
git add .
git commit -m "Add new connectivity analysis method"
```

### 2. Code Quality
```bash
# Format code
black src/brain_mapping tests/
isort src/brain_mapping tests/

# Type checking
mypy src/brain_mapping

# Linting
flake8 src/brain_mapping tests/
```

### 3. Testing
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_data_loader.py

# Run with coverage
pytest --cov=brain_mapping --cov-report=html
```

### 4. Documentation
```bash
# Build docs
cd docs
make html

# Serve locally
cd _build/html
python -m http.server 8000
```

## Adding New Features

### 1. New Analysis Method

Create a new analysis class in `src/brain_mapping/analysis/`:

```python
# src/brain_mapping/analysis/my_analysis.py
class MyAnalysisMethod:
    def __init__(self, parameters=None):
        self.parameters = parameters or {}
    
    def analyze(self, data):
        """Implement your analysis here."""
        results = {"metric": 0.95}
        return results
```

Register it in `src/brain_mapping/analysis/__init__.py`:

```python
from .my_analysis import MyAnalysisMethod

ANALYSIS_METHODS = {
    "my_method": MyAnalysisMethod,
    # ... other methods
}
```

### 2. New Visualization

Create a new visualization in `src/brain_mapping/visualization/`:

```python
# src/brain_mapping/visualization/my_viz.py
def create_my_visualization(data, **kwargs):
    """Create custom visualization."""
    # Implementation here
    return figure
```

### 3. GUI Integration

Add to the GUI in `src/brain_mapping/gui.py`:

```python
# Add to analysis method combo box
self.analysis_method.addItem("My Method")

# Handle in analysis method
if method == "My Method":
    from .analysis.my_analysis import MyAnalysisMethod
    analyzer = MyAnalysisMethod()
    results = analyzer.analyze(data)
```

### 4. CLI Integration

Add to CLI in `src/brain_mapping/cli.py`:

```python
# Add to analysis method choices
analysis_parser.add_argument(
    "--method", 
    choices=["connectivity", "activation", "my_method"], 
    default="connectivity"
)

# Handle in command
if args.method == "my_method":
    from .analysis.my_analysis import MyAnalysisMethod
    analyzer = MyAnalysisMethod()
    results = analyzer.analyze(data)
```

## Performance Optimization

### GPU Acceleration

Use CuPy for GPU-accelerated operations:

```python
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    import numpy as cp
    gpu_available = False

def process_with_gpu(data):
    if gpu_available:
        gpu_data = cp.asarray(data)
        result = cp.some_operation(gpu_data)
        return cp.asnumpy(result)
    else:
        return np.some_operation(data)
```

### Memory Management

For large datasets:

```python
def process_large_data(data_path):
    # Use memory mapping
    data = np.memmap(data_path, dtype='float32', mode='r')
    
    # Process in chunks
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        process_chunk(chunk)
```

### Parallel Processing

Use multiprocessing for CPU-intensive tasks:

```python
from multiprocessing import Pool
from functools import partial

def parallel_analysis(data, n_jobs=4):
    with Pool(n_jobs) as pool:
        chunks = np.array_split(data, n_jobs)
        results = pool.map(analyze_chunk, chunks)
    return np.concatenate(results)
```

## Testing Guidelines

### Unit Tests

Test individual components:

```python
# tests/test_data_loader.py
import pytest
from brain_mapping.core.data_loader import DataLoader

def test_load_nifti():
    loader = DataLoader()
    data = loader.load_brain_data("test_data.nii.gz")
    assert data.shape == (64, 64, 64)
```

### Integration Tests

Test component interactions:

```python
# tests/test_integration.py
def test_full_pipeline():
    # Load data
    loader = DataLoader()
    data = loader.load_brain_data("test_data.nii.gz")
    
    # Preprocess
    preprocessor = Preprocessor()
    processed = preprocessor.normalize(data)
    
    # Analyze
    analyzer = StatisticalAnalyzer()
    results = analyzer.compute_connectivity(processed)
    
    assert "connectivity_matrix" in results
```

### Performance Tests

Monitor performance regressions:

```python
# tests/test_performance.py
import time

def test_analysis_performance():
    start_time = time.time()
    
    # Run analysis
    results = run_analysis(large_dataset)
    
    elapsed = time.time() - start_time
    assert elapsed < 60  # Should complete in under 1 minute
```

## Debugging

### Logging

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

def analyze_data(data):
    logger.info(f"Starting analysis on data shape: {data.shape}")
    
    try:
        results = perform_analysis(data)
        logger.info("Analysis completed successfully")
        return results
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
```

### Profiling

Profile performance bottlenecks:

```python
import cProfile
import pstats

def profile_analysis():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run analysis
    results = analyze_large_dataset()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Maximum line length: 88 characters (Black default)

### Commit Messages

Use conventional commits:

```
feat: add new connectivity analysis method
fix: resolve memory leak in preprocessing
docs: update installation instructions
test: add unit tests for data loader
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release branch
4. Run full test suite
5. Build and test package
6. Create GitHub release
7. Publish to PyPI

```bash
# Build package
python -m build

# Test upload
twine upload --repository testpypi dist/*

# Production upload
twine upload dist/*
```
