"""
Brain Mapping Toolkit
=====================

A comprehensive, GPU-accelerated toolkit for integrating and visualizing 
large-scale brain imaging data (fMRI, DTI) in 3D.

This package provides:
- Multi-format data loading (DICOM, NIfTI, BIDS)
- GPU-accelerated preprocessing and analysis
- Interactive 3D visualization with VTK/Mayavi
- Machine learning pipelines for neuroimaging
- Cloud-based collaboration tools
- Clinical-grade security and compliance
"""

__version__ = "1.0.0"
__author__ = "Brain Mapping Toolkit Team"
__license__ = "MIT"

# Core imports
from .core.data_loader import DataLoader, BrainMapper
from .core.preprocessor import Preprocessor
from .visualization.renderer_3d import Visualizer
from .analysis.statistics import StatisticalAnalyzer
from .analysis.machine_learning import MLAnalyzer

# Main classes for quick access
__all__ = [
    "BrainMapper",
    "DataLoader", 
    "Preprocessor",
    "Visualizer",
    "StatisticalAnalyzer",
    "MLAnalyzer",
]

# Version info
def get_version():
    """Return the version string."""
    return __version__

def get_info():
    """Return package information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "GPU-accelerated brain imaging analysis toolkit"
    }
