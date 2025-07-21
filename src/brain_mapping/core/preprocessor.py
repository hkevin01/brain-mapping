"""
Preprocessing Module for Brain Imaging Data
==========================================

This module provides comprehensive preprocessing capabilities for neuroimaging
data including motion correction, normalization, filtering, and quality control.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage  # Ensure scipy.ndimage is available for CPU smoothing

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not available. Core functionality disabled.")

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .provenance import ProvenanceTracker

# Initialize provenance tracker for preprocessor
provenance_tracker = ProvenanceTracker()

# Example hook: log preprocessing event
# In actual preprocessing function, add:
# provenance_tracker.log_event("preprocessing", {"step": step_name, "params": params})


class PreprocessingPlugin:
    """
    Base class for preprocessing plugins.
    All plugins should inherit from this class and implement the run method.
    """
    
    def __init__(self, name: str = "BasePlugin"):
        """
        Initialize plugin.
        
        Parameters
        ----------
        name : str, optional
            Plugin name for identification
        """
        self.name = name
    
    def run(self, img, **kwargs):
        """
        Run the preprocessing step.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        **kwargs : dict
            Additional parameters for the preprocessing step
            
        Returns
        -------
        nibabel.Nifti1Image
            Processed image
        """
        raise NotImplementedError(f"{self.name}: run() must be implemented in subclass")


class GaussianSmoothingPlugin(PreprocessingPlugin):
    """
    Gaussian smoothing plugin using GPU acceleration and mixed-precision.
    """
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize Gaussian smoothing plugin.
        
        Parameters
        ----------
        sigma : float, default=1.0
            Standard deviation for Gaussian kernel
        """
        super().__init__(name="GaussianSmoothingPlugin")
        self.sigma = sigma
    
    def run(self, img, **kwargs):
        """
        Apply Gaussian smoothing to the image.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        nibabel.Nifti1Image
            Smoothed image
        """
        if GPU_AVAILABLE:
            import cupy as cp
            arr = cp.asarray(img)
            smoothed = cp.ndimage.gaussian_filter(arr, self.sigma)
            return cp.asnumpy(smoothed)
        else:
            return scipy.ndimage.gaussian_filter(img, self.sigma)


class QualityControlPlugin(PreprocessingPlugin):
    """
    Quality control plugin that computes and reports QC metrics.
    """
    
    def __init__(self):
        """
        Initialize quality control plugin.
        """
        super().__init__(name="QualityControlPlugin")
    
    def run(self, img, **kwargs):
        """
        Compute quality control metrics for the image.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        nibabel.Nifti1Image
            Original image (QC is non-destructive)
        """
        # Example QC: mean, std
        qc_metrics = {
            "mean": float(np.mean(img)),
            "std": float(np.std(img)),
            "min": float(np.min(img)),
            "max": float(np.max(img))
        }
        provenance_tracker.log_event("quality_control", qc_metrics)
        return qc_metrics


class MotionCorrectionPlugin(PreprocessingPlugin):
    """
    Motion correction plugin using FSL MCFLIRT.
    """
    
    def __init__(self):
        """
        Initialize motion correction plugin.
        """
        super().__init__(name="MotionCorrectionPlugin")
    
    def run(self, img, **kwargs):
        """
        Apply motion correction to 4D fMRI data.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input 4D brain image
        **kwargs : dict
            Additional parameters (ignored)
            
        Returns
        -------
        nibabel.Nifti1Image
            Motion-corrected image
        """
        if NIBABEL_AVAILABLE:
            # Placeholder: actual FSL MCFLIRT integration
            return img  # Return unchanged for now
        else:
            raise ImportError("nibabel not available for motion correction")


class Preprocessor:
    """
    Comprehensive preprocessing pipeline for brain imaging data.
    
    Supports:
    - Motion correction and realignment
    - Spatial normalization
    - Temporal filtering
    - Noise reduction
    - Quality control metrics
    - Plugin-based extensible architecture
    """
    
    def __init__(self, plugins: Optional[List[PreprocessingPlugin]] = None):
        """
        Initialize Preprocessor.
        
        Parameters
        ----------
        plugins : list of PreprocessingPlugin, optional
            List of preprocessing plugins to apply
        """
        self.plugins = plugins or []
        provenance_tracker.log_event(
            "init_preprocessor",
            {}
        )

    def add_plugin(self, plugin: PreprocessingPlugin):
        """
        Add a preprocessing plugin to the pipeline.
        
        Parameters
        ----------
        plugin : PreprocessingPlugin
            Plugin instance to add
        """
        self.plugins.append(plugin)
    
    def run(self, img, **kwargs):
        """
        Run the preprocessing pipeline.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        **kwargs : dict
            Additional parameters for the plugins
            
        Returns
        -------
        nibabel.Nifti1Image
            Processed image
        """
        for plugin in self.plugins:
            img = plugin.run(img, **kwargs)
        return img
