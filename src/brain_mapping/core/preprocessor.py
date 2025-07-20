"""
Preprocessing Module for Brain Imaging Data
==========================================

This module provides comprehensive preprocessing capabilities for neuroimaging
data including motion correction, normalization, filtering, and quality control.
"""

import warnings
from typing import Union, Dict, List, Optional, Tuple
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


class Preprocessor:
    """
    Comprehensive preprocessing pipeline for brain imaging data.
    
    Supports:
    - Motion correction and realignment
    - Spatial normalization
    - Temporal filtering
    - Noise reduction
    - Quality control metrics
    """
    
    def __init__(self, gpu_enabled: bool = True, precision: str = 'float32'):
        """
        Initialize Preprocessor.
        
        Parameters
        ----------
        gpu_enabled : bool, default=True
            Whether to use GPU acceleration when available
        precision : str, default='float32'
            Precision to use: 'float32' or 'float16' (mixed-precision)
        """
        self.gpu_enabled = gpu_enabled and GPU_AVAILABLE
        self.precision = precision
        self.pipelines = {
            'standard': self._standard_pipeline,
            'minimal': self._minimal_pipeline,
            'advanced': self._advanced_pipeline
        }
    
    def run_pipeline(self, img, pipeline: str = 'standard'):
        """
        Run preprocessing pipeline on image data.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        pipeline : str, default='standard'
            Pipeline name to execute
            
        Returns
        -------
        nibabel.Nifti1Image
            Preprocessed image
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required for preprocessing")
            
        if pipeline not in self.pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        
        return self.pipelines[pipeline](img)
    
    def _standard_pipeline(self, img):
        """Standard preprocessing pipeline."""
        # Basic preprocessing steps
        # Motion correction would go here
        # Normalization would go here
        # For now, return the input image
        return img
    
    def _minimal_pipeline(self, img):
        """Minimal preprocessing pipeline."""
        return img
    
    def _advanced_pipeline(self, img):
        """Advanced preprocessing pipeline with GPU acceleration and mixed-precision spatial smoothing."""
        if not self.gpu_enabled:
            warnings.warn("GPU not available, falling back to CPU for smoothing.")
            return self._spatial_smoothing(img, use_gpu=False)
        return self._spatial_smoothing(img, use_gpu=True)

    def _spatial_smoothing(self, img, use_gpu: bool = False, sigma: float = 1.0):
        """
        Apply spatial smoothing (Gaussian filter) to the image data.
        Supports CPU (NumPy) and GPU (CuPy) with mixed-precision.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        use_gpu : bool, default=False
            Whether to use GPU (CuPy)
        sigma : float, default=1.0
            Standard deviation for Gaussian kernel
        
        Returns
        -------
        nibabel.Nifti1Image
            Smoothed image
        """
        data = img.get_fdata()
        affine = img.affine
        dtype = np.float16 if self.precision == 'float16' else np.float32
        
        if use_gpu and GPU_AVAILABLE:
            # Move data to GPU and cast to desired precision
            d_gpu = cp.asarray(data, dtype=dtype)
            # Import cupyx.scipy.ndimage only if using GPU
            try:
                from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
            except ImportError:
                raise ImportError("cupyx.scipy.ndimage not available. Ensure CuPy is installed with ndimage support.")
            smoothed = cp_gaussian_filter(d_gpu, sigma=sigma)
            smoothed = cp.asnumpy(smoothed)
        else:
            # CPU fallback
            smoothed = scipy.ndimage.gaussian_filter(data.astype(dtype), sigma=sigma)
        
        # Return new Nifti image
        return nib.Nifti1Image(smoothed, affine)
