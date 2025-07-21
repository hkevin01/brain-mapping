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
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize plugin.
        
        Parameters
        ----------
        name : str, optional
            Plugin name for identification
        """
        self.name = name or self.__class__.__name__
    
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
        raise NotImplementedError("Subclasses must implement run method")


class GaussianSmoothingPlugin(PreprocessingPlugin):
    """
    Gaussian smoothing plugin using GPU acceleration and mixed-precision.
    """
    
    def __init__(self, sigma: float = 1.0, use_gpu: bool = True, precision: str = 'float32'):
        """
        Initialize Gaussian smoothing plugin.
        
        Parameters
        ----------
        sigma : float, default=1.0
            Standard deviation for Gaussian kernel
        use_gpu : bool, default=True
            Whether to use GPU acceleration
        precision : str, default='float32'
            Precision to use: 'float32' or 'float16'
        """
        super().__init__("GaussianSmoothing")
        self.sigma = sigma
        self.use_gpu = use_gpu
        self.precision = precision
    
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
        # Create a temporary preprocessor to use the existing smoothing method
        temp_preproc = Preprocessor(gpu_enabled=self.use_gpu, precision=self.precision)
        return temp_preproc._spatial_smoothing(img, use_gpu=self.use_gpu, sigma=self.sigma)


class QualityControlPlugin(PreprocessingPlugin):
    """
    Quality control plugin that computes and reports QC metrics.
    """
    
    def __init__(self, save_report: bool = True, report_path: Optional[str] = None):
        """
        Initialize quality control plugin.
        
        Parameters
        ----------
        save_report : bool, default=True
            Whether to save QC report to file
        report_path : str, optional
            Path to save QC report (auto-generated if None)
        """
        super().__init__("QualityControl")
        self.save_report = save_report
        self.report_path = report_path
    
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
        try:
            # Import QC functionality
            from .quality_control import QualityControl
            
            qc = QualityControl()
            data = img.get_fdata()
            
            # Compute QC metrics
            qc_results = qc.comprehensive_qc(data, img.affine)
            
            # Print summary
            print(f"QC Results for {self.name}:")
            print(f"  Overall Quality: {qc_results['overall_quality']}")
            print(f"  SNR: {qc_results['metrics'].get('snr', 'N/A')}")
            print(f"  Warnings: {len(qc_results['warnings'])}")
            
            # Save report if requested
            if self.save_report:
                import json
                from pathlib import Path
                
                if self.report_path is None:
                    self.report_path = f"qc_report_{Path(img.get_filename()).stem}.json"
                
                with open(self.report_path, 'w') as f:
                    json.dump(qc_results, f, indent=2, default=str)
                print(f"QC report saved to: {self.report_path}")
            
        except Exception as e:
            warnings.warn(f"Quality control failed: {str(e)}")
        
        # Return original image (QC is non-destructive)
        return img


class MotionCorrectionPlugin(PreprocessingPlugin):
    """
    Motion correction plugin using FSL MCFLIRT.
    """
    
    def __init__(self, reference_volume: Optional[int] = None, save_motion_params: bool = True):
        """
        Initialize motion correction plugin.
        
        Parameters
        ----------
        reference_volume : int, optional
            Reference volume index (middle volume if None)
        save_motion_params : bool, default=True
            Whether to save motion parameters
        """
        super().__init__("MotionCorrection")
        self.reference_volume = reference_volume
        self.save_motion_params = save_motion_params
    
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
        try:
            # Import FSL integration
            from .fsl_integration import FSLIntegration
            
            fsl = FSLIntegration()
            if not fsl.fsl_available:
                warnings.warn("FSL not available. Skipping motion correction.")
                return img
            
            # Save temporary input file
            import os
            import tempfile
            from pathlib import Path
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                input_path = temp_dir / "input.nii.gz"
                output_path = temp_dir / "motion_corrected.nii.gz"
                
                # Save input image
                img.to_filename(str(input_path))
                
                # Run motion correction
                result = fsl.motion_correction(
                    input_path, 
                    output_path, 
                    reference_volume=self.reference_volume
                )
                
                if result['success']:
                    # Load corrected image
                    corrected_img = nib.load(str(output_path))
                    
                    # Save motion parameters if requested
                    if self.save_motion_params and 'motion_parameters' in result:
                        motion_file = f"motion_params_{Path(img.get_filename()).stem}.txt"
                        np.savetxt(motion_file, result['motion_parameters'])
                        print(f"Motion parameters saved to: {motion_file}")
                    
                    print(f"Motion correction completed successfully.")
                    return corrected_img
                else:
                    warnings.warn(f"Motion correction failed: {result.get('error', 'Unknown error')}")
                    return img
                    
        except Exception as e:
            warnings.warn(f"Motion correction failed: {str(e)}")
            return img


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
    
    def __init__(self, sigma: float = 1.0, use_gpu: bool = True, precision: str = 'float32'):
        """
        Initialize Preprocessor.
        
        Parameters
        ----------
        sigma : float, default=1.0
            Standard deviation for Gaussian kernel
        use_gpu : bool, default=True
            Whether to use GPU acceleration
        precision : str, default='float32'
            Precision to use: 'float32' or 'float16' (mixed-precision)
        """
        self.sigma = sigma
        self.use_gpu = use_gpu
        self.precision = precision
        provenance_tracker.log_event(
            "init_preprocessor",
            {"sigma": sigma, "use_gpu": use_gpu, "precision": precision}
        )

    def bias_correction(self, img_path):
        # Simulate bias correction
        provenance_tracker.log_event(
            "preprocessing",
            {"step": "bias_correction", "img_path": img_path}
        )
        return f"{img_path}_bias_corrected"

    def spatial_smoothing(self, img_path):
        # Simulate spatial smoothing
        provenance_tracker.log_event(
            "preprocessing",
            {
                "step": "spatial_smoothing",
                "img_path": img_path,
                "sigma": self.sigma
            }
        )
        return f"{img_path}_smoothed"
