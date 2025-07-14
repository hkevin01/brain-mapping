"""
Data Loading and Management Module
================================

This module provides comprehensive data loading capabilities for neuroimaging
formats including DICOM, NIfTI, and BIDS-compliant datasets.
"""

import os
import warnings
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import nibabel as nib

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    warnings.warn("pydicom not available. DICOM support disabled.")

class DataLoader:
    """
    Universal data loader for neuroimaging formats.
    
    Supports:
    - NIfTI files (.nii, .nii.gz)
    - DICOM files and directories
    - BIDS-compliant datasets
    - Custom data validation
    """
    
    def __init__(self, validate_data: bool = True):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        validate_data : bool, default=True
            Whether to validate loaded data integrity
        """
        self.validate_data = validate_data
        self.supported_formats = ['.nii', '.nii.gz', '.dcm']
        
    def load(self, file_path: Union[str, Path]) -> nib.Nifti1Image:
        """
        Load neuroimaging data from file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the neuroimaging file
            
        Returns
        -------
        nibabel.Nifti1Image
            Loaded neuroimaging data
            
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If file format not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Determine file type and load accordingly
        if file_path.suffix in ['.nii', '.gz']:
            return self._load_nifti(file_path)
        elif file_path.suffix == '.dcm' and DICOM_AVAILABLE:
            return self._load_dicom(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_nifti(self, file_path: Path) -> nib.Nifti1Image:
        """Load NIfTI file."""
        try:
            img = nib.load(str(file_path))
            if self.validate_data:
                self._validate_nifti(img)
            return img
        except Exception as e:
            raise ValueError(f"Error loading NIfTI file: {e}")
    
    def _load_dicom(self, file_path: Path) -> nib.Nifti1Image:
        """Load DICOM file or directory."""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM support")
            
        # Implementation for DICOM loading
        # This would require more complex DICOM to NIfTI conversion
        raise NotImplementedError("DICOM loading coming soon")
    
    def _validate_nifti(self, img: nib.Nifti1Image) -> None:
        """Validate NIfTI image integrity."""
        data = img.get_fdata()
        
        # Check for common issues
        if np.any(np.isnan(data)):
            warnings.warn("Image contains NaN values")
        
        if np.any(np.isinf(data)):
            warnings.warn("Image contains infinite values")
        
        if data.ndim < 3:
            warnings.warn("Image has less than 3 dimensions")
    
    def load_bids_dataset(self, bids_root: Union[str, Path]) -> Dict:
        """
        Load BIDS-compliant dataset.
        
        Parameters
        ----------
        bids_root : str or Path
            Root directory of BIDS dataset
            
        Returns
        -------
        dict
            Dictionary containing dataset structure and metadata
        """
        bids_root = Path(bids_root)
        
        if not bids_root.exists():
            raise FileNotFoundError(f"BIDS root not found: {bids_root}")
        
        # Basic BIDS structure discovery
        dataset = {
            'root': bids_root,
            'subjects': [],
            'sessions': {},
            'derivatives': None
        }
        
        # Find subjects
        for item in bids_root.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                subject_id = item.name
                dataset['subjects'].append(subject_id)
                dataset['sessions'][subject_id] = self._discover_sessions(item)
        
        # Check for derivatives
        derivatives_path = bids_root / 'derivatives'
        if derivatives_path.exists():
            dataset['derivatives'] = derivatives_path
        
        return dataset
    
    def _discover_sessions(self, subject_dir: Path) -> List[str]:
        """Discover sessions for a subject."""
        sessions = []
        for item in subject_dir.iterdir():
            if item.is_dir() and item.name.startswith('ses-'):
                sessions.append(item.name)
        return sessions


class BrainMapper:
    """
    High-level interface for brain mapping data operations.
    
    This class provides a simplified API for common brain mapping tasks
    including data loading, basic preprocessing, and quality control.
    """
    
    def __init__(self, gpu_enabled: bool = True):
        """
        Initialize BrainMapper.
        
        Parameters
        ----------
        gpu_enabled : bool, default=True
            Whether to enable GPU acceleration when available
        """
        self.gpu_enabled = gpu_enabled
        self.loader = DataLoader()
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """Check if GPU acceleration is available."""
        if self.gpu_enabled:
            try:
                import cupy as cp
                cp.cuda.Device(0).compute_capability
                self.gpu_available = True
            except:
                self.gpu_available = False
                warnings.warn("GPU not available, falling back to CPU")
        else:
            self.gpu_available = False
    
    def load_data(self, file_path: Union[str, Path]) -> nib.Nifti1Image:
        """
        Load neuroimaging data with automatic format detection.
        
        Parameters
        ----------
        file_path : str or Path
            Path to neuroimaging file
            
        Returns
        -------
        nibabel.Nifti1Image
            Loaded brain imaging data
        """
        return self.loader.load(file_path)
    
    def load_dataset(self, dataset_path: Union[str, Path]) -> Dict:
        """
        Load complete dataset (BIDS or custom structure).
        
        Parameters
        ----------
        dataset_path : str or Path
            Path to dataset root directory
            
        Returns
        -------
        dict
            Dataset structure and metadata
        """
        return self.loader.load_bids_dataset(dataset_path)
    
    def preprocess(self, 
                   img: nib.Nifti1Image, 
                   pipeline: str = 'standard') -> nib.Nifti1Image:
        """
        Apply preprocessing pipeline to image data.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        pipeline : str, default='standard'
            Preprocessing pipeline name
            
        Returns
        -------
        nibabel.Nifti1Image
            Preprocessed image
        """
        # Import here to avoid circular imports
        from .preprocessor import Preprocessor
        
        preprocessor = Preprocessor(gpu_enabled=self.gpu_available)
        return preprocessor.run_pipeline(img, pipeline)
    
    def get_info(self, img: nib.Nifti1Image) -> Dict:
        """
        Get comprehensive information about brain image.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Brain image to analyze
            
        Returns
        -------
        dict
            Image metadata and quality metrics
        """
        data = img.get_fdata()
        
        info = {
            'shape': data.shape,
            'voxel_size': img.header.get_zooms(),
            'data_type': str(data.dtype),
            'orientation': nib.aff2axcodes(img.affine),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'std_value': float(np.std(data)),
            'has_nan': bool(np.any(np.isnan(data))),
            'has_inf': bool(np.any(np.isinf(data))),
        }
        
        return info


# Utility functions
def validate_file_format(file_path: Union[str, Path]) -> bool:
    """
    Validate if file format is supported.
    
    Parameters
    ----------
    file_path : str or Path
        Path to file
        
    Returns
    -------
    bool
        True if format is supported
    """
    file_path = Path(file_path)
    supported_extensions = ['.nii', '.nii.gz']
    
    if DICOM_AVAILABLE:
        supported_extensions.append('.dcm')
    
    return any(str(file_path).endswith(ext) for ext in supported_extensions)


def discover_files(directory: Union[str, Path], 
                  pattern: str = "*.nii*") -> List[Path]:
    """
    Discover neuroimaging files in directory.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    pattern : str, default="*.nii*"
        File pattern to match
        
    Returns
    -------
    List[Path]
        List of discovered files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    return list(directory.rglob(pattern))
