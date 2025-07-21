"""
BIDS Dataset Loader
==================

This module provides comprehensive support for BIDS (Brain Imaging Data Structure)
datasets, including validation, loading, and metadata management.

BIDS is the community standard for organizing neuroimaging data and is widely
adopted in the neuroscience community.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import pandas as pd
import numpy as np

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not available. BIDS loading disabled.")

try:
    from bids import BIDSLayout
    PY_BIDS_AVAILABLE = True
except ImportError:
    PY_BIDS_AVAILABLE = False
    warnings.warn("pybids not available. Using basic BIDS validation.")


class BIDSDatasetLoader:
    """
    Loader for BIDS-compliant neuroimaging datasets.
    
    This class provides comprehensive support for loading and validating
    BIDS datasets, including participant metadata, session information,
    and multi-modal data.
    """
    
    def __init__(self, dataset_path: Union[str, Path], validate: bool = True):
        """
        Initialize BIDS dataset loader.
        
        Parameters
        ----------
        dataset_path : str or Path
            Path to the BIDS dataset directory
        validate : bool, default=True
            Whether to validate BIDS compliance on initialization
        """
        self.dataset_path = Path(dataset_path)
        self.layout = None
        self.participants_df = None
        self.validation_errors = []
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"BIDS dataset not found: {dataset_path}")
        
        if validate:
            self.validate_bids()
        
        if PY_BIDS_AVAILABLE:
            self._initialize_layout()
    
    def _initialize_layout(self):
        """Initialize pybids layout if available."""
        try:
            self.layout = BIDSLayout(self.dataset_path)
            print(f"✓ BIDS layout initialized for {self.dataset_path}")
        except Exception as e:
            warnings.warn(f"Failed to initialize BIDS layout: {e}")
            self.layout = None
    
    def validate_bids(self) -> bool:
        """
        Validate BIDS compliance of the dataset.
        
        Returns
        -------
        bool
            True if dataset is BIDS compliant, False otherwise
        """
        self.validation_errors = []
        
        # Check required files
        required_files = ['dataset_description.json']
        for file in required_files:
            if not (self.dataset_path / file).exists():
                self.validation_errors.append(f"Missing required file: {file}")
        
        # Check dataset_description.json
        desc_file = self.dataset_path / 'dataset_description.json'
        if desc_file.exists():
            try:
                with open(desc_file, 'r') as f:
                    desc = json.load(f)
                
                required_fields = ['Name', 'BIDSVersion']
                for field in required_fields:
                    if field not in desc:
                        self.validation_errors.append(f"Missing required field in dataset_description.json: {field}")
                
                # Check BIDS version compatibility
                bids_version = desc.get('BIDSVersion', '')
                if bids_version and not self._is_bids_version_supported(bids_version):
                    self.validation_errors.append(f"Unsupported BIDS version: {bids_version}")
                    
            except json.JSONDecodeError:
                self.validation_errors.append("Invalid JSON in dataset_description.json")
        
        # Check for participants.tsv
        participants_file = self.dataset_path / 'participants.tsv'
        if participants_file.exists():
            try:
                self.participants_df = pd.read_csv(participants_file, sep='\t')
                if 'participant_id' not in self.participants_df.columns:
                    self.validation_errors.append("participants.tsv missing required 'participant_id' column")
            except Exception as e:
                self.validation_errors.append(f"Error reading participants.tsv: {e}")
        
        # Check subject directories
        subject_dirs = [d for d in self.dataset_path.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')]
        
        if not subject_dirs:
            self.validation_errors.append("No subject directories found (should start with 'sub-')")
        
        # Validate each subject directory
        for sub_dir in subject_dirs:
            self._validate_subject_directory(sub_dir)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            print("✓ BIDS dataset validation passed")
        else:
            print(f"✗ BIDS dataset validation failed with {len(self.validation_errors)} errors:")
            for error in self.validation_errors:
                print(f"  - {error}")
        
        return is_valid
    
    def _validate_subject_directory(self, sub_dir: Path):
        """Validate a single subject directory."""
        subject_id = sub_dir.name
        
        # Check for session directories
        session_dirs = [d for d in sub_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('ses-')]
        
        if session_dirs:
            # Multi-session dataset
            for ses_dir in session_dirs:
                self._validate_session_directory(ses_dir)
        else:
            # Single session dataset
            self._validate_session_directory(sub_dir)
    
    def _validate_session_directory(self, session_dir: Path):
        """Validate a session directory."""
        # Check for modality directories
        modality_dirs = [d for d in session_dir.iterdir() 
                        if d.is_dir() and d.name in ['anat', 'func', 'dwi', 'fmap', 'meg', 'eeg']]
        
        for mod_dir in modality_dirs:
            self._validate_modality_directory(mod_dir)
    
    def _validate_modality_directory(self, mod_dir: Path):
        """Validate a modality directory."""
        # Check for valid file extensions
        valid_extensions = ['.nii.gz', '.nii', '.json', '.tsv', '.bval', '.bvec']
        
        for file in mod_dir.iterdir():
            if file.is_file():
                if not any(file.name.endswith(ext) for ext in valid_extensions):
                    self.validation_errors.append(f"Invalid file extension: {file}")
    
    def _is_bids_version_supported(self, version: str) -> bool:
        """Check if BIDS version is supported."""
        supported_versions = ['1.0.0', '1.0.1', '1.0.2', '1.1.0', '1.1.1', '1.2.0', '1.3.0', '1.4.0', '1.5.0', '1.6.0', '1.7.0']
        return version in supported_versions
    
    def get_participants(self) -> pd.DataFrame:
        """
        Get participant metadata.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing participant information
        """
        if self.participants_df is not None:
            return self.participants_df.copy()
        
        participants_file = self.dataset_path / 'participants.tsv'
        if participants_file.exists():
            self.participants_df = pd.read_csv(participants_file, sep='\t')
            return self.participants_df.copy()
        
        return pd.DataFrame()
    
    def list_subjects(self) -> List[str]:
        """
        List all subjects in the dataset.
        
        Returns
        -------
        List[str]
            List of subject IDs
        """
        if self.layout:
            return self.layout.get_subjects()
        
        # Fallback to directory scanning
        subject_dirs = [d.name for d in self.dataset_path.iterdir() 
                       if d.is_dir() and d.name.startswith('sub-')]
        return sorted(subject_dirs)
    
    def list_sessions(self, subject_id: str) -> List[str]:
        """
        List all sessions for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (with or without 'sub-' prefix)
            
        Returns
        -------
        List[str]
            List of session IDs
        """
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'
        
        if self.layout:
            return self.layout.get_sessions(subject=subject_id)
        
        # Fallback to directory scanning
        subject_dir = self.dataset_path / subject_id
        if not subject_dir.exists():
            return []
        
        session_dirs = [d.name for d in subject_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('ses-')]
        return sorted(session_dirs)
    
    def list_modalities(self, subject_id: str, session_id: Optional[str] = None) -> List[str]:
        """
        List available modalities for a subject/session.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (with or without 'sub-' prefix)
        session_id : str, optional
            Session ID (with or without 'ses-' prefix)
            
        Returns
        -------
        List[str]
            List of available modalities
        """
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'
        
        if session_id and not session_id.startswith('ses-'):
            session_id = f'ses-{session_id}'
        
        if self.layout:
            filters = {'subject': subject_id}
            if session_id:
                filters['session'] = session_id
            
            return list(set(self.layout.get_entities()['suffix']))
        
        # Fallback to directory scanning
        base_path = self.dataset_path / subject_id
        if session_id:
            base_path = base_path / session_id
        
        if not base_path.exists():
            return []
        
        modality_dirs = [d.name for d in base_path.iterdir() 
                        if d.is_dir() and d.name in ['anat', 'func', 'dwi', 'fmap', 'meg', 'eeg']]
        return sorted(modality_dirs)
    
    def load_subject(self, 
                    subject_id: str, 
                    session_id: Optional[str] = None,
                    modalities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load all data for a specific subject/session.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (with or without 'sub-' prefix)
        session_id : str, optional
            Session ID (with or without 'ses-' prefix)
        modalities : List[str], optional
            List of modalities to load. If None, loads all available.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing loaded data organized by modality
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for loading BIDS data")
        
        if not subject_id.startswith('sub-'):
            subject_id = f'sub-{subject_id}'
        
        if session_id and not session_id.startswith('ses-'):
            session_id = f'ses-{session_id}'
        
        # Get available modalities
        available_modalities = self.list_modalities(subject_id, session_id)
        if modalities:
            modalities = [mod for mod in modalities if mod in available_modalities]
        else:
            modalities = available_modalities
        
        data = {
            'subject_id': subject_id,
            'session_id': session_id,
            'modalities': {}
        }
        
        # Load data for each modality
        for modality in modalities:
            data['modalities'][modality] = self._load_modality_data(
                subject_id, session_id, modality
            )
        
        return data
    
    def _load_modality_data(self, 
                           subject_id: str, 
                           session_id: Optional[str], 
                           modality: str) -> Dict[str, Any]:
        """Load data for a specific modality."""
        if self.layout:
            return self._load_with_pybids(subject_id, session_id, modality)
        else:
            return self._load_with_directory_scan(subject_id, session_id, modality)
    
    def _load_with_pybids(self, 
                         subject_id: str, 
                         session_id: Optional[str], 
                         modality: str) -> Dict[str, Any]:
        """Load data using pybids layout."""
        filters = {'subject': subject_id, 'suffix': modality}
        if session_id:
            filters['session'] = session_id
        
        files = self.layout.get_files(**filters)
        
        data = {
            'files': files,
            'metadata': {}
        }
        
        # Load metadata for each file
        for file_path in files:
            metadata = self.layout.get_metadata(file_path)
            data['metadata'][file_path] = metadata
        
        return data
    
    def _load_with_directory_scan(self, 
                                subject_id: str, 
                                session_id: Optional[str], 
                                modality: str) -> Dict[str, Any]:
        """Load data by scanning directories."""
        base_path = self.dataset_path / subject_id
        if session_id:
            base_path = base_path / session_id
        
        modality_path = base_path / modality
        if not modality_path.exists():
            return {'files': [], 'metadata': {}}
        
        files = list(modality_path.glob('*.nii*'))
        data = {
            'files': [str(f) for f in files],
            'metadata': {}
        }
        
        # Load JSON metadata files
        for file_path in files:
            json_file = file_path.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data['metadata'][str(file_path)] = json.load(f)
                except json.JSONDecodeError:
                    data['metadata'][str(file_path)] = {}
        
        return data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing dataset metadata
        """
        info = {
            'dataset_path': str(self.dataset_path),
            'validation_errors': self.validation_errors,
            'is_valid': len(self.validation_errors) == 0
        }
        
        # Load dataset description
        desc_file = self.dataset_path / 'dataset_description.json'
        if desc_file.exists():
            try:
                with open(desc_file, 'r') as f:
                    info['description'] = json.load(f)
            except json.JSONDecodeError:
                info['description'] = {}
        
        # Get participant information
        participants_df = self.get_participants()
        if not participants_df.empty:
            info['participants'] = {
                'count': len(participants_df),
                'columns': list(participants_df.columns),
                'data': participants_df.to_dict('records')
            }
        
        # Get subject information
        subjects = self.list_subjects()
        info['subjects'] = {
            'count': len(subjects),
            'list': subjects
        }
        
        return info


class BIDSValidator:
    """
    Standalone BIDS validator for datasets.
    
    This class provides comprehensive BIDS validation without requiring
    pybids installation.
    """
    
    @staticmethod
    def validate_dataset(dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a BIDS dataset and return detailed results.
        
        Parameters
        ----------
        dataset_path : str or Path
            Path to the BIDS dataset
            
        Returns
        -------
        Dict[str, Any]
            Validation results including errors, warnings, and statistics
        """
        loader = BIDSDatasetLoader(dataset_path, validate=True)
        
        return {
            'is_valid': len(loader.validation_errors) == 0,
            'errors': loader.validation_errors,
            'dataset_info': loader.get_dataset_info()
        } 