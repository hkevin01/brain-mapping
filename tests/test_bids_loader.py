"""
Test suite for BIDS Dataset Loader
=================================

Comprehensive tests for BIDS dataset validation, loading, and metadata management.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_mapping.core.bids_loader import BIDSDatasetLoader, BIDSValidator
from src.brain_mapping.core.provenance import ProvenanceTracker


@pytest.fixture
def temp_bids_dataset():
    """Create a temporary BIDS-compliant dataset for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create dataset_description.json
        desc = {
            "Name": "Test BIDS Dataset",
            "BIDSVersion": "1.4.0",
            "Authors": ["Test Author"],
            "HowToAcknowledge": "Cite this dataset"
        }
        with open(temp_path / "dataset_description.json", "w") as f:
            json.dump(desc, f)
        # Create participants.tsv
        participants_data = {
            "participant_id": ["sub-01", "sub-02"],
            "age": [25, 30],
            "sex": ["M", "F"]
        }
        participants_df = pd.DataFrame(participants_data)
        participants_df.to_csv(
            temp_path / "participants.tsv", sep="\t", index=False
        )
        # Create subject directories
        for sub_id in ["sub-01", "sub-02"]:
            sub_dir = temp_path / sub_id
            sub_dir.mkdir()
            # Create session directory
            ses_dir = sub_dir / "ses-01"
            ses_dir.mkdir()
            # Create modality directories
            for modality in ["anat", "func"]:
                mod_dir = ses_dir / modality
                mod_dir.mkdir()
                # Create dummy NIfTI file
                nii_file = mod_dir / f"{sub_id}_ses-01_T1w.nii.gz"
                nii_file.touch()
                # Create JSON metadata file
                json_file = mod_dir / f"{sub_id}_ses-01_T1w.json"
                metadata = {
                    "RepetitionTime": 2.0,
                    "EchoTime": 0.03,
                    "FlipAngle": 90
                }
                with open(json_file, "w") as f:
                    json.dump(metadata, f)
        yield temp_path

class TestBIDSDatasetLoader:
    """Test cases for BIDSDatasetLoader class."""
    
    def test_initialization_valid_dataset(self, temp_bids_dataset):
        """Test initialization with a valid BIDS dataset."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        assert loader.dataset_path == temp_bids_dataset
        assert len(loader.validation_errors) == 0
    
    def test_initialization_invalid_path(self):
        """Test initialization with invalid path."""
        with pytest.raises(FileNotFoundError):
            BIDSDatasetLoader("/nonexistent/path")
    
    def test_validation_missing_dataset_description(self, temp_bids_dataset):
        """Test validation when dataset_description.json is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create subject directory without dataset_description.json
            sub_dir = temp_path / "sub-01"
            sub_dir.mkdir()
            
            loader = BIDSDatasetLoader(temp_path, validate=True)
            assert len(loader.validation_errors) > 0
            assert any("dataset_description.json" in error for error in loader.validation_errors)
    
    def test_validation_invalid_json(self, temp_bids_dataset):
        """Test validation with invalid JSON in dataset_description.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid JSON file
            with open(temp_path / "dataset_description.json", "w") as f:
                f.write("invalid json content")
            
            loader = BIDSDatasetLoader(temp_path, validate=True)
            assert len(loader.validation_errors) > 0
            assert any("Invalid JSON" in error for error in loader.validation_errors)
    
    def test_validation_missing_required_fields(self, temp_bids_dataset):
        """Test validation when required fields are missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dataset_description.json with missing required fields
            desc = {"Name": "Test Dataset"}  # Missing BIDSVersion
            with open(temp_path / "dataset_description.json", "w") as f:
                json.dump(desc, f)
            
            loader = BIDSDatasetLoader(temp_path, validate=True)
            assert len(loader.validation_errors) > 0
            assert any("BIDSVersion" in error for error in loader.validation_errors)
    
    def test_validation_unsupported_bids_version(self, temp_bids_dataset):
        """Test validation with unsupported BIDS version."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dataset_description.json with unsupported version
            desc = {
                "Name": "Test Dataset",
                "BIDSVersion": "0.9.0"  # Unsupported version
            }
            with open(temp_path / "dataset_description.json", "w") as f:
                json.dump(desc, f)
            
            loader = BIDSDatasetLoader(temp_path, validate=True)
            assert len(loader.validation_errors) > 0
            assert any("Unsupported BIDS version" in error for error in loader.validation_errors)
    
    def test_get_participants(self, temp_bids_dataset):
        """Test getting participant information."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        participants_df = loader.get_participants()
        
        assert not participants_df.empty
        assert "participant_id" in participants_df.columns
        assert "age" in participants_df.columns
        assert "sex" in participants_df.columns
        assert len(participants_df) == 2
    
    def test_list_subjects(self, temp_bids_dataset):
        """Test listing subjects."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        subjects = loader.list_subjects()
        
        assert len(subjects) == 2
        assert "sub-01" in subjects
        assert "sub-02" in subjects
    
    def test_list_sessions(self, temp_bids_dataset):
        """Test listing sessions for a subject."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        sessions = loader.list_sessions("sub-01")
        
        assert len(sessions) == 1
        assert "ses-01" in sessions
    
    def test_list_sessions_without_sub_prefix(self, temp_bids_dataset):
        """Test listing sessions with subject ID without 'sub-' prefix."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        sessions = loader.list_sessions("01")
        
        assert len(sessions) == 1
        assert "ses-01" in sessions
    
    def test_list_modalities(self, temp_bids_dataset):
        """Test listing modalities for a subject/session."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        modalities = loader.list_modalities("sub-01", "ses-01")
        
        assert len(modalities) == 2
        assert "anat" in modalities
        assert "func" in modalities
    
    def test_load_subject(self, temp_bids_dataset):
        """Test loading subject data."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        data = loader.load_subject("sub-01", "ses-01")
        
        assert data["subject_id"] == "sub-01"
        assert data["session_id"] == "ses-01"
        assert "modalities" in data
        assert "anat" in data["modalities"]
        assert "func" in data["modalities"]
    
    def test_load_subject_specific_modalities(self, temp_bids_dataset):
        """Test loading specific modalities for a subject."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        data = loader.load_subject("sub-01", "ses-01", modalities=["anat"])
        
        assert "anat" in data["modalities"]
        assert "func" not in data["modalities"]
    
    def test_get_dataset_info(self, temp_bids_dataset):
        """Test getting comprehensive dataset information."""
        loader = BIDSDatasetLoader(temp_bids_dataset)
        info = loader.get_dataset_info()
        
        assert info["dataset_path"] == str(temp_bids_dataset)
        assert info["is_valid"] is True
        assert "description" in info
        assert "participants" in info
        assert "subjects" in info
        assert info["subjects"]["count"] == 2


class TestBIDSValidator:
    """Test cases for BIDSValidator class."""
    
    def test_validate_dataset_valid(self, temp_bids_dataset):
        """Test validation of a valid dataset."""
        result = BIDSValidator.validate_dataset(temp_bids_dataset)
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert "dataset_info" in result
    
    def test_validate_dataset_invalid(self):
        """Test validation of an invalid dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid dataset (missing required files)
            sub_dir = temp_path / "sub-01"
            sub_dir.mkdir()
            
            result = BIDSValidator.validate_dataset(temp_path)
            
            assert result["is_valid"] is False
            assert len(result["errors"]) > 0


class TestBIDSErrorHandling:
    """Test cases for error handling in BIDS loader."""
    
    def test_missing_nibabel_import(self):
        """Test behavior when nibabel is not available."""
        with patch.dict('sys.modules', {'nibabel': None}):
            # Re-import to trigger the import error
            import importlib
            if 'brain_mapping.core.bids_loader' in sys.modules:
                del sys.modules['brain_mapping.core.bids_loader']
            
            # Should not raise an error, but should warn
            with pytest.warns(UserWarning, match="nibabel not available"):
                from brain_mapping.core.bids_loader import BIDSDatasetLoader
    
    def test_missing_pybids_import(self):
        """Test behavior when pybids is not available."""
        with patch.dict('sys.modules', {'bids': None}):
            # Re-import to trigger the import error
            import importlib
            if 'brain_mapping.core.bids_loader' in sys.modules:
                del sys.modules['brain_mapping.core.bids_loader']
            
            # Should not raise an error, but should warn
            with pytest.warns(UserWarning, match="pybids not available"):
                from brain_mapping.core.bids_loader import BIDSDatasetLoader


class TestBIDSIntegration:
    """Integration tests for BIDS loader with real data patterns."""
    
    def test_complete_workflow(self, temp_bids_dataset):
        """Test complete workflow from validation to data loading."""
        # Validate dataset
        validator_result = BIDSValidator.validate_dataset(temp_bids_dataset)
        assert validator_result["is_valid"] is True
        
        # Initialize loader
        loader = BIDSDatasetLoader(temp_bids_dataset)
        
        # Get dataset info
        info = loader.get_dataset_info()
        assert info["subjects"]["count"] == 2
        
        # Load participant data
        participants = loader.get_participants()
        assert len(participants) == 2
        
        # Load subject data
        subject_data = loader.load_subject("sub-01", "ses-01")
        assert subject_data["subject_id"] == "sub-01"
        assert len(subject_data["modalities"]) == 2
    
    def test_multi_session_dataset(self):
        """Test handling of multi-session datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dataset_description.json
            desc = {"Name": "Multi-Session Test", "BIDSVersion": "1.4.0"}
            with open(temp_path / "dataset_description.json", "w") as f:
                json.dump(desc, f)
            
            # Create multi-session subject
            sub_dir = temp_path / "sub-01"
            sub_dir.mkdir()
            
            for session in ["ses-01", "ses-02"]:
                ses_dir = sub_dir / session
                ses_dir.mkdir()
                
                anat_dir = ses_dir / "anat"
                anat_dir.mkdir()
                (anat_dir / f"sub-01_{session}_T1w.nii.gz").touch()
            
            loader = BIDSDatasetLoader(temp_path)
            sessions = loader.list_sessions("sub-01")
            
            assert len(sessions) == 2
            assert "ses-01" in sessions
            assert "ses-02" in sessions
    
    def test_bids_loader_provenance(self):
        tracker = ProvenanceTracker()
        tracker.log_event(
            "data_loaded",
            {"source": "bids", "file": "test_bids_file.nii.gz"}
        )
        events = tracker.get_events()
        assert any(e["event_type"] == "data_loaded" for e in events)


if __name__ == "__main__":
    pytest.main([__file__])