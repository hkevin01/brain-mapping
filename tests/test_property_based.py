"""
Property-based tests for core modules using Hypothesis
"""
from hypothesis import given, strategies as st
from src.brain_mapping.core.data_loader import DataLoader
from src.brain_mapping.core.preprocessor import Preprocessor

@given(st.text(min_size=1, max_size=100))
def test_data_loader_load_property(file_name):
    loader = DataLoader(data_root="/tmp")
    result = loader.load(file_name)
    assert isinstance(result["file"], str)
    assert result["file"].endswith(file_name)

@given(st.text(min_size=1, max_size=100))
def test_preprocessor_bias_correction_property(img_path):
    preproc = Preprocessor()
    result = preproc.bias_correction(img_path)
    assert isinstance(result, str)
    assert result.endswith("_bias_corrected")
