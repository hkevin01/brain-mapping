"""
Property-based and edge case tests for visualization modules.
Covers interactive atlas, multi-planar, and glass brain modules.
"""
from hypothesis import given, strategies as st
import numpy as np
from visualization import interactive_atlas, multi_planar, glass_brain


def test_interactive_atlas_basic():
    atlas = interactive_atlas.BrainAtlas()
    assert atlas.get_region('Hippocampus') is not None


@given(st.text())
def test_interactive_atlas_invalid_region(region):
    atlas = interactive_atlas.BrainAtlas()
    result = atlas.get_region(region)
    assert result is None or isinstance(result, dict)


def test_multi_planar_basic():
    data = np.random.rand(64, 64, 64)
    mp = multi_planar.MultiPlanarViewer(data)
    assert mp.get_slice('axial', 32) is not None


@given(st.integers(min_value=0, max_value=63))
def test_multi_planar_slice_index(idx):
    data = np.random.rand(64, 64, 64)
    mp = multi_planar.MultiPlanarViewer(data)
    result = mp.get_slice('axial', idx)
    assert result is not None


def test_glass_brain_projection():
    data = np.random.rand(64, 64, 64)
    gb = glass_brain.GlassBrain(data)
    assert gb.project() is not None
