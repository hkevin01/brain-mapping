"""
Tests for interactive atlas, multi-planar, and glass brain visualization modules.
"""
import pytest
from src.brain_mapping.visualization.interactive_atlas import InteractiveAtlas
from src.brain_mapping.visualization.multi_planar import MultiPlanarVisualizer
from src.brain_mapping.visualization.glass_brain import GlassBrainVisualizer


def test_interactive_atlas_selection():
    atlas = InteractiveAtlas()
    atlas.select_region("region_1")
    atlas.select_region("region_2")
    assert "region_1" in atlas.selected_regions
    assert "region_2" in atlas.selected_regions
    assert atlas.visualize_selected() == ["region_1", "region_2"]


def test_multi_planar_show_planes():
    viz = MultiPlanarVisualizer()
    assert viz.show_planes(data=None) is True


def test_glass_brain_show_projection():
    viz = GlassBrainVisualizer()
    assert viz.show_projection(data=None) is True
