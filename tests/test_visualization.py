import numpy as np
import pytest

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import nibabel as nib
    NIB_AVAILABLE = True
except ImportError:
    NIB_AVAILABLE = False

from brain_mapping.visualization.renderer_3d import VTKVisualizer
from brain_mapping.visualization.utils import (
    axial_projection,
    coronal_projection,
    create_slice_actor,
    load_stat_map_data,
    numpy_to_vtk_image,
    sagittal_projection,
)


@pytest.mark.skipif(not VTK_AVAILABLE, reason="VTK not available")
def test_numpy_to_vtk_image():
    arr = np.ones((10, 10, 10), dtype=np.float32)
    vtk_img = numpy_to_vtk_image(arr)
    assert vtk_img is not None
    assert vtk_img.GetDimensions() == (10, 10, 10)


@pytest.mark.skipif(not VTK_AVAILABLE, reason="VTK not available")
def test_create_slice_actor():
    arr = np.ones((10, 10, 10), dtype=np.float32)
    vtk_img = numpy_to_vtk_image(arr)
    actor = create_slice_actor(vtk_img, 'z', 5)
    assert actor is not None


@pytest.mark.skipif(not VTK_AVAILABLE, reason="VTK not available")
def test_vtk_visualizer_instantiation():
    vis = VTKVisualizer()
    arr = np.ones((10, 10, 10), dtype=np.float32)
    vis.set_data(arr)
    assert hasattr(vis, 'renderer')
    assert hasattr(vis, 'render_window')


@pytest.mark.skipif(
    not MPL_AVAILABLE or not NIB_AVAILABLE,
    reason="matplotlib or nibabel not available"
)
def test_load_stat_map_data_numpy():
    arr = np.arange(27).reshape((3, 3, 3)).astype(np.float32)
    out = load_stat_map_data(arr)
    assert np.allclose(arr, out)


def test_sagittal_projection():
    arr = np.arange(27).reshape((3, 3, 3)).astype(np.float32)
    proj = sagittal_projection(arr)
    assert proj.shape == (3, 3)


def test_coronal_projection():
    arr = np.arange(27).reshape((3, 3, 3)).astype(np.float32)
    proj = coronal_projection(arr)
    assert proj.shape == (3, 3)


def test_axial_projection():
    arr = np.arange(27).reshape((3, 3, 3)).astype(np.float32)
    proj = axial_projection(arr)
    assert proj.shape == (3, 3) 