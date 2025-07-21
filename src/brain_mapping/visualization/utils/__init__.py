import logging
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image

logger = logging.getLogger(__name__)

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


def numpy_to_vtk_image(data: np.ndarray):
    """Convert numpy array to VTK image data."""
    if not VTK_AVAILABLE:
        return None
    vtk_data = vtk.vtkImageData()
    vtk_data.SetDimensions(data.shape)
    vtk_data.SetSpacing(1.0, 1.0, 1.0)
    vtk_data.SetOrigin(0.0, 0.0, 0.0)
    vtk_array = vtk.vtkFloatArray()
    vtk_array.SetNumberOfComponents(1)
    vtk_array.SetName("ImageData")
    flat_data = data.flatten(order='F')
    vtk_array.SetNumberOfTuples(len(flat_data))
    for i, value in enumerate(flat_data):
        vtk_array.SetValue(i, float(value))
    vtk_data.GetPointData().SetScalars(vtk_array)
    return vtk_data


def create_slice_actor(vtk_data, plane: str, slice_idx: int):
    """Create a slice plane actor for VTK visualization."""
    if not VTK_AVAILABLE or vtk_data is None:
        return None
    try:
        plane_obj = vtk.vtkPlane()
        bounds = vtk_data.GetBounds()
        if plane == 'x':
            origin = [slice_idx, (bounds[2] + bounds[3]) / 2,
                      (bounds[4] + bounds[5]) / 2]
            normal = [1, 0, 0]
        elif plane == 'y':
            origin = [(bounds[0] + bounds[1]) / 2, slice_idx,
                      (bounds[4] + bounds[5]) / 2]
            normal = [0, 1, 0]
        else:  # z
            origin = [(bounds[0] + bounds[1]) / 2,
                      (bounds[2] + bounds[3]) / 2, slice_idx]
            normal = [0, 0, 1]
        plane_obj.SetOrigin(origin)
        plane_obj.SetNormal(normal)
        cutter = vtk.vtkCutter()
        cutter.SetInputData(vtk_data)
        cutter.SetCutFunction(plane_obj)
        cutter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cutter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.8)
        return actor
    except Exception:
        return None


def load_stat_map_data(
    statistical_map: Union[str, Path, np.ndarray, Nifti1Image]
) -> np.ndarray:
    """
    Load statistical map data from file, nibabel image, or numpy array.
    """
    if isinstance(statistical_map, np.ndarray):
        return statistical_map
    elif isinstance(statistical_map, Nifti1Image):
        return statistical_map.get_fdata()
    else:
        img = nib.load(str(statistical_map))
        return img.get_fdata()


def sagittal_projection(data: np.ndarray) -> np.ndarray:
    """
    Create sagittal (left-right) maximum intensity projection.
    """
    if data.ndim == 4:
        data = np.mean(data, axis=-1)
    projection = np.max(data, axis=0)
    return np.rot90(projection)


def coronal_projection(data: np.ndarray) -> np.ndarray:
    """
    Create coronal (front-back) maximum intensity projection.
    """
    if data.ndim == 4:
        data = np.mean(data, axis=-1)
    projection = np.max(data, axis=1)
    return np.rot90(projection)


def axial_projection(data: np.ndarray) -> np.ndarray:
    """
    Create axial (top-bottom) maximum intensity projection.
    """
    if data.ndim == 4:
        data = np.mean(data, axis=-1)
    projection = np.max(data, axis=2)
    return projection 