"""
3D Visualization and Rendering Module
====================================

This module provides advanced 3D visualization capabilities for brain imaging
data using VTK and Mayavi backends.
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    warnings.warn("VTK not available. 3D visualization disabled.")

try:
    import mayavi
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    warnings.warn("Mayavi not available. Some visualization features disabled.")

import logging

from brain_mapping.visualization.utils import create_slice_actor, numpy_to_vtk_image

logger = logging.getLogger(__name__)


class BaseVisualizer:
    """
    Abstract base class for 3D visualization backends.
    """
    def __init__(self):
        self.actors = []
        self.data = None

    def set_data(self, data: np.ndarray):
        self.data = data

    def render(self):
        if self.data is not None:
            print(f"Rendering data with shape: {self.data.shape}")
        else:
            print("No data set for rendering.")

    def save(self, filename: str):
        if self.data is not None:
            print(f"Saving visualization to {filename}")
        else:
            print("No data to save.")

    def clear_scene(self):
        self.actors.clear()

    def show(self):
        if self.data is not None:
            print("Showing visualization window.")
        else:
            print("No data to show.")


class VTKVisualizer(BaseVisualizer):
    """
    VTK-based 3D visualizer for brain imaging data.
    """
    def __init__(self):
        super().__init__()
        if not VTK_AVAILABLE:
            raise ImportError("VTK not available for VTKVisualizer")
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

    def set_data(self, data: np.ndarray):
        self.data = data

    def render_brain_surface(self, threshold: Optional[float] = None,
                            color: tuple = (0.8, 0.8, 0.9),
                            opacity: float = 0.7) -> bool:
        try:
            vtk_data = numpy_to_vtk_image(self.data)
            if threshold is None:
                threshold = np.max(self.data) * 0.3
            marching_cubes = vtk.vtkMarchingCubes()
            marching_cubes.SetInputData(vtk_data)
            marching_cubes.SetValue(0, threshold)
            marching_cubes.Update()
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputConnection(marching_cubes.GetOutputPort())
            cleaner.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cleaner.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(opacity)
            self.renderer.AddActor(actor)
            self.actors.append(actor)
            return True
        except Exception as e:
            logger.error(f"Failed to render brain surface: {e}")
            return False

    def add_slice_planes(self, x_slice: Optional[int] = None,
                        y_slice: Optional[int] = None,
                        z_slice: Optional[int] = None) -> bool:
        try:
            vtk_data = numpy_to_vtk_image(self.data)
            shape = self.data.shape
            slices = {
                'x': x_slice or shape[0] // 2,
                'y': y_slice or shape[1] // 2,
                'z': z_slice or shape[2] // 2
            }
            for plane, slice_idx in slices.items():
                if slice_idx is not None:
                    actor = create_slice_actor(vtk_data, plane, slice_idx)
                    if actor:
                        self.renderer.AddActor(actor)
                        self.actors.append(actor)
            return True
        except Exception as e:
            logger.error(f"Failed to add slice planes: {e}")
            return False

    def render(self):
        self.render_window.Render()

    def show(self):
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(self.render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        self.render_window.Render()
        interactor.Start()

    def save(self, filename: str, width: int = 1920, height: int = 1080):
        self.render_window.SetSize(width, height)
        self.render_window.Render()
        screenshot = vtk.vtkWindowToImageFilter()
        screenshot.SetInput(self.render_window)
        screenshot.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(screenshot.GetOutputPort())
        writer.Write()

    def clear_scene(self):
        self.renderer.RemoveAllViewProps()
        super().clear_scene()


class MayaviVisualizer(BaseVisualizer):
    """
    Mayavi-based 3D visualizer for brain imaging data.
    """
    def __init__(self):
        super().__init__()
        if not MAYAVI_AVAILABLE:
            raise ImportError("Mayavi not available for MayaviVisualizer")
        # Placeholder for Mayavi initialization
        self.data = None

    def set_data(self, data: np.ndarray):
        self.data = data

    def render(self):
        if hasattr(self, 'mayavi') and self.mayavi:
            self.mayavi.figure(bgcolor=(0, 0, 0))
            print("Rendering with Mayavi.")
        else:
            print("Mayavi not initialized. Cannot render.")

    def show(self):
        if hasattr(self, 'mayavi') and self.mayavi:
            self.mayavi.show()
        else:
            print("Mayavi not initialized. Cannot show visualization.")

    def save(self, filename: str):
        print(f"Saving Mayavi visualization to {filename}")

    def clear_scene(self):
        if hasattr(self, 'mayavi') and self.mayavi:
            self.mayavi.clf()
            print("Scene cleared.")
        else:
            print("Mayavi not initialized. Cannot clear scene.")


class Visualizer:
    """
    3D renderer for neuroimaging data using VTK/Mayavi.
    """
    def __init__(self, backend: str = 'vtk'):
        """
        Initialize Visualizer.
        
        Parameters
        ----------
        backend : str, default='vtk'
            Rendering backend ('vtk' or 'mayavi')
        """
        self.backend = backend
        self._check_backend_availability()
        
        # Initialize rendering components
        if self.backend == 'vtk' and VTK_AVAILABLE:
            self._init_vtk()
        elif self.backend == 'mayavi' and MAYAVI_AVAILABLE:
            self._init_mayavi()
    
    def _init_vtk(self):
        """Initialize VTK rendering components."""
        if not VTK_AVAILABLE:
            warnings.warn("VTK not available, skipping VTK initialization")
            return
            
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(800, 600)
        
        # Set dark background
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        
        # Storage for actors
        self.actors = []
        
    def _init_mayavi(self):
        try:
            import mayavi.mlab
            self.mayavi = mayavi.mlab
        except ImportError:
            self.mayavi = None
            print("Mayavi not available. Visualization features limited.")

    def set_data(self, data: np.ndarray):
        self.data = data
        # Stub: integrate with VTK/Mayavi
        logging.info(f"Data set for visualization. Shape: {data.shape}")
    
    def render_brain_surface(self, image_data: np.ndarray, 
                           threshold: Optional[float] = None,
                           color: tuple = (0.8, 0.8, 0.9),
                           opacity: float = 0.7) -> bool:
        """
        Render brain surface from 3D image data.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D brain image data
        threshold : float, optional
            Intensity threshold for surface extraction
        color : tuple
            RGB color for surface
        opacity : float
            Surface opacity (0-1)
            
        Returns
        -------
        bool
            True if successful
        """
        if not VTK_AVAILABLE and self.backend == 'vtk':
            warnings.warn("VTK not available")
            return False
            
        try:
            # Convert numpy to VTK image
            vtk_data = self._numpy_to_vtk_image(image_data)
            
            # Auto-threshold if not provided
            if threshold is None:
                threshold = np.max(image_data) * 0.3
            
            # Create surface using marching cubes
            marching_cubes = vtk.vtkMarchingCubes()
            marching_cubes.SetInputData(vtk_data)
            marching_cubes.SetValue(0, threshold)
            marching_cubes.Update()
            
            # Clean the mesh
            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputConnection(marching_cubes.GetOutputPort())
            cleaner.Update()
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cleaner.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetOpacity(opacity)
            
            # Add to renderer
            self.renderer.AddActor(actor)
            self.actors.append(actor)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to render brain surface: {str(e)}")
            return False
    
    def add_slice_planes(self, image_data: np.ndarray,
                        x_slice: Optional[int] = None,
                        y_slice: Optional[int] = None,
                        z_slice: Optional[int] = None) -> bool:
        """
        Add orthogonal slice planes to visualization.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D/4D image data
        x_slice, y_slice, z_slice : int, optional
            Slice indices for each plane
            
        Returns
        -------
        bool
            True if successful
        """
        if not VTK_AVAILABLE and self.backend == 'vtk':
            return False
        
        # Handle 4D data by taking first volume
        if image_data.ndim == 4:
            image_data = image_data[..., 0]
        
        try:
            vtk_data = self._numpy_to_vtk_image(image_data)
            shape = image_data.shape
            
            # Default to center slices
            slices = {
                'x': x_slice or shape[0] // 2,
                'y': y_slice or shape[1] // 2,
                'z': z_slice or shape[2] // 2
            }
            
            for plane, slice_idx in slices.items():
                if slice_idx is not None:
                    actor = self._create_slice_actor(vtk_data, plane, slice_idx)
                    if actor:
                        self.renderer.AddActor(actor)
                        self.actors.append(actor)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to add slice planes: {str(e)}")
            return False
    
    def create_glass_brain(self, stat_map: np.ndarray,
                          background: Optional[np.ndarray] = None,
                          threshold: float = 2.0) -> bool:
        """
        Create glass brain projection visualization.
        
        Parameters
        ----------
        stat_map : np.ndarray
            Statistical map to project
        background : np.ndarray, optional
            Background brain image
        threshold : float
            Statistical threshold
            
        Returns
        -------
        bool
            True if successful
        """
        # Placeholder for glass brain implementation
        # Would typically project maximum intensity along axes
        warnings.warn("Glass brain visualization not yet implemented")
        return False
    
    def _numpy_to_vtk_image(self, data: np.ndarray):
        """Convert numpy array to VTK image data."""
        if not VTK_AVAILABLE:
            warnings.warn("VTK not available for image conversion")
            return None
            
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(data.shape)
        vtk_data.SetSpacing(1.0, 1.0, 1.0)
        vtk_data.SetOrigin(0.0, 0.0, 0.0)
        
        # Convert to VTK array
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetName("ImageData")
        
        flat_data = data.flatten(order='F')
        vtk_array.SetNumberOfTuples(len(flat_data))
        
        for i, value in enumerate(flat_data):
            vtk_array.SetValue(i, float(value))
        
        vtk_data.GetPointData().SetScalars(vtk_array)
        return vtk_data
    
    def _create_slice_actor(self, vtk_data,
                           plane: str, slice_idx: int):
        """Create slice actor for visualization."""
        if not VTK_AVAILABLE or vtk_data is None:
            warnings.warn("VTK not available for slice actor creation")
            return None
            
        try:
            # Create cutting plane
            plane_obj = vtk.vtkPlane()
            
            bounds = vtk_data.GetBounds()
            if plane == 'x':
                origin = [slice_idx, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
                normal = [1, 0, 0]
            elif plane == 'y':
                origin = [(bounds[0] + bounds[1]) / 2, slice_idx, (bounds[4] + bounds[5]) / 2]
                normal = [0, 1, 0]
            else:  # z
                origin = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, slice_idx]
                normal = [0, 0, 1]
            
            plane_obj.SetOrigin(origin)
            plane_obj.SetNormal(normal)
            
            # Create cutter
            cutter = vtk.vtkCutter()
            cutter.SetInputData(vtk_data)
            cutter.SetCutFunction(plane_obj)
            cutter.Update()
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cutter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(0.8)
            
            return actor
            
        except Exception:
            return None
    
    def render(self):
        """Render the current scene."""
        if self.backend == 'vtk' and hasattr(self, 'render_window'):
            self.render_window.Render()
    
    def start_interactive(self):
        """Start interactive visualization."""
        if self.backend == 'vtk' and hasattr(self, 'render_window'):
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(self.render_window)
            
            style = vtk.vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
            
            self.render_window.Render()
            interactor.Start()
        else:
            print("Starting interactive visualization...")
            self.show()
    
    def save_screenshot(self, filename: str, width: int = 1920, height: int = 1080):
        """Save screenshot of current view."""
        if self.backend == 'vtk' and hasattr(self, 'render_window'):
            self.render_window.SetSize(width, height)
            self.render_window.Render()
            
            screenshot = vtk.vtkWindowToImageFilter()
            screenshot.SetInput(self.render_window)
            screenshot.Update()
            
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(filename)
            writer.SetInputConnection(screenshot.GetOutputPort())
            writer.Write()
    
    def clear_scene(self):
        """Clear all actors from the scene."""
        if hasattr(self, 'renderer'):
            self.renderer.RemoveAllViewProps()
            self.actors.clear()
        if hasattr(self, 'mayavi') and self.mayavi:
            self.mayavi.clf()
            print("Scene cleared.")
        else:
            print("Mayavi not initialized. Cannot clear scene.")
        
    def _check_backend_availability(self):
        """Check if selected backend is available."""
        if self.backend == 'vtk' and not VTK_AVAILABLE:
            raise ImportError("VTK required for vtk backend")
        elif self.backend == 'mayavi' and not MAYAVI_AVAILABLE:
            raise ImportError("Mayavi required for mayavi backend")
        try:
            import vtk
            print("VTK available.")
        except ImportError:
            print("VTK not available.")
        try:
            import mayavi
            print("Mayavi available.")
        except ImportError:
            print("Mayavi not available.")
    
    def plot_brain_3d(self, 
                      data: np.ndarray, 
                      threshold: Optional[float] = None,
                      colormap: str = 'hot') -> None:
        """
        Create 3D brain visualization.
        
        Parameters
        ----------
        data : numpy.ndarray
            3D brain imaging data
        threshold : float, optional
            Statistical threshold for display
        colormap : str, default='hot'
            Colormap for visualization
        """
        if self.backend == 'vtk':
            self._plot_vtk_3d(data, threshold, colormap)
        elif self.backend == 'mayavi':
            self._plot_mayavi_3d(data, threshold, colormap)
    
    def _plot_vtk_3d(self, data: np.ndarray, threshold: Optional[float], 
                     colormap: str) -> None:
        """VTK-based 3D plotting."""
        if not VTK_AVAILABLE:
            raise ImportError("VTK not available")
        # VTK plotting implementation would go here
        print(f"VTK 3D plot: shape={data.shape}, threshold={threshold}")
    
    def _plot_mayavi_3d(self, data: np.ndarray, threshold: Optional[float],
                        colormap: str) -> None:
        """Mayavi-based 3D plotting."""
        if not MAYAVI_AVAILABLE:
            raise ImportError("Mayavi not available")
        # Mayavi plotting implementation would go here
        print(f"Mayavi 3D plot: shape={data.shape}, threshold={threshold}")
    
    def show(self) -> None:
        """Display the visualization."""
        print("Displaying visualization...")
    
    def save(self, filename: str) -> None:
        """
        Save visualization to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        print(f"Saving to {filename}")


class InteractivePlotter:
    """Interactive plotting utilities."""
    
    def __init__(self):
        self.plots = {}
    
    def create_dashboard(self, data_dict: Dict[str, Any]) -> None:
        """Create interactive dashboard."""
        print("Creating interactive dashboard...")
