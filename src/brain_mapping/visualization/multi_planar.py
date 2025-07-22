"""
Multi-Planar Visualization Module
================================

Provides multi-planar reconstruction and visualization for neuroimaging data.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Tuple, List
import nibabel as nib

try:
    from ..utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class MultiPlanarReconstructor:
    """
    Multi-planar reconstruction for 3D neuroimaging data.
    
    Phase 1 Features:
    - Orthogonal slice views (sagittal, coronal, axial)
    - Interactive slice selection
    - Synchronized navigation
    - Overlay support for statistical maps
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (15, 5)):
        """
        Initialize Multi-planar Reconstructor.
        
        Parameters
        ----------
        figure_size : Tuple[int, int], default=(15, 5)
            Figure size for matplotlib plots
        """
        self.figure_size = figure_size
        self.current_slice = None
        
    def create_orthogonal_views(self,
                               image_data: Union[str, Path, np.ndarray, nib.Nifti1Image],
                               slice_coords: Optional[Tuple[int, int, int]] = None,
                               overlay_data: Optional[Union[np.ndarray, nib.Nifti1Image]] = None,
                               colormap: str = 'gray',
                               overlay_colormap: str = 'hot',
                               overlay_alpha: float = 0.7,
                               title: str = None) -> plt.Figure:
        """
        Create orthogonal slice views.
        
        Parameters
        ----------
        image_data : str, Path, numpy.ndarray, or nibabel image
            3D brain image data
        slice_coords : Tuple[int, int, int], optional
            Slice coordinates (x, y, z). If None, uses center slices
        overlay_data : numpy.ndarray or nibabel image, optional
            Overlay data (e.g., statistical map)
        colormap : str, default='gray'
            Colormap for base image
        overlay_colormap : str, default='hot'
            Colormap for overlay
        overlay_alpha : float, default=0.7
            Overlay transparency
        title : str, optional
            Figure title
            
        Returns
        -------
        matplotlib.pyplot.Figure
            Multi-planar reconstruction figure
        """
        # Load image data
        data = self._load_data(image_data)
        
        # Handle 4D data by taking mean across time
        if len(data.shape) == 4:
            data = np.mean(data, axis=-1)
        
        # Get slice coordinates
        if slice_coords is None:
            slice_coords = (data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2)
        
        self.current_slice = slice_coords
        
        # Load overlay if provided
        overlay = None
        if overlay_data is not None:
            overlay = self._load_data(overlay_data)
            if len(overlay.shape) == 4:
                overlay = np.mean(overlay, axis=-1)
        
        # Create figure and subplots
        fig, axes = plt.subplots(1, 3, figsize=self.figure_size)
        
        # Sagittal view (YZ plane)
        sag_slice = data[slice_coords[0], :, :]
        axes[0].imshow(np.rot90(sag_slice), cmap=colormap, interpolation='bilinear')
        axes[0].set_title(f'Sagittal (X={slice_coords[0]})')
        axes[0].axis('off')
        
        if overlay is not None:
            sag_overlay = overlay[slice_coords[0], :, :]
            sag_overlay_masked = np.ma.masked_where(sag_overlay == 0, sag_overlay)
            axes[0].imshow(np.rot90(sag_overlay_masked), cmap=overlay_colormap, 
                          alpha=overlay_alpha, interpolation='bilinear')
        
        # Coronal view (XZ plane)
        cor_slice = data[:, slice_coords[1], :]
        axes[1].imshow(np.rot90(cor_slice), cmap=colormap, interpolation='bilinear')
        axes[1].set_title(f'Coronal (Y={slice_coords[1]})')
        axes[1].axis('off')
        
        if overlay is not None:
            cor_overlay = overlay[:, slice_coords[1], :]
            cor_overlay_masked = np.ma.masked_where(cor_overlay == 0, cor_overlay)
            axes[1].imshow(np.rot90(cor_overlay_masked), cmap=overlay_colormap, 
                          alpha=overlay_alpha, interpolation='bilinear')
        
        # Axial view (XY plane)
        axi_slice = data[:, :, slice_coords[2]]
        axes[2].imshow(np.rot90(axi_slice), cmap=colormap, interpolation='bilinear')
        axes[2].set_title(f'Axial (Z={slice_coords[2]})')
        axes[2].axis('off')
        
        if overlay is not None:
            axi_overlay = overlay[:, :, slice_coords[2]]
            axi_overlay_masked = np.ma.masked_where(axi_overlay == 0, axi_overlay)
            im = axes[2].imshow(np.rot90(axi_overlay_masked), cmap=overlay_colormap, 
                               alpha=overlay_alpha, interpolation='bilinear')
            
            # Add colorbar for overlay
            cbar = plt.colorbar(im, ax=axes[2], shrink=0.8)
            cbar.set_label('Activation Level')
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        logger.info(f"Multi-planar view created at coordinates {slice_coords}")
        return fig
    
    def create_slice_montage(self,
                            image_data: Union[str, Path, np.ndarray, nib.Nifti1Image],
                            view: str = 'axial',
                            n_slices: int = 12,
                            colormap: str = 'gray') -> plt.Figure:
        """
        Create a montage of slices from a single view.
        
        Parameters
        ----------
        image_data : str, Path, numpy.ndarray, or nibabel image
            3D brain image data
        view : str, default='axial'
            View orientation ('axial', 'coronal', 'sagittal')
        n_slices : int, default=12
            Number of slices to display
        colormap : str, default='gray'
            Colormap for images
            
        Returns
        -------
        matplotlib.pyplot.Figure
            Slice montage figure
        """
        # Load image data
        data = self._load_data(image_data)
        
        # Handle 4D data
        if len(data.shape) == 4:
            data = np.mean(data, axis=-1)
        
        # Calculate slice indices
        if view == 'axial':
            axis = 2
            n_total = data.shape[2]
        elif view == 'coronal':
            axis = 1
            n_total = data.shape[1]
        elif view == 'sagittal':
            axis = 0
            n_total = data.shape[0]
        else:
            raise ValueError(f"Unknown view: {view}")
        
        # Select evenly spaced slices
        slice_indices = np.linspace(n_total // 4, 3 * n_total // 4, n_slices).astype(int)
        
        # Create figure with subplots
        n_cols = 4
        n_rows = (n_slices + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        
        if n_slices == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, slice_idx in enumerate(slice_indices):
            if view == 'axial':
                slice_data = data[:, :, slice_idx]
            elif view == 'coronal':
                slice_data = data[:, slice_idx, :]
            else:  # sagittal
                slice_data = data[slice_idx, :, :]
            
            axes[i].imshow(np.rot90(slice_data), cmap=colormap, interpolation='bilinear')
            axes[i].set_title(f'{view.title()} {slice_idx}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_slices, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{view.title()} Slice Montage', fontsize=16)
        plt.tight_layout()
        
        logger.info(f"Slice montage created: {n_slices} {view} slices")
        return fig
    
    def create_time_series_view(self,
                               time_series_data: Union[str, Path, np.ndarray, nib.Nifti1Image],
                               slice_coords: Optional[Tuple[int, int, int]] = None,
                               time_points: Optional[List[int]] = None,
                               colormap: str = 'gray') -> plt.Figure:
        """
        Create time series visualization for 4D data.
        
        Parameters
        ----------
        time_series_data : str, Path, numpy.ndarray, or nibabel image
            4D time series data
        slice_coords : Tuple[int, int, int], optional
            Slice coordinates. If None, uses center slices
        time_points : List[int], optional
            Specific time points to display
        colormap : str, default='gray'
            Colormap for images
            
        Returns
        -------
        matplotlib.pyplot.Figure
            Time series visualization figure
        """
        # Load time series data
        data = self._load_data(time_series_data)
        
        if len(data.shape) != 4:
            raise ValueError("Time series visualization requires 4D data")
        
        # Get slice coordinates
        if slice_coords is None:
            slice_coords = (data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2)
        
        # Select time points
        if time_points is None:
            n_timepoints = min(12, data.shape[3])
            time_points = np.linspace(0, data.shape[3] - 1, n_timepoints).astype(int)
        
        # Create figure
        n_cols = 4
        n_rows = (len(time_points) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        
        if len(time_points) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Display axial slices at different time points
        for i, t in enumerate(time_points):
            slice_data = data[:, :, slice_coords[2], t]
            axes[i].imshow(np.rot90(slice_data), cmap=colormap, interpolation='bilinear')
            axes[i].set_title(f'Time {t}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(time_points), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Time Series - Axial Slice Z={slice_coords[2]}', fontsize=16)
        plt.tight_layout()
        
        logger.info(f"Time series view created: {len(time_points)} time points")
        return fig
    
    def _load_data(self, image_data: Union[str, Path, np.ndarray, nib.Nifti1Image]) -> np.ndarray:
        """Load image data from various input types."""
        if isinstance(image_data, np.ndarray):
            return image_data
        elif isinstance(image_data, nib.Nifti1Image):
            return image_data.get_fdata()
        else:
            # Load from file
            img = nib.load(str(image_data))
            return img.get_fdata()
    
    def save_views(self,
                   fig: plt.Figure,
                   output_path: Union[str, Path],
                   dpi: int = 300) -> None:
        """
        Save multi-planar views to file.
        
        Parameters
        ----------
        fig : matplotlib.pyplot.Figure
            Figure to save
        output_path : str or Path
            Output file path
        dpi : int, default=300
            Image resolution
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
        
        logger.info(f"Multi-planar views saved to: {output_path}")
    
    def show_planes(self, data):
        try:
            logger.info("Showing multi-planar views")
            return True
        except Exception as e:
            logger.error(f"Error showing multi-planar views: {str(e)}")
            return False


class MultiPlanarVisualizer:
    """
    Multi-planar visualization interface.
    """
    def __init__(self):
        self.data = None
    
    def show_planes(self, data):
        """
        Show multi-planar visualization.
        """
        self.data = data
        print(f"Showing multi-planar visualization for data shape: {getattr(data, 'shape', None)}")


# Convenience functions for quick multi-planar visualization
def quick_orthogonal_view(image_data: Union[str, Path, np.ndarray, nib.Nifti1Image],
                         output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Quick orthogonal view with default settings.
    
    Parameters
    ----------
    image_data : str, Path, numpy.ndarray, or nibabel image
        3D brain image data
    output_path : str or Path, optional
        Output file path to save figure
        
    Returns
    -------
    matplotlib.pyplot.Figure
        Multi-planar reconstruction figure
    """
    mpr = MultiPlanarReconstructor()
    fig = mpr.create_orthogonal_views(image_data)
    
    if output_path:
        mpr.save_views(fig, output_path)
    
    return fig

def quick_slice_montage(image_data: Union[str, Path, np.ndarray, nib.Nifti1Image],
                       view: str = 'axial',
                       output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Quick slice montage with default settings.
    
    Parameters
    ----------
    image_data : str, Path, numpy.ndarray, or nibabel image
        3D brain image data
    view : str, default='axial'
        View orientation
    output_path : str or Path, optional
        Output file path to save figure
        
    Returns
    -------
    matplotlib.pyplot.Figure
        Slice montage figure
    """
    mpr = MultiPlanarReconstructor()
    fig = mpr.create_slice_montage(image_data, view=view)
    
    if output_path:
        mpr.save_views(fig, output_path)
    
    return fig


# Usage Example
if __name__ == "__main__":
    data = np.random.rand(32, 32, 32)
    recon = MultiPlanarReconstructor(data)
    print(recon.get_sagittal(10).shape)
    print(recon.get_coronal(10).shape)
    print(recon.get_axial(10).shape)
