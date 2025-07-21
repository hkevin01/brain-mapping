"""
Glass Brain Projection Module
============================

This module provides 2D glass brain projections for neuroimaging data
as part of Phase 1 visualization capabilities.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from brain_mapping.visualization.utils import (
    axial_projection,
    coronal_projection,
    load_stat_map_data,
    sagittal_projection,
)

from ..utils.logging import get_logger

logger = get_logger(__name__)


class GlassBrainProjector:
    """
    Creates glass brain projections for statistical maps and activations.
    
    Phase 1 Features:
    - Maximum intensity projections
    - Statistical overlay visualization
    - Customizable color maps
    - Multi-view projections (sagittal, coronal, axial)
    """
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize Glass Brain Projector.
        
        Parameters
        ----------
        figure_size : Tuple[int, int], default=(12, 8)
            Figure size for matplotlib plots
        """
        self.figure_size = figure_size
        self.default_colormap = 'hot'
        self.background_color = 'black'
        
    def create_projection(self,
                         statistical_map: Union[str, Path, np.ndarray, nib.Nifti1Image],
                         threshold: float = 0.0,
                         views: List[str] = None,
                         colormap: str = None,
                         title: str = None) -> plt.Figure:
        """
        Create glass brain projection.
        
        Parameters
        ----------
        statistical_map : str, Path, numpy.ndarray, or nibabel image
            Statistical map to project
        threshold : float, default=0.0
            Threshold for displaying activations
        views : List[str], optional
            Views to include ('sagittal', 'coronal', 'axial')
            Default: all three views
        colormap : str, optional
            Matplotlib colormap name
        title : str, optional
            Plot title
            
        Returns
        -------
        matplotlib.pyplot.Figure
            Glass brain projection figure
        """
        if views is None:
            views = ['sagittal', 'coronal', 'axial']
        
        if colormap is None:
            colormap = self.default_colormap
        
        # Load data
        data = load_stat_map_data(statistical_map)
        
        # Apply threshold
        if threshold > 0:
            data = np.where(np.abs(data) >= threshold, data, 0)
        
        # Create figure
        fig = plt.figure(figsize=self.figure_size, facecolor=self.background_color)
        
        # Create subplots for each view
        n_views = len(views)
        
        for i, view in enumerate(views):
            ax = fig.add_subplot(1, n_views, i + 1)
            
            if view == 'sagittal':
                projection = sagittal_projection(data)
                ax.set_title('Sagittal View')
            elif view == 'coronal':
                projection = coronal_projection(data)
                ax.set_title('Coronal View')
            elif view == 'axial':
                projection = axial_projection(data)
                ax.set_title('Axial View')
            else:
                logger.warning(f"Unknown view: {view}")
                continue
            
            # Display projection
            im = ax.imshow(projection, cmap=colormap, interpolation='bilinear')
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add colorbar for the last subplot
            if i == n_views - 1:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Activation Level', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                cbar.ax.yaxis.label.set_color('white')
        
        if title:
            fig.suptitle(title, color='white', fontsize=16)
        
        plt.tight_layout()
        
        logger.info(f"Glass brain projection created with {len(views)} views")
        return fig
    
    def create_comparison_plot(self,
                              maps: Dict[str, Union[str, Path, np.ndarray, nib.Nifti1Image]],
                              threshold: float = 0.0,
                              view: str = 'axial') -> plt.Figure:
        """
        Create comparison plot of multiple statistical maps.
        
        Parameters
        ----------
        maps : Dict[str, statistical_map]
            Dictionary mapping labels to statistical maps
        threshold : float, default=0.0
            Threshold for displaying activations
        view : str, default='axial'
            View to display ('sagittal', 'coronal', 'axial')
            
        Returns
        -------
        matplotlib.pyplot.Figure
            Comparison figure
        """
        n_maps = len(maps)
        fig, axes = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4),
                                facecolor=self.background_color)
        
        if n_maps == 1:
            axes = [axes]
        
        for i, (label, statistical_map) in enumerate(maps.items()):
            data = load_stat_map_data(statistical_map)
            
            # Apply threshold
            if threshold > 0:
                data = np.where(np.abs(data) >= threshold, data, 0)
            
            # Create projection
            if view == 'sagittal':
                projection = sagittal_projection(data)
            elif view == 'coronal':
                projection = coronal_projection(data)
            else:  # axial
                projection = axial_projection(data)
            
            # Display
            im = axes[i].imshow(projection, cmap=self.default_colormap, interpolation='bilinear')
            axes[i].set_title(label, color='white')
            axes[i].set_aspect('equal')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        logger.info(f"Comparison plot created with {n_maps} maps")
        return fig
    
    def save_projection(self,
                       fig: plt.Figure,
                       output_path: Union[str, Path],
                       dpi: int = 300) -> None:
        """
        Save glass brain projection to file.
        
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
        
        fig.savefig(str(output_path), dpi=dpi, facecolor=self.background_color,
                   edgecolor='none', bbox_inches='tight')
        
        logger.info(f"Glass brain projection saved to: {output_path}")
    
    def show_projection(self, data):
        try:
            logger.info("Showing glass brain projection")
            return True
        except Exception as e:
            logger.error(f"Error showing glass brain projection: {str(e)}")
            return False


class GlassBrainVisualizer:
    def __init__(self):
        self.data = None
    def show_projection(self, data):
        self.data = data
        print(f"Showing glass brain projection for data shape: {getattr(data, 'shape', None)}")


class InteractiveBrainAtlas:
    """
    Interactive brain atlas for anatomical reference.
    
    Phase 1 Features:
    - Load standard brain atlases
    - Interactive region selection
    - Anatomical labeling
    - Coordinate transformation
    """
    
    def __init__(self, atlas_name: str = 'MNI152'):
        """
        Initialize Interactive Brain Atlas.
        
        Parameters
        ----------
        atlas_name : str, default='MNI152'
            Name of the brain atlas to use
        """
        self.atlas_name = atlas_name
        self.atlas_data = None
        self.labels = None
        self._load_atlas()
    
    def _load_atlas(self):
        """Load brain atlas data."""
        # This is a placeholder - in a real implementation,
        # you would load actual atlas data from standard locations
        logger.info(f"Loading {self.atlas_name} atlas...")
        
        # For Phase 1, create a simple synthetic atlas
        self.atlas_data = np.zeros((91, 109, 91))  # MNI152 dimensions
        self.labels = {
            0: 'Background',
            1: 'Frontal Cortex',
            2: 'Parietal Cortex',
            3: 'Temporal Cortex',
            4: 'Occipital Cortex'
        }
        
        logger.info(f"Atlas {self.atlas_name} loaded with {len(self.labels)} regions")
    
    def get_region_mask(self, region_id: int) -> np.ndarray:
        """
        Get binary mask for specific atlas region.
        
        Parameters
        ----------
        region_id : int
            Region identifier
            
        Returns
        -------
        numpy.ndarray
            Binary mask for the region
        """
        return self.atlas_data == region_id
    
    def get_region_name(self, region_id: int) -> str:
        """
        Get region name from ID.
        
        Parameters
        ----------
        region_id : int
            Region identifier
            
        Returns
        -------
        str
            Region name
        """
        return self.labels.get(region_id, f"Unknown Region {region_id}")
    
    def extract_region_values(self,
                             statistical_map: Union[str, Path, np.ndarray, nib.Nifti1Image],
                             region_id: int) -> np.ndarray:
        """
        Extract values from statistical map within specific atlas region.
        
        Parameters
        ----------
        statistical_map : statistical map data
            Statistical map to extract from
        region_id : int
            Atlas region identifier
            
        Returns
        -------
        numpy.ndarray
            Values within the specified region
        """
        if isinstance(statistical_map, (str, Path)):
            img = nib.load(str(statistical_map))
            data = img.get_fdata()
        elif isinstance(statistical_map, nib.Nifti1Image):
            data = statistical_map.get_fdata()
        else:
            data = statistical_map
        
        mask = self.get_region_mask(region_id)
        
        # Ensure same dimensions
        if data.shape[:3] != mask.shape:
            logger.warning(f"Dimension mismatch: data {data.shape}, atlas {mask.shape}")
            return np.array([])
        
        return data[mask]
    
    def create_atlas_overlay(self,
                           background_image: Union[str, Path, np.ndarray, nib.Nifti1Image],
                           regions: List[int] = None,
                           alpha: float = 0.5) -> plt.Figure:
        """
        Create atlas overlay on background image.
        
        Parameters
        ----------
        background_image : background image data
            Anatomical background image
        regions : List[int], optional
            Specific regions to highlight
        alpha : float, default=0.5
            Overlay transparency
            
        Returns
        -------
        matplotlib.pyplot.Figure
            Atlas overlay figure
        """
        if regions is None:
            regions = list(self.labels.keys())[1:]  # Exclude background
        
        # Load background
        if isinstance(background_image, (str, Path)):
            bg_img = nib.load(str(background_image))
            bg_data = bg_img.get_fdata()
        elif isinstance(background_image, nib.Nifti1Image):
            bg_data = background_image.get_fdata()
        else:
            bg_data = background_image
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get middle slices
        mid_sag = bg_data.shape[0] // 2
        mid_cor = bg_data.shape[1] // 2
        mid_axi = bg_data.shape[2] // 2
        
        views = [
            ('Sagittal', bg_data[mid_sag, :, :], self.atlas_data[mid_sag, :, :]),
            ('Coronal', bg_data[:, mid_cor, :], self.atlas_data[:, mid_cor, :]),
            ('Axial', bg_data[:, :, mid_axi], self.atlas_data[:, :, mid_axi])
        ]
        
        for i, (title, bg_slice, atlas_slice) in enumerate(views):
            # Display background
            axes[i].imshow(np.rot90(bg_slice), cmap='gray', alpha=1.0)
            
            # Overlay atlas regions
            overlay = np.zeros_like(atlas_slice)
            for region_id in regions:
                overlay[atlas_slice == region_id] = region_id
            
            if np.any(overlay > 0):
                axes[i].imshow(np.rot90(overlay), cmap='jet', alpha=alpha, vmin=1, vmax=max(regions))
            
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        logger.info(f"Atlas overlay created for {len(regions)} regions")
        return fig

# Convenience functions for quick glass brain projections
def quick_glass_brain(statistical_map: Union[str, Path, np.ndarray, nib.Nifti1Image],
                     threshold: float = 0.0,
                     output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Quick glass brain projection with default settings.
    
    Parameters
    ----------
    statistical_map : statistical map data
        Statistical map to project
    threshold : float, default=0.0
        Threshold for displaying activations
    output_path : str or Path, optional
        Output file path to save figure
        
    Returns
    -------
    matplotlib.pyplot.Figure
        Glass brain projection figure
    """
    projector = GlassBrainProjector()
    fig = projector.create_projection(statistical_map, threshold=threshold)
    
    if output_path:
        projector.save_projection(fig, output_path)
    
    return fig

def compare_activations(maps: Dict[str, Union[str, Path, np.ndarray, nib.Nifti1Image]],
                       threshold: float = 0.0,
                       output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Compare multiple activation maps.
    
    Parameters
    ----------
    maps : Dict[str, statistical_map]
        Dictionary mapping labels to statistical maps
    threshold : float, default=0.0
        Threshold for displaying activations
    output_path : str or Path, optional
        Output file path to save figure
        
    Returns
    -------
    matplotlib.pyplot.Figure
        Comparison figure
    """
    projector = GlassBrainProjector()
    fig = projector.create_comparison_plot(maps, threshold=threshold)
    
    if output_path:
        projector.save_projection(fig, output_path)
    
    return fig
