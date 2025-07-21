"""
Interactive Brain Atlas Module
=============================

Provides interactive brain atlas functionality for anatomical reference
and region-based analysis as part of Phase 1 capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any
import nibabel as nib
from matplotlib.widgets import Slider, Button
import logging

try:
    from ..utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class InteractiveBrainAtlas:
    """
    Interactive brain atlas for Phase 1 visualization and analysis.
    
    Features:
    - Standard atlas loading (AAL, Harvard-Oxford)
    - Interactive region selection
    - Statistical value extraction
    - ROI analysis capabilities
    """
    
    def __init__(self, atlas_name: str = 'aal'):
        """
        Initialize interactive brain atlas.
        
        Parameters
        ----------
        atlas_name : str, default='aal'
            Atlas to load ('aal', 'harvard_oxford', 'mni152')
        """
        self.atlas_name = atlas_name
        self.atlas_data = None
        self.labels = {}
        self.region_colors = {}
        self.selected_regions = []  # Track selected regions
        self._load_atlas()
        
    def _load_atlas(self):
        """Load atlas data and labels."""
        logger.info(f"Loading {self.atlas_name} atlas...")
        
        # For Phase 1, create synthetic atlas data
        # In production, this would load actual atlas files
        if self.atlas_name == 'aal':
            self._create_aal_atlas()
        elif self.atlas_name == 'harvard_oxford':
            self._create_harvard_oxford_atlas()
        else:
            self._create_generic_atlas()
            
        logger.info(f"Atlas {self.atlas_name} loaded with {len(self.labels)} regions")
    
    def _create_aal_atlas(self):
        """Create synthetic AAL atlas for Phase 1."""
        # Standard AAL dimensions
        self.atlas_data = np.zeros((91, 109, 91), dtype=np.int16)
        
        # Define major brain regions
        self.labels = {
            0: 'Background',
            1: 'Precentral_L',
            2: 'Precentral_R',
            3: 'Frontal_Sup_L',
            4: 'Frontal_Sup_R',
            5: 'Frontal_Sup_Orb_L',
            6: 'Frontal_Sup_Orb_R',
            7: 'Frontal_Mid_L',
            8: 'Frontal_Mid_R',
            9: 'Frontal_Mid_Orb_L',
            10: 'Frontal_Mid_Orb_R',
            11: 'Frontal_Inf_Oper_L',
            12: 'Frontal_Inf_Oper_R',
            13: 'Frontal_Inf_Tri_L',
            14: 'Frontal_Inf_Tri_R',
            15: 'Frontal_Inf_Orb_L',
            16: 'Frontal_Inf_Orb_R',
            17: 'Rolandic_Oper_L',
            18: 'Rolandic_Oper_R',
            19: 'Supp_Motor_Area_L',
            20: 'Supp_Motor_Area_R'
        }
        
        # Create simplified regional assignments
        self._create_synthetic_regions()
    
    def _create_harvard_oxford_atlas(self):
        """Create synthetic Harvard-Oxford atlas."""
        self.atlas_data = np.zeros((91, 109, 91), dtype=np.int16)
        
        self.labels = {
            0: 'Background',
            1: 'Frontal Pole',
            2: 'Insular Cortex',
            3: 'Superior Frontal Gyrus',
            4: 'Middle Frontal Gyrus',
            5: 'Inferior Frontal Gyrus',
            6: 'Precentral Gyrus',
            7: 'Temporal Pole',
            8: 'Superior Temporal Gyrus',
            9: 'Middle Temporal Gyrus',
            10: 'Inferior Temporal Gyrus'
        }
        
        self._create_synthetic_regions()
    
    def _create_generic_atlas(self):
        """Create generic atlas structure."""
        self.atlas_data = np.zeros((91, 109, 91), dtype=np.int16)
        
        self.labels = {
            0: 'Background',
            1: 'Frontal Lobe',
            2: 'Parietal Lobe',
            3: 'Temporal Lobe',
            4: 'Occipital Lobe',
            5: 'Cerebellum'
        }
        
        self._create_synthetic_regions()
    
    def _create_synthetic_regions(self):
        """Create synthetic regional data for Phase 1 demonstration."""
        # This creates simplified brain regions for demonstration
        # In production, actual atlas data would be loaded
        
        height, width, depth = self.atlas_data.shape
        
        # Frontal regions (anterior)
        self.atlas_data[10:40, 20:80, 30:70] = 1
        self.atlas_data[15:35, 25:75, 35:65] = 3
        
        # Parietal regions (posterior-superior)
        self.atlas_data[45:75, 20:80, 40:80] = 2
        
        # Temporal regions (lateral)
        self.atlas_data[20:60, 10:30, 20:60] = 3
        self.atlas_data[20:60, 80:100, 20:60] = 3
        
        # Occipital regions (posterior)
        self.atlas_data[70:90, 30:70, 30:70] = 4
        
        # Generate colors for regions
        self._generate_region_colors()
    
    def _generate_region_colors(self):
        """Generate distinct colors for atlas regions."""
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
        
        for region_id in self.labels.keys():
            if region_id == 0:
                self.region_colors[region_id] = (0, 0, 0, 0)  # Transparent background
            else:
                color = cmap(region_id % 10)
                self.region_colors[region_id] = color
    
    def get_region_mask(self, region_id: int) -> np.ndarray:
        """
        Get binary mask for a specific region.
        
        Parameters
        ----------
        region_id : int
            Region identifier
            
        Returns
        -------
        np.ndarray
            Binary mask for the region
        """
        if self.atlas_data is None:
            raise ValueError("Atlas not loaded")
            
        mask = (self.atlas_data == region_id).astype(np.uint8)
        logger.debug(f"Region {region_id} mask: {np.sum(mask)} voxels")
        return mask
    
    def get_region_name(self, region_id: int) -> str:
        """Get the name of a region."""
        return self.labels.get(region_id, f"Unknown_Region_{region_id}")
    
    def extract_region_values(self,
                             statistical_map: Union[str, Path, np.ndarray, nib.Nifti1Image],
                             region_id: int) -> np.ndarray:
        """
        Extract statistical values from a specific brain region.
        
        Parameters
        ----------
        statistical_map : str, Path, numpy.ndarray, or nibabel image
            Statistical map data
        region_id : int
            Region identifier
            
        Returns
        -------
        np.ndarray
            Statistical values within the region
        """
        # Load statistical map
        if isinstance(statistical_map, np.ndarray):
            stat_data = statistical_map
        elif isinstance(statistical_map, nib.Nifti1Image):
            stat_data = statistical_map.get_fdata()
        else:
            img = nib.load(str(statistical_map))
            stat_data = img.get_fdata()
        
        # Get region mask
        mask = self.get_region_mask(region_id)
        
        # Extract values
        region_values = stat_data[mask == 1]
        
        logger.info(f"Extracted {len(region_values)} values from region {region_id}")
        return region_values
    
    def create_interactive_atlas_view(self,
                                     background_image: Union[str, Path, np.ndarray, nib.Nifti1Image] = None,
                                     alpha: float = 0.6) -> plt.Figure:
        """
        Create interactive atlas visualization with sliders.
        
        Parameters
        ----------
        background_image : str, Path, numpy.ndarray, or nibabel image, optional
            Background anatomical image
        alpha : float, default=0.6
            Atlas overlay transparency
            
        Returns
        -------
        matplotlib.pyplot.Figure
            Interactive atlas figure
        """
        # Load background if provided
        if background_image is not None:
            if isinstance(background_image, np.ndarray):
                bg_data = background_image
            elif isinstance(background_image, nib.Nifti1Image):
                bg_data = background_image.get_fdata()
            else:
                img = nib.load(str(background_image))
                bg_data = img.get_fdata()
        else:
            # Create synthetic background
            bg_data = np.random.randn(*self.atlas_data.shape) * 0.1 + 1.0
        
        # Create figure with sliders
        fig = plt.figure(figsize=(15, 10))
        
        # Main axes for three views
        ax_sag = plt.subplot2grid((3, 4), (0, 0), colspan=1)
        ax_cor = plt.subplot2grid((3, 4), (0, 1), colspan=1)
        ax_axi = plt.subplot2grid((3, 4), (0, 2), colspan=1)
        ax_info = plt.subplot2grid((3, 4), (0, 3), colspan=1, rowspan=3)
        
        # Slider axes
        ax_sag_slider = plt.subplot2grid((3, 4), (1, 0), colspan=1)
        ax_cor_slider = plt.subplot2grid((3, 4), (1, 1), colspan=1)
        ax_axi_slider = plt.subplot2grid((3, 4), (1, 2), colspan=1)
        
        # Control axes
        ax_controls = plt.subplot2grid((3, 4), (2, 0), colspan=3)
        
        # Initialize slice positions
        sag_pos = bg_data.shape[0] // 2
        cor_pos = bg_data.shape[1] // 2
        axi_pos = bg_data.shape[2] // 2
        
        # Create sliders
        sag_slider = Slider(ax_sag_slider, 'Sagittal', 0, bg_data.shape[0]-1, 
                           valinit=sag_pos, valfmt='%d')
        cor_slider = Slider(ax_cor_slider, 'Coronal', 0, bg_data.shape[1]-1, 
                           valinit=cor_pos, valfmt='%d')
        axi_slider = Slider(ax_axi_slider, 'Axial', 0, bg_data.shape[2]-1, 
                           valinit=axi_pos, valfmt='%d')
        
        # Store images for updating
        self.images = {}
        
        def update_display():
            """Update all slice displays."""
            sag_idx = int(sag_slider.val)
            cor_idx = int(cor_slider.val)
            axi_idx = int(axi_slider.val)
            
            # Clear axes
            ax_sag.clear()
            ax_cor.clear()
            ax_axi.clear()
            
            # Display background
            ax_sag.imshow(np.rot90(bg_data[sag_idx, :, :]), cmap='gray', alpha=1.0)
            ax_cor.imshow(np.rot90(bg_data[:, cor_idx, :]), cmap='gray', alpha=1.0)
            ax_axi.imshow(np.rot90(bg_data[:, :, axi_idx]), cmap='gray', alpha=1.0)
            
            # Overlay atlas
            sag_atlas = self.atlas_data[sag_idx, :, :]
            cor_atlas = self.atlas_data[:, cor_idx, :]
            axi_atlas = self.atlas_data[:, :, axi_idx]
            
            # Create colored overlay
            sag_overlay = self._create_colored_overlay(sag_atlas)
            cor_overlay = self._create_colored_overlay(cor_atlas)
            axi_overlay = self._create_colored_overlay(axi_atlas)
            
            ax_sag.imshow(np.rot90(sag_overlay), alpha=alpha)
            ax_cor.imshow(np.rot90(cor_overlay), alpha=alpha)
            ax_axi.imshow(np.rot90(axi_overlay), alpha=alpha)
            
            # Set titles and turn off axes
            ax_sag.set_title(f'Sagittal (X={sag_idx})')
            ax_cor.set_title(f'Coronal (Y={cor_idx})')
            ax_axi.set_title(f'Axial (Z={axi_idx})')
            
            ax_sag.axis('off')
            ax_cor.axis('off')
            ax_axi.axis('off')
            
            plt.draw()
        
        # Connect sliders to update function
        sag_slider.on_changed(lambda x: update_display())
        cor_slider.on_changed(lambda x: update_display())
        axi_slider.on_changed(lambda x: update_display())
        
        # Display region information
        self._display_region_info(ax_info)
        
        # Initial display
        update_display()
        
        plt.tight_layout()
        
        logger.info("Interactive atlas view created")
        return fig
    
    def _create_colored_overlay(self, atlas_slice: np.ndarray) -> np.ndarray:
        """Create colored overlay from atlas slice."""
        height, width = atlas_slice.shape
        overlay = np.zeros((height, width, 4))  # RGBA
        
        for region_id in np.unique(atlas_slice):
            if region_id == 0:
                continue  # Skip background
            
            mask = atlas_slice == region_id
            color = self.region_colors.get(region_id, (1, 0, 0, 1))
            overlay[mask] = color
        
        return overlay
    
    def _display_region_info(self, ax):
        """Display region information in info panel."""
        ax.clear()
        ax.text(0.05, 0.95, f"Atlas: {self.atlas_name.upper()}", 
                transform=ax.transAxes, fontsize=14, weight='bold')
        
        y_pos = 0.85
        for region_id, region_name in list(self.labels.items())[:10]:  # Show first 10
            color = self.region_colors.get(region_id, (0, 0, 0, 1))
            ax.text(0.05, y_pos, f"{region_id}: {region_name}", 
                   transform=ax.transAxes, fontsize=10, color=color[:3])
            y_pos -= 0.08
        
        if len(self.labels) > 10:
            ax.text(0.05, y_pos, f"... and {len(self.labels)-10} more regions", 
                   transform=ax.transAxes, fontsize=10, style='italic')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def compute_region_statistics(self,
                                 statistical_map: Union[str, Path, np.ndarray, nib.Nifti1Image],
                                 regions: Optional[List[int]] = None) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics for brain regions.
        
        Parameters
        ----------
        statistical_map : str, Path, numpy.ndarray, or nibabel image
            Statistical map data
        regions : List[int], optional
            List of region IDs to analyze. If None, analyzes all regions.
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Statistics for each region
        """
        if regions is None:
            regions = [r for r in self.labels.keys() if r != 0]
        
        # Load statistical map
        if isinstance(statistical_map, np.ndarray):
            stat_data = statistical_map
        elif isinstance(statistical_map, nib.Nifti1Image):
            stat_data = statistical_map.get_fdata()
        else:
            img = nib.load(str(statistical_map))
            stat_data = img.get_fdata()
        
        region_stats = {}
        
        for region_id in regions:
            values = self.extract_region_values(stat_data, region_id)
            
            if len(values) > 0:
                region_stats[region_id] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_voxels': len(values),
                    'region_name': self.get_region_name(region_id)
                }
            else:
                region_stats[region_id] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'median': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'n_voxels': 0,
                    'region_name': self.get_region_name(region_id)
                }
        
        logger.info(f"Computed statistics for {len(region_stats)} regions")
        return region_stats
    
    def save_atlas_overlay(self,
                          background_image: Union[str, Path, np.ndarray, nib.Nifti1Image],
                          output_path: Union[str, Path],
                          slice_coords: Optional[Tuple[int, int, int]] = None,
                          alpha: float = 0.6) -> None:
        """
        Save atlas overlay visualization.
        
        Parameters
        ----------
        background_image : str, Path, numpy.ndarray, or nibabel image
            Background anatomical image
        output_path : str or Path
            Output file path
        slice_coords : Tuple[int, int, int], optional
            Slice coordinates (x, y, z)
        alpha : float, default=0.6
            Atlas overlay transparency
        """
        fig = self.create_interactive_atlas_view(background_image, alpha)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Atlas overlay saved to: {output_path}")
    
    def select_region(self, region_id: str):
        """Select a region for analysis and visualization."""
        try:
            self.selected_regions.append(region_id)
            logger.info("Region selected: %s", region_id)
        except Exception as e:
            logger.error(f"Error selecting region: {str(e)}")
            raise

    def visualize_selected(self):
        """Visualize the currently selected regions."""
        try:
            logger.info("Visualizing regions: %s", self.selected_regions)
            return self.selected_regions
        except Exception as e:
            logger.error(f"Error visualizing regions: {str(e)}")
            return []


class InteractiveAtlas:
    def __init__(self):
        self.selected_region = None
    def select_region(self, region_id):
        self.selected_region = region_id
        print(f"Selected region: {region_id}")
    def visualize_selected(self):
        print(f"Visualizing selected region: {self.selected_region}")


# Convenience functions for quick atlas usage
def quick_atlas_overlay(background_image: Union[str, Path, np.ndarray, nib.Nifti1Image],
                       atlas_name: str = 'aal',
                       output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Quick atlas overlay with default settings.
    
    Parameters
    ----------
    background_image : str, Path, numpy.ndarray, or nibabel image
        Background anatomical image
    atlas_name : str, default='aal'
        Atlas to use
    output_path : str or Path, optional
        Output file path to save figure
        
    Returns
    -------
    matplotlib.pyplot.Figure
        Atlas overlay figure
    """
    atlas = InteractiveBrainAtlas(atlas_name)
    fig = atlas.create_interactive_atlas_view(background_image)
    
    if output_path:
        atlas.save_atlas_overlay(background_image, output_path)
    
    return fig

def analyze_regions(statistical_map: Union[str, Path, np.ndarray, nib.Nifti1Image],
                   atlas_name: str = 'aal',
                   regions: Optional[List[int]] = None) -> Dict[int, Dict[str, float]]:
    """
    Analyze statistical values by brain regions.
    
    Parameters
    ----------
    statistical_map : str, Path, numpy.ndarray, or nibabel image
        Statistical map data
    atlas_name : str, default='aal'
        Atlas to use for regions
    regions : List[int], optional
        Specific regions to analyze
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Regional statistics
    """
    atlas = InteractiveBrainAtlas(atlas_name)
    return atlas.compute_region_statistics(statistical_map, regions)
