"""
Atlas overlay visualization utilities for brain mapping toolkit.
Supports overlaying anatomical atlases on statistical maps and
interactive region highlighting.
"""

import numpy as np
import matplotlib.pyplot as plt


def overlay_atlas(
    stat_map: np.ndarray,
    atlas: np.ndarray,
    region_id: int,
    alpha: float = 0.5
):
    """
    Overlay a specific atlas region on a statistical map.
    Args:
        stat_map: 3D statistical map (numpy array)
        atlas: 3D atlas data (numpy array)
        region_id: Region to highlight
        alpha: Overlay transparency
    Returns:
        fig: Matplotlib figure with overlay
    """
    mask = atlas == region_id
    fig, ax = plt.subplots()
    ax.imshow(np.max(stat_map, axis=2), cmap='gray')
    ax.imshow(np.max(mask, axis=2), cmap='Reds', alpha=alpha)
    ax.set_title(f"Region {region_id} Overlay")
    plt.show()
    return fig
