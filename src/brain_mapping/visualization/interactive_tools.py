"""
Interactive visualization tools for brain mapping.
Includes region selection, atlas overlays, and real-time updates.
"""

import numpy as np
import matplotlib.pyplot as plt


class InteractiveRegionSelector:
    """Interactive tool for selecting brain regions on visualizations."""
    def __init__(self, atlas_data: np.ndarray, labels: dict):
        self.atlas_data = atlas_data
        self.labels = labels

    def select_region(self, region_id: int):
        """Return mask for selected region."""
        return self.atlas_data == region_id

    def show(self):
        """Display atlas with selectable regions."""
        fig, ax = plt.subplots()
        ax.imshow(np.max(self.atlas_data, axis=2))
        ax.set_title("Select a region")
        plt.show()

# Additional interactive tools can be added here
