"""
Interactive Brain Atlas and Region Selection Tools
=================================================

Provides interactive atlas navigation and region selection for visualization workflows.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging


class InteractiveBrainAtlas:
    """
    Interactive atlas for region selection and navigation.
    """
    def __init__(self, atlas_data: np.ndarray):
        self.atlas_data = atlas_data
        self.selected_region = None

    def select_region(self, region_id: int):
        if region_id < 0 or region_id >= self.atlas_data.max():
            logging.warning(f"Region {region_id} out of bounds.")
            return False
        self.selected_region = region_id
        return True

    def get_region_mask(self):
        if self.selected_region is None:
            return np.zeros_like(self.atlas_data)
        return (self.atlas_data == self.selected_region).astype(np.uint8)

    def list_regions(self):
        return np.unique(self.atlas_data)


class RegionSelectionTool:
    """
    Tool for selecting and visualizing brain regions interactively.
    """
    def __init__(self, atlas: InteractiveBrainAtlas):
        self.atlas = atlas

    def select_and_visualize(self, region_id: int):
        if self.atlas.select_region(region_id):
            mask = self.atlas.get_region_mask()
            # Visualization stub: integrate with renderer
            print(f"Region {region_id} selected. Mask shape: {mask.shape}")
        else:
            print(f"Failed to select region {region_id}")


# Usage Example
if __name__ == "__main__":
    atlas_data = np.random.randint(0, 10, size=(64, 64, 64))
    atlas = InteractiveBrainAtlas(atlas_data)
    tool = RegionSelectionTool(atlas)
    tool.select_and_visualize(3)
