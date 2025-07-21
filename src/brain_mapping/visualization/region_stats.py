"""
Advanced region statistics and visualization utilities
for brain mapping toolkit.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


class RegionStats:
    """Compute and visualize statistics for selected brain regions."""
    def __init__(self, region_data: np.ndarray, region_labels: Dict[int, str]):
        self.region_data = region_data
        self.region_labels = region_labels

    def compute_stats(self) -> Dict[str, Any]:
        stats = {}
        for idx, label in self.region_labels.items():
            region_values = self.region_data[idx]
            stats[label] = {
                "mean": float(np.mean(region_values)),
                "std": float(np.std(region_values)),
                "min": float(np.min(region_values)),
                "max": float(np.max(region_values)),
            }
        return stats

    def plot_stats(self, stats: Dict[str, Any], save_path: str = None):
        labels = list(stats.keys())
        means = [stats[label]["mean"] for label in labels]
        stds = [stats[label]["std"] for label in labels]
        plt.figure(figsize=(10, 6))
        plt.bar(labels, means, yerr=stds, capsize=5)
        plt.ylabel("Mean Value ± Std")
        plt.title("Region Statistics")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
