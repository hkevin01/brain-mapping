"""
Automated QC metrics for brain mapping toolkit.
Provides functions to compute SNR, motion, spikes, and intensity uniformity.
"""

import numpy as np


def compute_snr(image_data: np.ndarray) -> float:
    """Compute signal-to-noise ratio for neuroimaging data."""
    signal = np.mean(image_data[image_data > 0])
    noise = np.std(image_data[image_data == 0])
    return signal / (noise + 1e-8)


def compute_motion(image_data: np.ndarray) -> float:
    """Compute simple motion metric for 4D data."""
    if image_data.ndim != 4:
        return 0.0
    diffs = np.diff(image_data, axis=3)
    return np.mean(np.abs(diffs))


def detect_spikes(image_data: np.ndarray) -> int:
    """Detect spikes in neuroimaging data."""
    spikes = np.sum(np.abs(image_data) > 5 * np.std(image_data))
    return int(spikes)


def intensity_uniformity(image_data: np.ndarray) -> float:
    """Compute intensity uniformity metric."""
    mean_intensity = np.mean(image_data)
    std_intensity = np.std(image_data)
    return std_intensity / (mean_intensity + 1e-8)
