"""
Quality Control Module
====================

Provides comprehensive quality control metrics and validation for neuroimaging data.
"""

import numpy as np
import nibabel as nib
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import ndimage
from scipy.stats import zscore

logger = logging.getLogger(__name__)

class QualityControl:
    """
    Quality control and validation for neuroimaging data.
    """
    
    def __init__(self):
        """Initialize QC with default thresholds."""
        self.qc_thresholds = {
            'motion_threshold': 2.0,  # mm
            'snr_threshold': 10.0,
            'spike_threshold': 3.0,  # standard deviations
            'intensity_threshold': 0.1  # coefficient of variation
        }
    
    def comprehensive_qc(self, image_data: np.ndarray, 
                        affine: Optional[np.ndarray] = None,
                        metadata: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive quality control assessment.
        
        Args:
            image_data: 3D or 4D numpy array
            affine: Affine transformation matrix
            metadata: Additional metadata
            
        Returns:
            Dictionary containing QC metrics
        """
        qc_results = {
            'overall_quality': 'unknown',
            'warnings': [],
            'metrics': {}
        }
        
        try:
            # Basic image statistics
            qc_results['metrics']['basic_stats'] = self._basic_statistics(image_data)
            
            # Signal-to-noise ratio
            qc_results['metrics']['snr'] = self._calculate_snr(image_data)
            
            # Motion assessment (for 4D data)
            if image_data.ndim == 4:
                qc_results['metrics']['motion'] = self._assess_motion(image_data)
            
            # Spike detection
            qc_results['metrics']['spikes'] = self._detect_spikes(image_data)
            
            # Intensity uniformity
            qc_results['metrics']['intensity_uniformity'] = self._assess_intensity_uniformity(image_data)
            
            # Overall quality assessment
            qc_results['overall_quality'] = self._determine_overall_quality(qc_results['metrics'])
            
            logger.info(f"QC completed. Overall quality: {qc_results['overall_quality']}")
            
        except Exception as e:
            logger.error(f"QC assessment failed: {str(e)}")
            qc_results['overall_quality'] = 'error'
            qc_results['warnings'].append(f"QC failed: {str(e)}")
        
        return qc_results
    
    def _basic_statistics(self, image_data: np.ndarray) -> Dict:
        """Calculate basic image statistics."""
        # Remove zero/background voxels for meaningful statistics
        mask = image_data > 0
        if image_data.ndim == 4:
            mask = mask.any(axis=-1)
        
        masked_data = image_data[mask]
        
        stats = {
            'mean': float(np.mean(masked_data)),
            'std': float(np.std(masked_data)),
            'min': float(np.min(masked_data)),
            'max': float(np.max(masked_data)),
            'median': float(np.median(masked_data)),
            'volume_voxels': int(np.sum(mask)),
            'total_voxels': int(mask.size)
        }
        
        return stats
    
    def _calculate_snr(self, image_data: np.ndarray) -> Dict:
        """Calculate signal-to-noise ratio."""
        if image_data.ndim == 4:
            # For 4D data, calculate temporal SNR
            mean_signal = np.mean(image_data, axis=-1)
            std_signal = np.std(image_data, axis=-1)
            
            # Avoid division by zero
            snr = np.divide(mean_signal, std_signal, 
                          out=np.zeros_like(mean_signal), 
                          where=std_signal!=0)
            
            # Calculate SNR for brain voxels only
            brain_mask = mean_signal > (0.1 * np.max(mean_signal))
            brain_snr = snr[brain_mask]
            
            snr_metrics = {
                'temporal_snr_mean': float(np.mean(brain_snr)),
                'temporal_snr_std': float(np.std(brain_snr)),
                'temporal_snr_median': float(np.median(brain_snr))
            }
        else:
            # For 3D data, estimate SNR using background noise
            # Simple method: assume background is in corners
            bg_corners = self._get_background_corners(image_data)
            noise_std = np.std(bg_corners)
            signal_mean = np.mean(image_data[image_data > 0])
            
            snr_metrics = {
                'spatial_snr': float(signal_mean / noise_std) if noise_std > 0 else 0,
                'noise_std': float(noise_std),
                'signal_mean': float(signal_mean)
            }
        
        return snr_metrics
    
    def _assess_motion(self, image_data: np.ndarray) -> Dict:
        """Assess motion in 4D time series data."""
        motion_metrics = {
            'frame_displacement': [],
            'mean_displacement': 0.0,
            'max_displacement': 0.0,
            'motion_outliers': []
        }
        
        try:
            # Calculate frame-to-frame displacement
            for t in range(1, image_data.shape[-1]):
                current_vol = image_data[..., t]
                prev_vol = image_data[..., t-1]
                
                # Simple correlation-based displacement estimate
                displacement = self._estimate_displacement(current_vol, prev_vol)
                motion_metrics['frame_displacement'].append(displacement)
            
            displacements = np.array(motion_metrics['frame_displacement'])
            motion_metrics['mean_displacement'] = float(np.mean(displacements))
            motion_metrics['max_displacement'] = float(np.max(displacements))
            
            # Identify motion outliers
            motion_threshold = self.qc_thresholds['motion_threshold']
            outliers = np.where(displacements > motion_threshold)[0]
            motion_metrics['motion_outliers'] = outliers.tolist()
            
        except Exception as e:
            logger.warning(f"Motion assessment failed: {str(e)}")
        
        return motion_metrics
    
    def _detect_spikes(self, image_data: np.ndarray) -> Dict:
        """Detect intensity spikes/artifacts."""
        spike_metrics = {
            'spike_volumes': [],
            'spike_voxels': [],
            'total_spikes': 0
        }
        
        try:
            if image_data.ndim == 4:
                # For 4D data, detect spikes across time
                for t in range(image_data.shape[-1]):
                    vol = image_data[..., t]
                    z_scores = np.abs(zscore(vol[vol > 0]))
                    
                    spike_threshold = self.qc_thresholds['spike_threshold']
                    spikes = np.sum(z_scores > spike_threshold)
                    
                    if spikes > 0:
                        spike_metrics['spike_volumes'].append(t)
                        spike_metrics['spike_voxels'].append(int(spikes))
                
                spike_metrics['total_spikes'] = sum(spike_metrics['spike_voxels'])
            else:
                # For 3D data, detect spatial spikes
                z_scores = np.abs(zscore(image_data[image_data > 0]))
                spike_threshold = self.qc_thresholds['spike_threshold']
                spike_metrics['total_spikes'] = int(np.sum(z_scores > spike_threshold))
                
        except Exception as e:
            logger.warning(f"Spike detection failed: {str(e)}")
        
        return spike_metrics
    
    def _assess_intensity_uniformity(self, image_data: np.ndarray) -> Dict:
        """Assess intensity uniformity across the image."""
        uniformity_metrics = {
            'coefficient_of_variation': 0.0,
            'intensity_range': 0.0,
            'uniformity_score': 0.0
        }
        
        try:
            # Use brain mask for meaningful assessment
            if image_data.ndim == 4:
                mean_image = np.mean(image_data, axis=-1)
            else:
                mean_image = image_data
            
            brain_mask = mean_image > (0.1 * np.max(mean_image))
            brain_intensities = mean_image[brain_mask]
            
            if len(brain_intensities) > 0:
                mean_intensity = np.mean(brain_intensities)
                std_intensity = np.std(brain_intensities)
                
                uniformity_metrics['coefficient_of_variation'] = float(std_intensity / mean_intensity)
                uniformity_metrics['intensity_range'] = float(np.ptp(brain_intensities))
                
                # Uniformity score (higher is better)
                uniformity_metrics['uniformity_score'] = 1.0 / (1.0 + uniformity_metrics['coefficient_of_variation'])
                
        except Exception as e:
            logger.warning(f"Intensity uniformity assessment failed: {str(e)}")
        
        return uniformity_metrics
    
    def _determine_overall_quality(self, metrics: Dict) -> str:
        """Determine overall quality based on all metrics."""
        quality_score = 0
        max_score = 0
        
        # SNR assessment
        if 'snr' in metrics:
            if 'temporal_snr_mean' in metrics['snr']:
                snr_value = metrics['snr']['temporal_snr_mean']
            else:
                snr_value = metrics['snr'].get('spatial_snr', 0)
            
            if snr_value > self.qc_thresholds['snr_threshold']:
                quality_score += 2
            elif snr_value > self.qc_thresholds['snr_threshold'] / 2:
                quality_score += 1
            max_score += 2
        
        # Motion assessment
        if 'motion' in metrics:
            mean_motion = metrics['motion']['mean_displacement']
            if mean_motion < self.qc_thresholds['motion_threshold'] / 2:
                quality_score += 2
            elif mean_motion < self.qc_thresholds['motion_threshold']:
                quality_score += 1
            max_score += 2
        
        # Spike assessment
        if 'spikes' in metrics:
            if metrics['spikes']['total_spikes'] == 0:
                quality_score += 1
            max_score += 1
        
        # Intensity uniformity
        if 'intensity_uniformity' in metrics:
            uniformity_score = metrics['intensity_uniformity']['uniformity_score']
            if uniformity_score > 0.8:
                quality_score += 1
            max_score += 1
        
        # Determine overall quality
        if max_score == 0:
            return 'unknown'
        
        quality_ratio = quality_score / max_score
        
        if quality_ratio >= 0.8:
            return 'excellent'
        elif quality_ratio >= 0.6:
            return 'good'
        elif quality_ratio >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _get_background_corners(self, image_data: np.ndarray) -> np.ndarray:
        """Extract background voxels from image corners."""
        shape = image_data.shape
        corner_size = min(10, min(shape) // 10)
        
        corners = []
        # Extract 8 corners of the 3D image
        for i in [0, -corner_size]:
            for j in [0, -corner_size]:
                for k in [0, -corner_size]:
                    corner = image_data[i:i+corner_size if i == 0 else i:,
                                     j:j+corner_size if j == 0 else j:,
                                     k:k+corner_size if k == 0 else k:]
                    corners.append(corner.flatten())
        
        return np.concatenate(corners)
    
    def _estimate_displacement(self, current_vol: np.ndarray, 
                             prev_vol: np.ndarray) -> float:
        """Estimate displacement between two volumes."""
        try:
            # Simple correlation-based displacement
            correlation = np.corrcoef(current_vol.flatten(), prev_vol.flatten())[0, 1]
            # Convert correlation to approximate displacement (rough estimate)
            displacement = (1 - correlation) * 5.0  # Scale factor
            return float(displacement)
        except:
            return 0.0

def validate_neuroimaging_data(image_path: Union[str, Path]) -> Dict:
    """
    Quick validation function for neuroimaging data.
    
    Args:
        image_path: Path to neuroimaging file
        
    Returns:
        Dictionary containing validation results
    """
    qc = QualityControl()
    
    try:
        # Load image
        img = nib.load(str(image_path))
        data = img.get_fdata()
        
        # Perform QC
        results = qc.comprehensive_qc(data, img.affine)
        results['file_path'] = str(image_path)
        results['image_shape'] = data.shape
        results['voxel_size'] = img.header.get_zooms()
        
        return results
        
    except Exception as e:
        return {
            'file_path': str(image_path),
            'overall_quality': 'error',
            'error': str(e),
            'warnings': [f"Failed to load or validate: {str(e)}"]
        }
