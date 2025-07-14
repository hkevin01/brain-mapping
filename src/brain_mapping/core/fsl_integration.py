"""
FSL Integration Module
====================

Provides integration with FSL (FMRIB Software Library) for preprocessing
and analysis of neuroimaging data.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FSLIntegration:
    """
    Interface for FSL tools and preprocessing pipelines.
    """
    
    def __init__(self, fsl_dir: Optional[str] = None):
        """
        Initialize FSL integration.
        
        Args:
            fsl_dir: Path to FSL installation directory
        """
        self.fsl_dir = fsl_dir or os.environ.get('FSLDIR')
        self.fsl_available = self._check_fsl_availability()
        
        if not self.fsl_available:
            logger.warning("FSL not found. FSL-dependent features disabled.")
    
    def _check_fsl_availability(self) -> bool:
        """Check if FSL is available and properly configured."""
        try:
            # Try to run a simple FSL command
            result = subprocess.run(['fslinfo'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def skull_strip(self, input_path: Union[str, Path], 
                   output_path: Union[str, Path],
                   fractional_intensity: float = 0.5,
                   robust: bool = True) -> Dict:
        """
        Perform skull stripping using FSL BET.
        
        Args:
            input_path: Input image path
            output_path: Output brain-extracted image path
            fractional_intensity: Fractional intensity threshold (0-1)
            robust: Use robust brain center estimation
            
        Returns:
            Dictionary with processing results
        """
        if not self.fsl_available:
            raise RuntimeError("FSL not available")
        
        # Build BET command
        cmd = ['bet', str(input_path), str(output_path)]
        
        # Add options
        cmd.extend(['-f', str(fractional_intensity)])
        if robust:
            cmd.append('-R')
        
        # Add brain mask output
        cmd.extend(['-m'])  # Generate brain mask
        
        try:
            logger.info(f"Running skull stripping: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'mask_path': str(output_path).replace('.nii', '_mask.nii'),
                    'command': ' '.join(cmd),
                    'stdout': result.stdout
                }
            else:
                logger.error(f"BET failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'command': ' '.join(cmd)
                }
                
        except subprocess.TimeoutExpired:
            logger.error("BET operation timed out")
            return {
                'success': False,
                'error': "Operation timed out",
                'command': ' '.join(cmd)
            }
    
    def motion_correction(self, input_path: Union[str, Path],
                         output_path: Union[str, Path],
                         reference_volume: Optional[int] = None) -> Dict:
        """
        Perform motion correction using FSL MCFLIRT.
        
        Args:
            input_path: Input 4D image path
            output_path: Output motion-corrected image path
            reference_volume: Reference volume index (middle volume if None)
            
        Returns:
            Dictionary with processing results
        """
        if not self.fsl_available:
            raise RuntimeError("FSL not available")
        
        # Build MCFLIRT command
        cmd = ['mcflirt', '-in', str(input_path), '-out', str(output_path)]
        
        # Set reference volume
        if reference_volume is not None:
            cmd.extend(['-refvol', str(reference_volume)])
        
        # Generate motion parameters
        cmd.extend(['-plots', '-mats'])
        
        try:
            logger.info(f"Running motion correction: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=600)
            
            if result.returncode == 0:
                # Parse motion parameters
                motion_params = self._parse_motion_parameters(output_path)
                
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'motion_parameters': motion_params,
                    'command': ' '.join(cmd),
                    'stdout': result.stdout
                }
            else:
                logger.error(f"MCFLIRT failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'command': ' '.join(cmd)
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Motion correction timed out")
            return {
                'success': False,
                'error': "Operation timed out",
                'command': ' '.join(cmd)
            }
    
    def spatial_smoothing(self, input_path: Union[str, Path],
                         output_path: Union[str, Path],
                         fwhm: float = 6.0) -> Dict:
        """
        Apply spatial smoothing using FSL.
        
        Args:
            input_path: Input image path
            output_path: Output smoothed image path
            fwhm: Full-width half-maximum of smoothing kernel (mm)
            
        Returns:
            Dictionary with processing results
        """
        if not self.fsl_available:
            raise RuntimeError("FSL not available")
        
        # Convert FWHM to sigma (FWHM = 2.355 * sigma)
        sigma = fwhm / 2.355
        
        # Build fslmaths command for smoothing
        cmd = ['fslmaths', str(input_path), '-s', str(sigma), str(output_path)]
        
        try:
            logger.info(f"Running spatial smoothing: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'fwhm': fwhm,
                    'sigma': sigma,
                    'command': ' '.join(cmd)
                }
            else:
                logger.error(f"Smoothing failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'command': ' '.join(cmd)
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Smoothing operation timed out")
            return {
                'success': False,
                'error': "Operation timed out",
                'command': ' '.join(cmd)
            }
    
    def registration(self, input_path: Union[str, Path],
                    reference_path: Union[str, Path],
                    output_path: Union[str, Path],
                    dof: int = 12,
                    cost_function: str = 'corratio') -> Dict:
        """
        Perform registration using FSL FLIRT.
        
        Args:
            input_path: Input image to be registered
            reference_path: Reference image
            output_path: Output registered image path
            dof: Degrees of freedom (6, 7, 9, or 12)
            cost_function: Cost function (corratio, mutualinfo, etc.)
            
        Returns:
            Dictionary with processing results
        """
        if not self.fsl_available:
            raise RuntimeError("FSL not available")
        
        # Build FLIRT command
        cmd = ['flirt', 
               '-in', str(input_path),
               '-ref', str(reference_path),
               '-out', str(output_path),
               '-dof', str(dof),
               '-cost', cost_function]
        
        # Generate transformation matrix
        matrix_path = str(output_path).replace('.nii', '.mat')
        cmd.extend(['-omat', matrix_path])
        
        try:
            logger.info(f"Running registration: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=600)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output_path': str(output_path),
                    'transformation_matrix': matrix_path,
                    'dof': dof,
                    'cost_function': cost_function,
                    'command': ' '.join(cmd)
                }
            else:
                logger.error(f"FLIRT failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'command': ' '.join(cmd)
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Registration timed out")
            return {
                'success': False,
                'error': "Operation timed out",
                'command': ' '.join(cmd)
            }
    
    def preprocessing_pipeline(self, input_path: Union[str, Path],
                             output_dir: Union[str, Path],
                             steps: Optional[List[str]] = None) -> Dict:
        """
        Run a complete preprocessing pipeline.
        
        Args:
            input_path: Input 4D fMRI image
            output_dir: Output directory for processed files
            steps: List of preprocessing steps to perform
            
        Returns:
            Dictionary with pipeline results
        """
        if not self.fsl_available:
            raise RuntimeError("FSL not available")
        
        if steps is None:
            steps = ['skull_strip', 'motion_correction', 'spatial_smoothing']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline_results = {
            'success': True,
            'steps_completed': [],
            'steps_failed': [],
            'outputs': {},
            'errors': []
        }
        
        current_input = input_path
        
        try:
            # Step 1: Skull stripping (if requested)
            if 'skull_strip' in steps:
                skull_strip_output = output_dir / 'brain_extracted.nii.gz'
                result = self.skull_strip(current_input, skull_strip_output)
                
                if result['success']:
                    current_input = skull_strip_output
                    pipeline_results['steps_completed'].append('skull_strip')
                    pipeline_results['outputs']['skull_strip'] = result
                else:
                    pipeline_results['steps_failed'].append('skull_strip')
                    pipeline_results['errors'].append(result.get('error', 'Unknown error'))
                    pipeline_results['success'] = False
            
            # Step 2: Motion correction (if requested)
            if 'motion_correction' in steps and pipeline_results['success']:
                motion_output = output_dir / 'motion_corrected.nii.gz'
                result = self.motion_correction(current_input, motion_output)
                
                if result['success']:
                    current_input = motion_output
                    pipeline_results['steps_completed'].append('motion_correction')
                    pipeline_results['outputs']['motion_correction'] = result
                else:
                    pipeline_results['steps_failed'].append('motion_correction')
                    pipeline_results['errors'].append(result.get('error', 'Unknown error'))
                    pipeline_results['success'] = False
            
            # Step 3: Spatial smoothing (if requested)
            if 'spatial_smoothing' in steps and pipeline_results['success']:
                smoothed_output = output_dir / 'smoothed.nii.gz'
                result = self.spatial_smoothing(current_input, smoothed_output)
                
                if result['success']:
                    current_input = smoothed_output
                    pipeline_results['steps_completed'].append('spatial_smoothing')
                    pipeline_results['outputs']['spatial_smoothing'] = result
                else:
                    pipeline_results['steps_failed'].append('spatial_smoothing')
                    pipeline_results['errors'].append(result.get('error', 'Unknown error'))
                    pipeline_results['success'] = False
            
            # Set final output
            pipeline_results['final_output'] = str(current_input)
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}")
            pipeline_results['success'] = False
            pipeline_results['errors'].append(str(e))
        
        return pipeline_results
    
    def _parse_motion_parameters(self, output_path: Union[str, Path]) -> Dict:
        """Parse motion parameters from MCFLIRT output."""
        motion_params = {
            'parameters_file': None,
            'max_displacement': 0.0,
            'mean_displacement': 0.0,
            'parameters': []
        }
        
        try:
            # MCFLIRT creates .par file with motion parameters
            par_file = str(output_path).replace('.nii', '.par')
            if os.path.exists(par_file):
                motion_params['parameters_file'] = par_file
                
                # Read motion parameters
                params = np.loadtxt(par_file)
                if params.ndim == 1:
                    params = params.reshape(1, -1)
                
                motion_params['parameters'] = params.tolist()
                
                # Calculate displacement metrics
                if len(params) > 0:
                    # Calculate frame-to-frame displacement
                    translations = params[:, :3]  # x, y, z translations
                    displacements = np.sqrt(np.sum(np.diff(translations, axis=0)**2, axis=1))
                    
                    motion_params['max_displacement'] = float(np.max(displacements))
                    motion_params['mean_displacement'] = float(np.mean(displacements))
                    
        except Exception as e:
            logger.warning(f"Failed to parse motion parameters: {str(e)}")
        
        return motion_params


def check_fsl_installation() -> Dict:
    """
    Check FSL installation and return status information.
    
    Returns:
        Dictionary with FSL installation status
    """
    status = {
        'installed': False,
        'version': None,
        'fsl_dir': None,
        'tools_available': []
    }
    
    # Check environment variable
    fsl_dir = os.environ.get('FSLDIR')
    if fsl_dir:
        status['fsl_dir'] = fsl_dir
    
    # Check common FSL tools
    tools_to_check = ['fslinfo', 'bet', 'mcflirt', 'flirt', 'fslmaths']
    
    for tool in tools_to_check:
        try:
            result = subprocess.run([tool], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            # Most FSL tools return non-zero when run without arguments
            # but they should be available
            status['tools_available'].append(tool)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    status['installed'] = len(status['tools_available']) > 0
    
    # Try to get FSL version
    if status['installed']:
        try:
            result = subprocess.run(['cat', f"{fsl_dir}/etc/fslversion"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                status['version'] = result.stdout.strip()
        except:
            pass
    
    return status
