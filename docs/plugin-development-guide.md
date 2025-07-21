# Plugin Development Guide

## 🧩 Extending Brain Mapping Toolkit with Custom Plugins

This guide explains how to create custom preprocessing plugins for the brain mapping toolkit, enabling researchers to add their own analysis steps without modifying the core codebase.

---

## 📋 Overview

### What are Plugins?
Plugins are modular preprocessing steps that can be combined to create custom analysis pipelines. Each plugin:
- Inherits from `PreprocessingPlugin`
- Implements a `run()` method
- Can be chained together in any order
- Handles errors gracefully

### Why Use Plugins?
- **Extensibility**: Add new preprocessing steps without core code changes
- **Modularity**: Test and debug individual steps independently
- **Reusability**: Share plugins across different projects
- **Customization**: Create lab-specific preprocessing workflows

---

## 🚀 Quick Start

### Basic Plugin Template

```python
from brain_mapping.core.preprocessor import PreprocessingPlugin
import numpy as np
import nibabel as nib

class MyCustomPlugin(PreprocessingPlugin):
    """
    A custom preprocessing plugin.
    """
    
    def __init__(self, parameter1=1.0, parameter2=True):
        """
        Initialize plugin with parameters.
        
        Parameters
        ----------
        parameter1 : float, default=1.0
            Description of parameter1
        parameter2 : bool, default=True
            Description of parameter2
        """
        super().__init__("MyCustomPlugin")
        self.parameter1 = parameter1
        self.parameter2 = parameter2
    
    def run(self, img, **kwargs):
        """
        Apply custom preprocessing to the image.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Input brain image
        **kwargs : dict
            Additional parameters (optional)
            
        Returns
        -------
        nibabel.Nifti1Image
            Processed image
        """
        # Get image data
        data = img.get_fdata()
        affine = img.affine
        
        # Apply your custom processing here
        processed_data = self._apply_custom_processing(data)
        
        # Return new NIfTI image
        return nib.Nifti1Image(processed_data, affine)
    
    def _apply_custom_processing(self, data):
        """
        Apply custom processing logic.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data array
            
        Returns
        -------
        numpy.ndarray
            Processed data array
        """
        # Your custom processing logic here
        # Example: simple thresholding
        if self.parameter2:
            processed = np.where(data > self.parameter1, data, 0)
        else:
            processed = data * self.parameter1
            
        return processed
```

### Using Your Plugin

```python
from brain_mapping.core.preprocessor import Preprocessor

# Create your plugin
my_plugin = MyCustomPlugin(parameter1=2.0, parameter2=True)

# Use in a pipeline
preproc = Preprocessor(plugins=[my_plugin])
result = preproc.run_pipeline(img, pipeline='custom')
```

---

## 🔧 Built-in Plugin Examples

### 1. GaussianSmoothingPlugin

```python
class GaussianSmoothingPlugin(PreprocessingPlugin):
    """
    Gaussian smoothing plugin using GPU acceleration and mixed-precision.
    """
    
    def __init__(self, sigma=1.0, use_gpu=True, precision='float32'):
        super().__init__("GaussianSmoothing")
        self.sigma = sigma
        self.use_gpu = use_gpu
        self.precision = precision
    
    def run(self, img, **kwargs):
        # Create temporary preprocessor to use existing smoothing
        temp_preproc = Preprocessor(gpu_enabled=self.use_gpu, precision=self.precision)
        return temp_preproc._spatial_smoothing(img, use_gpu=self.use_gpu, sigma=self.sigma)
```

### 2. QualityControlPlugin

```python
class QualityControlPlugin(PreprocessingPlugin):
    """
    Quality control plugin that computes and reports QC metrics.
    """
    
    def __init__(self, save_report=True, report_path=None):
        super().__init__("QualityControl")
        self.save_report = save_report
        self.report_path = report_path
    
    def run(self, img, **kwargs):
        # Import QC functionality
        from .quality_control import QualityControl
        
        qc = QualityControl()
        data = img.get_fdata()
        
        # Compute QC metrics
        qc_results = qc.comprehensive_qc(data, img.affine)
        
        # Print summary and save report
        print(f"QC Results: {qc_results['overall_quality']}")
        
        if self.save_report:
            # Save QC report to file
            import json
            with open(self.report_path or "qc_report.json", 'w') as f:
                json.dump(qc_results, f, indent=2, default=str)
        
        # Return original image (QC is non-destructive)
        return img
```

### 3. MotionCorrectionPlugin

```python
class MotionCorrectionPlugin(PreprocessingPlugin):
    """
    Motion correction plugin using FSL MCFLIRT.
    """
    
    def __init__(self, reference_volume=None, save_motion_params=True):
        super().__init__("MotionCorrection")
        self.reference_volume = reference_volume
        self.save_motion_params = save_motion_params
    
    def run(self, img, **kwargs):
        # Import FSL integration
        from .fsl_integration import FSLIntegration
        
        fsl = FSLIntegration()
        if not fsl.fsl_available:
            warnings.warn("FSL not available. Skipping motion correction.")
            return img
        
        # Save temporary file and run FSL motion correction
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Implementation details...
            pass
        
        return corrected_img
```

---

## 🎯 Plugin Categories

### 1. Data Processing Plugins
- **Smoothing**: Gaussian, anisotropic, bilateral
- **Filtering**: Bandpass, high-pass, low-pass
- **Normalization**: Intensity, spatial, temporal
- **Denoising**: ICA, PCA, wavelet-based

### 2. Quality Control Plugins
- **Metrics**: SNR, motion, spikes, artifacts
- **Visualization**: QC plots, reports, summaries
- **Validation**: Data integrity, format compliance

### 3. Analysis Plugins
- **Statistical**: GLM, correlation, connectivity
- **Machine Learning**: Classification, regression, clustering
- **Feature Extraction**: ROI analysis, connectivity matrices

### 4. Integration Plugins
- **External Tools**: FSL, SPM, AFNI wrappers
- **Data Formats**: BIDS, DICOM, custom formats
- **Cloud Services**: AWS, Google Cloud, Azure

---

## 📝 Best Practices

### 1. Plugin Design

```python
class WellDesignedPlugin(PreprocessingPlugin):
    """
    Example of a well-designed plugin.
    """
    
    def __init__(self, param1=1.0, param2=True, verbose=True):
        """
        Initialize with clear parameter documentation.
        
        Parameters
        ----------
        param1 : float, default=1.0
            Description of what param1 does
        param2 : bool, default=True
            Description of what param2 controls
        verbose : bool, default=True
            Whether to print progress information
        """
        super().__init__("WellDesignedPlugin")
        self.param1 = param1
        self.param2 = param2
        self.verbose = verbose
    
    def run(self, img, **kwargs):
        """
        Main processing method with proper error handling.
        """
        try:
            if self.verbose:
                print(f"Running {self.name} with param1={self.param1}")
            
            # Validate input
            if img is None:
                raise ValueError("Input image cannot be None")
            
            # Process the image
            result = self._process_image(img)
            
            if self.verbose:
                print(f"{self.name} completed successfully")
            
            return result
            
        except Exception as e:
            warnings.warn(f"{self.name} failed: {str(e)}")
            # Return original image or raise exception based on your needs
            return img
    
    def _process_image(self, img):
        """
        Separate processing logic for better organization.
        """
        # Your processing logic here
        pass
    
    def _validate_parameters(self):
        """
        Validate plugin parameters.
        """
        if self.param1 < 0:
            raise ValueError("param1 must be non-negative")
```

### 2. Error Handling

```python
def run(self, img, **kwargs):
    """
    Robust error handling example.
    """
    try:
        # Check prerequisites
        if not self._check_dependencies():
            warnings.warn(f"{self.name}: Missing dependencies, skipping")
            return img
        
        # Validate input
        if not self._validate_input(img):
            raise ValueError(f"{self.name}: Invalid input image")
        
        # Process with timeout
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError(f"{self.name}: Processing timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout
        
        try:
            result = self._process_image(img)
        finally:
            signal.alarm(0)  # Cancel timeout
        
        return result
        
    except TimeoutError:
        warnings.warn(f"{self.name}: Processing timed out, returning original image")
        return img
    except Exception as e:
        warnings.warn(f"{self.name}: Failed with error: {str(e)}")
        return img
```

### 3. Performance Optimization

```python
def run(self, img, **kwargs):
    """
    Performance-optimized plugin example.
    """
    # Use GPU if available and beneficial
    if self._should_use_gpu(img):
        return self._run_gpu(img)
    else:
        return self._run_cpu(img)

def _should_use_gpu(self, img):
    """
    Determine if GPU processing would be beneficial.
    """
    data_size = img.get_fdata().size
    return data_size > 1e6  # Use GPU for large datasets

def _run_gpu(self, img):
    """
    GPU-accelerated processing.
    """
    try:
        import cupy as cp
        data = cp.asarray(img.get_fdata())
        # GPU processing here
        result = cp.asnumpy(processed_data)
        return nib.Nifti1Image(result, img.affine)
    except ImportError:
        return self._run_cpu(img)
```

---

## 🔄 Plugin Chaining

### Creating Complex Pipelines

```python
# Create multiple plugins
qc_plugin = QualityControlPlugin(save_report=True)
smoothing_plugin = GaussianSmoothingPlugin(sigma=2.0, use_gpu=True)
custom_plugin = MyCustomPlugin(parameter1=1.5)

# Chain them together
plugins = [qc_plugin, smoothing_plugin, custom_plugin]

# Run the pipeline
preproc = Preprocessor(plugins=plugins)
result = preproc.run_pipeline(img, pipeline='custom')
```

### Conditional Plugin Execution

```python
class ConditionalPlugin(PreprocessingPlugin):
    """
    Plugin that conditionally applies processing.
    """
    
    def __init__(self, condition_func, true_plugin, false_plugin=None):
        super().__init__("Conditional")
        self.condition_func = condition_func
        self.true_plugin = true_plugin
        self.false_plugin = false_plugin
    
    def run(self, img, **kwargs):
        if self.condition_func(img):
            return self.true_plugin.run(img, **kwargs)
        elif self.false_plugin:
            return self.false_plugin.run(img, **kwargs)
        else:
            return img

# Usage example
def should_smooth(img):
    """Determine if smoothing should be applied."""
    data = img.get_fdata()
    return np.std(data) > 0.1

conditional_smoothing = ConditionalPlugin(
    condition_func=should_smooth,
    true_plugin=GaussianSmoothingPlugin(sigma=1.0),
    false_plugin=None
)
```

---

## 📊 Testing Your Plugins

### Unit Testing

```python
import unittest
import numpy as np
import nibabel as nib

class TestMyCustomPlugin(unittest.TestCase):
    
    def setUp(self):
        """Create test data."""
        self.test_data = np.random.randn(32, 32, 32)
        self.test_img = nib.Nifti1Image(self.test_data, np.eye(4))
        self.plugin = MyCustomPlugin(parameter1=2.0)
    
    def test_basic_functionality(self):
        """Test basic plugin functionality."""
        result = self.plugin.run(self.test_img)
        
        # Check that result is a NIfTI image
        self.assertIsInstance(result, nib.Nifti1Image)
        
        # Check that data shape is preserved
        self.assertEqual(result.shape, self.test_img.shape)
    
    def test_parameter_effects(self):
        """Test that parameters affect the output."""
        plugin1 = MyCustomPlugin(parameter1=1.0)
        plugin2 = MyCustomPlugin(parameter1=2.0)
        
        result1 = plugin1.run(self.test_img)
        result2 = plugin2.run(self.test_img)
        
        # Results should be different
        self.assertFalse(np.array_equal(
            result1.get_fdata(), 
            result2.get_fdata()
        ))
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with None input
        with self.assertWarns(UserWarning):
            result = self.plugin.run(None)
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
def test_plugin_in_pipeline():
    """Test plugin integration in preprocessing pipeline."""
    # Create test image
    test_img = create_test_brain_data()
    
    # Create plugin
    plugin = MyCustomPlugin(parameter1=1.5)
    
    # Test in pipeline
    preproc = Preprocessor(plugins=[plugin])
    result = preproc.run_pipeline(test_img, pipeline='custom')
    
    # Verify result
    assert result is not None
    assert result.shape == test_img.shape
    print("Plugin integration test passed!")

# Run the test
test_plugin_in_pipeline()
```

---

## 📦 Sharing Your Plugins

### 1. GitHub Repository

```bash
# Create a plugin repository
mkdir my-brain-mapping-plugins
cd my-brain-mapping-plugins

# Create plugin package
mkdir my_plugins
touch my_plugins/__init__.py
touch my_plugins/my_custom_plugin.py

# Add setup.py
cat > setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="my-brain-mapping-plugins",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "brain-mapping-toolkit",
        "numpy",
        "nibabel"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom plugins for brain mapping toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/my-brain-mapping-plugins",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
EOF
```

### 2. Plugin Registry

Consider contributing your plugins to a central registry:

```python
# In your plugin package
PLUGIN_REGISTRY = {
    "my_custom_plugin": {
        "class": MyCustomPlugin,
        "description": "Custom preprocessing for specific analysis",
        "author": "Your Name",
        "version": "0.1.0",
        "dependencies": ["numpy", "scipy"]
    }
}
```

### 3. Documentation

Create comprehensive documentation for your plugins:

```markdown
# My Custom Plugin

## Description
Brief description of what the plugin does.

## Installation
```bash
pip install my-brain-mapping-plugins
```

## Usage
```python
from my_plugins import MyCustomPlugin
from brain_mapping.core.preprocessor import Preprocessor

plugin = MyCustomPlugin(parameter1=2.0)
preproc = Preprocessor(plugins=[plugin])
result = preproc.run_pipeline(img, pipeline='custom')
```

## Parameters
- `parameter1` (float): Description of parameter1
- `parameter2` (bool): Description of parameter2

## Examples
[Include usage examples]

## Citation
If you use this plugin, please cite:
[Your citation information]
```

---

## 🎯 Advanced Topics

### 1. GPU-Accelerated Plugins

```python
class GPUAcceleratedPlugin(PreprocessingPlugin):
    """
    Example of a GPU-accelerated plugin.
    """
    
    def __init__(self, use_gpu=True, precision='float32'):
        super().__init__("GPUAccelerated")
        self.use_gpu = use_gpu
        self.precision = precision
    
    def run(self, img, **kwargs):
        if self.use_gpu and self._gpu_available():
            return self._run_gpu(img)
        else:
            return self._run_cpu(img)
    
    def _gpu_available(self):
        """Check if GPU is available."""
        try:
            import cupy as cp
            return True
        except ImportError:
            return False
    
    def _run_gpu(self, img):
        """GPU-accelerated processing."""
        import cupy as cp
        
        # Convert to GPU array
        data = cp.asarray(img.get_fdata(), dtype=self.precision)
        
        # GPU processing
        processed = self._gpu_processing(data)
        
        # Convert back to CPU
        result = cp.asnumpy(processed)
        
        return nib.Nifti1Image(result, img.affine)
    
    def _gpu_processing(self, data):
        """GPU processing logic."""
        # Your GPU-accelerated algorithm here
        return data  # Placeholder
```

### 2. Memory-Efficient Plugins

```python
class MemoryEfficientPlugin(PreprocessingPlugin):
    """
    Plugin that processes data in chunks to save memory.
    """
    
    def __init__(self, chunk_size=1000):
        super().__init__("MemoryEfficient")
        self.chunk_size = chunk_size
    
    def run(self, img, **kwargs):
        data = img.get_fdata()
        
        if data.ndim == 3:
            return self._process_3d(data, img.affine)
        elif data.ndim == 4:
            return self._process_4d(data, img.affine)
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
    
    def _process_4d(self, data, affine):
        """Process 4D data in chunks."""
        result = np.zeros_like(data)
        
        for i in range(0, data.shape[-1], self.chunk_size):
            end_idx = min(i + self.chunk_size, data.shape[-1])
            chunk = data[..., i:end_idx]
            
            # Process chunk
            processed_chunk = self._process_chunk(chunk)
            result[..., i:end_idx] = processed_chunk
            
            print(f"Processed chunk {i//self.chunk_size + 1}/{(data.shape[-1] + self.chunk_size - 1)//self.chunk_size}")
        
        return nib.Nifti1Image(result, affine)
    
    def _process_chunk(self, chunk):
        """Process a single chunk of data."""
        # Your chunk processing logic here
        return chunk  # Placeholder
```

---

## 🚀 Getting Help

### Resources
- **GitHub Issues**: Report bugs or request features
- **Documentation**: [Link to main documentation]
- **Examples**: [Link to example plugins]
- **Community**: [Link to community forum]

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add your plugin with tests
4. Submit a pull request

### Contact
- **Email**: plugins@brain-mapping.org
- **GitHub**: https://github.com/your-org/brain-mapping
- **Discussions**: https://github.com/your-org/brain-mapping/discussions

---

**Happy plugin development!** 🧩🚀 