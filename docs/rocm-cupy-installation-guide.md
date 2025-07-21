# ROCm & CuPy Installation Guide for Brain Mapping Toolkit

## 🚀 AMD GPU Acceleration Setup

This guide provides step-by-step instructions for setting up ROCm and CuPy to enable GPU acceleration in the brain mapping toolkit on AMD hardware.

---

## 📋 System Requirements

### Hardware Requirements
- **GPU**: AMD Radeon RX 5000 series or newer (RX 5500, RX 5600, RX 5700, RX 6000, RX 7000)
- **CPU**: x86_64 architecture (Intel/AMD)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB free space for ROCm installation

### Software Requirements
- **OS**: Ubuntu 20.04 LTS or newer (recommended)
- **Kernel**: Linux 5.4+ (for ROCm 5.x)
- **Python**: 3.8, 3.9, or 3.10
- **Package Manager**: apt (Ubuntu/Debian)

### Supported GPU Models
| Series | Models | ROCm Support | Performance |
|--------|--------|--------------|-------------|
| RX 5000 | RX 5500, RX 5600, RX 5700 | ✅ Full | Good |
| RX 6000 | RX 6600, RX 6700, RX 6800, RX 6900 | ✅ Full | Excellent |
| RX 7000 | RX 7600, RX 7700, RX 7800, RX 7900 | ✅ Full | Excellent |
| Pro | W5500, W6600, W6800, W7900 | ✅ Full | Professional |

---

## 🔧 Step 1: System Preparation

### 1.1 Update System Packages

```bash
# Update package list
sudo apt update

# Upgrade existing packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential cmake git wget curl
```

### 1.2 Check GPU Compatibility

```bash
# Check if your GPU is detected
lspci | grep -i amd

# Check kernel version
uname -r

# Check system architecture
uname -m
```

**Expected output for compatible system:**
```
01:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT]
Linux 5.15.0-generic
x86_64
```

### 1.3 Install Required Dependencies

```bash
# Install ROCm dependencies
sudo apt install -y \
    libnuma-dev \
    libpci-dev \
    libdrm-dev \
    libelf-dev \
    libssl-dev \
    libudev-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv
```

---

## 🎯 Step 2: ROCm Installation

### 2.1 Add ROCm Repository

```bash
# Add AMD ROCm repository
wget https://repo.radeon.com/amdgpu-install/5.7.3/ubuntu/jammy/amdgpu-install_5.7.3.50700-1_all.deb
sudo apt install ./amdgpu-install_5.7.3.50700-1_all.deb

# Add ROCm repository
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    sudo gpg --dearmor | sudo tee /etc/apt/keyrings/rocm-keyring.gpg > /dev/null

echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/debian jammy main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Update package list
sudo apt update
```

### 2.2 Install ROCm

```bash
# Install ROCm (this may take 10-20 minutes)
sudo apt install -y rocm-hip-sdk

# Install additional ROCm packages
sudo apt install -y \
    rocm-opencl-sdk \
    rocm-utils \
    rocm-dev
```

### 2.3 Configure Environment

```bash
# Add ROCm to PATH
echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' >> ~/.bashrc

# Reload shell configuration
source ~/.bashrc
```

### 2.4 Verify ROCm Installation

```bash
# Check ROCm version
rocm-smi

# Check HIP installation
hipconfig

# Test GPU detection
rocm-smi --showproductname
```

**Expected output:**
```
==================== ROCm System Management Interface ====================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
0    45.0c  45.0W   1800M   1750M    0%   auto  180.0W    0%   0%   
================================================================================
============================= End of ROCm SMI Log ==============================
```

---

## 🐍 Step 3: Python Environment Setup

### 3.1 Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv brain_mapping_env

# Activate environment
source brain_mapping_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3.2 Install Core Dependencies

```bash
# Install core scientific packages
pip install numpy scipy matplotlib

# Install neuroimaging packages
pip install nibabel pydicom

# Install development tools
pip install pytest black flake8
```

---

## ⚡ Step 4: CuPy Installation

### 4.1 Install CuPy with ROCm Support

```bash
# Install CuPy with ROCm backend
pip install cupy-cuda12x

# Alternative: Install from source for latest features
# pip install cupy-cuda12x --pre
```

### 4.2 Verify CuPy Installation

```python
# Test CuPy installation
python3 -c "
import cupy as cp
print('CuPy version:', cp.__version__)
print('GPU devices:', cp.cuda.runtime.getDeviceCount())
print('Current device:', cp.cuda.Device().id)
print('GPU memory:', cp.cuda.runtime.memGetInfo())
"
```

**Expected output:**
```
CuPy version: 12.0.0
GPU devices: 1
Current device: 0
GPU memory: (8589934592, 8589934592)
```

### 4.3 Test GPU Acceleration

```python
# Test basic GPU operations
import cupy as cp
import numpy as np

# Create test arrays
a_cpu = np.random.randn(1000, 1000)
a_gpu = cp.asarray(a_cpu)

# Test GPU computation
b_gpu = cp.dot(a_gpu, a_gpu.T)
b_cpu = b_gpu.get()

print("GPU computation successful!")
print("Result shape:", b_cpu.shape)
```

---

## 🧠 Step 5: Brain Mapping Toolkit Installation

### 5.1 Install Toolkit

```bash
# Clone repository (if not already done)
git clone https://github.com/your-org/brain-mapping.git
cd brain-mapping

# Install in development mode
pip install -e .
```

### 5.2 Test GPU Acceleration

```python
# Test brain mapping toolkit GPU features
from brain_mapping.core.preprocessor import Preprocessor, GaussianSmoothingPlugin
import nibabel as nib
import numpy as np

# Create test data
test_data = np.random.randn(64, 64, 64)
test_img = nib.Nifti1Image(test_data, np.eye(4))

# Test GPU smoothing
plugin = GaussianSmoothingPlugin(sigma=1.0, use_gpu=True, precision='float16')
preproc = Preprocessor(plugins=[plugin])

print("Testing GPU-accelerated smoothing...")
result = preproc.run_pipeline(test_img, pipeline='custom')
print("✅ GPU acceleration working!")
```

---

## 🔍 Step 6: Performance Optimization

### 6.1 GPU Memory Management

```python
# Monitor GPU memory usage
import cupy as cp

def print_gpu_memory():
    """Print current GPU memory usage."""
    free, total = cp.cuda.runtime.memGetInfo()
    used = total - free
    print(f"GPU Memory: {used/1e9:.1f}GB used / {total/1e9:.1f}GB total")

# Use in your code
print_gpu_memory()
```

### 6.2 Mixed-Precision Optimization

```python
# Use mixed-precision for memory efficiency
from brain_mapping.core.preprocessor import Preprocessor

# For memory-constrained GPUs, use float16
preproc = Preprocessor(gpu_enabled=True, precision='float16')

# For maximum precision, use float32
preproc = Preprocessor(gpu_enabled=True, precision='float32')
```

### 6.3 Batch Processing

```python
# Process large datasets in batches
def process_large_dataset(data_paths, batch_size=4):
    """Process multiple images in batches."""
    results = []
    
    for i in range(0, len(data_paths), batch_size):
        batch = data_paths[i:i+batch_size]
        
        # Process batch
        for path in batch:
            img = nib.load(path)
            result = preproc.run_pipeline(img, pipeline='advanced')
            results.append(result)
        
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()
    
    return results
```

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue 1: ROCm Installation Fails

```bash
# Check system compatibility
lspci | grep -i amd
uname -r

# Reinstall with specific version
sudo apt remove rocm-hip-sdk
sudo apt install rocm-hip-sdk=5.7.3.50700-1
```

#### Issue 2: CuPy Import Error

```bash
# Reinstall CuPy with correct backend
pip uninstall cupy-cuda12x
pip install cupy-cuda12x --force-reinstall

# Check CUDA compatibility
python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

#### Issue 3: GPU Memory Errors

```python
# Clear GPU memory
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()

# Use smaller batch sizes
batch_size = 2  # Reduce from 4 to 2
```

#### Issue 4: Performance Issues

```bash
# Check GPU utilization
rocm-smi

# Monitor system resources
htop
nvidia-smi  # If you have both NVIDIA and AMD GPUs
```

### Performance Tuning

#### 1. GPU Frequency Settings

```bash
# Set GPU to performance mode
sudo rocm-smi --setperflevel high

# Check current settings
rocm-smi --showperflevel
```

#### 2. Memory Allocation

```python
# Optimize memory allocation
import cupy as cp

# Set memory pool size
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# Monitor memory usage
print_gpu_memory()
```

#### 3. Thread Configuration

```bash
# Set number of CPU threads for preprocessing
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

---

## 📊 Benchmarking

### Performance Testing Script

```python
import time
import numpy as np
import nibabel as nib
from brain_mapping.core.preprocessor import Preprocessor

def benchmark_smoothing(img, sigma=1.0):
    """Benchmark smoothing performance."""
    
    # CPU test
    print("Testing CPU smoothing...")
    preproc_cpu = Preprocessor(gpu_enabled=False)
    start_time = time.time()
    result_cpu = preproc_cpu.run_pipeline(img, pipeline='advanced')
    cpu_time = time.time() - start_time
    
    # GPU test
    print("Testing GPU smoothing...")
    preproc_gpu = Preprocessor(gpu_enabled=True, precision='float32')
    start_time = time.time()
    result_gpu = preproc_gpu.run_pipeline(img, pipeline='advanced')
    gpu_time = time.time() - start_time
    
    # Mixed-precision test
    print("Testing mixed-precision smoothing...")
    preproc_mixed = Preprocessor(gpu_enabled=True, precision='float16')
    start_time = time.time()
    result_mixed = preproc_mixed.run_pipeline(img, pipeline='advanced')
    mixed_time = time.time() - start_time
    
    # Results
    print(f"\n📊 Performance Results:")
    print(f"CPU (float32):     {cpu_time:.3f}s")
    print(f"GPU (float32):     {gpu_time:.3f}s")
    print(f"GPU (float16):     {mixed_time:.3f}s")
    print(f"Speedup (float32): {cpu_time/gpu_time:.1f}x")
    print(f"Speedup (float16): {cpu_time/mixed_time:.1f}x")
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'mixed_time': mixed_time,
        'speedup_32': cpu_time/gpu_time,
        'speedup_16': cpu_time/mixed_time
    }

# Run benchmark
test_img = create_test_brain_data(shape=(128, 128, 128))
results = benchmark_smoothing(test_img)
```

---

## 🔗 Additional Resources

### Official Documentation
- [AMD ROCm Documentation](https://rocmdocs.amd.com/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [Brain Mapping Toolkit Documentation](https://github.com/your-org/brain-mapping)

### Community Support
- [ROCm GitHub Issues](https://github.com/RadeonOpenCompute/ROCm/issues)
- [CuPy GitHub Issues](https://github.com/cupy/cupy/issues)
- [Brain Mapping Discussions](https://github.com/your-org/brain-mapping/discussions)

### Performance Optimization
- [AMD GPU Performance Tuning](https://rocmdocs.amd.com/en/latest/deploy/linux/prerequisites.html)
- [CuPy Performance Tips](https://docs.cupy.dev/en/stable/user_guide/performance.html)
- [Memory Management Best Practices](https://docs.cupy.dev/en/stable/user_guide/memory.html)

---

## ✅ Verification Checklist

Before proceeding with brain mapping analysis, verify:

- [ ] ROCm installation: `rocm-smi` shows GPU
- [ ] CuPy installation: `import cupy` works
- [ ] GPU memory: Sufficient VRAM for your datasets
- [ ] Brain mapping toolkit: GPU acceleration enabled
- [ ] Performance: Benchmark shows expected speedup
- [ ] Stability: No crashes during extended use

---

## 🚀 Next Steps

After successful installation:

1. **Run the demo notebook** to test all features
2. **Benchmark on your datasets** to measure performance gains
3. **Create custom plugins** for your specific analysis needs
4. **Share results** with the community
5. **Contribute improvements** to the toolkit

**Happy GPU-accelerated brain mapping!** 🧠⚡ 