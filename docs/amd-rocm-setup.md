# AMD ROCm GPU Setup for Brain Mapping Toolkit

## Overview

This document provides guidance for setting up the Brain Mapping Toolkit with AMD GPU acceleration using ROCm (Radeon Open Compute) and HIP (Heterogeneous-Compute Interface for Portability).

## AMD GPU Requirements

### Supported AMD GPUs (ROCm 5.x+)
- [ ] **AMD RX 6000/7000 Series**: RX 6800 XT, RX 6900 XT, RX 7800 XT, RX 7900 XTX
- [ ] **AMD Instinct Series**: MI100, MI200, MI250, MI300
- [ ] **AMD Radeon Pro**: W6800, W6900, V620, V340
- [ ] **APU Support**: Ryzen 7000 series with RDNA2/3 integrated graphics

### System Requirements
- [ ] **OS**: Ubuntu 20.04/22.04, RHEL 8/9, SLES 15
- [ ] **Kernel**: Linux kernel 5.4+
- [ ] **Memory**: 16GB+ system RAM, 8GB+ VRAM
- [ ] **ROCm Version**: 5.4.0 or later

## Installation Steps

### 1. ROCm Installation

#### Ubuntu/Debian:
```bash
# Add ROCm repository
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install ROCm
sudo apt update
sudo apt install rocm-dev rocm-libs rocm-utils

# Add user to render group
sudo usermod -a -G render,video $USER
```

#### RHEL/CentOS:
```bash
# Add ROCm repository
sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=rocm
baseurl=https://repo.radeon.com/rocm/yum/5.7/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

# Install ROCm
sudo dnf install rocm-dev rocm-libs rocm-utils
```

### 2. Python Environment Setup

```bash
# Create conda environment with ROCm support
conda create -n brain-mapping-amd python=3.10 -y
conda activate brain-mapping-amd

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install CuPy with ROCm support
pip install cupy-rocm-5-0

# Install other ROCm-enabled packages
pip install numba[rocm]
```

### 3. Verification

```python
# Test ROCm/HIP availability
import torch
print("PyTorch ROCm available:", torch.cuda.is_available())
print("ROCm device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name())

# Test CuPy ROCm
import cupy as cp
print("CuPy ROCm available:", cp.cuda.is_available())
```

## Code Adaptations for AMD

### 1. GPU Detection and Fallback

```python
def detect_gpu_vendor():
    """Detect GPU vendor and return appropriate backend."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name().lower()
            if 'nvidia' in device_name or 'tesla' in device_name or 'quadro' in device_name:
                return 'cuda'
            elif 'amd' in device_name or 'radeon' in device_name or 'rx' in device_name:
                return 'rocm'
    except ImportError:
        pass
    
    # Check for ROCm directly
    try:
        import subprocess
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return 'rocm'
    except FileNotFoundError:
        pass
    
    return 'cpu'

# Usage in code
GPU_BACKEND = detect_gpu_vendor()
print(f"Using GPU backend: {GPU_BACKEND}")
```

### 2. Memory Management

```python
def get_optimal_batch_size():
    """Calculate optimal batch size based on available GPU memory."""
    if GPU_BACKEND == 'rocm':
        import torch
        if torch.cuda.is_available():
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # Conservative allocation (use 80% of available memory)
            return min(32, max(1, int(total_memory * 0.8 / (1024**3))))  # GB to batch size
    return 8  # Default batch size for CPU
```

### 3. Kernel Compilation

```python
# HIP kernel for image processing
hip_kernel = """
extern "C" __global__
void gaussian_filter_hip(float* input, float* output, int width, int height, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // Gaussian filter implementation
        float sum = 0.0f;
        float norm = 0.0f;
        
        for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
                int nx = idx + dx;
                int ny = idy + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float weight = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
                    sum += input[ny * width + nx] * weight;
                    norm += weight;
                }
            }
        }
        
        output[idy * width + idx] = sum / norm;
    }
}
"""

def compile_hip_kernel():
    """Compile HIP kernel for AMD GPUs."""
    if GPU_BACKEND == 'rocm':
        try:
            import cupy as cp
            return cp.RawKernel(hip_kernel, 'gaussian_filter_hip')
        except ImportError:
            print("CuPy not available, falling back to CPU")
    return None
```

## Performance Optimization

### 1. Memory Allocation Strategies

```python
class AMDMemoryManager:
    """Optimized memory management for AMD GPUs."""
    
    def __init__(self):
        self.memory_pool = None
        if GPU_BACKEND == 'rocm':
            try:
                import cupy as cp
                self.memory_pool = cp.get_default_memory_pool()
                # Set memory pool size to 80% of GPU memory
                self.memory_pool.set_limit(size=int(self._get_gpu_memory() * 0.8))
            except ImportError:
                pass
    
    def _get_gpu_memory(self):
        """Get total GPU memory in bytes."""
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0
    
    def allocate(self, shape, dtype=cp.float32):
        """Allocate GPU memory with pooling."""
        if GPU_BACKEND == 'rocm' and self.memory_pool:
            return cp.zeros(shape, dtype=dtype)
        else:
            import numpy as np
            return np.zeros(shape, dtype=dtype)
```

### 2. Kernel Optimization

```python
def optimize_for_amd():
    """AMD-specific optimizations."""
    if GPU_BACKEND == 'rocm':
        # Set ROCm-specific environment variables
        import os
        os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Use first GPU
        os.environ['ROCR_VISIBLE_DEVICES'] = '0'
        
        # Optimize CuPy for AMD
        try:
            import cupy as cp
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            # Enable memory pool for better performance
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=2**30)  # 1GB limit
        except ImportError:
            pass
```

## Benchmarking

### Performance Comparison

```python
def benchmark_operations():
    """Benchmark key operations on AMD vs CPU."""
    import time
    import numpy as np
    
    # Test data
    data_size = (1024, 1024, 64)  # Typical brain volume
    
    if GPU_BACKEND == 'rocm':
        import cupy as cp
        # GPU benchmark
        gpu_data = cp.random.random(data_size, dtype=cp.float32)
        
        start_time = time.time()
        result = cp.fft.fftn(gpu_data)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"AMD GPU FFT time: {gpu_time:.4f}s")
    
    # CPU benchmark
    cpu_data = np.random.random(data_size).astype(np.float32)
    
    start_time = time.time()
    result = np.fft.fftn(cpu_data)
    cpu_time = time.time() - start_time
    
    print(f"CPU FFT time: {cpu_time:.4f}s")
    
    if GPU_BACKEND == 'rocm':
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
```

## Troubleshooting

### Common Issues

1. [ ] **ROCm Not Detected**
   - Verify GPU compatibility: `rocm-smi`
   - Check driver version: `cat /sys/module/amdgpu/version`
   - Reinstall ROCm if necessary

2. [ ] **PyTorch Not Using GPU**
   - Check PyTorch installation: `pip show torch`
   - Verify ROCm support: `python -c "import torch; print(torch.cuda.is_available())"`
   - Reinstall with correct ROCm version

3. [ ] **Memory Errors**
   - Reduce batch size in config
   - Enable memory pooling
   - Check available GPU memory: `rocm-smi --showmeminfo vram`

4. [ ] **Performance Issues**
   - Enable mixed precision training
   - Optimize kernel launch parameters
   - Use async memory transfers

### Environment Variables

```bash
# ROCm optimization settings
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # For unsupported GPUs
export PYTORCH_ROCM_ARCH="gfx1030;gfx1100"  # Specify GPU architecture
```

## Integration with Brain Mapping Toolkit

### Configuration Updates

Update `src/brain_mapping/utils/config.py`:

```python
# Add AMD-specific settings
AMD_SETTINGS = {
    'enable_rocm': True,
    'memory_fraction': 0.8,
    'kernel_optimization': 'aggressive',
    'mixed_precision': True,
    'async_operations': True
}

# Update GPU detection
def configure_gpu():
    """Configure GPU backend based on available hardware."""
    backend = detect_gpu_vendor()
    
    if backend == 'rocm':
        print("Configuring for AMD ROCm...")
        optimize_for_amd()
        return AMD_SETTINGS
    elif backend == 'cuda':
        print("Configuring for NVIDIA CUDA...")
        return CUDA_SETTINGS
    else:
        print("Using CPU backend...")
        return CPU_SETTINGS
```

This setup ensures optimal performance on AMD hardware while maintaining compatibility with NVIDIA and CPU-only systems.
