# 🔧 ACE-Step Device Compatibility Guide

## 🚨 MPS Limitation Issue

**Problem**: MPS (Metal Performance Shaders) on Apple Silicon has a limitation with large output channels (>65536) in Conv1d operations, which ACE-Step requires.

**Error**: `NotImplementedError: Output channels > 65536 not supported at the MPS device.`

## ✅ Solution

### 1. Automatic Device Detection
The service now automatically detects the best available device:

```python
# Device priority:
1. CUDA (NVIDIA GPU) - Full support
2. MPS (Apple Silicon) - Tested, fallback to CPU if incompatible
3. CPU (Any system) - Full support, slower but compatible
```

### 2. MPS Fallback
When MPS is detected but incompatible:
- Automatically falls back to CPU
- Disables torch.compile for better compatibility
- Enables CPU offloading for memory efficiency

### 3. Environment Variables
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
```

## 🧪 Testing Device Compatibility

### Run Device Test
```bash
python test_device.py
```

### Expected Output
```
🚀 ACE-Step Device Compatibility Test
==================================================
🔍 Testing device compatibility...
==================================================
PyTorch version: 2.6.0.dev20241112
CUDA available: False
MPS available: True

🧪 Testing MPS operations...
  - Testing basic tensor operations...
  ✅ Basic operations work
  - Testing Conv1d with large output channels...
  ❌ Large Conv1d failed: Output channels > 65536 not supported at the MPS device. 
  🔄 MPS has limitations, will use CPU fallback

📋 Test Results:
  Recommended device: cpu
  ⚠️  CPU is recommended (slower but compatible)
```

## 🚀 Rebuilding with Device Fix

### 1. Rebuild Service
```bash
./rebuild_with_device_fix.sh
```

### 2. Manual Rebuild
```bash
# Test device compatibility
python test_device.py

# Clean previous builds
bentoml delete ace-step-audio-service --yes

# Build new service
bentoml build --force

# Rebuild Docker image
docker build -f Dockerfile.simple -t ace-step-audio:latest .
```

## 📊 Performance Comparison

| Device | Speed | Memory | Compatibility |
|--------|-------|--------|----------------|
| CUDA   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| MPS    | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ (Limited) |
| CPU    | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🔧 Troubleshooting

### MPS Issues
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Test MPS operations
python -c "
import torch
try:
    x = torch.randn(1, 1, 1000, device='mps')
    conv = torch.nn.Conv1d(1, 100000, 3, device='mps')
    y = conv(x)
    print('MPS works')
except Exception as e:
    print(f'MPS failed: {e}')
"
```

### CPU Fallback
```bash
# Force CPU usage
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""

# Run service
python -m bentoml serve service:acestepaudoservice
```

### Memory Issues
```bash
# Increase memory limit
docker run --memory=8g -p 3000:3000 ace-step-audio:latest

# Enable CPU offloading
export CPU_OFFLOAD=true
```

## 🎯 Best Practices

### 1. Development
- Use CPU for development and testing
- Test with small models first
- Monitor memory usage

### 2. Production
- Use CUDA if available (NVIDIA GPU)
- Use CPU with sufficient RAM (16GB+)
- Avoid MPS for production workloads

### 3. Monitoring
```bash
# Check device usage
docker stats

# Monitor memory
htop

# Check GPU usage (if CUDA)
nvidia-smi
```

## 📝 Configuration

### Service Configuration
```python
# In service.py
DEVICE = get_optimal_device()  # Auto-detection
config.device_id = 0 if DEVICE in ["cpu", "mps"] else config.device_id
```

### Docker Configuration
```dockerfile
# In Dockerfile
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV TOKENIZERS_PARALLELISM=false
```

## 🆘 Support

### Common Issues
1. **MPS Error**: Use CPU fallback
2. **Memory Error**: Increase Docker memory limit
3. **Slow Performance**: Use CUDA if available
4. **Model Loading Error**: Check checkpoint path

### Debug Commands
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check device availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"

# Test service
curl http://localhost:3000/health
```

---

**🎉 With these fixes, ACE-Step will work on any system with proper device fallback!**
