#!/usr/bin/env python3
"""
Test script to check device compatibility and MPS fallback
"""

import torch
import sys
import os

def test_device_compatibility():
    """Test device compatibility and MPS fallback"""
    print("🔍 Testing device compatibility...")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
    
    # Check MPS
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    # Test MPS with problematic operations
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("\n🧪 Testing MPS operations...")
        try:
            # Test 1: Simple tensor operations
            print("  - Testing basic tensor operations...")
            x = torch.randn(1000, 1000, device='mps')
            y = torch.randn(1000, 1000, device='mps')
            z = x + y
            print("  ✅ Basic operations work")
            
            # Test 2: Conv1d with large output channels
            print("  - Testing Conv1d with large output channels...")
            try:
                conv = torch.nn.Conv1d(1, 100000, 3, device='mps')
                x = torch.randn(1, 1, 1000, device='mps')
                y = conv(x)
                print("  ✅ Large Conv1d works")
            except Exception as e:
                print(f"  ❌ Large Conv1d failed: {e}")
                print("  🔄 MPS has limitations, will use CPU fallback")
                return "cpu"
            
            # Test 3: Complex operations
            print("  - Testing complex operations...")
            x = torch.randn(1, 1, 1000, device='mps')
            conv1 = torch.nn.Conv1d(1, 64, 3, device='mps')
            conv2 = torch.nn.Conv1d(64, 128, 3, device='mps')
            y = conv2(conv1(x))
            print("  ✅ Complex operations work")
            
            print("  ✅ MPS is fully compatible")
            return "mps"
            
        except Exception as e:
            print(f"  ❌ MPS test failed: {e}")
            print("  🔄 Falling back to CPU")
            return "cpu"
    
    # Test CPU
    print("\n🧪 Testing CPU operations...")
    try:
        x = torch.randn(1000, 1000, device='cpu')
        y = torch.randn(1000, 1000, device='cpu')
        z = x + y
        print("  ✅ CPU operations work")
        return "cpu"
    except Exception as e:
        print(f"  ❌ CPU test failed: {e}")
        return None

def get_optimal_device():
    """Get the optimal device for inference"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS compatibility
        result = test_device_compatibility()
        return result
    else:
        return "cpu"

def main():
    """Main test function"""
    print("🚀 ACE-Step Device Compatibility Test")
    print("=" * 50)
    
    # Test device compatibility
    device = get_optimal_device()
    
    print(f"\n📋 Test Results:")
    print(f"  Recommended device: {device}")
    
    if device == "mps":
        print("  ✅ MPS is recommended for Apple Silicon")
    elif device == "cuda":
        print("  ✅ CUDA is recommended for NVIDIA GPU")
    elif device == "cpu":
        print("  ⚠️  CPU is recommended (slower but compatible)")
    else:
        print("  ❌ No compatible device found")
        sys.exit(1)
    
    # Test with actual model loading
    print(f"\n🧪 Testing model loading on {device}...")
    try:
        # Set environment variables
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Test basic model operations
        if device == "mps":
            # Test with smaller model first
            x = torch.randn(1, 1, 1000, device='mps')
            conv = torch.nn.Conv1d(1, 64, 3, device='mps')
            y = conv(x)
            print("  ✅ MPS model operations work")
        elif device == "cuda":
            x = torch.randn(1, 1, 1000, device='cuda')
            conv = torch.nn.Conv1d(1, 64, 3, device='cuda')
            y = conv(x)
            print("  ✅ CUDA model operations work")
        else:
            x = torch.randn(1, 1, 1000, device='cpu')
            conv = torch.nn.Conv1d(1, 64, 3, device='cpu')
            y = conv(x)
            print("  ✅ CPU model operations work")
            
        print(f"\n🎉 Device compatibility test passed!")
        print(f"   Use device: {device}")
        
    except Exception as e:
        print(f"  ❌ Model loading test failed: {e}")
        print(f"  🔄 Try using CPU fallback")
        return "cpu"
    
    return device

if __name__ == "__main__":
    device = main()
    print(f"\n📝 Recommendation: Use device='{device}' in your service")
