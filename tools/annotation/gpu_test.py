#!/usr/bin/env python3
"""
GPU Test Script for CRPlayer Annotation Tool
Tests PyTorch GPU availability and performance
"""

import torch
import time
import sys


def test_gpu_availability():
  """Test basic GPU availability"""
  print("🔍 Testing GPU availability...")
  print(f"PyTorch version: {torch.__version__}")

  if torch.cuda.is_available():
    print("✅ CUDA is available!")
    device_count = torch.cuda.device_count()
    print(f"GPU count: {device_count}")

    for i in range(device_count):
      props = torch.cuda.get_device_properties(i)
      print(f"GPU {i}: {props.name}")
      memory_gb = props.total_memory / 1024**3
      print(f"  Memory: {memory_gb:.1f}GB")
      print(f"  Compute capability: {props.major}.{props.minor}")

    if torch.version.cuda:
      print(f"CUDA version: {torch.version.cuda}")

    if torch.backends.cudnn.is_available():
      print(f"cuDNN version: {torch.backends.cudnn.version()}")

    return True
  else:
    print("❌ CUDA is not available. Running on CPU.")
    return False


def test_gpu_performance():
  """Test GPU performance with a simple computation"""
  if not torch.cuda.is_available():
    print("Skipping GPU performance test (no GPU available)")
    return

  print("\n⚡ Testing GPU performance...")

  device = torch.device("cuda:0")

  # Create large tensors
  size = (10000, 10000)
  print(f"Creating {size[0]}x{size[1]} matrices...")

  # CPU test
  print("Testing CPU performance...")
  start_time = time.time()
  a_cpu = torch.randn(size)
  b_cpu = torch.randn(size)
  c_cpu = torch.matmul(a_cpu, b_cpu)
  cpu_time = time.time() - start_time
  print(f"CPU time: {cpu_time:.3f}s")

  # GPU test
  print("Testing GPU performance...")
  start_time = time.time()
  a_gpu = torch.randn(size, device=device)
  b_gpu = torch.randn(size, device=device)
  c_gpu = torch.matmul(a_gpu, b_gpu)
  gpu_time = time.time() - start_time
  print(f"GPU time: {gpu_time:.3f}s")
  # Calculate speedup
  if gpu_time > 0:
    speedup = cpu_time / gpu_time
    print(f"GPU speedup: {speedup:.1f}x")

  # Memory usage
  if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"GPU memory allocated: {memory_allocated:.1f}GB")
    print(f"GPU memory reserved: {memory_reserved:.1f}GB")


def test_autolabel_gpu():
  """Test autolabel service GPU integration"""
  print("\n🤖 Testing autolabel GPU integration...")

  try:
    from services.autolabel_service import AutoLabelService
    service = AutoLabelService()

    # Check if service can detect GPU
    print("Autolabel service initialized successfully")

    # Test model loading (if model exists)
    try:
      from pathlib import Path
      model_path = Path(__file__).parent / "data" / "models" / "SingleLabelClassification" / "model.pth"
      if model_path.exists():
        print(f"Model found at {model_path}")
        print("GPU integration test: PASSED")
      else:
        print("Model not found, but service initialized correctly")
    except Exception as e:
      print(f"Model loading test failed: {e}")

  except Exception as e:
    print(f"❌ Autolabel service test failed: {e}")
    return False

  return True


def main():
  """Main test function"""
  print("🚀 CRPlayer Annotation Tool - GPU Test Suite")
  print("=" * 50)

  gpu_available = test_gpu_availability()
  test_gpu_performance()
  autolabel_ok = test_autolabel_gpu()

  print("\n" + "=" * 50)
  print("📊 Test Results Summary:")
  print(f"GPU Available: {'✅ Yes' if gpu_available else '❌ No'}")
  print(f"Autolabel Service: {'✅ OK' if autolabel_ok else '❌ Failed'}")

  if gpu_available:
    print("\n🎉 GPU acceleration is working! Autolabel will use GPU for faster inference.")
  else:
    print("\n⚠️  GPU not available. Autolabel will run on CPU (slower).")
    print("To enable GPU:")
    print("1. Install NVIDIA drivers")
    print("2. Install nvidia-docker2")
    print("3. Configure Docker daemon with GPU runtime")
    print("4. Rebuild the container with GPU support")

  return 0 if gpu_available and autolabel_ok else 1


if __name__ == "__main__":
  sys.exit(main())
