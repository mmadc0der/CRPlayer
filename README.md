# Android RL Agent Streaming

High-performance GPU-accelerated Android screen streaming for reinforcement learning agents. Optimized for RTX 3060 with 60fps H265 streaming using scrcpy protocol.

## Features

- **GPU-Accelerated H265 Decoding** - Hardware acceleration using NVDEC on RTX 3060
- **60fps Real-time Streaming** - Low-latency streaming optimized for RL training
- **PyTorch Integration** - Direct tensor output on GPU for RL agents
- **scrcpy Protocol** - Raw stream handling without display overhead
- **Performance Monitoring** - Built-in FPS and latency tracking

## Requirements

- Linux environment with ADB access
- NVIDIA RTX 3060 (or compatible GPU)
- Python 3.8+
- CUDA-enabled PyTorch
- scrcpy installed and accessible

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install scrcpy:**
```bash
# Ubuntu/Debian
sudo apt install scrcpy

# Or build from source for latest version
```

3. **Setup ADB connection:**
```bash
adb devices  # Verify your Android device is connected
```

## Quick Start

### Basic Streaming Test

```python
from android_stream_gpu import GPUAndroidStreamer

# Create streamer
streamer = GPUAndroidStreamer(
    max_fps=60,
    video_codec="h265",
    use_gpu=True
)

# Frame callback
def process_frame(tensor, pts, timestamp):
    print(f"Frame: {tensor.shape}, Device: {tensor.device}")

# Start streaming
streamer.start_streaming(frame_callback=process_frame)
```

### RL Agent Integration

```python
from rl_agent_integration import AndroidGameEnvironment

# Create environment
env = AndroidGameEnvironment(
    observation_size=(224, 224),
    frame_stack=4,
    max_fps=60
)

# Standard RL loop
observation = env.reset()
for step in range(1000):
    action = agent.act(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break
```

## Performance Testing

Run comprehensive performance tests:

```bash
python test_stream.py
```

Expected performance on RTX 3060:
- **60fps** sustained streaming
- **<50ms** end-to-end latency  
- **<2GB** GPU memory usage
- **<30%** CPU utilization

## Configuration

### Optimal Settings for RTX 3060

```python
streamer = GPUAndroidStreamer(
    max_fps=60,           # Target framerate
    max_size=1920,        # Resolution (1920x1080)
    video_codec="h265",   # Better compression
    bit_rate="12M",       # High quality for RTX 3060
    use_gpu=True,         # Enable hardware acceleration
    buffer_size=5         # Low latency buffer
)
```

### Device-Specific Optimization

```bash
# Check available encoders on device
scrcpy --list-encoders

# Use specific encoder if needed
scrcpy --video-encoder=OMX.qcom.video.encoder.hevc
```

## Troubleshooting

### Common Issues

**No frames received:**
- Verify ADB connection: `adb devices`
- Check scrcpy works: `scrcpy --no-display`
- Ensure device screen is unlocked

**Low FPS performance:**
- Check GPU utilization: `nvidia-smi`
- Reduce resolution: `max_size=1280`
- Lower bitrate: `bit_rate="8M"`

**High latency:**
- Reduce buffer size: `buffer_size=3`
- Use USB connection instead of WiFi
- Close other GPU applications

**GPU acceleration not working:**
- Verify CUDA installation: `torch.cuda.is_available()`
- Check NVDEC support: `ffmpeg -hwaccels`
- Fallback to CPU decoding automatically

### Performance Optimization

1. **USB Connection** - Use USB instead of WiFi ADB
2. **Dedicated GPU** - Close other GPU applications
3. **Process Priority** - Run with higher priority
4. **Memory Management** - Monitor GPU memory usage

## Architecture

```
Android Device → scrcpy-server → ADB Tunnel → Python Client
                                                    ↓
GPU Decoder (NVDEC) → PyTorch Tensors → RL Agent
```

### Key Components

- **`android_stream_gpu.py`** - Main streaming class with GPU acceleration
- **`rl_agent_integration.py`** - RL environment wrapper and example agent
- **`test_stream.py`** - Performance testing and validation

## Advanced Usage

### Custom Frame Processing

```python
def advanced_frame_callback(tensor, pts, timestamp):
    # tensor: (3, H, W) on GPU, normalized [0,1]
    
    # Resize for RL agent
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), 
        size=(84, 84), 
        mode='bilinear'
    ).squeeze(0)
    
    # Convert to grayscale
    grayscale = torch.mean(resized, dim=0, keepdim=True)
    
    # Feed to RL agent
    agent.process_observation(grayscale)
```

### Multi-Device Streaming

```python
# Stream from multiple devices
devices = ["device1", "device2"]
streamers = []

for device_id in devices:
    streamer = GPUAndroidStreamer(device_id=device_id)
    streamer.start_streaming(lambda t, p, ts, d=device_id: process_device_frame(d, t))
    streamers.append(streamer)
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Support

For issues and questions:
- Check troubleshooting section
- Run performance tests
- Check GPU compatibility
- Verify ADB connection
