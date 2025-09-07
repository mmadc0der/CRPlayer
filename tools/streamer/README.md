### Streamer Tool (Android GPU-accelerated streaming)

High-performance Android screen streaming using the scrcpy server and PyAV, optionally decoding on GPU and returning PyTorch tensors suitable for RL agents.

### Features

- **Direct scrcpy socket**: minimal overhead raw H264 stream
- **Optional GPU decode**: NVDEC via PyAV when available
- **PyTorch tensors**: normalized, CHW format on device (CPU/GPU)
- **Callback pipeline**: per-frame hook for downstream processing

### Requirements

- ADB-accessible Android device (USB or over TCP/IP)
- Python 3.10+
- System `ffmpeg` available (installed in Dockerfile)

### Install (local)

```bash
pip install -r tools/streamer/requirements.txt
```

### Quick start

```python
from streamer import GPUAndroidStreamer

def on_frame(tensor, pts, ts):
    # tensor: torch.FloatTensor, CHW, [0,1], on CPU/GPU depending on config
    pass

streamer = GPUAndroidStreamer(max_fps=60, max_size=1920, video_codec="h264", use_gpu=True)
streamer.start_streaming(frame_callback=on_frame)
```

### Docker (CPU by default)

Build from the `tools/streamer` directory:

```bash
docker build -t crplayer-streamer .
```

Run with host ADB available (recommended: use `adb connect <device_ip>:5555` on the host):

```bash
docker run --rm -it \
  --network=host \
  -e ADB_SERVER_SOCKET=tcp:127.0.0.1:5037 \
  crplayer-streamer
```

If you need to run ADB inside the container, ensure device access is configured on your host OS; on Windows, prefer ADB over TCP/IP.

### GPU notes

- For CUDA/NVDEC, use an NVIDIA CUDA base image and run with `--gpus=all`.
- Install PyTorch CUDA wheels per upstream docs, and provide FFmpeg with NVDEC.
- When GPU is unavailable or disabled, the streamer automatically falls back to CPU decoding.

### Troubleshooting

- No frames: verify scrcpy server pushes and that `adb forward` is active.
- Timeout/no data: the tool sends periodic wake signals; make sure the device is unlocked.
- Decoding errors: ensure codec matches (`video_codec`), and FFmpeg/PyAV versions are recent.

### Development

- Single module entrypoint: `streamer/GPUAndroidStreamer` in `android_stream_gpu.py`.
- Keep imports minimal; avoid unused heavy deps.
- Tests and CI can mock ADB and socket layers; decode paths can use small test bitstreams.
