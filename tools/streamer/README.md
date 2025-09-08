# Streamer Module

This module streams Android frames via the scrcpy socket and decodes them with PyAV. It exposes a simple API for RL pipelines and provides a CLI for basic operations.

## API

```python
from streamer import AndroidStreamer, StreamerConfig

streamer = AndroidStreamer(StreamerConfig(use_gpu=True, buffer_size=8))
streamer.start()
frame = streamer.get_latest_frame()
stats = streamer.get_stats()
streamer.stop()
```

## CLI

```bash
python -m streamer.cli start --device-id XYZ --use-gpu
```

## Docker

- CPU image: `python:3.12-slim` with `adb` and `ffmpeg` installed
- For GPU: follow Dockerfile notes to switch to an NVIDIA CUDA base and run with `--gpus all`

## Requirements

- System: `adb`, `ffmpeg`
- Python: `av`, `numpy`, `torch` (optional if you only need numpy outputs)

## Notes

- Non-blocking socket reads with bounded buffer and frame-drop policy
- Minimal logging by default; enable `--debug` for verbose logs

