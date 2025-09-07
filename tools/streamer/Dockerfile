FROM python:3.12-slim

# Install runtime dependencies: ADB (for device control) and FFmpeg (PyAV codecs)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  android-tools-adb \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements for this tool and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r /app/requirements.txt

# Copy the streamer module
COPY . /app

USER madc0der
# Default command: import check
CMD ["python", "-c", "from streamer import GPUAndroidStreamer; print('Streamer module ready')"]

# Notes for GPU/NVIDIA usage:
# - To enable CUDA/NVDEC inside Docker, use an NVIDIA CUDA base image and ensure
#   nvidia-container-toolkit is installed on the host; run with --gpus=all.
# - PyTorch with CUDA can be installed by overriding the base and pip indexes.
# - Hardware decode via PyAV depends on FFmpeg build with CUDA/NVDEC.

