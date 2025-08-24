#!/bin/bash

# Zero-latency Android screenrecord streaming
# Usage: ./stream_screenrecord.sh [width] [height] [bitrate]

WIDTH=${1:-360}
HEIGHT=${2:-800}
BITRATE=${3:-6M}

echo "Starting zero-latency streaming: ${WIDTH}x${HEIGHT} @ ${BITRATE}"

# Android screenrecord with correct syntax - outputs MP4 to stdout
adb shell screenrecord --size ${WIDTH}x${HEIGHT} --bit-rate ${BITRATE} --output-format=h264 - | \
python -m inference.cli stdin --expected-width "$WIDTH" --expected-height "$HEIGHT" --monitor
