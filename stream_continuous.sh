#!/bin/bash

# Continuous scrcpy streaming using time-limited recordings
# Usage: ./stream_continuous.sh [width] [height] [bitrate]

WIDTH=${1:-360}
HEIGHT=${2:-800}
BITRATE=${3:-6000000}

echo "Starting continuous streaming: ${WIDTH}x${HEIGHT} @ ${BITRATE}bps"

# Create output FIFO
OUTPUT_FIFO="/tmp/continuous_stream"
mkfifo "$OUTPUT_FIFO" 2>/dev/null || true

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    pkill -f "scrcpy.*record.*tmp"
    rm -f "$OUTPUT_FIFO" /tmp/scrcpy_chunk_*
    exit
}
trap cleanup INT TERM

# Background continuous recording loop
{
    chunk=0
    while true; do
        chunk_file="/tmp/scrcpy_chunk_$chunk"
        echo "[$(date '+%H:%M:%S')] Recording chunk $chunk..."
        
        # Record 10-second chunks to avoid FIFO issues
        timeout 10s scrcpy --no-window --no-audio \
            --video-bit-rate "$BITRATE" \
            --record-format=mkv \
            --record "$chunk_file" 2>/dev/null
        
        # Stream chunk to output FIFO if it exists and has data
        if [[ -f "$chunk_file" && -s "$chunk_file" ]]; then
            cat "$chunk_file" >> "$OUTPUT_FIFO" &
            rm -f "$chunk_file"
        fi
        
        chunk=$((chunk + 1))
        sleep 0.5  # Brief pause between recordings
    done
} &

# Wait for first chunk
sleep 3

# Stream to Python pipeline
echo "Starting pipeline..."
cat "$OUTPUT_FIFO" | python -m inference.cli stdin \
    --expected-width "$WIDTH" --expected-height "$HEIGHT" --monitor

cleanup
