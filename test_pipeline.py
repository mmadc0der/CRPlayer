#!/usr/bin/env python3
"""Test script to verify pipeline works with known H.264 data"""

import subprocess
import sys
import os

def test_with_sample_video():
    """Test pipeline with a sample H.264 stream"""
    
    # Create a simple test H.264 stream using FFmpeg
    print("Creating test H.264 stream...")
    
    # Generate a simple test pattern video and encode to H.264
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=5:size=360x800:rate=30",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "h264",
        "-"
    ]
    
    try:
        # Run FFmpeg to generate test H.264 stream
        ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Pipe the output to our CRPlayer
        crplayer_cmd = [
            sys.executable, "-m", "inference.cli", "stdin",
            "--expected-width", "360",
            "--expected-height", "800"
        ]
        
        crplayer_proc = subprocess.Popen(
            crplayer_cmd,
            stdin=ffmpeg_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Close FFmpeg stdout in parent to avoid deadlock
        ffmpeg_proc.stdout.close()
        
        # Wait for processes to complete
        crplayer_output, crplayer_error = crplayer_proc.communicate(timeout=10)
        ffmpeg_output, ffmpeg_error = ffmpeg_proc.communicate()
        
        print("FFmpeg stderr:", ffmpeg_error.decode())
        print("CRPlayer stdout:", crplayer_output.decode())
        print("CRPlayer stderr:", crplayer_error.decode())
        
    except subprocess.TimeoutExpired:
        print("Test completed (timeout reached)")
        crplayer_proc.kill()
        ffmpeg_proc.kill()
    except Exception as e:
        print(f"Test failed: {e}")

def test_scrcpy_output():
    """Test what scrcpy actually outputs"""
    print("Testing scrcpy output format...")
    
    cmd = [
        "scrcpy", "--no-window", "--no-audio", 
        "--max-fps", "30", "--video-bit-rate", "2M",
        "--record", "-"
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Read first 1KB to see what format scrcpy outputs
        data = proc.stdout.read(1024)
        print(f"First 1KB from scrcpy: {len(data)} bytes")
        print(f"Hex dump: {data[:64].hex()}")
        
        # Check for H.264 NAL unit headers
        if data.startswith(b'\x00\x00\x00\x01'):
            print("✓ Looks like H.264 stream (starts with NAL unit)")
        elif data.startswith(b'ftypmp4'):
            print("✗ This is MP4 format, not raw H.264")
        else:
            print(f"? Unknown format, first 4 bytes: {data[:4].hex()}")
            
        proc.kill()
        
    except Exception as e:
        print(f"Scrcpy test failed: {e}")

if __name__ == "__main__":
    print("=== CRPlayer Pipeline Test ===")
    
    if len(sys.argv) > 1 and sys.argv[1] == "scrcpy":
        test_scrcpy_output()
    else:
        test_with_sample_video()
