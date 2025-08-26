"""
Streamer Module - Android Stream Processing

Contains GPU-accelerated Android streaming components for capturing
and decoding game frames in real-time with hardware acceleration.
"""

from .android_stream_gpu import GPUAndroidStreamer

__all__ = ['GPUAndroidStreamer']
