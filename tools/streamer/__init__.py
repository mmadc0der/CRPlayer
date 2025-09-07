"""
Streamer Module - Android Stream Processing

Exposes GPU-accelerated Android streaming components for capturing
and decoding game frames in real-time with optional hardware acceleration.
"""

from .android_stream_gpu import GPUAndroidStreamer

__all__ = ['GPUAndroidStreamer']
