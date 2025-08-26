#!/usr/bin/env python3
"""
Shared Memory Streaming Pipeline Architecture
Efficient producer-consumer pattern for real-time frame streaming with multiple consumers.
"""

import threading
import time
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass
from collections import deque
import torch
import numpy as np
from abc import ABC, abstractmethod
import weakref
import gc


@dataclass
class FrameData:
    """Immutable frame data container."""
    frame_id: int
    tensor: torch.Tensor  # GPU tensor [C, H, W]
    timestamp: float
    pts: int
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Ensure tensor is on GPU and properly formatted."""
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("tensor must be torch.Tensor")
        
        # Ensure tensor is on GPU if available
        if torch.cuda.is_available() and self.tensor.device.type != 'cuda':
            self.tensor = self.tensor.cuda()
        
        # Ensure CHW format
        if len(self.tensor.shape) != 3:
            raise ValueError(f"Expected CHW tensor, got shape {self.tensor.shape}")


class StreamConsumerIterator:
    """Iterator for a specific consumer to track its position in the stream."""
    
    def __init__(self, consumer_id: str, stream_buffer: 'SharedStreamBuffer'):
        self.consumer_id = consumer_id
        self.stream_buffer = stream_buffer
        self.position = 0  # Current read position
        self.last_access = time.time()
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Optional[FrameData]:
        """Get next frame for this consumer."""
        self.last_access = time.time()
        frame = self.stream_buffer._get_frame_at_position(self.consumer_id, self.position)
        
        if frame is not None:
            self.position += 1
            
        return frame
    
    def peek(self) -> Optional[FrameData]:
        """Peek at next frame without advancing position."""
        return self.stream_buffer._get_frame_at_position(self.consumer_id, self.position)
    
    def skip(self, count: int = 1):
        """Skip frames without processing."""
        self.position += count
        self.last_access = time.time()
    
    def get_available_count(self) -> int:
        """Get number of frames available to read."""
        return max(0, self.stream_buffer.write_position - self.position)
    
    def is_alive(self) -> bool:
        """Check if consumer is still active (accessed recently)."""
        return time.time() - self.last_access < 30.0  # 30 second timeout


class SharedStreamBuffer:
    """
    Efficient shared memory buffer for streaming frames.
    Single producer, multiple consumers with automatic memory management.
    """
    
    def __init__(self, max_buffer_size: int = 100):
        """
        Initialize shared stream buffer.
        
        Args:
            max_buffer_size: Maximum number of frames to keep in memory
        """
        self.max_buffer_size = max_buffer_size
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.write_position = 0
        self.consumers: Dict[str, StreamConsumerIterator] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._start_cleanup_thread()
        
        # Statistics
        self.stats = {
            'frames_produced': 0,
            'frames_consumed': 0,
            'memory_cleanups': 0,
            'active_consumers': 0
        }
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                time.sleep(5.0)  # Cleanup every 5 seconds
                self._cleanup_inactive_consumers()
                self._cleanup_old_frames()
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def add_frame(self, frame_data: FrameData):
        """
        Add new frame to buffer (producer only).
        
        Args:
            frame_data: Frame to add to stream
        """
        with self._lock:
            # Ensure frame_id matches write position
            frame_data.frame_id = self.write_position
            
            # Add to buffer
            self.buffer.append(frame_data)
            self.write_position += 1
            self.stats['frames_produced'] += 1
            
            # Trigger cleanup if buffer is getting full
            if len(self.buffer) >= self.max_buffer_size * 0.8:
                self._cleanup_old_frames()
    
    def create_consumer(self, consumer_id: str, start_from_latest: bool = True) -> StreamConsumerIterator:
        """
        Create new consumer iterator.
        
        Args:
            consumer_id: Unique identifier for consumer
            start_from_latest: If True, start from latest frame; if False, start from oldest available
            
        Returns:
            Consumer iterator
        """
        with self._lock:
            if consumer_id in self.consumers:
                # Return existing consumer
                return self.consumers[consumer_id]
            
            iterator = StreamConsumerIterator(consumer_id, self)
            
            # Set starting position
            if start_from_latest:
                iterator.position = self.write_position
            else:
                # Start from oldest available frame
                oldest_frame_id = self.write_position - len(self.buffer)
                iterator.position = max(0, oldest_frame_id)
            
            self.consumers[consumer_id] = iterator
            self.stats['active_consumers'] = len(self.consumers)
            
            return iterator
    
    def remove_consumer(self, consumer_id: str):
        """Remove consumer and clean up resources."""
        with self._lock:
            if consumer_id in self.consumers:
                del self.consumers[consumer_id]
                self.stats['active_consumers'] = len(self.consumers)
    
    def _get_frame_at_position(self, consumer_id: str, position: int) -> Optional[FrameData]:
        """Get frame at specific position for consumer."""
        with self._lock:
            # Check if position is available in buffer
            oldest_available = self.write_position - len(self.buffer)
            
            if position < oldest_available:
                # Frame too old, already cleaned up
                return None
            
            if position >= self.write_position:
                # Frame not yet available
                return None
            
            # Calculate buffer index
            buffer_index = position - oldest_available
            
            if 0 <= buffer_index < len(self.buffer):
                self.stats['frames_consumed'] += 1
                return self.buffer[buffer_index]
            
            return None
    
    def _cleanup_inactive_consumers(self):
        """Remove consumers that haven't accessed data recently."""
        with self._lock:
            inactive_consumers = [
                consumer_id for consumer_id, iterator in self.consumers.items()
                if not iterator.is_alive()
            ]
            
            for consumer_id in inactive_consumers:
                print(f"[CLEANUP] Removing inactive consumer: {consumer_id}")
                del self.consumers[consumer_id]
            
            if inactive_consumers:
                self.stats['active_consumers'] = len(self.consumers)
    
    def _cleanup_old_frames(self):
        """Clean up frames that all consumers have passed."""
        with self._lock:
            if not self.consumers:
                return
            
            # Find minimum position across all active consumers
            min_position = min(iterator.position for iterator in self.consumers.values())
            
            # Calculate how many frames we can safely remove
            oldest_available = self.write_position - len(self.buffer)
            frames_to_keep = max(10, self.write_position - min_position)  # Keep at least 10 frames
            
            if len(self.buffer) > frames_to_keep:
                frames_to_remove = len(self.buffer) - frames_to_keep
                
                # Remove old frames
                for _ in range(frames_to_remove):
                    if self.buffer:
                        old_frame = self.buffer.popleft()
                        # Help garbage collection
                        del old_frame
                
                self.stats['memory_cleanups'] += 1
                
                # Force garbage collection periodically
                if self.stats['memory_cleanups'] % 10 == 0:
                    gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                **self.stats,
                'buffer_size': len(self.buffer),
                'write_position': self.write_position,
                'consumer_positions': {
                    consumer_id: iterator.position 
                    for consumer_id, iterator in self.consumers.items()
                },
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if not self.buffer:
            return 0.0
        
        # Estimate based on first frame
        sample_frame = self.buffer[0]
        tensor_size = sample_frame.tensor.element_size() * sample_frame.tensor.numel()
        total_size = tensor_size * len(self.buffer)
        
        return total_size / (1024 * 1024)  # Convert to MB


class StreamConsumer(ABC):
    """Abstract base class for stream consumers."""
    
    def __init__(self, consumer_id: str, stream_buffer: SharedStreamBuffer):
        self.consumer_id = consumer_id
        self.stream_buffer = stream_buffer
        self.iterator = stream_buffer.create_consumer(consumer_id)
        self.is_running = False
        self._thread = None
    
    @abstractmethod
    def process_frame(self, frame_data: FrameData) -> bool:
        """
        Process a single frame.
        
        Args:
            frame_data: Frame to process
            
        Returns:
            True to continue processing, False to stop
        """
        pass
    
    def start(self):
        """Start consumer in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop consumer."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Remove from stream buffer
        self.stream_buffer.remove_consumer(self.consumer_id)
    
    def _run_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                frame_data = next(self.iterator)
                
                if frame_data is None:
                    # No frame available, wait briefly
                    time.sleep(0.001)
                    continue
                
                # Process frame
                should_continue = self.process_frame(frame_data)
                if not should_continue:
                    break
                    
            except Exception as e:
                print(f"[ERROR] Consumer {self.consumer_id} error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def get_available_frames(self) -> int:
        """Get number of frames available to process."""
        return self.iterator.get_available_count()


class StreamProducer:
    """Stream producer that captures and decodes frames."""
    
    def __init__(self, stream_buffer: SharedStreamBuffer):
        self.stream_buffer = stream_buffer
        self.is_producing = False
        self._thread = None
        
        # Import here to avoid circular dependency
        from android_stream_gpu import GPUAndroidStreamer
        
        self.streamer = GPUAndroidStreamer(
            max_fps=60,
            max_size=1080,
            video_codec="h264",
            bit_rate="80M",
            use_gpu=True,
            buffer_size=5  # Small buffer since we're using shared buffer
        )
    
    def start_production(self):
        """Start producing frames."""
        if self.is_producing:
            return
        
        self.is_producing = True
        
        # Start streaming with frame callback
        self.streamer.start_streaming(frame_callback=self._frame_callback)
        
        print("Stream producer started")
    
    def stop_production(self):
        """Stop producing frames."""
        if not self.is_producing:
            return
        
        self.is_producing = False
        self.streamer.stop_streaming()
        
        print("Stream producer stopped")
    
    def _frame_callback(self, tensor: torch.Tensor, pts: int, timestamp: float):
        """Callback to handle new frames from streamer."""
        if not self.is_producing:
            return
        
        try:
            # Create frame data
            frame_data = FrameData(
                frame_id=0,  # Will be set by buffer
                tensor=tensor,
                timestamp=timestamp,
                pts=pts,
                metadata={}
            )
            
            # Add to shared buffer
            self.stream_buffer.add_frame(frame_data)
            
        except Exception as e:
            print(f"[ERROR] Producer frame callback error: {e}")


# Consumer registry will be managed externally
# Specific consumer implementations should be in separate modules
