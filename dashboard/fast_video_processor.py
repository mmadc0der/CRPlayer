"""
Fast video frame processor using OpenCV for real-time dashboard streaming.
Replaces slow FFmpeg-based approach with optimized frame extraction.
"""

import cv2
import numpy as np
import base64
import threading
import queue
import time
from typing import Optional, Callable, Tuple
import tempfile
import os


class FastVideoProcessor:
    """High-performance video frame processor using OpenCV."""
    
    def __init__(self, max_fps: int = 10, target_width: int = 400):
        self.max_fps = max_fps
        self.target_width = target_width
        self.frame_interval = 1.0 / max_fps
        
        # OpenCV video capture
        self._cap: Optional[cv2.VideoCapture] = None
        self._temp_file: Optional[str] = None
        
        # Frame processing
        self._frame_queue = queue.Queue(maxsize=5)
        self._processor_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Buffer for incoming MKV data
        self._mkv_buffer = bytearray()
        self._last_frame_time = 0
        
        # Statistics
        self.frames_processed = 0
        self.frames_dropped = 0
        
    def start(self) -> None:
        """Start the video processor."""
        if self._running:
            return
            
        self._running = True
        self._processor_thread = threading.Thread(target=self._process_frames, daemon=True)
        self._processor_thread.start()
        print("Fast video processor started")
        
    def stop(self) -> None:
        """Stop the video processor."""
        self._running = False
        
        if self._processor_thread:
            self._processor_thread.join(timeout=2.0)
            
        self._cleanup_opencv()
        print("Fast video processor stopped")
        
    def add_mkv_data(self, data: bytes) -> None:
        """Add MKV data to processing buffer."""
        self._mkv_buffer.extend(data)
        
        # Process when we have enough data
        if len(self._mkv_buffer) > 32768:  # 32KB threshold
            self._update_video_source()
            
    def get_frame(self, timeout: float = 0.1) -> Optional[dict]:
        """Get next processed frame if available."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _update_video_source(self) -> None:
        """Update OpenCV video source with new MKV data."""
        if len(self._mkv_buffer) < 1024:
            return
            
        try:
            # Write buffer to temporary file for OpenCV
            if not self._temp_file:
                fd, self._temp_file = tempfile.mkstemp(suffix='.mkv')
                os.close(fd)
                
            # Write accumulated data
            with open(self._temp_file, 'wb') as f:
                f.write(self._mkv_buffer)
                
            # Reinitialize OpenCV capture
            self._cleanup_opencv()
            self._cap = cv2.VideoCapture(self._temp_file)
            
            if self._cap.isOpened():
                # Configure for performance
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
                print(f"OpenCV capture initialized with {len(self._mkv_buffer)} bytes")
            else:
                print("Failed to open video with OpenCV")
                
            # Clear buffer after processing
            self._mkv_buffer.clear()
            
        except Exception as e:
            print(f"Error updating video source: {e}")
            
    def _process_frames(self) -> None:
        """Main frame processing loop."""
        while self._running:
            current_time = time.time()
            
            # Throttle frame rate
            if current_time - self._last_frame_time < self.frame_interval:
                time.sleep(0.01)
                continue
                
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                
                if ret and frame is not None:
                    try:
                        # Process frame
                        processed_frame = self._process_frame(frame)
                        
                        if processed_frame:
                            # Try to queue frame
                            try:
                                self._frame_queue.put_nowait({
                                    "type": "frame",
                                    "data": processed_frame,
                                    "timestamp": current_time,
                                    "width": self.target_width
                                })
                                self.frames_processed += 1
                                self._last_frame_time = current_time
                                
                            except queue.Full:
                                # Drop frame if queue is full
                                self.frames_dropped += 1
                                
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        
            time.sleep(0.01)
            
    def _process_frame(self, frame: np.ndarray) -> Optional[str]:
        """Process single frame for web transmission."""
        try:
            # Get original dimensions
            height, width = frame.shape[:2]
            
            # Calculate target height maintaining aspect ratio
            aspect_ratio = height / width
            target_height = int(self.target_width * aspect_ratio)
            
            # Resize frame
            resized = cv2.resize(frame, (self.target_width, target_height), 
                               interpolation=cv2.INTER_LINEAR)
            
            # Encode as JPEG (much faster than PNG)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Good quality/speed balance
            success, buffer = cv2.imencode('.jpg', resized, encode_params)
            
            if success:
                # Convert to base64 for web transmission
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                return jpg_as_text
            else:
                print("Failed to encode frame as JPEG")
                return None
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
            
    def _cleanup_opencv(self) -> None:
        """Clean up OpenCV resources."""
        if self._cap:
            self._cap.release()
            self._cap = None
            
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
            except OSError:
                pass
            self._temp_file = None
            
    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'queue_size': self._frame_queue.qsize(),
            'buffer_size': len(self._mkv_buffer)
        }


class StreamingVideoProcessor:
    """Alternative processor that streams H.264 directly to browser."""
    
    def __init__(self):
        self._h264_buffer = bytearray()
        self._frame_callback: Optional[Callable] = None
        
    def set_frame_callback(self, callback: Callable[[bytes], None]) -> None:
        """Set callback for raw H.264 frames."""
        self._frame_callback = callback
        
    def add_mkv_data(self, data: bytes) -> None:
        """Extract H.264 from MKV container."""
        # Simple MKV parser to extract H.264 NAL units
        # This is a simplified implementation - production would need full MKV parsing
        self._h264_buffer.extend(data)
        
        # Look for H.264 NAL unit start codes (0x00000001)
        while len(self._h264_buffer) > 4:
            # Find start code
            start_idx = self._find_nal_start()
            if start_idx == -1:
                break
                
            # Find next start code
            next_idx = self._find_nal_start(start_idx + 4)
            if next_idx == -1:
                break
                
            # Extract NAL unit
            nal_unit = bytes(self._h264_buffer[start_idx:next_idx])
            
            if self._frame_callback:
                self._frame_callback(nal_unit)
                
            # Remove processed data
            self._h264_buffer = self._h264_buffer[next_idx:]
            
    def _find_nal_start(self, offset: int = 0) -> int:
        """Find H.264 NAL unit start code."""
        for i in range(offset, len(self._h264_buffer) - 3):
            if (self._h264_buffer[i] == 0x00 and 
                self._h264_buffer[i+1] == 0x00 and
                self._h264_buffer[i+2] == 0x00 and
                self._h264_buffer[i+3] == 0x01):
                return i
        return -1


# Factory function for easy integration
def create_fast_processor(method: str = "opencv", **kwargs) -> object:
    """Create optimized video processor.
    
    Args:
        method: "opencv" for OpenCV processor, "streaming" for H.264 streaming
        **kwargs: Additional parameters for processor
    """
    if method == "opencv":
        return FastVideoProcessor(**kwargs)
    elif method == "streaming":
        return StreamingVideoProcessor()
    else:
        raise ValueError(f"Unknown processor method: {method}")
