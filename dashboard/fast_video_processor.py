"""
Fast video frame processor using OpenCV for real-time dashboard streaming.
Replaces slow FFmpeg-based approach with optimized frame extraction.
"""

import base64
import threading
import queue
import time
import subprocess
from typing import Optional, Callable, Tuple
import tempfile
import os


class FastVideoProcessor:
    """High-performance video frame processor using OpenCV."""
    
    def __init__(self, max_fps: int = 10, target_width: int = 400):
        self.max_fps = max_fps
        self.target_width = target_width
        self.frame_interval = 1.0 / max_fps
        
        # FFmpeg subprocess for video processing
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
            
        self._cleanup_resources()
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
        """Update video source with new MKV data using FFmpeg subprocess."""
        if len(self._mkv_buffer) < 1024:
            return
            
        try:
            # Use FFmpeg subprocess with optimized settings for real-time processing
            if not self._temp_file:
                fd, self._temp_file = tempfile.mkstemp(suffix='.mkv')
                os.close(fd)
                
            # Write accumulated data
            with open(self._temp_file, 'wb') as f:
                f.write(self._mkv_buffer)
                
            # Check if file has enough data for FFmpeg
            file_size = os.path.getsize(self._temp_file)
            if file_size < 4096:  # Need minimum data for FFmpeg header parsing
                return
                
            print(f"Processing MKV file with {file_size} bytes")
            
            # Clear buffer after writing
            self._mkv_buffer.clear()
            
        except Exception as e:
            print(f"Error updating video source: {e}")
            
    def _process_frames(self) -> None:
        """Main frame processing loop using FFmpeg subprocess."""
        ffmpeg_process = None
        
        while self._running:
            current_time = time.time()
            
            # Throttle frame rate
            if current_time - self._last_frame_time < self.frame_interval:
                time.sleep(0.01)
                continue
                
            # Check if we have a temp file to process
            if self._temp_file and os.path.exists(self._temp_file):
                file_size = os.path.getsize(self._temp_file)
                
                if file_size > 4096 and not ffmpeg_process:
                    # Start FFmpeg process with optimized settings
                    ffmpeg_process = self._start_ffmpeg_extraction()
                    
                if ffmpeg_process:
                    try:
                        # Read frame from FFmpeg stdout
                        frame_data = self._read_ffmpeg_frame(ffmpeg_process)
                        
                        if frame_data:
                            # Try to queue frame
                            try:
                                self._frame_queue.put_nowait({
                                    "type": "frame",
                                    "data": frame_data,
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
                        # Restart FFmpeg on error
                        if ffmpeg_process:
                            ffmpeg_process.terminate()
                            ffmpeg_process = None
                        
            time.sleep(0.01)
            
        # Cleanup FFmpeg process
        if ffmpeg_process:
            ffmpeg_process.terminate()
            ffmpeg_process.wait()
            
            
    def _start_ffmpeg_extraction(self) -> Optional[object]:
        """Start FFmpeg process for frame extraction with optimized settings."""
        try:
            cmd = [
                'ffmpeg',
                '-i', self._temp_file,
                '-vf', f'scale={self.target_width}:-1',  # Scale to target width, auto height
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',  # Use MJPEG for faster encoding
                '-q:v', '8',  # Good quality/speed balance (2-31, lower = better)
                '-r', str(self.max_fps),  # Frame rate
                '-an',  # No audio
                'pipe:1'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            print(f"Started FFmpeg extraction process (PID: {process.pid})")
            return process
            
        except Exception as e:
            print(f"Failed to start FFmpeg extraction: {e}")
            return None
            
    def _read_ffmpeg_frame(self, process) -> Optional[str]:
        """Read a single JPEG frame from FFmpeg stdout."""
        try:
            # Read JPEG header (FF D8)
            header = process.stdout.read(2)
            if len(header) != 2 or header != b'\xff\xd8':
                return None
                
            # Read until JPEG end marker (FF D9)
            frame_data = header
            while True:
                byte = process.stdout.read(1)
                if not byte:
                    break
                    
                frame_data += byte
                
                # Check for JPEG end marker
                if len(frame_data) >= 2 and frame_data[-2:] == b'\xff\xd9':
                    # Convert to base64
                    return base64.b64encode(frame_data).decode('utf-8')
                    
                # Prevent reading too much data
                if len(frame_data) > 1024 * 1024:  # 1MB limit
                    break
                    
            return None
            
        except Exception as e:
            print(f"Error reading FFmpeg frame: {e}")
            return None
            
    def _cleanup_resources(self) -> None:
        """Clean up resources."""
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
