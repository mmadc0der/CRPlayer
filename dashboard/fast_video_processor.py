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
    """High-performance video frame processor using FFmpeg stdin."""
    
    def __init__(self, max_fps: int = 10, target_width: int = 400):
        self.max_fps = max_fps
        self.target_width = target_width
        self.frame_interval = 1.0 / max_fps
        
        # FFmpeg subprocess for streaming processing
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        
        # Frame processing
        self._frame_queue = queue.Queue(maxsize=5)
        self._processor_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Buffer for incoming MKV data
        self._mkv_buffer = bytearray()
        self._last_frame_time = 0
        self._buffer_lock = threading.Lock()
        
        # Statistics
        self.frames_processed = 0
        self.frames_dropped = 0
        
    def start(self) -> None:
        """Start the video processor."""
        if self._running:
            return
            
        self._running = True
        
        # Start FFmpeg process for streaming
        self._start_ffmpeg_streaming()
        
        # Start frame reader thread
        self._reader_thread = threading.Thread(target=self._read_frames, daemon=True)
        self._reader_thread.start()
        
        print("Fast video processor started")
        
    def stop(self) -> None:
        """Stop the video processor."""
        self._running = False
        
        # Stop FFmpeg process
        if self._ffmpeg_process:
            try:
                if self._ffmpeg_process.stdin:
                    self._ffmpeg_process.stdin.close()
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=3)
            except:
                if self._ffmpeg_process:
                    self._ffmpeg_process.kill()
            self._ffmpeg_process = None
        
        # Wait for threads
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
            
        print("Fast video processor stopped")
        
    def add_mkv_data(self, data: bytes) -> None:
        """Add MKV data to FFmpeg stdin."""
        if not self._ffmpeg_process or not self._ffmpeg_process.stdin:
            # Buffer data until FFmpeg is ready
            with self._buffer_lock:
                self._mkv_buffer.extend(data)
            return
            
        try:
            # Send data directly to FFmpeg stdin
            self._ffmpeg_process.stdin.write(data)
            self._ffmpeg_process.stdin.flush()
            
            # Also send any buffered data
            with self._buffer_lock:
                if self._mkv_buffer:
                    self._ffmpeg_process.stdin.write(self._mkv_buffer)
                    self._ffmpeg_process.stdin.flush()
                    self._mkv_buffer.clear()
                    
        except (BrokenPipeError, OSError) as e:
            print(f"[FastVideoProcessor] FFmpeg stdin error: {e}")
            # Restart FFmpeg
            self._restart_ffmpeg()
            
    def get_frame(self, timeout: float = 0.1) -> Optional[dict]:
        """Get next processed frame if available."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _start_ffmpeg_streaming(self) -> None:
        """Start FFmpeg process for streaming MKV input."""
        try:
            cmd = [
                'ffmpeg',
                '-f', 'matroska',  # Input format
                '-i', 'pipe:0',    # Read from stdin
                '-vf', f'scale={self.target_width}:-1',  # Scale to target width
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-q:v', '8',  # Balanced quality/speed
                '-r', '3',    # 3 FPS for dashboard
                '-an',        # No audio
                '-loglevel', 'error',  # Minimal logging
                'pipe:1'
            ]
            
            print(f"[FastVideoProcessor] Starting streaming FFmpeg: {' '.join(cmd)}")
            
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            print(f"[FastVideoProcessor] FFmpeg streaming process started (PID: {self._ffmpeg_process.pid})")
            
        except Exception as e:
            print(f"[FastVideoProcessor] Failed to start streaming FFmpeg: {e}")
            self._ffmpeg_process = None
            
    def _read_frames(self) -> None:
        """Read frames from FFmpeg stdout."""
        while self._running:
            if not self._ffmpeg_process:
                time.sleep(0.1)
                continue
                
            try:
                # Read frame from FFmpeg stdout
                frame_data = self._read_ffmpeg_frame(self._ffmpeg_process)
                
                if frame_data:
                    current_time = time.time()
                    
                    # Throttle frame rate
                    if current_time - self._last_frame_time >= self.frame_interval:
                        print(f"[FastVideoProcessor] Frame extracted, base64 length: {len(frame_data)}")
                        
                        try:
                            self._frame_queue.put_nowait({
                                "type": "frame",
                                "data": frame_data,
                                "timestamp": current_time,
                                "width": self.target_width
                            })
                            self.frames_processed += 1
                            self._last_frame_time = current_time
                            print(f"[FastVideoProcessor] Frame queued (total: {self.frames_processed})")
                            
                        except queue.Full:
                            self.frames_dropped += 1
                            print(f"[FastVideoProcessor] Frame dropped - queue full")
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"[FastVideoProcessor] Frame reading error: {e}")
                time.sleep(0.1)
            
            
    def _restart_ffmpeg(self) -> None:
        """Restart FFmpeg process."""
        print("[FastVideoProcessor] Restarting FFmpeg process...")
        
        # Stop current process
        if self._ffmpeg_process:
            try:
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=2)
            except:
                pass
            self._ffmpeg_process = None
            
        # Start new process
        time.sleep(0.5)  # Brief delay
        self._start_ffmpeg_streaming()
            
    def _read_ffmpeg_frame(self, process) -> Optional[str]:
        """Read a single JPEG frame from FFmpeg stdout."""
        try:
            # Check if process is still running
            if process.poll() is not None:
                print(f"FFmpeg process ended with return code: {process.returncode}")
                # Print stderr for debugging
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                if stderr_output:
                    print(f"FFmpeg stderr: {stderr_output[:500]}...")
                return None
            
            # Try to read a chunk of data instead of byte-by-byte
            chunk = process.stdout.read(4096)
            if not chunk:
                return None
                
            # Look for JPEG markers in the chunk
            # JPEG starts with FF D8 and ends with FF D9
            start_marker = b'\xff\xd8'
            end_marker = b'\xff\xd9'
            
            start_idx = chunk.find(start_marker)
            if start_idx == -1:
                return None
                
            # Find end marker
            end_idx = chunk.find(end_marker, start_idx + 2)
            if end_idx == -1:
                # Read more data to find the end
                additional_data = process.stdout.read(8192)
                if additional_data:
                    chunk += additional_data
                    end_idx = chunk.find(end_marker, start_idx + 2)
                    
            if end_idx != -1:
                # Extract complete JPEG frame
                frame_data = chunk[start_idx:end_idx + 2]
                print(f"[FastVideoProcessor] Extracted JPEG frame: {len(frame_data)} bytes")
                return base64.b64encode(frame_data).decode('utf-8')
                
            return None
            
        except Exception as e:
            print(f"Error reading FFmpeg frame: {e}")
            return None
            
            
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
