"""
Fast video frame processor using OpenCV for real-time dashboard streaming.
Replaces slow FFmpeg-based approach with optimized frame extraction.
"""

import base64
import threading
import queue
import os
import sys
import subprocess
import time
from typing import Optional, Callable, Tuple
import tempfile


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
        
        # MKV stream synchronization
        self._has_valid_header = False
        self._segment_buffer = bytearray()
        self._looking_for_header = True
        self._restart_count = 0
        self._max_restarts = 3
        
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
        """Add MKV data with improved streaming."""
        if not self._ffmpeg_process or not self._ffmpeg_process.stdin:
            return
            
        try:
            # Send data immediately in smaller chunks for better streaming
            chunk_size = 4096  # 4KB chunks
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                self._ffmpeg_process.stdin.write(chunk)
                # Force immediate write to pipe
                self._ffmpeg_process.stdin.flush()
                
        except (BrokenPipeError, OSError) as e:
            print(f"[FastVideoProcessor] FFmpeg stdin error: {e}")
            # Don't restart immediately, let the monitoring handle it
            
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
                '-analyzeduration', '1000000',  # 1 second analysis
                '-probesize', '1000000',        # 1MB probe size
                '-fflags', '+igndts+ignidx',    # Ignore timestamps and index
                '-i', 'pipe:0',    # Read from stdin
                '-vf', f'scale={self.target_width}:-1',  # Scale to target width
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-q:v', '10',  # Lower quality for stability
                '-r', '2',     # 2 FPS to reduce load
                '-an',         # No audio
                '-flush_packets', '1',  # Flush packets immediately
                '-loglevel', 'error',   # Minimal logging
                'pipe:1'
            ]
            
            print(f"[FastVideoProcessor] Starting streaming FFmpeg: {' '.join(cmd)}")
            
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered
                preexec_fn=None if sys.platform == 'win32' else lambda: os.setpgrp()
            )
            print(f"[FastVideoProcessor] FFmpeg streaming process started (PID: {self._ffmpeg_process.pid})")
            
        except Exception as e:
            print(f"[FastVideoProcessor] Failed to start streaming FFmpeg: {e}")
            self._ffmpeg_process = None
            
    def _read_frames(self) -> None:
        """Read frames from FFmpeg stdout."""
        frame_count_since_restart = 0
        last_restart_time = time.time()
        last_frame_time = time.time()  # Initialize properly
        
        while self._running:
            if not self._ffmpeg_process:
                time.sleep(0.1)
                continue
                
            try:
                # Read frame from FFmpeg stdout
                frame_data = self._read_ffmpeg_frame(self._ffmpeg_process)
                
                if frame_data:
                    current_time = time.time()
                    frame_count_since_restart += 1
                    last_frame_time = current_time  # Update local frame time
                    
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
                    # No frame available - check if FFmpeg might be stuck
                    current_time = time.time()
                    time_since_last_frame = current_time - last_frame_time
                    time_since_restart = current_time - last_restart_time
                    
                    # Only restart if we've been running for at least 10 seconds without frames
                    # OR after processing 150 frames successfully
                    if ((time_since_last_frame > 20.0 and time_since_restart > 10.0) or 
                        frame_count_since_restart > 150):
                        print(f"[FastVideoProcessor] FFmpeg appears stuck (no frames for {time_since_last_frame:.1f}s, {frame_count_since_restart} frames processed, {time_since_restart:.1f}s since restart)")
                        self._restart_ffmpeg()
                        frame_count_since_restart = 0
                        last_restart_time = current_time
                        last_frame_time = current_time  # Reset frame timer
                        continue
                        
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
                if self._ffmpeg_process.stdin:
                    self._ffmpeg_process.stdin.close()
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=2)
            except:
                try:
                    self._ffmpeg_process.kill()
                except:
                    pass
            self._ffmpeg_process = None
            
        # Track restart count
        self._restart_count += 1
        print(f"[FastVideoProcessor] Restart #{self._restart_count}")
        
        # Clear any buffered data
        with self._buffer_lock:
            self._mkv_buffer.clear()
            self._has_valid_header = False
            self._looking_for_header = True
            
        # Start new process after brief delay
        time.sleep(1.0)
        self._start_ffmpeg_streaming()
            
    def _read_ffmpeg_frame(self, process) -> Optional[str]:
        """Read a single JPEG frame from FFmpeg stdout with timeout."""
        try:
            # Check if process is still running
            if process.poll() is not None:
                print(f"[FastVideoProcessor] FFmpeg process ended with return code: {process.returncode}")
                # Print stderr for debugging
                try:
                    stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                    if stderr_output:
                        print(f"[FastVideoProcessor] FFmpeg stderr: {stderr_output[:300]}")
                except:
                    pass
                # Trigger restart
                self._restart_ffmpeg()
                return None
            
            # Use non-blocking read with timeout
            import select
            import sys
            
            # Check if data is available (Linux/Unix only)
            if hasattr(select, 'select'):
                ready, _, _ = select.select([process.stdout], [], [], 0.1)  # 100ms timeout
                if not ready:
                    return None
            
            # Try to read a chunk of data
            try:
                chunk = process.stdout.read(4096)
            except:
                return None
                
            if not chunk:
                return None
                
            # Look for JPEG markers in the chunk
            start_marker = b'\xff\xd8'
            end_marker = b'\xff\xd9'
            
            start_idx = chunk.find(start_marker)
            if start_idx == -1:
                return None
                
            # Find end marker
            end_idx = chunk.find(end_marker, start_idx + 2)
            if end_idx == -1:
                # Read more data to find the end, but with limit
                try:
                    additional_data = process.stdout.read(8192)
                    if additional_data:
                        chunk += additional_data
                        end_idx = chunk.find(end_marker, start_idx + 2)
                except:
                    return None
                    
            if end_idx != -1:
                # Extract complete JPEG frame
                frame_data = chunk[start_idx:end_idx + 2]
                print(f"[FastVideoProcessor] Extracted JPEG frame: {len(frame_data)} bytes")
                return base64.b64encode(frame_data).decode('utf-8')
                
            return None
            
        except Exception as e:
            print(f"[FastVideoProcessor] Error reading FFmpeg frame: {e}")
            return None
            
    def _find_mkv_segment_start(self) -> None:
        """Find a valid MKV segment start in the buffer."""
        # Look for EBML header (0x1A45DFA3) or Segment header (0x18538067)
        ebml_header = b'\x1a\x45\xdf\xa3'
        segment_header = b'\x18\x53\x80\x67'
        
        # Search for EBML header first (start of new file)
        ebml_pos = self._mkv_buffer.find(ebml_header)
        if ebml_pos != -1:
            print(f"[FastVideoProcessor] Found EBML header at position {ebml_pos}")
            # Remove data before EBML header
            if ebml_pos > 0:
                self._mkv_buffer = self._mkv_buffer[ebml_pos:]
            self._has_valid_header = True
            self._looking_for_header = False
            self._restart_count = 0  # Reset restart counter on success
            return
            
        # Look for segment header (continuation)
        segment_pos = self._mkv_buffer.find(segment_header)
        if segment_pos != -1 and len(self._mkv_buffer) > segment_pos + 100:  # Need some data after header
            print(f"[FastVideoProcessor] Found Segment header at position {segment_pos}")
            # Remove data before segment header
            if segment_pos > 0:
                self._mkv_buffer = self._mkv_buffer[segment_pos:]
            self._has_valid_header = True
            self._looking_for_header = False
            self._restart_count = 0  # Reset restart counter on success
            return
            
        # If buffer is too large without finding headers, try direct mode
        if len(self._mkv_buffer) > 128 * 1024:  # 128KB (reduced threshold)
            print(f"[FastVideoProcessor] No valid headers found in {len(self._mkv_buffer)} bytes")
            if self._restart_count >= 2:  # After 2 failed attempts, go direct
                print(f"[FastVideoProcessor] Switching to direct streaming mode")
                self._restart_count = self._max_restarts
                self._has_valid_header = True
                self._looking_for_header = False
            else:
                self._mkv_buffer = self._mkv_buffer[-32*1024:]  # Keep last 32KB
            
            
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
