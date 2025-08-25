"""
High-performance Android screen streaming with GPU acceleration for RL agents.
Optimized for RTX 3060 with H265 hardware decoding and PyTorch integration.
"""

import subprocess
import socket
import struct
import threading
import time
import queue
import random
from typing import Optional, Tuple, Callable
import numpy as np
import torch
import cv2
import av
from collections import deque


class GPUAndroidStreamer:
    """GPU-accelerated Android screen streamer using scrcpy protocol."""
    
    def __init__(self, 
                 device_id: Optional[str] = None,
                 max_fps: int = 60,
                 max_size: int = 1920,
                 video_codec: str = "h265",
                 bit_rate: str = "8M",
                 use_gpu: bool = True,
                 buffer_size: int = 30):
        """
        Initialize GPU-accelerated Android streamer.
        
        Args:
            device_id: ADB device ID (None for first available)
            max_fps: Maximum frame rate (60 recommended)
            max_size: Maximum resolution (1920 recommended for performance)
            video_codec: Video codec (h265 for better quality/compression)
            bit_rate: Video bitrate (8M recommended for 60fps)
            use_gpu: Enable GPU acceleration
            buffer_size: Frame buffer size for smooth playback
        """
        self.device_id = device_id
        self.max_fps = max_fps
        self.max_size = max_size
        self.video_codec = video_codec
        self.bit_rate = bit_rate
        self.use_gpu = use_gpu
        self.buffer_size = buffer_size
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using device: {self.device}")
        
        # Streaming state
        self.is_streaming = False
        self.scrcpy_process = None
        self.video_socket = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stats_queue = queue.Queue(maxsize=100)
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = None
        self.fps_history = deque(maxlen=60)
        
        # Callbacks
        self.frame_callback: Optional[Callable] = None
        
    def cleanup_existing_servers(self):
        """Kill any existing scrcpy server processes."""
        try:
            # Kill any existing scrcpy server processes on device
            kill_cmd = ["adb", "shell", "pkill", "-f", "scrcpy-server"]
            subprocess.run(kill_cmd, capture_output=True, text=True)
            
            # Clear all tunnels (both forward and reverse)
            clear_reverse_cmd = ["adb", "reverse", "--remove-all"]
            subprocess.run(clear_reverse_cmd, capture_output=True, text=True)
            clear_forward_cmd = ["adb", "forward", "--remove-all"]
            subprocess.run(clear_forward_cmd, capture_output=True, text=True)
            
            print("[CLEANUP] Cleaned up existing servers and tunnels")
            time.sleep(1)  # Wait for cleanup
            
        except Exception as e:
            print(f"Cleanup warning: {e}")
    
    def setup_adb_tunnel(self) -> Tuple[int, int]:
        """Setup ADB forward tunnel for scrcpy connection."""
        # Clean up any existing servers first
        self.cleanup_existing_servers()
        
        # Generate random port and session ID (decimal for scid)
        local_port = random.randint(27000, 28000)
        device_port = random.randint(27000, 28000)
        scid = random.randint(0x10000000, 0xFFFFFFFF)
        
        # Setup forward tunnel (PC -> device)
        tunnel_cmd = ["adb", "forward", f"tcp:{local_port}", f"tcp:{device_port}"]
        print(f"Setting up ADB forward tunnel: {' '.join(tunnel_cmd)}")
        result = subprocess.run(tunnel_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ADB tunnel failed - STDOUT: {result.stdout}")
            print(f"ADB tunnel failed - STDERR: {result.stderr}")
            raise RuntimeError(f"Failed to setup ADB tunnel: {result.stderr}")
        
        print(f"[OK] ADB forward tunnel setup: localhost:{local_port} -> device:{device_port}")
        
        # Verify tunnel is active
        list_cmd = ["adb", "forward", "--list"]
        list_result = subprocess.run(list_cmd, capture_output=True, text=True)
        print(f"Active tunnels: {list_result.stdout.strip()}")
        
        return local_port, scid
    
    def start_scrcpy_server(self, scid: int) -> subprocess.Popen:
        """Start scrcpy server manually for direct socket access."""
        device_arg = f"-s {self.device_id}" if self.device_id else ""
        
        # Push scrcpy server to device
        server_locations = [
            "/usr/share/scrcpy/scrcpy-server",
            "/usr/local/bin/scrcpy-server", 
            "/opt/scrcpy/scrcpy-server"
        ]
        
        server_pushed = False
        for location in server_locations:
            try:
                push_cmd = f"adb {device_arg} push {location} /data/local/tmp/scrcpy-server.jar"
                result = subprocess.run(push_cmd.split(), capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    server_pushed = True
                    break
            except:
                continue
        
        if not server_pushed:
            raise RuntimeError("Could not find or push scrcpy-server")
        
        # Start server with scrcpy 3.3.1 minimal valid arguments
        server_cmd = [
            "adb", device_arg, "shell",
            f"CLASSPATH=/data/local/tmp/scrcpy-server.jar",
            "app_process", "/", "com.genymobile.scrcpy.Server",
            "3.3.1",  # version
            f"scid={scid:08x}",
            "log_level=debug",
            f"max_size={self.max_size}",
            f"video_bit_rate={self.bit_rate.replace('M', '000000')}",  # correct parameter name
            f"max_fps={self.max_fps}",
            "tunnel_forward=true",
            "send_frame_meta=false",
            "control=false",
            "display_id=0",
            "show_touches=false",
            "stay_awake=true",
            "video_codec=h265",
            "audio=false"
        ]
        
        # Remove empty args
        server_cmd = [arg for arg in server_cmd if arg.strip()]
        
        print(f"Starting scrcpy server: {' '.join(server_cmd)}")
        
        process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=0  # Unbuffered for real-time output
        )
        
        # Start monitoring immediately
        self.scrcpy_process = process
        self.monitor_server_output()
        
        # Check if server process actually started
        time.sleep(1)
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"Server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            raise RuntimeError(f"Server process exited with code {process.returncode}")
        
        return process
    
    def wait_for_server_ready(self, port: int, max_attempts: int = 30) -> bool:
        """Wait for server to be ready and listening on port."""
        for attempt in range(max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("localhost", port))
                sock.close()
                
                if result == 0:
                    print(f"Server ready on port {port} after {attempt + 1} attempts")
                    return True
                    
            except Exception as e:
                pass
            
            print(f"Attempt {attempt + 1}/{max_attempts}: Server not ready on port {port}")
            time.sleep(1)
        
        return False
    
    def check_server_status(self):
        """Check actual server process status and output."""
        if not self.scrcpy_process:
            print("No server process found")
            return
        
        poll_result = self.scrcpy_process.poll()
        print(f"Server process status: {'Running' if poll_result is None else f'Exited with code {poll_result}'}")
        
        if poll_result is not None:
            # Process has exited, get output
            try:
                stdout, stderr = self.scrcpy_process.communicate(timeout=1)
                if stdout:
                    print(f"[SERVER STDOUT] {stdout}")
                if stderr:
                    print(f"[SERVER STDERR] {stderr}")
            except subprocess.TimeoutExpired:
                print("Timeout reading server output")
    
    def monitor_server_output(self):
        """Monitor server stdout/stderr in background thread."""
        if not self.scrcpy_process:
            return
        
        def read_output():
            print("[MONITOR] Starting server output monitoring...")
            while self.is_streaming and self.scrcpy_process:
                try:
                    poll_result = self.scrcpy_process.poll()
                    if poll_result is not None:
                        print(f"[ERROR] Server process exited with code: {poll_result}")
                        # Get remaining output
                        try:
                            remaining_output = self.scrcpy_process.stdout.read()
                            if remaining_output:
                                print(f"[SERVER FINAL] {remaining_output}")
                        except:
                            pass
                        break
                    
                    # Read output line by line
                    try:
                        if self.scrcpy_process.stdout:
                            # Use readline with timeout simulation
                            import select
                            import sys
                            
                            if sys.platform != 'win32':
                                # Unix-like systems - use select
                                ready, _, _ = select.select([self.scrcpy_process.stdout], [], [], 0.1)
                                if ready:
                                    line = self.scrcpy_process.stdout.readline()
                                    if line:
                                        print(f"[SERVER] {line.strip()}")
                            else:
                                # Windows - read with timeout
                                import threading
                                import queue
                                
                                def enqueue_output(out, queue):
                                    try:
                                        for line in iter(out.readline, ''):
                                            queue.put(line)
                                    except:
                                        pass
                                
                                if not hasattr(self, '_output_queue'):
                                    self._output_queue = queue.Queue()
                                    self._output_thread = threading.Thread(
                                        target=enqueue_output, 
                                        args=(self.scrcpy_process.stdout, self._output_queue),
                                        daemon=True
                                    )
                                    self._output_thread.start()
                                
                                try:
                                    line = self._output_queue.get_nowait()
                                    if line:
                                        print(f"[SERVER] {line.strip()}")
                                except queue.Empty:
                                    pass
                                    
                    except Exception as e:
                        print(f"Error reading server output: {e}")
                        
                    time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error monitoring server: {e}")
                    break
        
        monitor_thread = threading.Thread(target=read_output, daemon=True)
        monitor_thread.start()
    
    def connect_video_socket(self, port: int) -> socket.socket:
        """Connect to scrcpy video socket with server monitoring."""
        # Check server status before waiting
        self.check_server_status()
        
        # Wait for server to be ready
        if not self.wait_for_server_ready(port):
            print("Server readiness check failed, checking status again...")
            self.check_server_status()
            raise RuntimeError(f"Server not ready on port {port} after waiting")
        
        # Start monitoring server output
        self.monitor_server_output()
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)  # 10 second timeout
        
        try:
            sock.connect(("localhost", port))
            print(f"Connected to video socket on port {port}")
            
            # For reverse tunnel, device sends dummy byte first
            dummy = sock.recv(1)
            if not dummy:
                raise RuntimeError("Failed to receive dummy byte")
            print(f"Received dummy byte: {dummy.hex()}")
            
            # Read device name length and name
            name_length_data = sock.recv(4)
            if len(name_length_data) != 4:
                raise RuntimeError("Failed to receive device name length")
            
            name_length = struct.unpack(">I", name_length_data)[0]
            print(f"Device name length: {name_length}")
            
            if name_length > 0:
                device_name = sock.recv(name_length).decode("utf-8")
                print(f"Connected to device: {device_name}")
            
            sock.settimeout(None)  # Remove timeout for streaming
            return sock
            
        except Exception as e:
            sock.close()
            raise RuntimeError(f"Failed to connect to video socket: {e}")
    
    def read_frame_header(self, sock: socket.socket) -> Tuple[bool, bool, int, int]:
        """Read scrcpy frame header."""
        if not sock:
            raise RuntimeError("Socket is None")
        
        header_data = sock.recv(12)
        if len(header_data) != 12:
            raise RuntimeError("Failed to read frame header")
        
        # Parse frame header
        pts_and_flags = struct.unpack(">Q", header_data[:8])[0]
        packet_size = struct.unpack(">I", header_data[8:])[0]
        
        # Extract flags from PTS
        config_packet = bool(pts_and_flags & (1 << 63))
        key_frame = bool(pts_and_flags & (1 << 62))
        pts = pts_and_flags & ((1 << 62) - 1)
        
        return config_packet, key_frame, pts, packet_size
    
    def setup_gpu_decoder(self) -> av.CodecContext:
        """Setup GPU-accelerated H265 decoder."""
        if self.use_gpu and torch.cuda.is_available():
            try:
                # Try NVDEC hardware acceleration
                codec = av.CodecContext.create("hevc", "r")
                codec.options = {"hwaccel": "cuda", "hwaccel_device": "0"}
                return codec
            except:
                print("GPU decoding not available, falling back to CPU")
        
        # Fallback to CPU decoding
        return av.CodecContext.create("hevc" if self.video_codec == "h265" else "h264", "r")
    
    def decode_frame_gpu(self, packet_data: bytes, decoder: av.CodecContext) -> Optional[torch.Tensor]:
        """Decode frame using GPU acceleration and return PyTorch tensor."""
        try:
            packet = av.Packet(packet_data)
            frames = decoder.decode(packet)
            
            for frame in frames:
                # Convert to numpy array
                img_array = frame.to_ndarray(format='rgb24')
                
                # Convert to PyTorch tensor on GPU
                tensor = torch.from_numpy(img_array).to(self.device)
                
                # Normalize to [0, 1] for RL processing
                tensor = tensor.float() / 255.0
                
                # Rearrange to CHW format (channels first)
                tensor = tensor.permute(2, 0, 1)
                
                return tensor
                
        except Exception as e:
            print(f"Decode error: {e}")
            return None
        
        return None
    
    def stream_worker(self):
        """Main streaming worker thread."""
        try:
            # Setup decoder
            decoder = self.setup_gpu_decoder()
            
            # Setup ADB tunnel and connect
            local_port, scid = self.setup_adb_tunnel()
            time.sleep(1)  # Wait for tunnel
            
            # Start scrcpy server
            self.scrcpy_process = self.start_scrcpy_server(scid)
            print(f"Server process started with PID: {self.scrcpy_process.pid}")
            
            # Connect to video socket (includes server readiness check)
            self.video_socket = self.connect_video_socket(local_port)
            
            # Read codec metadata
            codec_data = self.video_socket.recv(12)
            codec_id, width, height = struct.unpack(">III", codec_data)
            print(f"Video stream: {width}x{height}, codec_id: {codec_id}")
            
            self.start_time = time.time()
            
            while self.is_streaming:
                try:
                    # Validate socket connection
                    if not self.video_socket:
                        print("[ERROR] Video socket is None, stopping stream")
                        break
                    
                    # Read frame header
                    config_packet, key_frame, pts, packet_size = self.read_frame_header(self.video_socket)
                    
                    # Read frame data
                    frame_data = b""
                    remaining = packet_size
                    while remaining > 0:
                        if not self.video_socket:
                            print("[ERROR] Socket disconnected during frame read")
                            break
                        chunk = self.video_socket.recv(min(remaining, 8192))
                        if not chunk:
                            print("[ERROR] No data received, connection lost")
                            break
                        frame_data += chunk
                        remaining -= len(chunk)
                    
                    if len(frame_data) != packet_size:
                        continue
                    
                    # Decode frame
                    tensor = self.decode_frame_gpu(frame_data, decoder)
                    if tensor is not None:
                        # Update statistics
                        self.frame_count += 1
                        current_time = time.time()
                        if self.start_time:
                            fps = self.frame_count / (current_time - self.start_time)
                            self.fps_history.append(fps)
                        
                        # Add to queue (non-blocking)
                        try:
                            self.frame_queue.put((tensor, pts, current_time), block=False)
                        except queue.Full:
                            # Remove oldest frame if buffer full
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put((tensor, pts, current_time), block=False)
                            except queue.Empty:
                                pass
                        
                        # Call frame callback if set
                        if self.frame_callback:
                            self.frame_callback(tensor, pts, current_time)
                            
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
                    
        except Exception as e:
            print(f"Streaming error: {e}")
        finally:
            self.cleanup()
    
    def start_streaming(self, frame_callback: Optional[Callable] = None):
        """Start the streaming process."""
        if self.is_streaming:
            return
        
        self.frame_callback = frame_callback
        self.is_streaming = True
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self.stream_worker, daemon=True)
        self.stream_thread.start()
        
        print("Streaming started...")
    
    def stop_streaming(self):
        """Stop the streaming process."""
        self.is_streaming = False
        
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join(timeout=5)
        
        self.cleanup()
        print("Streaming stopped.")
    
    def get_latest_frame(self) -> Optional[Tuple[torch.Tensor, int, float]]:
        """Get the latest frame from the buffer."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_fps_stats(self) -> dict:
        """Get current FPS statistics."""
        if not self.fps_history:
            return {"current_fps": 0, "avg_fps": 0, "frame_count": self.frame_count}
        
        current_fps = self.fps_history[-1] if self.fps_history else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        return {
            "current_fps": round(current_fps, 2),
            "avg_fps": round(avg_fps, 2),
            "frame_count": self.frame_count,
            "buffer_size": self.frame_queue.qsize()
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.is_streaming = False
        
        if self.video_socket:
            self.video_socket.close()
            self.video_socket = None
            # Start scrcpy server
            self.scrcpy_process = self.start_scrcpy_server(scid)
            print(f"Server process started with PID: {self.scrcpy_process.pid}")
            
            # Connect to video socket (includes server readiness check)
            self.video_socket = self.connect_video_socket(port)
        
        if self.scrcpy_process:
            self.scrcpy_process.terminate()
            self.scrcpy_process.wait()
            self.scrcpy_process = None
        
        # Clean up server processes and tunnels
        self.cleanup_existing_servers()


# Example usage for RL agent
def example_rl_integration():
    """Example of how to integrate with RL agent."""
    
    def frame_processor(tensor: torch.Tensor, pts: int, timestamp: float):
        """Process frame for RL agent."""
        # tensor is already on GPU and normalized [0,1]
        # Shape: (3, height, width) - CHW format
        
        # Example: resize for RL agent input
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),  # Add batch dimension
            size=(224, 224),      # Standard RL input size
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        print(f"Frame processed: {resized.shape}, GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
        # Here you would feed the tensor to your RL agent
        # agent.process_observation(resized)
    
    # Create streamer
    streamer = GPUAndroidStreamer(
        max_fps=60,
        max_size=1920,
        video_codec="h265",
        use_gpu=True
    )
    
    # Start streaming
    streamer.start_streaming(frame_callback=frame_processor)
    
    try:
        # Monitor performance
        while True:
            time.sleep(5)
            stats = streamer.get_fps_stats()
            print(f"FPS Stats: {stats}")
            
            # Get latest frame manually if needed
            frame_data = streamer.get_latest_frame()
            if frame_data:
                tensor, pts, timestamp = frame_data
                print(f"Latest frame: {tensor.shape}")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        streamer.stop_streaming()


if __name__ == "__main__":
    example_rl_integration()
