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
        
    def setup_adb_tunnel(self) -> tuple:
        """Setup ADB tunnel for scrcpy connection."""
        import random
        port = random.randint(27183, 27283)
        scid = random.randint(1, 2**31 - 1)  # 31-bit random number
        
        device_arg = f"-s {self.device_id}" if self.device_id else ""
        # Use reverse tunnel - device connects to computer
        cmd = f"adb {device_arg} reverse localabstract:scrcpy_{scid:08x} tcp:{port}"
        
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to setup ADB tunnel: {result.stderr}")
        
        return port, scid
    
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
        
        # Start server with correct argument format (space-separated values)
        server_cmd = [
            "adb", device_arg, "shell",
            f"CLASSPATH=/data/local/tmp/scrcpy-server.jar",
            "app_process", "/", "com.genymobile.scrcpy.Server",
            "2.1",  # version
            f"{scid:08x}",  # scid without key=value format
            "info",  # log_level
            str(self.max_size),  # max_size
            self.bit_rate.replace('M', '000000'),  # bit_rate
            str(self.max_fps),  # max_fps
            "-1",  # lock_video_orientation
            "false",  # tunnel_forward
            "-",  # crop
            "false",  # send_frame_meta
            "false",  # control
            "0",  # display_id
            "false",  # show_touches
            "true",  # stay_awake
            f"{1 if self.video_codec == 'h265' else 0}",  # video_codec
            "0",  # video_encoder
            "false",  # audio
            "0",  # audio_codec
            "0",  # audio_encoder
            "false"  # camera
        ]
        
        # Remove empty args
        server_cmd = [arg for arg in server_cmd if arg.strip()]
        
        print(f"Starting scrcpy server: {' '.join(server_cmd)}")
        
        process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return process
    
    def connect_video_socket(self, port: int) -> socket.socket:
        """Connect to scrcpy video socket."""
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
            port, scid = self.setup_adb_tunnel()
            time.sleep(1)  # Wait for tunnel
            
            # Start scrcpy server
            self.scrcpy_process = self.start_scrcpy_server(scid)
            time.sleep(3)  # Wait for server startup
            
            # Connect to video socket
            self.video_socket = self.connect_video_socket(port)
            
            # Read codec metadata
            codec_data = self.video_socket.recv(12)
            codec_id, width, height = struct.unpack(">III", codec_data)
            print(f"Video stream: {width}x{height}, codec_id: {codec_id}")
            
            self.start_time = time.time()
            
            while self.is_streaming:
                try:
                    # Read frame header
                    config_packet, key_frame, pts, packet_size = self.read_frame_header(self.video_socket)
                    
                    # Read frame data
                    frame_data = b""
                    remaining = packet_size
                    while remaining > 0:
                        chunk = self.video_socket.recv(min(remaining, 8192))
                        if not chunk:
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
        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass
            self.video_socket = None
        
        if self.scrcpy_process:
            try:
                self.scrcpy_process.terminate()
                self.scrcpy_process.wait(timeout=5)
            except:
                try:
                    self.scrcpy_process.kill()
                except:
                    pass
            self.scrcpy_process = None
        
        # Clear ADB tunnel
        try:
            subprocess.run(["adb", "reverse", "--remove-all"], capture_output=True)
        except:
            pass


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
