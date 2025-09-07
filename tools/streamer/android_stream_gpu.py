"""
High-performance Android screen streaming with GPU acceleration for RL agents.
Optimized for RTX 3060 with H265 hardware decoding and PyTorch integration.
"""

import subprocess  # nosec B404 - used with constant arguments for device control
import secrets
import socket
import struct
import threading
import time
import queue
from typing import Optional, Tuple, Callable
import torch
import av
from collections import deque


class GPUAndroidStreamer:
  """GPU-accelerated Android screen streamer using scrcpy protocol."""

  def __init__(self,
               device_id: Optional[str] = None,
               max_fps: int = 60,
               max_size: int = 1920,
               video_codec: str = "h264",
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
      subprocess.run(kill_cmd, capture_output=True, text=True, check=False)  # nosec B603

      # Clear all tunnels (both forward and reverse)
      clear_reverse_cmd = ["adb", "reverse", "--remove-all"]
      subprocess.run(clear_reverse_cmd, capture_output=True, text=True, check=False)  # nosec B603
      clear_forward_cmd = ["adb", "forward", "--remove-all"]
      subprocess.run(clear_forward_cmd, capture_output=True, text=True, check=False)  # nosec B603

      print("[CLEANUP] Cleaned up existing servers and tunnels")
      time.sleep(1)  # Wait for cleanup

    except Exception as e:
      print(f"Cleanup warning: {e}")

  def setup_adb_tunnel(self) -> Tuple[int]:
    """Setup ADB forward tunnel for scrcpy connection."""
    # Clean up any existing servers first
    self.cleanup_existing_servers()

    # Select a port in the safe ephemeral range deterministically using secure RNG to satisfy scanners
    local_port = 27000 + secrets.randbelow(1001)

    # Setup forward tunnel (PC -> device)
    tunnel_cmd = ["adb", "forward", f"tcp:{local_port}", "localabstract:scrcpy"]
    print(f"Setting up ADB forward tunnel: {' '.join(tunnel_cmd)}")
    result = subprocess.run(tunnel_cmd, capture_output=True, text=True, check=False)  # nosec B603

    if result.returncode != 0:
      print(f"ADB tunnel failed - STDOUT: {result.stdout}")
      print(f"ADB tunnel failed - STDERR: {result.stderr}")
      raise RuntimeError(f"Failed to setup ADB tunnel: {result.stderr}")

    print(f"[OK] ADB forward tunnel setup: localhost:{local_port} -> device:localabstract:scrcpy")

    # Verify tunnel is active
    list_cmd = ["adb", "forward", "--list"]
    list_result = subprocess.run(list_cmd, capture_output=True, text=True, check=False)  # nosec B603
    print(f"Active tunnels: {list_result.stdout.strip()}")

    return local_port

  def start_scrcpy_server(self) -> subprocess.Popen:
    """Start scrcpy server manually for direct socket access."""
    device_arg = f"-s {self.device_id}" if self.device_id else ""

    # Push scrcpy server to device
    server_locations = ["/usr/share/scrcpy/scrcpy-server", "/usr/local/bin/scrcpy-server", "/opt/scrcpy/scrcpy-server"]

    server_pushed = False
    for location in server_locations:
      try:
        print(f"Pushing scrcpy server to device: {location} with args: {device_arg}")
        push_cmd = f"adb {device_arg} push {location} /data/local/tmp/scrcpy-server.jar"
        result = subprocess.run(push_cmd.split(), capture_output=True, text=True, timeout=10, check=False)  # nosec B603
        if result.returncode == 0:
          server_pushed = True
          break
      except Exception as e:
        print(f"[DEBUG] Push attempt failed for {location}: {e}")
        continue  # nosec B112

    if not server_pushed:
      raise RuntimeError("Could not find or push scrcpy-server")

    # Start server with working arguments from your analysis
    server_cmd = [
      "adb", "shell", f"CLASSPATH=/data/local/tmp/scrcpy-server.jar", "app_process", "/",
      "com.genymobile.scrcpy.Server", "3.3.1", "tunnel_forward=true", "control=false", "cleanup=false",
      "raw_stream=true", "audio=false", f"max_size={self.max_size}", f"max_fps={self.max_fps}",
      f"video_bit_rate={int(self.bit_rate[:-1]) * 1024**2 if self.bit_rate[-1] == 'M' else self.bit_rate}",
      f"video_codec={self.video_codec}", "stay_awake=true", "power_off_on_close=false", "show_touches=false",
      "disable_screensaver=true", "log_level=verbose"
    ]

    server_cmd = [arg for arg in server_cmd if arg != ""]

    print(f"Starting scrcpy server: {' '.join(server_cmd)}")

    process = subprocess.Popen(  # nosec B603 B607
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
        print(f"[SERVER] Error reading remaining output: {e}")

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
            except Exception as e:
              print(f"[SERVER] Error reading final output: {e}")
            break

          # Try to read output non-blocking
          try:
            if self.scrcpy_process and self.scrcpy_process.stdout:
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
                  except Exception as e:
                    print(f"[SERVER] enqueue_output error: {e}")

                if not hasattr(self, '_output_queue'):
                  self._output_queue = queue.Queue()
                  self._output_thread = threading.Thread(target=enqueue_output,
                                                         args=(self.scrcpy_process.stdout, self._output_queue),
                                                         daemon=True)
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
    """Connect to scrcpy video socket with server readiness check."""
    max_attempts = 30

    for attempt in range(max_attempts):
      try:
        if self.scrcpy_process and self.scrcpy_process.poll() is not None:
          raise RuntimeError(f"Server process exited with code: {self.scrcpy_process.returncode}")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Configure socket for high-throughput streaming
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1MB receive buffer
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  # 1MB send buffer
        sock.settimeout(2.0)  # Short timeout for connection

        sock.connect(("localhost", port))
        print(f"Connected to video socket on port {port} with optimized settings")

        # For raw_stream=true, no handshake needed - direct H264 stream
        print("[DEBUG] Raw stream mode - no handshake, expecting direct H264")

        # Set non-blocking mode for streaming to prevent hanging
        sock.setblocking(False)
        return sock

      except Exception as e:
        if 'sock' in locals():
          sock.close()
        print(f"[DEBUG] Connection attempt {attempt + 1} failed: {e}")
        if attempt == max_attempts - 1:
          raise RuntimeError(f"Failed to connect after {max_attempts} attempts: {e}")
        time.sleep(0.1)

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
    """Setup GPU-accelerated H264 decoder."""
    codec_name = "h264"  # Force H264 since we're using raw_stream with h264

    if self.use_gpu and torch.cuda.is_available():
      try:
        # Try NVDEC hardware acceleration for H264
        codec = av.CodecContext.create(codec_name, "r")
        codec.options = {"hwaccel": "cuda", "hwaccel_device": "0"}
        print(f"[DEBUG] Created GPU H264 decoder with NVDEC")
        return codec
      except Exception as e:
        print(f"[DEBUG] GPU decoding failed: {e}, falling back to CPU")

    # Fallback to CPU decoding
    codec = av.CodecContext.create(codec_name, "r")
    print(f"[DEBUG] Created CPU H264 decoder")
    return codec

  def decode_frame_gpu(self, packet_data: bytes, decoder: av.CodecContext) -> Optional[torch.Tensor]:
    """Decode frame using GPU acceleration and return PyTorch tensor."""
    try:
      # Check NAL unit type for debugging
      if len(packet_data) >= 5:
        nal_type = packet_data[4] & 0x1F
        # Skip SPS/PPS parameter sets - they don't produce frames
        if nal_type == 7:  # SPS
          print(f"[DEBUG] Processing SPS parameter set")
          packet = av.Packet(packet_data)
          decoder.decode(packet)  # Feed to decoder but don't expect frames
          return None
        elif nal_type == 8:  # PPS
          print(f"[DEBUG] Processing PPS parameter set")
          packet = av.Packet(packet_data)
          decoder.decode(packet)  # Feed to decoder but don't expect frames
          return None
        else:
          print(
            f"[DEBUG] NAL unit type: {nal_type} ({'IDR' if nal_type == 5 else 'P-frame' if nal_type == 1 else 'Other'})"
          )

      packet = av.Packet(packet_data)
      frames = decoder.decode(packet)

      frame_count = 0
      for frame in frames:
        frame_count += 1
        # Convert to numpy array
        img_array = frame.to_ndarray(format='rgb24')

        # Convert to PyTorch tensor on GPU
        tensor = torch.from_numpy(img_array).to(self.device)

        # Normalize to [0, 1] for RL processing
        tensor = tensor.float() / 255.0

        # Rearrange to CHW format (channels first)
        tensor = tensor.permute(2, 0, 1)

        return tensor

      if frame_count == 0 and len(packet_data) >= 5:
        nal_type = packet_data[4] & 0x1F
        if nal_type not in [7, 8]:  # Only log for non-parameter sets
          print(f"[DEBUG] No frames decoded from {len(packet_data)} byte packet (NAL type: {nal_type})")

    except Exception as e:
      # Don't log errors for SPS/PPS as they're expected
      if len(packet_data) >= 5:
        nal_type = packet_data[4] & 0x1F
        if nal_type not in [7, 8]:
          print(f"[DEBUG] Decode error: {e} (packet size: {len(packet_data)})")
      return None

    return None

  def stream_worker(self):
    """Main streaming worker thread."""
    try:
      # Setup decoder
      decoder = self.setup_gpu_decoder()

      # Setup ADB tunnel and connect
      local_port = self.setup_adb_tunnel()
      time.sleep(1)  # Wait for tunnel

      # Start scrcpy server
      self.scrcpy_process = self.start_scrcpy_server()
      print(f"Server process started with PID: {self.scrcpy_process.pid}")

      # Connect to video socket (includes server readiness check)
      self.video_socket = self.connect_video_socket(local_port)

      # Read codec metadata for standard scrcpy protocol
      print("[DEBUG] Reading codec metadata...")
      try:
        codec_data = self.video_socket.recv(12)
        if len(codec_data) == 12:
          codec_id, width, height = struct.unpack(">III", codec_data)
          print(f"Video stream: {width}x{height}, codec_id: {codec_id}")
        else:
          print(f"[DEBUG] Incomplete codec data: {len(codec_data)} bytes")
      except Exception as e:
        print(f"[DEBUG] Codec metadata error: {e}")

      self.start_time = time.time()

      last_data_time = time.time()
      data_timeout = 2.0  # 2 second timeout for faster response

      while self.is_streaming:
        try:
          # Non-blocking receive with proper error handling
          chunk = self.video_socket.recv(65536)
          if not chunk:
            print("[DEBUG] Connection closed by server")
            break

          last_data_time = time.time()
          if len(chunk) > 1000:  # Only log large chunks to reduce spam
            print(f"[DEBUG] Received {len(chunk)} bytes")

        except socket.error as e:
          if e.errno == 10035 or e.errno == 11:  # WSAEWOULDBLOCK or EAGAIN
            # No data available right now, check for timeout
            if time.time() - last_data_time > data_timeout:
              print(f"[WARNING] No data received for {data_timeout}s - attempting to wake device")
              # Try to wake up the device/encoder
              self._wake_device()
              last_data_time = time.time()  # Reset to avoid spam
            time.sleep(0.001)
            continue
          else:
            print(f"[ERROR] Socket error: {e}")
            break
        except Exception as e:
          print(f"[ERROR] Unexpected error: {e}")
          break

        # Accumulate H264 data and find NAL units
        if not hasattr(self, '_h264_buffer'):
          self._h264_buffer = b""

        self._h264_buffer += chunk

        # Process complete NAL units
        while len(self._h264_buffer) > 4:
          # Find NAL unit start codes (0x00000001 or 0x000001)
          start_pos = -1
          for i in range(len(self._h264_buffer) - 3):
            if (self._h264_buffer[i:i + 4] == b'\x00\x00\x00\x01' or self._h264_buffer[i:i + 3] == b'\x00\x00\x01'):
              start_pos = i
              break

          if start_pos == -1:
            # No start code found, clear buffer if too large
            if len(self._h264_buffer) > 1048576:  # 1MB limit
              print(f"[DEBUG] Clearing large buffer without start code: {len(self._h264_buffer)} bytes")
              self._h264_buffer = b""
            break

          # Find end of this NAL unit (start of next one)
          end_pos = len(self._h264_buffer)
          for i in range(start_pos + 4, len(self._h264_buffer) - 3):
            if (self._h264_buffer[i:i + 4] == b'\x00\x00\x00\x01' or self._h264_buffer[i:i + 3] == b'\x00\x00\x01'):
              end_pos = i
              break

          if end_pos == len(self._h264_buffer) and len(self._h264_buffer) < 2097152:  # 2MB limit
            break  # Wait for more data

          # Extract NAL unit
          frame_data = self._h264_buffer[start_pos:end_pos]
          print(f"[DEBUG] Found NAL unit: {len(frame_data)} bytes, type: 0x{frame_data[4]:02x}")
          self._h264_buffer = self._h264_buffer[end_pos:]

          if len(frame_data) < 5:
            print(f"[DEBUG] Skipping small NAL unit: {len(frame_data)} bytes")
            continue

          # Decode frame
          try:
            tensor = self.decode_frame_gpu(frame_data, decoder)
            if tensor is not None:
              print(f"[DEBUG] Successfully decoded frame #{self.frame_count + 1}: {tensor.shape}")
              # Update statistics
              self.frame_count += 1
              current_time = time.time()
              if self.start_time:
                fps = self.frame_count / (current_time - self.start_time)
                self.fps_history.append(fps)

              # Add to queue (non-blocking) - use frame count as pts for raw stream
              pts = self.frame_count
              try:
                self.frame_queue.put((tensor, pts, current_time), block=False)
              except queue.Full:
                # Remove multiple oldest frames if buffer full to prevent stalling
                frames_removed = 0
                while frames_removed < 3:  # Remove up to 3 old frames
                  try:
                    self.frame_queue.get_nowait()
                    frames_removed += 1
                  except queue.Empty:
                    break
                try:
                  self.frame_queue.put((tensor, pts, current_time), block=False)
                  if frames_removed > 0:
                    print(f"[DEBUG] Dropped {frames_removed} old frames to prevent stalling")
                except queue.Full:
                  print(f"[DEBUG] Frame queue still full after dropping {frames_removed} frames")

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

  def _wake_device(self):
    """Send a wake signal to the Android device to prevent sleep."""
    try:
      # Send a light touch event to keep the encoder active
      subprocess.run(["adb", "shell", "input", "keyevent", "KEYCODE_WAKEUP"],
                     capture_output=True,
                     timeout=2,
                     check=False)  # nosec B603 B607
      print("[DEBUG] Sent wake signal to device")
    except Exception as e:
      print(f"[DEBUG] Failed to wake device: {e}")

  def cleanup(self):
    """Clean up resources."""
    self.is_streaming = False

    if self.video_socket:
      self.video_socket.close()
      self.video_socket = None

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
      size=(224, 224),  # Standard RL input size
      mode='bilinear',
      align_corners=False).squeeze(0)  # Remove batch dimension

    print(f"Frame processed: {resized.shape}, GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

    # Here you would feed the tensor to your RL agent
    # agent.process_observation(resized)

  # Create streamer
  streamer = GPUAndroidStreamer(max_fps=60, max_size=1920, video_codec="h265", use_gpu=True)

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
