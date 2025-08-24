import asyncio
import websockets
import threading
import json
import time
from typing import Set, Optional, Dict, Any
import subprocess
import base64
from io import BytesIO
import queue


class WebSocketDashboardSubscriber:
    """WebSocket server subscriber that integrates with the pipeline."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server: Optional[websockets.WebSocketServer] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        
        # Data accumulation for frame processing
        self._mkv_buffer = bytearray()
        self._frame_count = 0
        self._last_stats_time = 0
        
        # FFmpeg process for frame extraction
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._frame_queue = queue.Queue(maxsize=10)
        self._frame_thread: Optional[threading.Thread] = None
        self._device_resolution = None
        
        # Pipeline reference for stats
        self._pipeline = None
        
    def set_pipeline(self, pipeline):
        """Set reference to pipeline for stats access."""
        self._pipeline = pipeline
        
    def start(self) -> None:
        """Start the WebSocket server in a separate thread."""
        if self._running:
            return
            
        self._running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"WebSocket server starting on {self.host}:{self.port}")
        
    def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return
            
        print("Stopping WebSocket server...")
        self._running = False
        
        # Stop FFmpeg process
        self._stop_ffmpeg()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            
            # Schedule server close in the event loop
            future = asyncio.run_coroutine_threadsafe(self._close_server(), self.loop)
            try:
                future.result(timeout=2.0)
            except Exception as e:
                print(f"Error closing server: {e}")
        if self.thread:
            self.thread.join(timeout=2.0)
        print("WebSocket server stopped")
        
    def _start_ffmpeg_process(self) -> None:
        """Start FFmpeg process for frame extraction."""
        try:
            # FFmpeg command to extract frames from MKV stream
            cmd = [
                'ffmpeg',
                '-f', 'matroska',  # Input format
                '-i', 'pipe:0',    # Read from stdin
                '-vf', self._get_scale_filter(),  # Dynamic scaling based on device
                '-f', 'image2pipe',      # Output format
                '-vcodec', 'png',        # PNG output
                '-r', '5',               # 5 FPS for dashboard
                'pipe:1'                 # Write to stdout
            ]
            
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            print("FFmpeg process started for frame extraction")
            
        except Exception as e:
            print(f"Failed to start FFmpeg: {e}")
            
    def _stop_ffmpeg(self) -> None:
        """Stop FFmpeg process."""
        if self._ffmpeg_process:
            try:
                self._ffmpeg_process.terminate()
                self._ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ffmpeg_process.kill()
            except Exception as e:
                print(f"Error stopping FFmpeg: {e}")
            finally:
                self._ffmpeg_process = None
                
    def _restart_ffmpeg(self) -> None:
        """Restart FFmpeg process."""
        self._stop_ffmpeg()
        self._start_ffmpeg_process()
        
    def _frame_processor(self) -> None:
        """Process frames from FFmpeg output."""
        while self._running:
            # Process FFmpeg output
            if self._ffmpeg_process and self._ffmpeg_process.stdout:
                try:
                    # Read PNG header to determine frame size
                    png_header = self._ffmpeg_process.stdout.read(8)
                    if len(png_header) == 8 and png_header[:4] == b'\x89PNG':
                        # Read rest of PNG data (simplified - assumes small frames)
                        frame_data = png_header + self._ffmpeg_process.stdout.read(65536)
                        
                        if frame_data:
                            # Convert to base64 for transmission
                            frame_b64 = base64.b64encode(frame_data).decode('utf-8')
                            
                            # Queue frame for transmission
                            try:
                                self._frame_queue.put_nowait({
                                    "type": "frame",
                                    "data": frame_b64,
                                    "timestamp": time.time()
                                })
                                self._frame_count += 1
                            except queue.Full:
                                # Drop frame if queue is full
                                pass
                                
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    
            # Send queued frames to clients
            try:
                frame_data = self._frame_queue.get(timeout=0.1)
                if self.loop and self._running:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast(frame_data),
                        self.loop
                    )
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Frame transmission error: {e}")
        
    async def _close_server(self) -> None:
        """Close the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
    def _run_server(self) -> None:
        """Run the WebSocket server in its own event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            print(f"WebSocket server thread started, binding to {self.host}:{self.port}")
            
            # Start frame processing thread
            self._frame_thread = threading.Thread(target=self._frame_processor, daemon=True)
            self._frame_thread.start()
            
            self.loop.run_until_complete(self._start_server())
        except Exception as e:
            print(f"WebSocket server error: {e}")
        finally:
            self.loop.close()
            
    async def _start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            )
            print(f"[OK] WebSocket server running on ws://{self.host}:{self.port}")
            
            # Keep server running
            while self._running:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"[ERROR] Failed to start WebSocket server on {self.host}:{self.port}: {e}")
            raise
            
    async def _handle_client(self, websocket, path=None) -> None:
        """Handle new WebSocket client connection."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"Client connected: {client_addr}")
        
        try:
            # Send initial connection message
            await self._send_to_client(websocket, {
                "type": "log",
                "level": "info",
                "message": f"Connected to pipeline server"
            })
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_addr}")
        except Exception as e:
            print(f"Client error {client_addr}: {e}")
        finally:
            self.clients.discard(websocket)
            
    async def _handle_message(self, websocket, message: str) -> None:
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "touch":
                await self._handle_touch_event(data)
            elif msg_type == "ping":
                await self._send_to_client(websocket, {"type": "pong"})
            else:
                print(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON from client: {message}")
            
    async def _handle_touch_event(self, data: Dict[str, Any]) -> None:
        """Handle touch event from dashboard."""
        x = data.get("x", 0)
        y = data.get("y", 0)
        
        # Send ADB touch command
        try:
            subprocess.run([
                "adb", "shell", "input", "tap", str(x), str(y)
            ], check=True, capture_output=True)
            
            # Broadcast touch event to all clients
            await self._broadcast({
                "type": "log",
                "level": "info", 
                "message": f"Touch sent: ({x}, {y})"
            })
            
        except subprocess.CalledProcessError as e:
            await self._broadcast({
                "type": "log",
                "level": "error",
                "message": f"Touch failed: {e}"
            })
            
    async def _send_to_client(self, websocket, data: Dict[str, Any]) -> None:
        """Send data to specific client."""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            print(f"Failed to send to client: {e}")
            
    async def _broadcast(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected clients."""
        if not self.clients:
            return
            
        message = json.dumps(data)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)
                
        # Remove disconnected clients
        self.clients -= disconnected
        
    def handle_chunk(self, chunk) -> None:
        """Handle data chunk from pipeline (subscriber callback)."""
        # Accumulate MKV data
        self._mkv_buffer.extend(chunk.data.tobytes())
        
        # Try to extract frame every 64KB or so
        if len(self._mkv_buffer) > 65536:
            self._process_mkv_buffer()
            
        # Send stats periodically
        current_time = time.time()
        if current_time - self._last_stats_time >= 1.0:
            self._send_stats()
            self._last_stats_time = current_time
            
    def _process_mkv_buffer(self) -> None:
        """Process accumulated MKV data to extract frames."""
        if len(self._mkv_buffer) < 1024:  # Wait for more data
            return
            
        # Start FFmpeg process if not running
        if self._ffmpeg_process is None or self._ffmpeg_process.poll() is not None:
            self._start_ffmpeg_process()
            
        # Send data to FFmpeg
        if self._ffmpeg_process and self._ffmpeg_process.stdin:
            try:
                self._ffmpeg_process.stdin.write(bytes(self._mkv_buffer))
                self._ffmpeg_process.stdin.flush()
                self._mkv_buffer.clear()
            except (BrokenPipeError, OSError) as e:
                print(f"FFmpeg pipe error: {e}")
                self._restart_ffmpeg()
                
    def _get_device_resolution(self) -> tuple:
        """Get device resolution from ADB."""
        try:
            result = subprocess.run(
                ['adb', 'shell', 'wm', 'size'],
                capture_output=True, text=True, check=True
            )
            # Parse output like "Physical size: 1080x2400"
            for line in result.stdout.strip().split('\n'):
                if 'size:' in line:
                    size_part = line.split('size:')[1].strip()
                    width, height = map(int, size_part.split('x'))
                    return (width, height)
        except Exception as e:
            print(f"Could not get device resolution: {e}")
        
        # Default fallback
        return (1080, 1920)
        
    def _get_scale_filter(self) -> str:
        """Generate FFmpeg scale filter maintaining aspect ratio."""
        if not self._device_resolution:
            self._device_resolution = self._get_device_resolution()
            
        width, height = self._device_resolution
        
        # Calculate target size maintaining aspect ratio
        # Max width/height for dashboard display
        max_width = 400
        max_height = 600
        
        # Calculate scale factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        target_width = int(width * scale)
        target_height = int(height * scale)
        
        # Ensure even dimensions for video encoding
        target_width = target_width - (target_width % 2)
        target_height = target_height - (target_height % 2)
        
        print(f"Device resolution: {width}x{height}, scaling to: {target_width}x{target_height}")
        return f'scale={target_width}:{target_height}'
                
        # Send frame processing status
        if self.loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({
                    "type": "log",
                    "level": "info",
                    "message": f"Processing MKV data ({len(self._mkv_buffer)} bytes buffered, {self._frame_count} frames, {self._device_resolution})"
                }),
                self.loop
            )
            
    def _send_stats(self) -> None:
        """Send pipeline statistics to clients."""
        if not self._pipeline or not self.loop or not self._running:
            return
            
        stats = self._pipeline.get_stats()
        
        # Send metrics to dashboard
        asyncio.run_coroutine_threadsafe(
            self._broadcast({
                "type": "metrics",
                "data": {
                    "dataRate": stats.get("avg_rate_bps", 0),
                    "frameRate": self._frame_count,  # Frames processed this second
                    "totalData": stats.get("total_bytes", 0),
                    "totalChunks": stats.get("total_chunks", 0),
                    "uptime": stats.get("elapsed_time", 0)
                }
            }),
            self.loop
        )
        
        # Reset frame count for next second
        self._frame_count = 0


# Integration helper
def create_dashboard_subscriber(host: str = "0.0.0.0", port: int = 8765) -> WebSocketDashboardSubscriber:
    """Create and return a dashboard subscriber ready for pipeline integration."""
    return WebSocketDashboardSubscriber(host, port)
