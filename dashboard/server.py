import asyncio
import json
import websockets
import threading
import time
from typing import Set, Optional, Dict, Any
import subprocess
import base64
from io import BytesIO


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
        self._running = False
        if self.loop and self.server:
            # Schedule server close in the event loop
            future = asyncio.run_coroutine_threadsafe(self._close_server(), self.loop)
            try:
                future.result(timeout=2.0)
            except Exception as e:
                print(f"Error closing server: {e}")
        if self.thread:
            self.thread.join(timeout=2.0)
        print("WebSocket server stopped")
        
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
        # For now, just clear buffer and increment frame count
        # TODO: Implement proper MKV parsing and frame extraction
        self._frame_count += 1
        
        # Keep only last 32KB to prevent memory growth
        if len(self._mkv_buffer) > 32768:
            self._mkv_buffer = self._mkv_buffer[-32768:]
            
        # Send frame placeholder (for testing)
        if self.loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._broadcast({
                    "type": "log",
                    "level": "info",
                    "message": f"Frame {self._frame_count} processed ({len(self._mkv_buffer)} bytes buffered)"
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
