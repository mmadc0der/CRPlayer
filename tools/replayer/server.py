import asyncio
from typing import List

import websockets


class ReplayerServer:
  """Low-latency WebSocket server broadcasting frame and metadata packets."""

  def __init__(self, host: str = "0.0.0.0", port: int = 8765):
    self.host = host
    self.port = port
    self._server: websockets.server.Serve = None
    self._clients: List[websockets.WebSocketServerProtocol] = []
    # Protect client list in async context
    self._clients_lock = asyncio.Lock()

  async def _handler(self, websocket: websockets.WebSocketServerProtocol):
    """Handle incoming WebSocket connection."""
    async with self._clients_lock:
      self._clients.append(websocket)
    try:
      # Keep connection open; this server is write-only for now
      async for _ in websocket:  # read to detect close
        pass
    finally:
      async with self._clients_lock:
        if websocket in self._clients:
          self._clients.remove(websocket)

  async def broadcast(self, data: bytes):
    """Broadcast binary packet to all connected clients."""
    if not self._clients:
      return
    async with self._clients_lock:
      # Use copy to prevent modification during iteration
      clients_snapshot = list(self._clients)
    # Send concurrently but don't await gather errors strictly
    coros = []
    for ws in clients_snapshot:
      if ws.closed:
        async with self._clients_lock:
          if ws in self._clients:
            self._clients.remove(ws)
        continue
      coros.append(ws.send(data))
    if coros:
      await asyncio.gather(*coros, return_exceptions=True)

  async def start(self):
    """Start WebSocket server asynchronously."""
    if self._server is not None:
      return
    self._server = await websockets.serve(self._handler, self.host, self.port, ping_interval=None)
    print(f"[ReplayerServer] Listening on ws://{self.host}:{self.port}")

  async def stop(self):
    """Stop server and close client connections."""
    if self._server is None:
      return
    self._server.close()
    await self._server.wait_closed()
    async with self._clients_lock:
      clients_snapshot = list(self._clients)
    for ws in clients_snapshot:
      await ws.close(code=1001, reason="Server shutdown")
    self._server = None
    print("[ReplayerServer] Server stopped")
