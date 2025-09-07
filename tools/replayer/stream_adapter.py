import asyncio
import time
from typing import Optional

import torch

from core.stream_pipeline import SharedStreamBuffer, FrameData
from .server import ReplayerServer

PACKET_MAGIC = b"CRPL"  # 4 bytes magic prefix for packet identification


def encode_packet(frame: FrameData) -> bytes:
  """Encode FrameData to binary packet: magic|frame_id(u32)|pts(u64)|ts_ms(u64)|height(u16)|width(u16)|channels(u8)|tensor_bytes."""
  tensor_cpu = frame.tensor.cpu().contiguous()
  c, h, w = tensor_cpu.shape
  ts_ms = int(frame.timestamp * 1000)
  header = (PACKET_MAGIC + frame.frame_id.to_bytes(4, "big") + frame.pts.to_bytes(8, "big") + ts_ms.to_bytes(8, "big") +
            h.to_bytes(2, "big") + w.to_bytes(2, "big") + c.to_bytes(1, "big"))
  return header + tensor_cpu.numpy().tobytes()


class StreamToWebsocketAdapter:
  """Connect SharedStreamBuffer producer output to ReplayerServer broadcast."""

  def __init__(self, buffer: SharedStreamBuffer, server: ReplayerServer, fps: int = 60):
    self.buffer = buffer
    self.server = server
    self.fps = fps
    self._task: Optional[asyncio.Task] = None

  async def _loop(self):
    consumer = self.buffer.create_consumer("websocket_broadcast", start_from_latest=True)
    frame_interval = 1.0 / self.fps
    last_sent = 0.0
    while True:
      frame: FrameData = consumer.peek()
      if frame is None:
        # No new frame yet
        await asyncio.sleep(0.001)
        continue
      # Rate-limit to fps
      now = time.time()
      if now - last_sent < frame_interval:
        await asyncio.sleep(frame_interval - (now - last_sent))
        continue
      consumer.skip(1)
      packet = encode_packet(frame)
      await self.server.broadcast(packet)
      last_sent = time.time()

  async def start(self):
    if self._task is not None:
      return
    await self.server.start()
    self._task = asyncio.create_task(self._loop())

  async def stop(self):
    if self._task:
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass
      self._task = None
    await self.server.stop()
