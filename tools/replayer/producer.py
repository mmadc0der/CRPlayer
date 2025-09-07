from __future__ import annotations

import glob
import os
import threading
import time
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from PIL import Image

from core.stream_pipeline import FrameData, SharedStreamBuffer
from tools.streamer import AndroidStreamer, StreamerConfig


class LiveProducer:
  """Produce frames from an Android device into a SharedStreamBuffer using tools.streamer."""

  def __init__(self, stream_buffer: SharedStreamBuffer, device_id: Optional[str] = None,
               use_gpu: bool = True, max_fps: Optional[int] = 60, max_size: Optional[int] = 1600,
               codec: str = "h264", bitrate: Optional[int] = None) -> None:
    self.stream_buffer = stream_buffer
    self.is_producing = False
    self._lock = threading.Lock()
    self._streamer = AndroidStreamer(StreamerConfig(
      device_id=device_id,
      use_gpu=use_gpu,
      max_fps=max_fps,
      max_size=max_size,
      codec=codec,
      bitrate=bitrate,
      buffer_size=4,
    ))

  def start(self) -> None:
    with self._lock:
      if self.is_producing:
        return
      self.is_producing = True

    def _on_frame(frame_tuple) -> None:
      if not self.is_producing:
        return
      tensor = frame_tuple[0]
      ts = float(frame_tuple[2]) if len(frame_tuple) >= 3 else time.time()
      if not isinstance(tensor, torch.Tensor):
        # Convert numpy HxWx3 uint8 -> CHW float32
        np_img = np.asarray(tensor)
        tensor_t = torch.from_numpy(np_img).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
      else:
        tensor_t = tensor
      frame = FrameData(frame_id=0, tensor=tensor_t, timestamp=ts, pts=0, metadata={})
      self.stream_buffer.add_frame(frame)

    self._streamer.start(frame_callback=_on_frame)

  def stop(self) -> None:
    with self._lock:
      if not self.is_producing:
        return
      self.is_producing = False
    self._streamer.stop()


class ReplayProducer:
  """Replay frames from a directory of images or a .npz/.npy file into SharedStreamBuffer."""

  def __init__(self, stream_buffer: SharedStreamBuffer, source_path: str, fps: int = 60, loop: bool = True) -> None:
    self.stream_buffer = stream_buffer
    self.source_path = source_path
    self.fps = fps
    self.loop = loop
    self.is_producing = False
    self._thread: Optional[threading.Thread] = None

  def _iter_frames(self) -> Iterable[torch.Tensor]:
    path = self.source_path
    if os.path.isdir(path):
      # Load images sorted lexicographically
      patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
      files: list[str] = []
      for p in patterns:
        files.extend(glob.glob(os.path.join(path, p)))
      files.sort()
      for fp in files:
        img = Image.open(fp).convert("RGB")
        np_img = np.asarray(img)
        yield torch.from_numpy(np_img).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
    else:
      # Attempt to load numpy stack
      if path.endswith(".npz"):
        data = np.load(path)
        arr = data[list(data.keys())[0]]  # first array
      else:
        arr = np.load(path)
      # Expect NHWC uint8 or float32
      if arr.ndim != 4:
        raise ValueError("Expected 4D array for replay (N,H,W,C)")
      if arr.shape[-1] in (1, 3):
        # NHWC
        for frame in arr:
          fr = frame
          if fr.dtype != np.float32:
            fr = fr.astype(np.float32) / 255.0
          yield torch.from_numpy(fr).permute(2, 0, 1).contiguous()
      else:
        # Assume NCHW
        for frame in arr:
          fr = frame
          if fr.dtype != np.float32:
            fr = fr.astype(np.float32) / 255.0
          yield torch.from_numpy(fr).contiguous()

  def _run(self) -> None:
    frame_interval = 1.0 / max(1, self.fps)
    while self.is_producing:
      for tensor in self._iter_frames():
        if not self.is_producing:
          break
        ts = time.time()
        frame = FrameData(frame_id=0, tensor=tensor, timestamp=ts, pts=0, metadata={})
        self.stream_buffer.add_frame(frame)
        time.sleep(frame_interval)
      if not self.loop:
        break

  def start(self) -> None:
    if self.is_producing:
      return
    self.is_producing = True
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    if not self.is_producing:
      return
    self.is_producing = False
    if self._thread is not None:
      self._thread.join(timeout=5.0)
      self._thread = None

