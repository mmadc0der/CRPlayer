from __future__ import annotations

import collections
import threading
import time
from typing import Deque, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


FrameTuple = Tuple["torch.Tensor", int, float] if _TORCH_AVAILABLE else Tuple[np.ndarray, int, float]


class FrameBuffer:
    def __init__(self, maxlen: int = 8, drop_policy: str = "drop_oldest") -> None:
        self._frames: Deque[FrameTuple] = collections.deque(maxlen=maxlen)
        self._drop_policy = drop_policy
        self._lock = threading.Lock()
        self._frame_count = 0
        self._start_ts = time.monotonic()

    def push(self, frame: FrameTuple) -> None:
        with self._lock:
            if len(self._frames) == self._frames.maxlen:
                if self._drop_policy == "drop_oldest":
                    self._frames.popleft()
                elif self._drop_policy == "drop_newest":
                    return
            self._frames.append(frame)
            self._frame_count += 1

    def get_latest(self) -> Optional[FrameTuple]:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            elapsed = max(1e-6, time.monotonic() - self._start_ts)
            fps = self._frame_count / elapsed
            return {
                "fps": fps,
                "queue_size": float(len(self._frames)),
                "elapsed_s": elapsed,
            }

