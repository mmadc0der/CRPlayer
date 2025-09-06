from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

try:  # Optional, convert to torch for RL pipelines
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

import av  # PyAV


class H264Decoder:
    def __init__(self, use_gpu: bool = False, logger: Optional[logging.Logger] = None) -> None:
        self._use_gpu = use_gpu and _TORCH_AVAILABLE
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._codec = av.CodecContext.create("h264", "r")
        # Try to enable hardware acceleration if requested; ignore on failure
        if self._use_gpu:
            try:
                # NVDEC via ffmpeg if compiled; PyAV exposes options on codec context
                self._codec.options = self._codec.options or {}
                self._codec.options.update({"hwaccel": "cuda"})
            except Exception:
                self._logger.info("GPU decode not available; falling back to CPU")
                self._use_gpu = False

    def decode_nal(self, nal_unit: bytes) -> List[Tuple[object, int, float]]:
        """
        Decode a single Annex B NAL unit. Returns list of tuples: (frame, width, height).
        Frame is either numpy.ndarray (H, W, 3) RGB or torch.Tensor CHW float32 in [0,1].
        """
        if not nal_unit:
            return []

        packet = av.Packet(nal_unit)
        frames = self._codec.decode(packet)
        outputs: List[Tuple[object, int, float]] = []
        for frame in frames:
            # Convert to RGB numpy array
            rgb = frame.to_ndarray(format="rgb24")  # HxWx3 uint8
            height, width = rgb.shape[:2]
            ts = float(frame.time) if getattr(frame, "time", None) is not None else time.monotonic()
            if _TORCH_AVAILABLE:
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(dtype=torch.float32) / 255.0
                if self._use_gpu:
                    tensor = tensor.cuda(non_blocking=True)
                outputs.append((tensor, width, ts))
            else:
                outputs.append((rgb, width, ts))
        return outputs

