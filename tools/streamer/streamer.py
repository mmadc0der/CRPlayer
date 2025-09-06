from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

from .buffer import FrameBuffer
from .decoder import H264Decoder
from .pipeline import BitstreamPipeline
from .transport import ScrcpyTransport


FrameTuple = Tuple["torch.Tensor", int, float] if _TORCH_AVAILABLE else Tuple[np.ndarray, int, float]


@dataclass
class StreamerConfig:
    device_id: Optional[str] = None
    socket_host: str = "127.0.0.1"
    socket_port: int = 27183
    adb_path: str = "adb"
    max_fps: Optional[int] = None
    max_size: Optional[int] = None
    codec: str = "h264"
    bitrate: Optional[int] = None
    use_gpu: bool = False
    buffer_size: int = 8
    drop_policy: str = "drop_oldest"  # or "drop_newest"
    log_level: int = logging.INFO
    debug: bool = False


class AndroidStreamer:
    def __init__(self, config: StreamerConfig) -> None:
        self._config = config
        logging.basicConfig(level=config.log_level, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        self._logger = logging.getLogger(self.__class__.__name__)

        self._transport = ScrcpyTransport(
            adb_path=config.adb_path,
            device_id=config.device_id,
            host=config.socket_host,
            port=config.socket_port,
            max_fps=config.max_fps,
            max_size=config.max_size,
            codec=config.codec,
            bitrate=config.bitrate,
            debug=config.debug,
        )

        self._buffer = FrameBuffer(maxlen=config.buffer_size, drop_policy=config.drop_policy)
        self._decoder = H264Decoder(use_gpu=config.use_gpu)
        self._pipeline = BitstreamPipeline(
            decoder=self._decoder,
            frame_buffer=self._buffer,
            logger=logging.getLogger("BitstreamPipeline"),
        )

        self._run_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def start(self, frame_callback: Optional[Callable[[FrameTuple], None]] = None) -> None:
        if self._running.is_set():
            self._logger.info("AndroidStreamer already running")
            return

        self._logger.info("Starting transport ...")
        self._transport.start()
        sock = self._transport.get_socket()

        self._logger.info("Starting pipeline thread ...")
        self._running.set()

        def _runner() -> None:
            try:
                self._pipeline.run(socket_obj=sock, running_event=self._running, frame_callback=frame_callback)
            except Exception as exc:  # pragma: no cover - safety net
                self._logger.exception("Pipeline crashed: %s", exc)
                self._running.clear()

        self._run_thread = threading.Thread(target=_runner, name="AndroidStreamerPipeline", daemon=True)
        self._run_thread.start()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._logger.info("Stopping pipeline and transport ...")
        self._running.clear()
        if self._run_thread is not None:
            self._run_thread.join(timeout=5.0)
            self._run_thread = None
        self._transport.stop()

    def get_latest_frame(self) -> Optional[FrameTuple]:
        return self._buffer.get_latest()

    def get_stats(self) -> Dict[str, float]:
        stats = self._buffer.get_stats()
        # Attach decoder stats if needed later
        return stats

