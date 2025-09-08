from __future__ import annotations

import logging
import select
import socket
import time
from typing import Callable, Optional

from .buffer import FrameTuple
from .buffer import FrameBuffer


START3 = b"\x00\x00\x01"
START4 = b"\x00\x00\x00\x01"


def _find_next_start_code(data: bytes, start_index: int) -> int:
    i = start_index
    max_i = max(0, len(data) - 3)
    while i <= max_i:
        if data[i : i + 3] == START3:
            return i
        if i + 4 <= len(data) and data[i : i + 4] == START4:
            return i
        i += 1
    return -1


class BitstreamPipeline:
    def __init__(self, decoder, frame_buffer: FrameBuffer, logger: Optional[logging.Logger] = None) -> None:
        self._decoder = decoder
        self._buffer = frame_buffer
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def run(
        self,
        socket_obj: socket.socket,
        running_event,
        frame_callback: Optional[Callable[[FrameTuple], None]] = None,
    ) -> None:
        sock = socket_obj
        sock.setblocking(False)
        buf = bytearray()
        last_log = time.monotonic()

        while running_event.is_set():
            rlist, _, _ = select.select([sock], [], [], 0.25)
            if not rlist:
                # Periodic heartbeat log in debug
                if time.monotonic() - last_log > 5.0 and self._logger.isEnabledFor(logging.DEBUG):
                    self._logger.debug("pipeline idle, buffer=%d bytes", len(buf))
                    last_log = time.monotonic()
                continue

            try:
                chunk = sock.recv(65536)
            except BlockingIOError:
                continue
            if not chunk:
                self._logger.info("socket closed by peer")
                break
            buf.extend(chunk)

            # Consume complete NAL units based on Annex B start codes
            cursor = 0
            while True:
                start = _find_next_start_code(buf, cursor)
                if start < 0:
                    # keep remaining bytes for next chunk
                    break
                # determine start code length
                sc_len = 3 if buf[start : start + 3] == START3 else 4
                next_start = _find_next_start_code(buf, start + sc_len)
                if next_start < 0:
                    # incomplete NAL at end; wait for more data
                    # compact buffer by slicing from start to end
                    if start > 0:
                        del buf[:start]
                    break
                # we have a complete NAL from start to next_start
                nal = bytes(buf[start:next_start])
                cursor = next_start

                # Decode and push frames
                try:
                    frames = self._decoder.decode_nal(nal)
                except Exception as exc:  # pragma: no cover - decoding safety net
                    self._logger.exception("decoder error: %s", exc)
                    continue

                for f in frames:
                    self._buffer.push(f)
                    if frame_callback is not None:
                        try:
                            frame_callback(f)
                        except Exception:  # pragma: no cover - user callback
                            self._logger.exception("frame_callback raised an exception")

