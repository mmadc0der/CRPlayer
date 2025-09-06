from __future__ import annotations

import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransportConfig:
    adb_path: str
    device_id: Optional[str]
    host: str
    port: int
    max_fps: Optional[int]
    max_size: Optional[int]
    codec: str
    bitrate: Optional[int]
    debug: bool


class ScrcpyTransport:
    """
    Minimal ADB-based transport for scrcpy socket connection.

    This implementation sets up an ADB reverse tunnel to local TCP and then
    connects a standard TCP socket. Actual scrcpy server startup is kept out
    of scope for simplicity and testability; it can be added in future versions.
    """

    def __init__(
        self,
        adb_path: str = "adb",
        device_id: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 27183,
        max_fps: Optional[int] = None,
        max_size: Optional[int] = None,
        codec: str = "h264",
        bitrate: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        self._config = TransportConfig(
            adb_path=adb_path,
            device_id=device_id,
            host=host,
            port=port,
            max_fps=max_fps,
            max_size=max_size,
            codec=codec,
            bitrate=bitrate,
            debug=debug,
        )
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _adb_base(self) -> list[str]:
        base = [self._config.adb_path]
        if self._config.device_id:
            base += ["-s", self._config.device_id]
        return base

    def _adb_reverse(self) -> None:
        # Map local tcp:port to device abstract scrcpy when server is active.
        cmd = self._adb_base() + ["reverse", f"tcp:{self._config.port}", f"localabstract:scrcpy"]
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def start(self) -> None:
        # Setup reverse and connect socket
        self._adb_reverse()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.settimeout(5.0)
        self._sock.connect((self._config.host, self._config.port))
        self._sock.setblocking(False)

    def get_socket(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError("Transport not started")
        return self._sock

    def stop(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
        # Best-effort unmap reverse
        cmd = self._adb_base() + ["reverse", "--remove", f"tcp:{self._config.port}"]
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

