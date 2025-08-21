import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Generator, Optional, Tuple

# Minimal utils kept in this file to limit file count

def _which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        fp = os.path.join(p, cmd)
        if os.path.isfile(fp) and os.access(fp, os.X_OK):
            return fp
    return None


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class CaptureConfig:
    size: str = "720x1600"  # WxH
    bitrate: int = 6_000_000  # bps
    device_id: Optional[str] = None
    crop: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h relative to size
    show: bool = False


class AdbScreenCapture:
    def __init__(self, cfg: CaptureConfig) -> None:
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        adb = _which("adb")
        if not adb:
            raise RuntimeError("adb not found in PATH")
        base_cmd = [adb]
        if self.cfg.device_id:
            base_cmd += ["-s", self.cfg.device_id]
        # Use screenrecord to output H264 elementary stream to stdout
        sr = [
            "exec-out",
            "screenrecord",
            f"--bit-rate",
            str(self.cfg.bitrate),
            "--output-format=h264",
            "--size",
            self.cfg.size,
            "-",
        ]
        cmd = base_cmd + sr
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass

    def raw_stream(self):
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("capture not started")
        return self.proc.stdout


class FFmpegDecoder:
    def __init__(self, size: str, crop: Optional[Tuple[int, int, int, int]] = None) -> None:
        self.width, self.height = [int(x) for x in size.split("x", 1)]
        self.crop = crop
        self.proc: Optional[subprocess.Popen] = None

    def start(self, input_pipe) -> None:
        ffmpeg = _which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH (needed for decode)")
        filters = []
        # Ensure exact incoming size to avoid scaler surprises
        # Apply crop if requested
        if self.crop:
            x, y, w, h = self.crop
            filters.append(f"crop={w}:{h}:{x}:{y}")
            # Update expected output size to cropped size
            self.width, self.height = w, h
        # Convert to rgb24 for general processing
        vf = []
        if filters:
            vf = ["-vf", ",".join(filters)]
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "h264",
            "-i",
            "-",
            *vf,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=input_pipe,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def frames(self) -> Generator[bytes, None, None]:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("decoder not started")
        frame_bytes = self.width * self.height * 3  # rgb24
        read = self.proc.stdout.read
        while True:
            chunk = read(frame_bytes)
            if not chunk or len(chunk) < frame_bytes:
                break
            yield chunk

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass


def run_loop(cfg: CaptureConfig) -> None:
    cap = AdbScreenCapture(cfg)
    cap.start()
    dec = FFmpegDecoder(cfg.size, cfg.crop)
    dec.start(cap.raw_stream())

    # Lazy import cv2 only if show is requested
    if cfg.show:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

    frames = 0
    t0 = time.time()
    last = t0
    try:
        for raw in dec.frames():
            frames += 1
            if cfg.show:
                import numpy as np  # type: ignore
                import cv2  # type: ignore
                arr = np.frombuffer(raw, dtype=np.uint8)
                arr = arr.reshape((dec.height, dec.width, 3))
                cv2.imshow("CRPlayer", arr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if frames % 60 == 0:
                now = time.time()
                fps = 60.0 / (now - last)
                last = now
                print(f"FPS ~ {fps:.1f}")
    finally:
        dec.stop()
        cap.stop()
        if cfg.show:
            try:
                import cv2  # type: ignore
                cv2.destroyAllWindows()
            except Exception:
                pass


def record_raw_h264(cfg: CaptureConfig, out_path: str, duration_sec: Optional[float] = None) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cap = AdbScreenCapture(cfg)
    cap.start()
    fout = open(out_path, "wb")
    t_start = _now_ms()
    total = 0
    try:
        stream = cap.raw_stream()
        assert stream is not None
        while True:
            data = stream.read(4096)
            if not data:
                break
            fout.write(data)
            total += len(data)
            if total % (1024 * 1024 * 8) == 0:
                elapsed = _now_ms() - t_start
                print(f"Wrote {total/1e6:.1f} MB in {elapsed/1000:.1f}s")
            if duration_sec is not None:
                if (_now_ms() - t_start) >= int(duration_sec * 1000):
                    break
    finally:
        fout.close()
        cap.stop()
