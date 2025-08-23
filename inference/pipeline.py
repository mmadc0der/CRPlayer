import os
import subprocess
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


class ScrcpyRecorder:
    """Start scrcpy recording to an MP4 file. Optionally limit duration.

    Provides a `start()`/`stop()` lifecycle. This is the single capture backend.
    """

    def __init__(self, cfg: CaptureConfig, out_path: str, duration_sec: Optional[float] = None) -> None:
        self.cfg = cfg
        self.out_path = out_path
        self.duration_sec = duration_sec
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        scrcpy = _which("scrcpy")
        if not scrcpy:
            raise RuntimeError("scrcpy not found in PATH; install it and ensure it's accessible")
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        sc_cmd = [
            scrcpy,
            "--no-window",
            "--no-audio",
            "--video-bit-rate", str(self.cfg.bitrate),
            "--record", self.out_path,
        ]
        if self.cfg.device_id:
            sc_cmd += ["-s", self.cfg.device_id]
        # Note: scrcpy will run until terminated. If duration is provided, we'll stop later.
        self.proc = subprocess.Popen(sc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


class FFmpegFileFollower:
    """Decode frames from a growing MP4 file produced by scrcpy using ffmpeg.

    Warning: This relies on ffmpeg being able to read from a file while it's being written.
    In practice this often works, but if it doesn't on your setup, consider post-hoc decoding.
    """

    def __init__(self, crop: Optional[Tuple[int, int, int, int]] = None, expected_size: Optional[Tuple[int, int]] = None) -> None:
        self.crop = crop
        self.proc: Optional[subprocess.Popen] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        if expected_size:
            self.width, self.height = expected_size

    def start(self, mp4_path: str) -> None:
        ffmpeg = _which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH (needed for decode)")
        filters = []
        if self.crop:
            x, y, w, h = self.crop
            filters.append(f"crop={w}:{h}:{x}:{y}")
            self.width, self.height = w, h
        vf = []
        if filters:
            vf = ["-vf", ",".join(filters)]
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-i", mp4_path,
            *vf,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def frames(self) -> Generator[bytes, None, None]:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("decoder not started")
        if self.width is None or self.height is None:
            # Without known size, we cannot segment frames; require crop or explicit size elsewhere
            raise RuntimeError("expected_size not known; provide crop or set expected_size when constructing")
        frame_bytes = self.width * self.height * 3
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


def capture_frames_with_scrcpy(cfg: CaptureConfig, out_mp4: str, crop: Optional[Tuple[int, int, int, int]] = None,
                               expected_size: Optional[Tuple[int, int]] = None) -> Generator[bytes, None, None]:
    """Start scrcpy recording to MP4 and simultaneously decode frames from the growing file.

    Yields raw rgb24 frames (cropped if specified). Intended for feeding CNNs.
    """
    recorder = ScrcpyRecorder(cfg, out_mp4)
    follower = FFmpegFileFollower(crop=crop, expected_size=expected_size)
    recorder.start()
    try:
        # Give scrcpy a brief moment to create the MP4 file on disk
        t0 = _now_ms()
        while not os.path.isfile(out_mp4) and (_now_ms() - t0) < 3000:
            time.sleep(0.05)
        follower.start(out_mp4)
        try:
            for frame in follower.frames():
                yield frame
        finally:
            follower.stop()
    finally:
        recorder.stop()


def record_mp4_with_scrcpy(cfg: CaptureConfig, out_mp4: str, duration_sec: Optional[float] = None) -> None:
    """Record MP4 using scrcpy only. Optionally stop after duration.

    Stores the MP4 replay for later labeling/training.
    """
    scrcpy = _which("scrcpy")
    if not scrcpy:
        raise RuntimeError("scrcpy not found in PATH; install it and ensure it's accessible")
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    sc_cmd = [
        scrcpy,
        "--no-window",
        "--no-audio",
        "--video-bit-rate", str(cfg.bitrate),
        "--record", out_mp4,
    ]
    if cfg.device_id:
        sc_cmd += ["-s", cfg.device_id]

    proc = subprocess.Popen(sc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if duration_sec is None:
        # Run until the user kills the process externally
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
    else:
        try:
            proc.wait(timeout=max(1.0, duration_sec))
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
