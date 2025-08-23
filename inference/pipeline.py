import os
import subprocess
import time
import threading
import queue
import socket
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, Callable, List
from .monitoring import PerformanceMonitor, ConsoleReporter

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


class StreamHub:
    """Fan-out hub that reads H.264 bytes from an input stream and distributes them.

    - Uses a dedicated reader thread.
    - Sends memoryview chunks to each subscriber's bounded queue.
    - Backpressure policy: 'drop' (default) drops chunks for slow subscribers; 'block' waits.
    """

    def __init__(self, input_stream, chunk_size: int = 16 * 1024, monitor: Optional[PerformanceMonitor] = None) -> None:
        self.input_stream = input_stream
        self.chunk_size = chunk_size
        self.subs: List[Tuple[queue.Queue, str]] = []
        self.policy = 'drop'
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.monitor = monitor
        self._bytes_read = 0

    def add_subscriber(self, max_queue: int = 256) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.subs.append((q, 'drop'))
        return q

    def add_blocking_subscriber(self, max_queue: int = 256) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.subs.append((q, 'block'))
        return q

    def start(self) -> None:
        def _run():
            read = self.input_stream.read
            while not self._stop.is_set():
                data = read(self.chunk_size)
                if not data:
                    break
                    
                self._bytes_read += len(data)
                mv = memoryview(data)
                
                # Update monitoring metrics
                if self.monitor:
                    total_queue_size = sum(q.qsize() for q, _ in self.subs)
                    self.monitor.update_stream_metrics(len(data), total_queue_size)
                
                for q, mode in list(self.subs):
                    try:
                        if mode == 'block':
                            q.put(mv, timeout=0.1)
                        else:
                            q.put_nowait(mv)
                    except queue.Full:
                        if self.monitor:
                            self.monitor.report_error("queue_full", f"Dropped chunk for {mode} subscriber")
                        # drop
                        pass
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


class FFmpegStdinDecoder:
    """Decode raw H.264 from stdin into rgb24 frames, optionally cropped.

    Use `consume(q)` to feed data from a hub subscriber queue and iterate `frames()` to pull decoded frames.
    """

    def __init__(self, crop: Optional[Tuple[int, int, int, int]] = None, expected_size: Optional[Tuple[int, int]] = None, monitor: Optional[PerformanceMonitor] = None) -> None:
        self.crop = crop
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        if expected_size:
            self.width, self.height = expected_size
        self.proc: Optional[subprocess.Popen] = None
        self._feeder: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.monitor = monitor

    def start(self) -> None:
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
            "-f", "h264",
            "-i", "-",
            *vf,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def consume(self, q: queue.Queue) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("decoder not started")
        def _feed():
            write = self.proc.stdin.write
            flush = self.proc.stdin.flush
            while not self._stop.is_set():
                try:
                    mv = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    write(mv)
                    flush()
                except BrokenPipeError:
                    break
        self._feeder = threading.Thread(target=_feed, daemon=True)
        self._feeder.start()

    def frames(self) -> Generator[bytes, None, None]:
        if not self.proc or not self.proc.stdout:
            raise RuntimeError("decoder not started")
        if self.width is None or self.height is None:
            raise RuntimeError("expected_size not known; provide crop or expected_size")
        frame_bytes = self.width * self.height * 3
        read = self.proc.stdout.read
        while True:
            frame_start = time.time()
            chunk = read(frame_bytes)
            if not chunk or len(chunk) < frame_bytes:
                break
                
            # Update monitoring metrics
            if self.monitor:
                processing_time = time.time() - frame_start
                self.monitor.update_frame_metrics(len(chunk), processing_time)
                
            yield chunk

    def stop(self) -> None:
        self._stop.set()
        if self._feeder:
            self._feeder.join(timeout=1.0)
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass


class MP4Writer:
    """Subscriber that writes raw H.264 to MP4 via ffmpeg copy muxing."""

    def __init__(self, out_mp4: str) -> None:
        self.out_mp4 = out_mp4
        self.proc: Optional[subprocess.Popen] = None
        self._feeder: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        ffmpeg = _which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH")
        os.makedirs(os.path.dirname(self.out_mp4), exist_ok=True)
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-f", "h264",
            "-i", "-",
            "-an",
            "-c:v", "copy",
            "-movflags", "+faststart",
            self.out_mp4,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def consume(self, q: queue.Queue) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("writer not started")
        def _feed():
            write = self.proc.stdin.write
            flush = self.proc.stdin.flush
            while not self._stop.is_set():
                try:
                    mv = q.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    write(mv)
                    flush()
                except BrokenPipeError:
                    break
        self._feeder = threading.Thread(target=_feed, daemon=True)
        self._feeder.start()

    def stop(self) -> None:
        self._stop.set()
        if self._feeder:
            self._feeder.join(timeout=1.0)
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # finalize mp4
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.kill()
            except Exception:
                pass


def run_hub_from_stream(input_stream,
                        decode_crop: Optional[Tuple[int, int, int, int]] = None,
                        expected_size: Optional[Tuple[int, int]] = None,
                        mp4_out: Optional[str] = None,
                        report_fps: bool = True,
                        monitor: Optional[PerformanceMonitor] = None) -> Generator[bytes, None, None]:
    """Run a hub on an input H.264 stream and attach optional subscribers.

    Yields decoded rgb24 frames if decoder is attached. If no decoder, yields nothing.
    """
    hub = StreamHub(input_stream, monitor=monitor)
    # Subscribers
    dec: Optional[FFmpegStdinDecoder] = None
    dec_q: Optional[queue.Queue] = None
    if decode_crop or expected_size:
        dec = FFmpegStdinDecoder(crop=decode_crop, expected_size=expected_size, monitor=monitor)
        dec.start()
        dec_q = hub.add_blocking_subscriber(max_queue=256)
        dec.consume(dec_q)

    writer: Optional[MP4Writer] = None
    if mp4_out:
        writer = MP4Writer(mp4_out)
        writer.start()
        wq = hub.add_subscriber(max_queue=1024)
        writer.consume(wq)

    hub.start()
    try:
        if dec:
            for frame in dec.frames():
                yield frame
        else:
            # If no decoder, just keep the hub running until input ends
            while True:
                time.sleep(0.1)
    finally:
        hub.stop()
        if dec:
            dec.stop()
        if writer:
            writer.stop()
