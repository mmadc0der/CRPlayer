import os
import time
import typer
from typing import Optional
from .pipeline import CaptureConfig, record_mp4_with_scrcpy, capture_frames_with_scrcpy

app = typer.Typer(add_completion=False, help="CRPlayer Inference CLI")


@app.command()
def capture(
    out: str = typer.Option("recordings/session.mp4", help="Path to MP4 file to record via scrcpy"),
    device: Optional[str] = typer.Option(None, help="ADB device id (-s)"),
    bitrate: int = typer.Option(6_000_000, help="Bitrate in bps"),
    expected_width: Optional[int] = typer.Option(None, help="Expected frame width after crop (required if crop not provided)"),
    expected_height: Optional[int] = typer.Option(None, help="Expected frame height after crop (required if crop not provided)"),
    crop: Optional[str] = typer.Option(None, help="Crop as x,y,w,h in source coordinates (optional)"),
):
    """Start scrcpy recording and stream frames from the growing MP4, printing FPS.

    This is a research utility to validate the frame pipeline for CNNs. It does not display frames.
    """
    crop_tuple = None
    if crop:
        try:
            x, y, w, h = [int(v) for v in crop.split(",")]
            crop_tuple = (x, y, w, h)
        except Exception:
            raise typer.BadParameter("crop must be x,y,w,h")
    expected_size = None
    if expected_width and expected_height:
        expected_size = (expected_width, expected_height)
    elif not crop_tuple:
        raise typer.BadParameter("Provide either crop or expected_width and expected_height so frames can be segmented")

    cfg = CaptureConfig(bitrate=bitrate, device_id=device)
    frames = 0
    t_last = time.time()
    for _ in capture_frames_with_scrcpy(cfg, out_mp4=out, crop=crop_tuple, expected_size=expected_size):
        frames += 1
        if frames % 60 == 0:
            now = time.time()
            fps = 60.0 / (now - t_last)
            t_last = now
            typer.echo(f"FPS ~ {fps:.1f}")


@app.command()
def record(
    out: str = typer.Option("recordings/session.mp4", help="Output MP4 path (scrcpy)"),
    device: Optional[str] = typer.Option(None, help="ADB device id (-s)"),
    bitrate: int = typer.Option(6_000_000, help="Bitrate in bps"),
    duration: Optional[float] = typer.Option(None, help="Duration in seconds to record (default: unlimited)"),
):
    """Record MP4 (scrcpy-only) for replay and offline training."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cfg = CaptureConfig(bitrate=bitrate, device_id=device)
    record_mp4_with_scrcpy(cfg, out, duration_sec=duration)


if __name__ == "__main__":
    app()
