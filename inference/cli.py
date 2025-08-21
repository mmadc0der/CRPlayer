import os
import typer
from typing import Optional
from .pipeline import CaptureConfig, run_loop, record_raw_h264

app = typer.Typer(add_completion=False, help="CRPlayer Inference CLI")


@app.command()
def run(
    device: Optional[str] = typer.Option(None, help="ADB device id (-s)"),
    size: str = typer.Option("720x1600", help="Capture size WxH"),
    bitrate: int = typer.Option(6_000_000, help="Bitrate in bps"),
    crop: Optional[str] = typer.Option(None, help="Crop as x,y,w,h relative to capture size"),
    show: bool = typer.Option(False, help="Display window (OpenCV)"),
):
    """Run capture->decode loop and print FPS."""
    crop_tuple = None
    if crop:
        try:
            x, y, w, h = [int(v) for v in crop.split(",")]
            crop_tuple = (x, y, w, h)
        except Exception:
            raise typer.BadParameter("crop must be x,y,w,h")
    cfg = CaptureConfig(size=size, bitrate=bitrate, device_id=device, crop=crop_tuple, show=show)
    run_loop(cfg)


@app.command()
def record(
    out: str = typer.Option("recordings/stream.h264", help="Output raw h264 path"),
    device: Optional[str] = typer.Option(None, help="ADB device id (-s)"),
    size: str = typer.Option("720x1600", help="Capture size WxH"),
    bitrate: int = typer.Option(6_000_000, help="Bitrate in bps"),
):
    """Record raw H264 stream to file."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cfg = CaptureConfig(size=size, bitrate=bitrate, device_id=device)
    record_raw_h264(cfg, out)


if __name__ == "__main__":
    app()
