import sys
import os
import time
import typer
from typing import Optional
from .pipeline import run_hub_from_stream
from .monitoring import PerformanceMonitor, ConsoleReporter

app = typer.Typer(add_completion=False, help="CRPlayer Inference CLI")


@app.command()
def stdin(
    crop: Optional[str] = typer.Option(None, help="Crop as x,y,w,h in source coordinates (optional)"),
    expected_width: Optional[int] = typer.Option(None, help="Expected frame width after crop (required if crop not provided)"),
    expected_height: Optional[int] = typer.Option(None, help="Expected frame height after crop (required if crop not provided)"),
    mp4_out: Optional[str] = typer.Option(None, help="Optional MP4 output path; writes a replay via subscriber"),
    no_fps: bool = typer.Option(False, help="Disable FPS reporting"),
    monitor: bool = typer.Option(True, help="Enable advanced monitoring and device status"),
    quiet: bool = typer.Option(False, help="Minimal output (overrides monitor)"),
):
    """Consume raw H.264 from stdin, fan-out to subscribers.

    Example (PowerShell): scrcpy --no-window --no-audio --v4l2-buffer 0 --video-bit-rate 6000000 --output-format=h264 --record - | python -m inference.cli stdin --mp4-out recordings/replay.mp4 --expected-width 360 --expected-height 800
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
    # If neither crop nor expected_size provided, we won't decode, only optional MP4 write.

    # Ensure directory for mp4
    if mp4_out:
        os.makedirs(os.path.dirname(mp4_out), exist_ok=True)

    # Setup monitoring
    perf_monitor = None
    if monitor and not quiet:
        perf_monitor = PerformanceMonitor(update_interval=1.0)
        
        # Add console reporter for real-time updates
        console_reporter = ConsoleReporter(update_interval=2.0)
        perf_monitor.add_callback(console_reporter)
        
        # Start monitoring
        perf_monitor.start_monitoring()
        
        # Print initial status
        typer.echo("\n🚀 Starting CRPlayer with advanced monitoring...")
        typer.echo("Press Ctrl+C to stop\n")
        
        # Show detailed status every 10 seconds
        def detailed_status_callback(status):
            if status["metrics"]["frames_processed"] > 0 and status["metrics"]["frames_processed"] % 300 == 0:  # Every ~10 seconds at 30fps
                perf_monitor.print_status()
        
        perf_monitor.add_callback(detailed_status_callback)
    
    try:
        frames = 0
        for _ in run_hub_from_stream(
            sys.stdin.buffer,
            decode_crop=crop_tuple,
            expected_size=expected_size,
            mp4_out=mp4_out,
            report_fps=False,  # Let monitor handle FPS reporting
            monitor=perf_monitor,
        ):
            frames += 1
            
            # Legacy FPS reporting (only if monitoring disabled)
            if not monitor and not no_fps and frames % 60 == 0:
                now = time.time()
                if 't_last' in locals():
                    fps = 60.0 / (now - t_last)
                    typer.echo(f"FPS ~ {fps:.1f}")
                t_last = now
    
    except KeyboardInterrupt:
        if perf_monitor:
            typer.echo("\n\n📊 Final Status:")
            perf_monitor.print_status()
    
    finally:
        if perf_monitor:
            perf_monitor.stop_monitoring()


@app.command()
def monitor_device():
    """Monitor device connection and stream health without processing."""
    monitor = PerformanceMonitor(update_interval=1.0)
    console_reporter = ConsoleReporter(update_interval=1.0)
    monitor.add_callback(console_reporter)
    
    typer.echo("🔍 CRPlayer Device Monitor")
    typer.echo("Monitoring device connection and readiness...")
    typer.echo("Press Ctrl+C to stop\n")
    
    monitor.start_monitoring()
    
    try:
        while True:
            time.sleep(5.0)
            monitor.print_status()
    except KeyboardInterrupt:
        typer.echo("\n👋 Monitoring stopped")
    finally:
        monitor.stop_monitoring()


# Note: separate explicit record/capture commands removed in favor of stdin hub.


if __name__ == "__main__":
    app()
