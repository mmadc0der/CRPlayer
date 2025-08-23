import time
import threading
import queue
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import deque


@dataclass
class StreamMetrics:
    """Real-time stream performance metrics"""
    fps: float = 0.0
    bitrate_kbps: float = 0.0
    frame_drops: int = 0
    latency_ms: float = 0.0
    queue_depth: int = 0
    bytes_processed: int = 0
    frames_processed: int = 0
    errors: int = 0
    last_frame_time: float = 0.0
    connection_status: str = "unknown"
    device_id: Optional[str] = None


@dataclass
class DeviceInfo:
    """Device connection information"""
    device_id: str
    model: str = "unknown"
    resolution: str = "unknown"
    android_version: str = "unknown"
    connected: bool = False
    last_seen: float = 0.0


class PerformanceMonitor:
    """Advanced monitoring for CRPlayer stream processing"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics = StreamMetrics()
        self.device_info: Optional[DeviceInfo] = None
        
        # Rolling windows for trend analysis
        self.fps_history = deque(maxlen=60)  # 1 minute at 1Hz
        self.bitrate_history = deque(maxlen=60)
        self.latency_history = deque(maxlen=60)
        
        # Threading
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks for real-time updates
        self.callbacks: List[callable] = []
        
    def add_callback(self, callback: callable):
        """Add callback for real-time metric updates"""
        self.callbacks.append(callback)
        
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Update device info
                self._update_device_info()
                
                # Calculate trends
                self._calculate_trends()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(self.get_current_status())
                    except Exception as e:
                        print(f"Monitor callback error: {e}")
                        
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(self.update_interval)
                
    def _update_device_info(self):
        """Check device connection status via ADB"""
        try:
            result = subprocess.run(
                ["adb", "devices", "-l"],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                devices = []
                
                for line in lines:
                    if line.strip() and '\tdevice' in line:
                        parts = line.split('\t')
                        device_id = parts[0]
                        
                        # Try to get device info
                        info_result = subprocess.run(
                            ["adb", "-s", device_id, "shell", "getprop"],
                            capture_output=True,
                            text=True,
                            timeout=3.0
                        )
                        
                        model = "unknown"
                        android_version = "unknown"
                        resolution = "unknown"
                        
                        if info_result.returncode == 0:
                            props = info_result.stdout
                            # Parse key properties
                            for prop_line in props.split('\n'):
                                if '[ro.product.model]' in prop_line:
                                    model = prop_line.split(': [')[1].rstrip(']')
                                elif '[ro.build.version.release]' in prop_line:
                                    android_version = prop_line.split(': [')[1].rstrip(']')
                        
                        # Get screen resolution
                        res_result = subprocess.run(
                            ["adb", "-s", device_id, "shell", "wm", "size"],
                            capture_output=True,
                            text=True,
                            timeout=2.0
                        )
                        
                        if res_result.returncode == 0 and 'Physical size:' in res_result.stdout:
                            resolution = res_result.stdout.split('Physical size: ')[1].strip()
                        
                        self.device_info = DeviceInfo(
                            device_id=device_id,
                            model=model,
                            android_version=android_version,
                            resolution=resolution,
                            connected=True,
                            last_seen=time.time()
                        )
                        
                        with self._lock:
                            self.metrics.connection_status = "connected"
                            self.metrics.device_id = device_id
                        break
                else:
                    # No devices found
                    with self._lock:
                        self.metrics.connection_status = "disconnected"
                        self.metrics.device_id = None
            else:
                with self._lock:
                    self.metrics.connection_status = "adb_error"
                    
        except subprocess.TimeoutExpired:
            with self._lock:
                self.metrics.connection_status = "timeout"
        except FileNotFoundError:
            with self._lock:
                self.metrics.connection_status = "adb_not_found"
        except Exception as e:
            with self._lock:
                self.metrics.connection_status = f"error: {str(e)}"
                
    def _calculate_trends(self):
        """Calculate performance trends"""
        with self._lock:
            self.fps_history.append(self.metrics.fps)
            self.bitrate_history.append(self.metrics.bitrate_kbps)
            self.latency_history.append(self.metrics.latency_ms)
            
    def update_frame_metrics(self, frame_size: int, processing_time: float):
        """Update metrics when a frame is processed"""
        now = time.time()
        
        with self._lock:
            self.metrics.frames_processed += 1
            self.metrics.bytes_processed += frame_size
            
            # Calculate FPS
            if self.metrics.last_frame_time > 0:
                frame_interval = now - self.metrics.last_frame_time
                if frame_interval > 0:
                    self.metrics.fps = 1.0 / frame_interval
                    
            self.metrics.last_frame_time = now
            self.metrics.latency_ms = processing_time * 1000
            
    def update_stream_metrics(self, bytes_received: int, queue_size: int):
        """Update stream-level metrics"""
        now = time.time()
        
        with self._lock:
            # Calculate bitrate (rough estimate)
            if hasattr(self, '_last_bitrate_update'):
                time_diff = now - self._last_bitrate_update
                if time_diff > 0:
                    self.metrics.bitrate_kbps = (bytes_received * 8) / (time_diff * 1000)
            
            self._last_bitrate_update = now
            self.metrics.queue_depth = queue_size
            
    def report_error(self, error_type: str, details: str = ""):
        """Report an error occurrence"""
        with self._lock:
            self.metrics.errors += 1
        print(f"[ERROR] {error_type}: {details}")
        
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        with self._lock:
            status = {
                "timestamp": time.time(),
                "metrics": {
                    "fps": round(self.metrics.fps, 1),
                    "bitrate_kbps": round(self.metrics.bitrate_kbps, 1),
                    "latency_ms": round(self.metrics.latency_ms, 1),
                    "queue_depth": self.metrics.queue_depth,
                    "frames_processed": self.metrics.frames_processed,
                    "bytes_processed": self.metrics.bytes_processed,
                    "errors": self.metrics.errors,
                    "connection_status": self.metrics.connection_status
                },
                "device": None,
                "trends": {
                    "avg_fps": round(sum(self.fps_history) / len(self.fps_history), 1) if self.fps_history else 0,
                    "avg_bitrate": round(sum(self.bitrate_history) / len(self.bitrate_history), 1) if self.bitrate_history else 0,
                    "avg_latency": round(sum(self.latency_history) / len(self.latency_history), 1) if self.latency_history else 0
                }
            }
            
            if self.device_info:
                status["device"] = {
                    "id": self.device_info.device_id,
                    "model": self.device_info.model,
                    "android_version": self.device_info.android_version,
                    "resolution": self.device_info.resolution,
                    "connected": self.device_info.connected
                }
                
        return status
        
    def print_status(self):
        """Print current status to console"""
        status = self.get_current_status()
        
        print("\n" + "="*60)
        print("CRPlayer Stream Monitor")
        print("="*60)
        
        # Connection status
        conn_status = status["metrics"]["connection_status"]
        status_color = "🟢" if conn_status == "connected" else "🔴"
        print(f"{status_color} Connection: {conn_status}")
        
        # Device info
        if status["device"]:
            dev = status["device"]
            print(f"📱 Device: {dev['model']} (Android {dev['android_version']})")
            print(f"📐 Resolution: {dev['resolution']}")
        
        # Performance metrics
        metrics = status["metrics"]
        trends = status["trends"]
        
        print(f"\n📊 Performance:")
        print(f"   FPS: {metrics['fps']} (avg: {trends['avg_fps']})")
        print(f"   Bitrate: {metrics['bitrate_kbps']} kbps (avg: {trends['avg_bitrate']})")
        print(f"   Latency: {metrics['latency_ms']} ms (avg: {trends['avg_latency']})")
        print(f"   Queue Depth: {metrics['queue_depth']}")
        
        print(f"\n📈 Totals:")
        print(f"   Frames: {metrics['frames_processed']:,}")
        print(f"   Data: {metrics['bytes_processed'] / 1024 / 1024:.1f} MB")
        print(f"   Errors: {metrics['errors']}")
        
        print("="*60)


class ConsoleReporter:
    """Real-time console reporting for monitoring data"""
    
    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self.last_update = 0.0
        
    def __call__(self, status: Dict[str, Any]):
        """Callback for monitor updates"""
        now = time.time()
        if now - self.last_update >= self.update_interval:
            self._print_compact_status(status)
            self.last_update = now
            
    def _print_compact_status(self, status: Dict[str, Any]):
        """Print compact status line"""
        metrics = status["metrics"]
        conn = "🟢" if metrics["connection_status"] == "connected" else "🔴"
        
        print(f"\r{conn} FPS: {metrics['fps']:5.1f} | "
              f"Bitrate: {metrics['bitrate_kbps']:6.1f}k | "
              f"Latency: {metrics['latency_ms']:5.1f}ms | "
              f"Queue: {metrics['queue_depth']:3d} | "
              f"Frames: {metrics['frames_processed']:6d}", end="", flush=True)
