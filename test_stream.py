"""
Test script for GPU-accelerated Android streaming.
Verifies RTX 3060 performance and 60fps capability.
"""

import time
import torch
import psutil
import threading
from android_stream_gpu import GPUAndroidStreamer


class StreamingTester:
    """Test harness for Android streaming performance."""
    
    def __init__(self):
        self.frame_count = 0
        self.total_latency = 0
        self.max_latency = 0
        self.min_latency = float('inf')
        self.gpu_memory_peak = 0
        self.start_time = None
        self.test_duration = 30  # seconds
        
    def frame_callback(self, tensor: torch.Tensor, pts: int, timestamp: float):
        """Callback to measure performance metrics."""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            return
        
        # Calculate latency (rough estimate)
        latency = current_time - timestamp
        self.total_latency += latency
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)
        
        # Track GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            self.gpu_memory_peak = max(self.gpu_memory_peak, gpu_memory)
        
        self.frame_count += 1
        
        # Print stats every 60 frames
        if self.frame_count % 60 == 0:
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed
            avg_latency = self.total_latency / self.frame_count * 1000  # ms
            
            print(f"Frame {self.frame_count}: {fps:.1f} FPS, "
                  f"Latency: {avg_latency:.1f}ms avg, {latency*1000:.1f}ms current, "
                  f"GPU: {gpu_memory:.1f}MB, Shape: {tensor.shape}")
    
    def run_performance_test(self):
        """Run comprehensive performance test."""
        print("=== Android Streaming Performance Test ===")
        print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU only'}")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        print(f"Test duration: {self.test_duration}s")
        print()
        
        # Create streamer with optimal settings for RTX 3060
        streamer = GPUAndroidStreamer(
            max_fps=60,
            max_size=800,
            video_codec="h264",
            bit_rate="80M",  # Higher bitrate for RTX 3060
            use_gpu=True,
            buffer_size=10   # Smaller buffer for lower latency
        )
        
        try:
            # Start streaming
            print("Starting stream...")
            streamer.start_streaming(frame_callback=self.frame_callback)
            
            # Wait for initial connection
            time.sleep(3)
            
            # Monitor system resources
            monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            monitor_thread.start()
            
            # Run test
            test_start = time.time()
            last_frame_count = 0
            stats_interval = 2.0  # Print stats every 2 seconds
            last_stats_time = test_start
            
            while time.time() - test_start < self.test_duration:
                # Consume frames from buffer to prevent stalling
                frame_data = streamer.get_latest_frame()
                if frame_data:
                    tensor, pts, timestamp = frame_data
                    # Process frame for RL (just consume it for now)
                    pass
                
                time.sleep(0.001)
                
                # Print stats less frequently
                current_time = time.time()
                if current_time - last_stats_time >= stats_interval:
                    stats = streamer.get_fps_stats()
                    if stats['frame_count'] != last_frame_count:
                        print(f"Stream stats: {stats}")
                        last_frame_count = stats['frame_count']
                    else:
                        print(f"[WARNING] Frame count stuck at {stats['frame_count']} - streaming may have stalled")
                    last_stats_time = current_time
            
            # Final results
            self._print_results(streamer)
            
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"Test error: {e}")
        finally:
            streamer.stop_streaming()
    
    def _monitor_system(self):
        """Monitor system resources during test."""
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 80:
                print(f"WARNING: High CPU usage: {cpu_percent}%")
            
            if memory.percent > 80:
                print(f"WARNING: High memory usage: {memory.percent}%")
            
            time.sleep(5)
    
    def _print_results(self, streamer):
        """Print final test results."""
        if self.frame_count == 0:
            print("No frames received!")
            return
        
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed
        avg_latency = self.total_latency / self.frame_count * 1000
        
        print("\n=== PERFORMANCE RESULTS ===")
        print(f"Total frames: {self.frame_count}")
        print(f"Test duration: {elapsed:.1f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Target FPS: 60")
        print(f"Performance: {(avg_fps/60)*100:.1f}% of target")
        print()
        print(f"Latency - Avg: {avg_latency:.1f}ms, Min: {self.min_latency*1000:.1f}ms, Max: {self.max_latency*1000:.1f}ms")
        print(f"GPU Memory Peak: {self.gpu_memory_peak:.1f}MB")
        
        # Performance assessment
        if avg_fps >= 55:
            print("✅ EXCELLENT: Suitable for real-time RL training")
        elif avg_fps >= 45:
            print("✅ GOOD: Suitable for most RL applications")
        elif avg_fps >= 30:
            print("⚠️  MODERATE: May impact RL training speed")
        else:
            print("❌ POOR: Not suitable for real-time RL")
        
        if avg_latency < 50:
            print("✅ LOW LATENCY: Excellent for RL")
        elif avg_latency < 100:
            print("⚠️  MODERATE LATENCY: Acceptable for RL")
        else:
            print("❌ HIGH LATENCY: May impact RL performance")


def test_basic_functionality():
    """Test basic streaming functionality."""
    print("=== Basic Functionality Test ===")
    
    streamer = GPUAndroidStreamer(max_fps=30, max_size=800)
    
    frame_received = False
    
    def simple_callback(tensor, pts, timestamp):
        nonlocal frame_received
        frame_received = True
        print(f"✅ Frame received: {tensor.shape}, device: {tensor.device}")
    
    try:
        streamer.start_streaming(frame_callback=simple_callback)
        
        # Wait for first frame
        timeout = 10
        start = time.time()
        while not frame_received and time.time() - start < timeout:
            time.sleep(0.1)
        
        if frame_received:
            print("✅ Basic streaming works!")
        else:
            print("❌ No frames received within timeout")
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
    finally:
        streamer.stop_streaming()


def test_gpu_acceleration():
    """Test GPU acceleration specifically."""
    print("=== GPU Acceleration Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Test tensor operations on GPU
    test_tensor = torch.randn(3, 1920, 1080).cuda()
    resized = torch.nn.functional.interpolate(
        test_tensor.unsqueeze(0),
        size=(224, 224),
        mode='bilinear'
    ).squeeze(0)
    
    print(f"✅ GPU tensor operations work: {resized.shape}")


if __name__ == "__main__":
    print("Android Streaming Test Suite")
    print("=" * 40)
    
    # Run tests
    test_gpu_acceleration()
    print()
    
    test_basic_functionality()
    print()
    
    # Ask user for performance test
    response = input("Run full performance test? (y/n): ").lower().strip()
    if response == 'y':
        tester = StreamingTester()
        tester.run_performance_test()
    else:
        print("Skipping performance test")
