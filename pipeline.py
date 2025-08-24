import threading
import time
from typing import List, Optional, Callable
from inference.fifo_reader import NonBlockingFifoReader


class DataChunk:
    """Zero-copy read-only view of data owned by the pipeline."""
    
    def __init__(self, data: bytes, timestamp: float):
        self._data = memoryview(data)
        self.timestamp = timestamp
        self.size = len(data)
    
    @property
    def data(self) -> memoryview:
        """Read-only view of the data."""
        return self._data
    
    def hex_preview(self, length: int = 16) -> str:
        """Get hex preview of first N bytes."""
        preview_len = min(length, self.size)
        return self._data[:preview_len].tobytes().hex()


class StreamPipeline:
    """Main pipeline process that owns all read data and distributes to subscribers."""
    
    def __init__(self, fifo_path: str):
        self.fifo_path = fifo_path
        self.reader = NonBlockingFifoReader(fifo_path)
        self.subscribers: List[Callable[[DataChunk], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_bytes = 0
        self.total_chunks = 0
        self.start_time = 0.0
        
    def add_subscriber(self, callback: Callable[[DataChunk], None]) -> None:
        """Add a subscriber that will receive data chunks."""
        self.subscribers.append(callback)
        
    def start(self) -> None:
        """Start the pipeline."""
        if self._running:
            return
            
        self._running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"Pipeline started, reading from: {self.fifo_path}")
        
    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.reader.close()
        print("Pipeline stopped")
        
    def _run(self) -> None:
        """Main pipeline loop."""
        try:
            self.reader.open()
            
            while self._running:
                # Read available data
                data = self.reader.read_available()
                
                if data:
                    # Create zero-copy chunk
                    chunk = DataChunk(data, time.time())
                    
                    # Update statistics
                    self.total_bytes += chunk.size
                    self.total_chunks += 1
                    
                    # Distribute to subscribers
                    for subscriber in self.subscribers:
                        try:
                            subscriber(chunk)
                        except Exception as e:
                            print(f"Subscriber error: {e}")
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Pipeline error: {e}")
        finally:
            self.reader.close()
            
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        avg_rate = self.total_bytes / elapsed if elapsed > 0 else 0
        
        return {
            'total_bytes': self.total_bytes,
            'total_chunks': self.total_chunks,
            'elapsed_time': elapsed,
            'avg_rate_bps': avg_rate,
            'running': self._running
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <fifo_path>")
        sys.exit(1)
    
    fifo_path = sys.argv[1]
    
    # Create pipeline
    pipeline = StreamPipeline(fifo_path)
    
    # Add debug subscriber
    def debug_subscriber(chunk: DataChunk) -> None:
        if pipeline.total_chunks <= 3:
            print(f"[CHUNK {pipeline.total_chunks}] {chunk.size} bytes at {chunk.timestamp:.3f}s")
            print(f"  First 16 bytes: {chunk.hex_preview()}")
    
    # Add stats subscriber (every 5 seconds)
    last_stats_time = 0
    def stats_subscriber(chunk: DataChunk) -> None:
        global last_stats_time
        if chunk.timestamp - last_stats_time >= 5.0:
            stats = pipeline.get_stats()
            print(f"[STATS] {stats['total_bytes']} bytes, {stats['total_chunks']} chunks, "
                  f"{stats['avg_rate_bps']:.1f} B/s avg")
            last_stats_time = chunk.timestamp
    
    pipeline.add_subscriber(debug_subscriber)
    pipeline.add_subscriber(stats_subscriber)
    
    try:
        pipeline.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pipeline.stop()
