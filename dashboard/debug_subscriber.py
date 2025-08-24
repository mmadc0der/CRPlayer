#!/usr/bin/env python3
"""
Debug Subscriber - Logs detailed stream information to file for debugging
"""

import time
import os
from typing import Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import DataChunk

class DebugSubscriber:
    """Debug subscriber that logs detailed stream information to file."""
    
    def __init__(self, log_file: str = "stream_debug.log", max_log_size: int = 10*1024*1024):
        self.log_file = log_file
        self.max_log_size = max_log_size
        self.start_time = 0.0
        
        # Statistics
        self.total_bytes = 0
        self.chunk_count = 0
        self.last_chunk_time = 0.0
        
        # Open log file
        self.log_fd = open(self.log_file, 'w')
        self._log("=== DEBUG SUBSCRIBER STARTED ===")
        
    def __call__(self, chunk: DataChunk) -> None:
        """Process incoming data chunk and log debug info."""
        if self.start_time == 0:
            self.start_time = chunk.timestamp
            
        self.total_bytes += chunk.size
        self.chunk_count += 1
        
        # Calculate timing
        elapsed = chunk.timestamp - self.start_time
        gap = chunk.timestamp - self.last_chunk_time if self.last_chunk_time > 0 else 0
        
        # Convert memoryview to bytes for analysis
        data = chunk.data.tobytes()
        
        # Log chunk info
        self._log(f"CHUNK {self.chunk_count:4d}: {chunk.size:6d} bytes at {elapsed:8.3f}s (gap: {gap:6.3f}s)")
        
        # Log first few bytes for pattern analysis
        if chunk.size > 0:
            hex_preview = data[:32].hex()
            self._log(f"  HEX: {hex_preview}")
            
            # Check for specific patterns
            patterns = self._analyze_patterns(data)
            if patterns:
                self._log(f"  PATTERNS: {', '.join(patterns)}")
        
        # Log timing anomalies
        if gap > 0.5:  # More than 500ms gap
            self._log(f"  WARNING: Large gap detected ({gap:.3f}s)")
        
        # Log every 10th chunk with more details
        if self.chunk_count % 10 == 0:
            rate = self.total_bytes / elapsed if elapsed > 0 else 0
            self._log(f"  STATS: Total {self.total_bytes} bytes, {rate:.1f} B/s avg")
            
        # Check for data corruption indicators
        if chunk.size == 0:
            self._log("  WARNING: Empty chunk received")
        elif chunk.size > 100*1024:  # Very large chunk
            self._log(f"  WARNING: Large chunk ({chunk.size} bytes)")
            
        self.last_chunk_time = chunk.timestamp
        
        # Rotate log if too large
        self._check_log_rotation()
    
    def _analyze_patterns(self, data: bytes) -> list:
        """Analyze data for known patterns."""
        patterns = []
        
        # MKV signatures
        if data.startswith(b'\x1a\x45\xdf\xa3'):
            patterns.append("EBML_Header")
        if data.startswith(b'\x18\x53\x80\x67'):
            patterns.append("MKV_Segment")
        if data.startswith(b'\x1f\x43\xb6\x75'):
            patterns.append("MKV_Cluster")
        if b'\xa3' in data[:10]:
            patterns.append("SimpleBlock")
        if b'\xa1' in data[:10]:
            patterns.append("Block")
            
        # Video format signatures
        if data.startswith(b'\x00\x00\x00\x01'):
            patterns.append("H264_NAL")
        if data.startswith(b'\xff\xd8'):
            patterns.append("JPEG_Start")
        if data.endswith(b'\xff\xd9'):
            patterns.append("JPEG_End")
            
        # Check for repeated patterns (possible corruption)
        if len(set(data[:100])) < 5:  # Very low entropy
            patterns.append("LOW_ENTROPY")
            
        return patterns
    
    def _log(self, message: str) -> None:
        """Write message to log file with timestamp."""
        timestamp = time.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        self.log_fd.write(f"[{timestamp}] {message}\n")
        self.log_fd.flush()  # Ensure immediate write
    
    def _check_log_rotation(self) -> None:
        """Rotate log file if it gets too large."""
        try:
            if os.path.getsize(self.log_file) > self.max_log_size:
                self.log_fd.close()
                
                # Rename old log
                backup_name = f"{self.log_file}.old"
                if os.path.exists(backup_name):
                    os.remove(backup_name)
                os.rename(self.log_file, backup_name)
                
                # Create new log
                self.log_fd = open(self.log_file, 'w')
                self._log("=== LOG ROTATED ===")
        except Exception as e:
            self._log(f"Log rotation error: {e}")
    
    def close(self) -> None:
        """Close the debug subscriber."""
        if hasattr(self, 'log_fd') and self.log_fd:
            elapsed = time.time() - self.start_time
            rate = self.total_bytes / elapsed if elapsed > 0 else 0
            
            self._log("=== FINAL STATISTICS ===")
            self._log(f"Duration: {elapsed:.1f} seconds")
            self._log(f"Total bytes: {self.total_bytes}")
            self._log(f"Total chunks: {self.chunk_count}")
            self._log(f"Average rate: {rate:.1f} B/s")
            self._log(f"Average chunk size: {self.total_bytes/max(1,self.chunk_count):.1f} bytes")
            self._log("=== DEBUG SUBSCRIBER STOPPED ===")
            
            self.log_fd.close()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()


def main():
    """Test the debug subscriber with pipeline."""
    import sys
    from pipeline import StreamPipeline
    
    if len(sys.argv) != 2:
        print("Usage: python debug_subscriber.py <fifo_path>")
        sys.exit(1)
    
    fifo_path = sys.argv[1]
    
    # Create pipeline and debug subscriber
    pipeline = StreamPipeline(fifo_path)
    debug_sub = DebugSubscriber("debug_stream.log")
    
    # Add debug subscriber
    pipeline.add_subscriber(debug_sub)
    
    try:
        pipeline.start()
        print(f"Started debug logging for {fifo_path}")
        print(f"Log file: {debug_sub.log_file}")
        print("Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping debug subscriber...")
    finally:
        debug_sub.close()
        pipeline.stop()


if __name__ == "__main__":
    main()
