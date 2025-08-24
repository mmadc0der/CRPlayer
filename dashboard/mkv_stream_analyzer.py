#!/usr/bin/env python3
"""
MKV Stream Analyzer - Debug tool to understand the actual structure of incoming MKV data
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inference.fifo_reader import NonBlockingFifoReader

class MKVStreamAnalyzer:
    def __init__(self, fifo_path: str = "/tmp/scrcpy_stream"):
        self.fifo_path = fifo_path
        self.reader = NonBlockingFifoReader(fifo_path)
        self.total_bytes = 0
        self.chunk_count = 0
        
    def analyze_stream(self, duration_seconds: int = 30):
        """Analyze MKV stream for specified duration."""
        print(f"Starting MKV stream analysis for {duration_seconds} seconds...")
        print(f"Reading from: {self.fifo_path}")
        print("-" * 60)
        
        start_time = time.time()
        last_report = start_time
        buffer = bytearray()
        
        while time.time() - start_time < duration_seconds:
            try:
                data = self.reader.read()
                if data:
                    self.total_bytes += len(data)
                    self.chunk_count += 1
                    buffer.extend(data)
                    
                    # Analyze this chunk
                    self._analyze_chunk(data, self.chunk_count)
                    
                    # Keep buffer manageable
                    if len(buffer) > 1024 * 1024:  # 1MB
                        self._analyze_buffer_structure(buffer)
                        buffer = buffer[-512*1024:]  # Keep last 512KB
                
                # Report every 5 seconds
                current_time = time.time()
                if current_time - last_report >= 5.0:
                    self._report_progress(current_time - start_time)
                    last_report = current_time
                    
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error reading stream: {e}")
                time.sleep(1)
        
        # Final analysis
        print("\n" + "=" * 60)
        print("FINAL ANALYSIS")
        print("=" * 60)
        self._analyze_buffer_structure(buffer)
        self._final_report(start_time)
        
    def _analyze_chunk(self, data: bytes, chunk_num: int):
        """Analyze individual data chunk."""
        if len(data) == 0:
            return
            
        # Look for known MKV signatures
        signatures = {
            b'\x1a\x45\xdf\xa3': 'EBML Header',
            b'\x18\x53\x80\x67': 'Segment',
            b'\x11\x4d\x9b\x74': 'SeekHead',
            b'\x15\x49\xa9\x66': 'Info',
            b'\x16\x54\xae\x6b': 'Tracks',
            b'\x1c\x53\xbb\x6b': 'Cues',
            b'\x1f\x43\xb6\x75': 'Cluster',
            b'\xa3': 'SimpleBlock',
            b'\xa1': 'Block',
        }
        
        found_signatures = []
        for sig, name in signatures.items():
            if sig in data:
                pos = data.find(sig)
                found_signatures.append(f"{name} at {pos}")
        
        if found_signatures or chunk_num <= 5 or chunk_num % 20 == 0:
            print(f"Chunk {chunk_num:3d}: {len(data):6d} bytes", end="")
            if found_signatures:
                print(f" - FOUND: {', '.join(found_signatures)}")
            else:
                print(f" - First 16 bytes: {data[:16].hex()}")
    
    def _analyze_buffer_structure(self, buffer: bytearray):
        """Analyze accumulated buffer for MKV structure."""
        print(f"\nBUFFER ANALYSIS ({len(buffer)} bytes):")
        print("-" * 40)
        
        # Look for all MKV signatures in buffer
        signatures = {
            b'\x1a\x45\xdf\xa3': 'EBML Header',
            b'\x18\x53\x80\x67': 'Segment',
            b'\x11\x4d\x9b\x74': 'SeekHead',
            b'\x15\x49\xa9\x66': 'Info',
            b'\x16\x54\xae\x6b': 'Tracks',
            b'\x1c\x53\xbb\x6b': 'Cues',
            b'\x1f\x43\xb6\x75': 'Cluster',
        }
        
        found_any = False
        for sig, name in signatures.items():
            positions = []
            start = 0
            while True:
                pos = buffer.find(sig, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
                if len(positions) >= 10:  # Limit output
                    break
            
            if positions:
                found_any = True
                print(f"  {name:12}: {len(positions)} occurrences at {positions[:5]}")
        
        if not found_any:
            print("  NO MKV SIGNATURES FOUND!")
            print(f"  First 64 bytes: {buffer[:64].hex()}")
            print(f"  Last 64 bytes:  {buffer[-64:].hex()}")
            
            # Check if it might be raw H.264
            if buffer.startswith(b'\x00\x00\x00\x01') or b'\x00\x00\x01' in buffer[:100]:
                print("  POSSIBLE H.264 NAL units detected!")
            
            # Check for other video formats
            if buffer.startswith(b'ftyp') or b'ftyp' in buffer[:100]:
                print("  POSSIBLE MP4 format detected!")
    
    def _report_progress(self, elapsed: float):
        """Report current progress."""
        rate = self.total_bytes / elapsed if elapsed > 0 else 0
        print(f"Progress: {elapsed:.1f}s, {self.total_bytes} bytes, {self.chunk_count} chunks, {rate:.1f} B/s")
    
    def _final_report(self, start_time):
        """Print final analysis report."""
        duration = time.time() - start_time
        print(f"Total duration: {duration:.1f} seconds")
        print(f"Total bytes: {self.total_bytes}")
        print(f"Total chunks: {self.chunk_count}")
        print(f"Average chunk size: {self.total_bytes / max(1, self.chunk_count):.1f} bytes")
        print(f"Average rate: {self.total_bytes / max(1, duration):.1f} B/s")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MKV stream structure")
    parser.add_argument("--fifo", default="/tmp/scrcpy_stream", help="FIFO path")
    parser.add_argument("--duration", type=int, default=30, help="Analysis duration in seconds")
    
    args = parser.parse_args()
    
    analyzer = MKVStreamAnalyzer(args.fifo)
    
    try:
        analyzer.analyze_stream(args.duration)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
