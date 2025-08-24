#!/usr/bin/env python3
"""
MKV Analyzer Subscriber - Analyzes MKV stream structure as a pipeline subscriber
"""

import time
from typing import Dict, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import DataChunk

class MKVAnalyzerSubscriber:
    """Analyzes MKV stream structure and reports findings."""
    
    def __init__(self, report_interval: float = 5.0):
        self.report_interval = report_interval
        self.last_report_time = 0.0
        self.start_time = 0.0
        
        # Statistics
        self.total_bytes = 0
        self.chunk_count = 0
        self.signature_counts: Dict[str, List[int]] = {}
        
        # Buffer for structure analysis
        self.analysis_buffer = bytearray()
        self.max_buffer_size = 512 * 1024  # 512KB
        
        # MKV signatures to look for
        self.mkv_signatures = {
            b'\x1a\x45\xdf\xa3': 'EBML_Header',
            b'\x18\x53\x80\x67': 'Segment',
            b'\x11\x4d\x9b\x74': 'SeekHead',
            b'\x15\x49\xa9\x66': 'Info',
            b'\x16\x54\xae\x6b': 'Tracks',
            b'\x1c\x53\xbb\x6b': 'Cues',
            b'\x1f\x43\xb6\x75': 'Cluster',
            b'\xa3': 'SimpleBlock',
            b'\xa1': 'Block',
        }
        
        print("[MKVAnalyzer] Initialized - looking for MKV signatures")
    
    def __call__(self, chunk: DataChunk) -> None:
        """Process incoming data chunk."""
        if self.start_time == 0:
            self.start_time = chunk.timestamp
            
        self.total_bytes += chunk.size
        self.chunk_count += 1
        
        # Convert memoryview to bytes for analysis
        data = chunk.data.tobytes()
        
        # Analyze chunk for signatures
        self._analyze_chunk_signatures(data, self.chunk_count)
        
        # Add to analysis buffer
        self.analysis_buffer.extend(data)
        
        # Keep buffer manageable
        if len(self.analysis_buffer) > self.max_buffer_size:
            self._analyze_buffer_structure()
            # Keep last 256KB
            self.analysis_buffer = self.analysis_buffer[-256*1024:]
        
        # Report periodically
        if chunk.timestamp - self.last_report_time >= self.report_interval:
            self._report_progress(chunk.timestamp)
            self.last_report_time = chunk.timestamp
    
    def _analyze_chunk_signatures(self, data: bytes, chunk_num: int) -> None:
        """Look for MKV signatures in this chunk."""
        found_signatures = []
        
        for signature, name in self.mkv_signatures.items():
            positions = []
            start = 0
            while True:
                pos = data.find(signature, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
                if len(positions) >= 5:  # Limit to prevent spam
                    break
            
            if positions:
                if name not in self.signature_counts:
                    self.signature_counts[name] = []
                self.signature_counts[name].extend(positions)
                found_signatures.append(f"{name}@{positions}")
        
        # Report interesting chunks
        if found_signatures:
            print(f"[MKVAnalyzer] Chunk {chunk_num}: {len(data)} bytes - FOUND: {', '.join(found_signatures)}")
        elif chunk_num <= 3:
            # Show first few chunks regardless
            print(f"[MKVAnalyzer] Chunk {chunk_num}: {len(data)} bytes - First 32 bytes: {data[:32].hex()}")
    
    def _analyze_buffer_structure(self) -> None:
        """Analyze accumulated buffer for overall MKV structure."""
        print(f"\n[MKVAnalyzer] === BUFFER ANALYSIS ({len(self.analysis_buffer)} bytes) ===")
        
        found_any = False
        for signature, name in self.mkv_signatures.items():
            count = self.analysis_buffer.count(signature)
            if count > 0:
                found_any = True
                first_pos = self.analysis_buffer.find(signature)
                print(f"  {name:12}: {count} occurrences, first at {first_pos}")
        
        if not found_any:
            print("  ❌ NO MKV SIGNATURES FOUND!")
            print(f"  📊 Buffer start: {self.analysis_buffer[:64].hex()}")
            print(f"  📊 Buffer end:   {self.analysis_buffer[-64:].hex()}")
            
            # Check for other formats
            self._check_alternative_formats()
        else:
            print("  ✅ MKV signatures detected!")
    
    def _check_alternative_formats(self) -> None:
        """Check if data might be in other formats."""
        buffer = self.analysis_buffer
        
        # Check for H.264 NAL units
        if (buffer.startswith(b'\x00\x00\x00\x01') or 
            buffer.startswith(b'\x00\x00\x01') or
            b'\x00\x00\x00\x01' in buffer[:200]):
            print("  🎥 POSSIBLE H.264 NAL units detected!")
        
        # Check for MP4
        if buffer.startswith(b'ftyp') or b'ftyp' in buffer[:200]:
            print("  📦 POSSIBLE MP4 format detected!")
        
        # Check for AVI
        if buffer.startswith(b'RIFF') and b'AVI ' in buffer[:200]:
            print("  🎬 POSSIBLE AVI format detected!")
        
        # Check for WebM (which is MKV-based)
        if b'webm' in buffer[:200].lower():
            print("  🌐 POSSIBLE WebM format detected!")
        
        # Check for raw video patterns
        if len(set(buffer[:100])) < 10:  # Very low entropy
            print("  📺 POSSIBLE raw/uncompressed video data!")
    
    def _report_progress(self, current_time: float) -> None:
        """Report current analysis progress."""
        elapsed = current_time - self.start_time
        rate = self.total_bytes / elapsed if elapsed > 0 else 0
        
        print(f"\n[MKVAnalyzer] === PROGRESS REPORT ===")
        print(f"  Time: {elapsed:.1f}s | Bytes: {self.total_bytes} | Chunks: {self.chunk_count}")
        print(f"  Rate: {rate:.1f} B/s | Avg chunk: {self.total_bytes/max(1,self.chunk_count):.1f} bytes")
        
        if self.signature_counts:
            print("  Signatures found:")
            for name, positions in self.signature_counts.items():
                print(f"    {name}: {len(positions)} times")
        else:
            print("  ❌ No MKV signatures found yet!")
    
    def get_final_report(self) -> Dict:
        """Get final analysis report."""
        return {
            'total_bytes': self.total_bytes,
            'chunk_count': self.chunk_count,
            'signatures_found': dict(self.signature_counts),
            'has_mkv_structure': len(self.signature_counts) > 0,
            'buffer_size': len(self.analysis_buffer)
        }


def main():
    """Test the analyzer with pipeline."""
    import sys
    from pipeline import StreamPipeline
    
    if len(sys.argv) != 2:
        print("Usage: python mkv_analyzer_subscriber.py <fifo_path>")
        sys.exit(1)
    
    fifo_path = sys.argv[1]
    
    # Create pipeline and analyzer
    pipeline = StreamPipeline(fifo_path)
    analyzer = MKVAnalyzerSubscriber(report_interval=3.0)
    
    # Add analyzer as subscriber
    pipeline.add_subscriber(analyzer)
    
    try:
        pipeline.start()
        print(f"Started MKV analysis on {fifo_path}")
        print("Press Ctrl+C to stop...")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[MKVAnalyzer] Stopping analysis...")
        report = analyzer.get_final_report()
        print(f"Final report: {report}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
