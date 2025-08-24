#!/usr/bin/env python3

import sys
import time
from pipeline import StreamPipeline
from dashboard import create_dashboard_subscriber


def main():
    if len(sys.argv) != 2:
        print("Usage: python example_with_dashboard.py <fifo_path>")
        sys.exit(1)
    
    fifo_path = sys.argv[1]
    
    # Create pipeline
    pipeline = StreamPipeline(fifo_path)
    
    # Create dashboard subscriber
    dashboard = create_dashboard_subscriber(host="localhost", port=8765)
    dashboard.set_pipeline(pipeline)  # Give dashboard access to pipeline stats
    
    # Add dashboard as subscriber
    pipeline.add_subscriber(dashboard.handle_chunk)
    
    # Add debug subscriber for console output
    def debug_subscriber(chunk):
        if pipeline.total_chunks <= 3:
            print(f"[CHUNK {pipeline.total_chunks}] {chunk.size} bytes at {chunk.timestamp:.3f}s")
            print(f"  First 16 bytes: {chunk.hex_preview()}")
        elif pipeline.total_chunks == 4:
            print("[DEBUG] Stream active - debug output reduced")
    
    pipeline.add_subscriber(debug_subscriber)
    
    try:
        print(f"Starting pipeline with dashboard integration...")
        print(f"FIFO: {fifo_path}")
        print(f"Dashboard: http://localhost:8080")
        print(f"WebSocket: ws://localhost:8765")
        print("Press Ctrl+C to stop\n")
        
        # Start dashboard server
        dashboard.start()
        time.sleep(0.5)  # Let server start
        
        # Start pipeline
        pipeline.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Print stats every 10 seconds
            if int(time.time()) % 10 == 0:
                stats = pipeline.get_stats()
                print(f"[STATS] {stats['total_bytes']} bytes, "
                      f"{stats['total_chunks']} chunks, "
                      f"{stats['avg_rate_bps']:.1f} B/s avg, "
                      f"{len(dashboard.clients)} clients connected")
                time.sleep(1)  # Prevent multiple prints in same second
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        dashboard.stop()
        pipeline.stop()
        print("Shutdown complete")


if __name__ == "__main__":
    main()
