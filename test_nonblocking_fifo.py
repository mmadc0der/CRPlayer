#!/usr/bin/env python3

import os
import time
import threading
from inference.fifo_reader import NonBlockingFifoReader
from inference.pipeline import StreamHub
from inference.monitoring import PerformanceMonitor


def test_nonblocking_fifo():
    """Test non-blocking FIFO reading with scrcpy."""
    
    fifo_path = "/tmp/scrcpy_nonblock"
    
    # Create FIFO
    try:
        os.mkfifo(fifo_path)
        print(f"Created FIFO: {fifo_path}")
    except FileExistsError:
        print(f"FIFO already exists: {fifo_path}")
    
    # Start scrcpy in background
    print("Starting scrcpy recording...")
    scrcpy_cmd = [
        "scrcpy", "--no-window", "--no-audio", 
        "--video-bit-rate", "6000000",
        "--record-format=mkv", 
        "--record", fifo_path
    ]
    
    import subprocess
    scrcpy_proc = subprocess.Popen(scrcpy_cmd, 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
    
    # Give scrcpy time to start
    time.sleep(2)
    
    try:
        # Test non-blocking FIFO reader
        print("Testing non-blocking FIFO reader...")
        
        with NonBlockingFifoReader(fifo_path) as reader:
            bytes_read = 0
            chunks_read = 0
            start_time = time.time()
            
            while time.time() - start_time < 10:  # Test for 10 seconds
                data = reader.read(4096)
                
                if data:
                    bytes_read += len(data)
                    chunks_read += 1
                    
                    if chunks_read <= 3:
                        print(f"[DEBUG] Chunk {chunks_read}: {len(data)} bytes, "
                              f"first 16 bytes: {data[:16].hex()}")
                    elif chunks_read == 4:
                        print(f"[DEBUG] Stream active - {chunks_read} chunks, "
                              f"{bytes_read} total bytes")
                else:
                    # No data available - sleep briefly
                    time.sleep(0.01)
                    
            print(f"\nTest completed: {bytes_read} bytes in {chunks_read} chunks")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        scrcpy_proc.terminate()
        scrcpy_proc.wait()
        try:
            os.unlink(fifo_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    test_nonblocking_fifo()
