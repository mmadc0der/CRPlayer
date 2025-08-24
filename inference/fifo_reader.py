import os
import errno
import time
from typing import Optional


class NonBlockingFifoReader:
    """Non-blocking FIFO reader that handles EAGAIN/EWOULDBLOCK gracefully."""
    
    def __init__(self, fifo_path: str, chunk_size: int = 4096):
        self.fifo_path = fifo_path
        self.chunk_size = chunk_size
        self.fd: Optional[int] = None
        self._buffer = b""
        
    def open(self) -> None:
        """Open FIFO in non-blocking read mode."""
        if self.fd is not None:
            self.close()
        
        # Open with O_NONBLOCK to prevent blocking on empty FIFO
        self.fd = os.open(self.fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        
    def close(self) -> None:
        """Close the FIFO file descriptor."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            
    def read(self, size: int) -> bytes:
        """Read data from FIFO, handling EAGAIN/EWOULDBLOCK gracefully.
        
        Returns empty bytes if no data available (non-blocking).
        """
        if self.fd is None:
            raise RuntimeError("FIFO not opened")
            
        try:
            data = os.read(self.fd, size)
            return data
        except OSError as err:
            if err.errno == errno.EAGAIN or err.errno == errno.EWOULDBLOCK:
                # No data available - return empty bytes
                return b""
            else:
                # Real error - reraise
                raise
                
    def read_available(self) -> bytes:
        """Read all currently available data from FIFO."""
        if self.fd is None:
            raise RuntimeError("FIFO not opened")
            
        chunks = []
        while True:
            try:
                chunk = os.read(self.fd, self.chunk_size)
                if not chunk:  # EOF
                    break
                chunks.append(chunk)
            except OSError as err:
                if err.errno == errno.EAGAIN or err.errno == errno.EWOULDBLOCK:
                    # No more data available
                    break
                else:
                    raise
                    
        return b"".join(chunks)
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) != 2:
        print("Usage: python fifo_reader.py <fifo_path>")
        sys.exit(1)
    
    fifo_path = sys.argv[1]
    
    print(f"Starting FIFO debug reader for: {fifo_path}")
    print("Press Ctrl+C to stop\n")
    
    try:
        with NonBlockingFifoReader(fifo_path) as reader:
            start_time = time.time()
            last_report_time = start_time
            total_bytes = 0
            second_bytes = 0
            
            while True:
                # Read all available data
                data = reader.read_available()
                data_len = len(data)
                
                if data_len > 0:
                    total_bytes += data_len
                    second_bytes += data_len
                    
                    # Show first few bytes for debugging
                    if total_bytes <= 32:
                        print(f"[DATA] First {data_len} bytes: {data[:min(16, data_len)].hex()}")
                
                current_time = time.time()
                
                # Report every second
                if current_time - last_report_time >= 1.0:
                    elapsed = current_time - start_time
                    avg_rate = total_bytes / elapsed if elapsed > 0 else 0
                    
                    print(f"[{elapsed:6.1f}s] This second: {second_bytes:6d} bytes | "
                          f"Total: {total_bytes:8d} bytes | "
                          f"Avg rate: {avg_rate:7.1f} B/s")
                    
                    second_bytes = 0
                    last_report_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nStopped by user")
        exit()
    except Exception as e:
        print(f"Error: {e}")
