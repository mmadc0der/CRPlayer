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
