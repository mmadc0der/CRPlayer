"""
Scrcpy Socket Demux - Direct connection to scrcpy video socket
Replaces FIFO-based approach with zero-copy socket streaming
"""

import asyncio
import struct
import socket
import subprocess
import time
import logging
from typing import Optional, AsyncIterator, NamedTuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NALUnit:
    """H.264/H.265 NAL unit with metadata"""
    data: bytes
    pts: int  # Presentation timestamp
    size: int
    timestamp: float  # Local timestamp
    is_keyframe: bool = False

class ScrcpyProtocolError(Exception):
    """Scrcpy protocol parsing error"""
    pass

class ScrcpySocketDemux:
    """Direct TCP connection to scrcpy video socket with proper protocol handling"""
    
    def __init__(self, adb_port: int = 27183, device_id: Optional[str] = None):
        self.adb_port = adb_port
        self.device_id = device_id
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._stats = {
            'frames_received': 0,
            'bytes_received': 0,
            'keyframes': 0,
            'start_time': 0.0
        }
        
    async def connect(self) -> bool:
        """Connect to scrcpy video socket"""
        try:
            # Setup ADB port forward
            await self._setup_adb_forward()
            
            # Connect to video socket
            self.reader, self.writer = await asyncio.open_connection(
                'localhost', self.adb_port
            )
            
            # Read and validate device metadata
            await self._read_device_metadata()
            
            self._connected = True
            self._stats['start_time'] = time.time()
            logger.info(f"Connected to scrcpy video socket on port {self.adb_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to scrcpy socket: {e}")
            await self.disconnect()
            return False
    
    async def disconnect(self):
        """Close socket connection"""
        self._connected = False
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.reader = None
        self.writer = None
        
    async def stream_nal_units(self) -> AsyncIterator[NALUnit]:
        """Stream NAL units from scrcpy socket"""
        if not self._connected:
            raise ScrcpyProtocolError("Not connected to scrcpy socket")
            
        logger.info("Starting NAL unit stream...")
        
        try:
            while self._connected:
                nal_unit = await self._read_next_frame()
                if nal_unit:
                    self._update_stats(nal_unit)
                    yield nal_unit
                else:
                    # No data available, small delay
                    await asyncio.sleep(0.001)
                    
        except asyncio.CancelledError:
            logger.info("NAL unit stream cancelled")
        except Exception as e:
            logger.error(f"Error in NAL unit stream: {e}")
            raise
    
    async def _setup_adb_forward(self):
        """Setup ADB port forwarding for scrcpy video socket"""
        try:
            # Kill any existing forward
            subprocess.run([
                'adb', 'forward', '--remove', f'tcp:{self.adb_port}'
            ], capture_output=True)
            
            # Setup new forward
            cmd = ['adb']
            if self.device_id:
                cmd.extend(['-s', self.device_id])
            cmd.extend([
                'forward', f'tcp:{self.adb_port}', 'localabstract:scrcpy'
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"ADB forward setup: {result.stdout.strip()}")
            
        except subprocess.CalledProcessError as e:
            raise ScrcpyProtocolError(f"ADB forward failed: {e.stderr}")
    
    async def _read_device_metadata(self):
        """Read initial device metadata from scrcpy socket"""
        try:
            # Read device name length (2 bytes)
            name_len_data = await self.reader.readexactly(2)
            name_len = struct.unpack('>H', name_len_data)[0]
            
            # Read device name
            if name_len > 0:
                device_name = await self.reader.readexactly(name_len)
                logger.info(f"Connected to device: {device_name.decode('utf-8')}")
            
            # Read initial resolution (8 bytes)
            resolution_data = await self.reader.readexactly(8)
            width, height = struct.unpack('>II', resolution_data)
            logger.info(f"Device resolution: {width}x{height}")
            
        except Exception as e:
            raise ScrcpyProtocolError(f"Failed to read device metadata: {e}")
    
    async def _read_next_frame(self) -> Optional[NALUnit]:
        """Read next video frame from scrcpy socket"""
        try:
            # Read frame header (12 bytes)
            # Format: PTS (8 bytes) + frame_size (4 bytes)
            header_data = await self.reader.readexactly(12)
            pts, frame_size = struct.unpack('>QI', header_data)
            
            if frame_size == 0:
                logger.warning("Received frame with zero size")
                return None
                
            if frame_size > 10 * 1024 * 1024:  # 10MB sanity check
                logger.error(f"Frame size too large: {frame_size} bytes")
                raise ScrcpyProtocolError(f"Invalid frame size: {frame_size}")
            
            # Read frame data
            frame_data = await self.reader.readexactly(frame_size)
            
            # Convert to Annex-B format
            nal_data = self._to_annexb(frame_data)
            
            # Check if keyframe
            is_keyframe = self._is_keyframe(nal_data)
            
            return NALUnit(
                data=nal_data,
                pts=pts,
                size=len(nal_data),
                timestamp=time.time(),
                is_keyframe=is_keyframe
            )
            
        except asyncio.IncompleteReadError:
            logger.warning("Incomplete read from scrcpy socket")
            return None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            raise
    
    def _to_annexb(self, frame_data: bytes) -> bytes:
        """Convert scrcpy frame data to Annex-B format with start codes"""
        if not frame_data:
            return b''
        
        # Check if already in Annex-B format
        if frame_data.startswith(b'\x00\x00\x00\x01') or frame_data.startswith(b'\x00\x00\x01'):
            return frame_data
        
        # Convert length-prefixed NALs to Annex-B
        annexb_data = bytearray()
        offset = 0
        
        while offset < len(frame_data):
            if offset + 4 > len(frame_data):
                break
                
            # Read NAL length (4 bytes, big-endian)
            nal_length = struct.unpack('>I', frame_data[offset:offset+4])[0]
            offset += 4
            
            if offset + nal_length > len(frame_data):
                logger.warning(f"Invalid NAL length: {nal_length}")
                break
            
            # Add start code and NAL data
            annexb_data.extend(b'\x00\x00\x00\x01')
            annexb_data.extend(frame_data[offset:offset+nal_length])
            offset += nal_length
        
        return bytes(annexb_data)
    
    def _is_keyframe(self, nal_data: bytes) -> bool:
        """Check if NAL unit contains a keyframe (IDR)"""
        # Look for IDR NAL unit (type 5 for H.264)
        for i in range(len(nal_data) - 4):
            if nal_data[i:i+4] == b'\x00\x00\x00\x01':
                if i + 4 < len(nal_data):
                    nal_type = nal_data[i + 4] & 0x1F
                    if nal_type == 5:  # IDR frame
                        return True
        return False
    
    def _update_stats(self, nal_unit: NALUnit):
        """Update streaming statistics"""
        self._stats['frames_received'] += 1
        self._stats['bytes_received'] += nal_unit.size
        if nal_unit.is_keyframe:
            self._stats['keyframes'] += 1
    
    def get_stats(self) -> dict:
        """Get streaming statistics"""
        elapsed = time.time() - self._stats['start_time']
        fps = self._stats['frames_received'] / elapsed if elapsed > 0 else 0
        bitrate = (self._stats['bytes_received'] * 8) / elapsed if elapsed > 0 else 0
        
        return {
            'frames_received': self._stats['frames_received'],
            'bytes_received': self._stats['bytes_received'],
            'keyframes': self._stats['keyframes'],
            'elapsed_time': elapsed,
            'fps': fps,
            'bitrate_bps': bitrate,
            'connected': self._connected
        }


async def main():
    """Test scrcpy socket connection"""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    device_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    demux = ScrcpySocketDemux(device_id=device_id)
    
    try:
        if not await demux.connect():
            print("Failed to connect to scrcpy socket")
            return
        
        print("Connected! Streaming NAL units...")
        print("Press Ctrl+C to stop\n")
        
        frame_count = 0
        async for nal_unit in demux.stream_nal_units():
            frame_count += 1
            
            if frame_count <= 5:
                print(f"Frame {frame_count}: {nal_unit.size} bytes, "
                      f"PTS: {nal_unit.pts}, Keyframe: {nal_unit.is_keyframe}")
                print(f"  First 32 bytes: {nal_unit.data[:32].hex()}")
            
            # Print stats every 60 frames
            if frame_count % 60 == 0:
                stats = demux.get_stats()
                print(f"\n[STATS] Frames: {stats['frames_received']}, "
                      f"FPS: {stats['fps']:.1f}, "
                      f"Bitrate: {stats['bitrate_bps']/1000000:.1f} Mbps, "
                      f"Keyframes: {stats['keyframes']}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await demux.disconnect()
        stats = demux.get_stats()
        print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
