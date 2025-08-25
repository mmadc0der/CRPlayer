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
        self._initial_buffer = b''
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
            # Check if scrcpy is already running and using the port
            result = subprocess.run([
                'adb', 'forward', '--list'
            ], capture_output=True, text=True)
            
            # Look for existing scrcpy forward
            existing_forward = None
            for line in result.stdout.strip().split('\n'):
                if 'localabstract:scrcpy' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        existing_forward = parts[1].split(':')[-1]  # Extract port
                        break
            
            if existing_forward:
                self.adb_port = int(existing_forward)
                logger.info(f"Using existing scrcpy forward on port: {self.adb_port}")
            else:
                # Setup new forward - but scrcpy should already be running
                logger.warning("No existing scrcpy forward found - make sure scrcpy is running")
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
            # Scrcpy v3+ protocol: first read is the video stream header
            # Try to read first few bytes to detect stream format
            initial_data = await asyncio.wait_for(self.reader.read(16), timeout=2.0)
            
            if not initial_data:
                raise ScrcpyProtocolError("No data received from scrcpy socket")
            
            logger.info(f"Received initial data: {len(initial_data)} bytes")
            logger.debug(f"Initial bytes: {initial_data.hex()}")
            
            # Put the data back for frame reading (create a buffer)
            self._initial_buffer = initial_data
            
            # For now, skip metadata parsing and go straight to frame reading
            logger.info("Connected to scrcpy video stream")
            
        except asyncio.TimeoutError:
            raise ScrcpyProtocolError("Timeout waiting for initial data from scrcpy")
        except Exception as e:
            raise ScrcpyProtocolError(f"Failed to read initial data: {e}")
    
    async def _read_next_frame(self) -> Optional[NALUnit]:
        """Read next video frame from scrcpy socket"""
        try:
            # Use initial buffer if available
            if self._initial_buffer:
                data = self._initial_buffer
                self._initial_buffer = b''
                logger.info(f"Processing initial buffer: {len(data)} bytes")
                
                # Try to find frame boundaries in initial data
                nal_data = self._extract_nal_from_raw(data)
                if nal_data:
                    is_keyframe = self._is_keyframe(nal_data)
                    return NALUnit(
                        data=nal_data,
                        pts=0,  # Unknown PTS for initial data
                        size=len(nal_data),
                        timestamp=time.time(),
                        is_keyframe=is_keyframe
                    )
            
            # Try to read raw data and find NAL units
            data = await asyncio.wait_for(self.reader.read(8192), timeout=0.1)
            if not data:
                return None
            
            # Extract NAL units from raw data
            nal_data = self._extract_nal_from_raw(data)
            if nal_data:
                is_keyframe = self._is_keyframe(nal_data)
                return NALUnit(
                    data=nal_data,
                    pts=int(time.time() * 1000000),  # Generate PTS from timestamp
                    size=len(nal_data),
                    timestamp=time.time(),
                    is_keyframe=is_keyframe
                )
            
            return None
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
    
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
    
    def _extract_nal_from_raw(self, data: bytes) -> Optional[bytes]:
        """Extract NAL units from raw scrcpy stream data"""
        if not data:
            return None
        
        # Look for NAL unit start codes in the data
        start_codes = [b'\x00\x00\x00\x01', b'\x00\x00\x01']
        
        for start_code in start_codes:
            start_idx = data.find(start_code)
            if start_idx != -1:
                # Found a start code, try to find the next one
                search_start = start_idx + len(start_code)
                next_idx = -1
                
                for next_start_code in start_codes:
                    pos = data.find(next_start_code, search_start)
                    if pos != -1:
                        if next_idx == -1 or pos < next_idx:
                            next_idx = pos
                
                if next_idx != -1:
                    # Extract complete NAL unit
                    return data[start_idx:next_idx]
                else:
                    # Take rest of data if no next start code found
                    return data[start_idx:]
        
        # If no start codes found, check if this might be length-prefixed
        if len(data) >= 4:
            try:
                # Try to parse as length-prefixed NAL
                nal_length = struct.unpack('>I', data[:4])[0]
                if 4 + nal_length <= len(data) and nal_length > 0:
                    # Convert to Annex-B
                    nal_data = b'\x00\x00\x00\x01' + data[4:4+nal_length]
                    return nal_data
            except:
                pass
        
        # Return None if no valid NAL unit found
        return None
    
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
