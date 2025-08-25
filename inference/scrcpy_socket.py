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
from typing import Optional, AsyncIterator, NamedTuple, Dict, Any
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
        self._tunnel_type = 'reverse'  # Track tunnel type for protocol handling
        self._stats = {
            'frames_received': 0,
            'bytes_received': 0,
            'keyframes': 0,
            'start_time': 0.0
        }
        
    async def _setup_adb_tunnel(self):
        """Setup ADB forward tunnel to scrcpy socket (simpler for programmatic access)"""
        # First, check running scrcpy processes to get the actual SCID
        scid = await self._get_scrcpy_scid()
        if not scid:
            raise Exception("No running scrcpy process found")
        
        # The correct socket name uses the SCID in 8-character hex format (scrcpy 2.0+)
        correct_socket = f"localabstract:scrcpy_{scid}"
        logger.info(f"Looking for scrcpy socket: {correct_socket} (SCID: {scid})")
        
        # Use forward tunnel for programmatic access (simpler)
        self._tunnel_type = 'forward'
        self.adb_port = 27184
        
        # Check existing forwards first
        result = subprocess.run(['adb', 'forward', '--list'], 
                              capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if correct_socket in line and line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    port_part = parts[1]
                    if port_part.startswith('tcp:'):
                        self.adb_port = int(port_part.split(':')[1])
                        logger.info(f"[OK] Found existing forward: tcp:{self.adb_port} -> {correct_socket}")
                        return
        
        # Create new forward tunnel
        logger.info(f"Creating forward tunnel: tcp:{self.adb_port} -> {correct_socket}")
        result = subprocess.run([
            'adb', 'forward', f'tcp:{self.adb_port}', correct_socket
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"[OK] Forward tunnel established: tcp:{self.adb_port} -> {correct_socket}")
            return
        
        raise Exception(f"Failed to create forward tunnel: {result.stderr}")
    
    async def _get_scrcpy_scid(self) -> Optional[str]:
        """Extract SCID from running scrcpy process"""
        try:
            # Try Windows and Unix process listing
            if os.name == 'nt':
                result = subprocess.run(['wmic', 'process', 'get', 'commandline'], 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                
            for line in result.stdout.split('\n'):
                if 'scrcpy-server.jar' in line and 'scid=' in line:
                    # Extract scid=XXXXXXXX from command line
                    parts = line.split()
                    for part in parts:
                        if part.startswith('scid='):
                            scid_value = part.split('=')[1]
                            # Validate SCID is valid hex format (8 characters)
                            try:
                                # SCID can be hex string or decimal - handle both
                                if len(scid_value) == 8 and all(c in '0123456789abcdefABCDEF' for c in scid_value):
                                    # Already in hex format
                                    scid = scid_value.lower()
                                else:
                                    # Try as decimal and convert to hex
                                    scid_int = int(scid_value)
                                    scid = format(scid_int, '08x')
                                logger.info(f"Found scrcpy SCID: {scid}")
                                return scid
                            except ValueError:
                                logger.warning(f"Invalid SCID format: {scid_value}")
                                continue
            return None
        except Exception as e:
            logger.error(f"Failed to get SCID: {e}")
            return None

    async def connect(self, max_retries: int = 3) -> bool:
        """Connect to scrcpy video socket with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Connection attempt {attempt + 1}/{max_retries}")
                
                # Setup ADB tunnel (reverse or forward)
                await self._setup_adb_tunnel()
                
                # Connect to video socket with timeout (forward tunnel)
                logger.info(f"Connecting to localhost:{self.adb_port}...")
                self.reader, self.writer = await asyncio.wait_for(
                    asyncio.open_connection('localhost', self.adb_port),
                    timeout=5.0
                )
                
                # Validate connection is actually working
                if not await self._validate_connection():
                    await self.disconnect()
                    if attempt < max_retries - 1:
                        logger.warning(f"Connection validation failed, retrying in 2s...")
                        await asyncio.sleep(2)
                        continue
                    else:
                        return False
                
                # Read and validate device metadata with timeout
                await asyncio.wait_for(self._read_device_metadata(), timeout=3.0)
                
                self._connected = True
                self._stats['start_time'] = time.time()
                logger.info(f"Successfully connected to scrcpy video socket on port {self.adb_port}")
                return True
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout on attempt {attempt + 1} - scrcpy may not be running or video disabled")
                if attempt == max_retries - 1:
                    logger.error("Use: scrcpy --no-audio --max-fps=60 --max-size=1600 --video-bit-rate=20M")
                await self.disconnect()
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                await self.disconnect()
                
            if attempt < max_retries - 1:
                await asyncio.sleep(2)  # Wait before retry
                
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
    
    
    async def _validate_connection(self) -> bool:
        """Validate that the socket connection is working"""
        try:
            if self.writer.is_closing():
                logger.debug("Writer is closing")
                return False
                
            # Skip validation for forward tunnels - just check if connection is alive
            # The real validation happens during metadata reading
            logger.debug("Skipping validation for forward tunnel - will validate during metadata read")
            return True
                
        except Exception as e:
            logger.debug(f"Connection validation failed: {e}")
            return False
    
    async def _read_device_metadata(self):
        """Read initial scrcpy protocol data with improved error handling"""
        try:
            # According to scrcpy protocol, the first socket opened gets:
            # 1. Dummy byte (if forward tunnel) 
            # 2. Device metadata
            # 3. Video codec metadata (if this is video socket)
            
            # Use any buffered data first
            if self._initial_buffer:
                logger.debug(f"Using buffered data: {len(self._initial_buffer)} bytes")
            
            # Try to read dummy byte for forward tunnel (may not always be present)
            dummy_byte = None
            if self._initial_buffer:
                # Check if we have buffered data that might be dummy byte
                if len(self._initial_buffer) > 0:
                    first_byte = self._initial_buffer[:1]
                    # If it looks like a dummy byte (usually 0x00), consume it
                    if first_byte == b'\x00':
                        dummy_byte = first_byte
                        self._initial_buffer = self._initial_buffer[1:]
                        logger.debug(f"[OK] Forward tunnel dummy byte from buffer: {dummy_byte.hex()}")
            else:
                try:
                    # Try to read a byte with short timeout
                    potential_byte = await asyncio.wait_for(self.reader.read(1), timeout=0.5)
                    if potential_byte == b'\x00':
                        dummy_byte = potential_byte
                        logger.debug(f"[OK] Forward tunnel dummy byte: {dummy_byte.hex()}")
                    else:
                        # Not a dummy byte, put it back in buffer for metadata reading
                        self._initial_buffer = potential_byte + self._initial_buffer
                        logger.debug("No dummy byte found, proceeding to metadata")
                except asyncio.TimeoutError:
                    logger.debug("No dummy byte received, proceeding to metadata")
            
            # Read device metadata length (4 bytes)
            logger.debug("Reading device metadata length...")
            meta_len_data = await self._read_exact_with_buffer(4, timeout=5.0)
            if not meta_len_data or len(meta_len_data) != 4:
                raise Exception("Failed to read metadata length - connection may be to wrong socket type")
            
            meta_len = struct.unpack('>I', meta_len_data)[0]
            logger.debug(f"Device metadata length: {meta_len}")
            
            # Validate metadata length is reasonable
            if meta_len > 1024 or meta_len < 0:
                raise Exception(f"Invalid metadata length: {meta_len}")
            
            # Read device name
            logger.debug(f"Reading device name ({meta_len} bytes)...")
            device_name_data = await self._read_exact_with_buffer(meta_len, timeout=5.0)
            if not device_name_data or len(device_name_data) != meta_len:
                raise Exception("Failed to read device name")
            
            device_name = device_name_data.decode('utf-8', errors='ignore').rstrip('\x00')
            logger.debug(f"Device name: {device_name}")
            
            # Try to read video codec metadata (12 bytes)
            # This will only succeed if this is the video socket
            try:
                codec_data = await asyncio.wait_for(
                    self._read_exact_with_buffer(12), timeout=5.0
                )
                if codec_data and len(codec_data) == 12:
                    codec_id, width, height = struct.unpack('>III', codec_data)
                    codec_names = {0: 'H264', 1: 'H265', 2: 'AV1'}
                    codec_name = codec_names.get(codec_id, f'Unknown({codec_id})')
                    
                    logger.info(f"[OK] Video socket confirmed - {codec_name} {width}x{height} ({self._tunnel_type} tunnel)")
                    self._stats['codec_info'] = f"{codec_name} {width}x{height}"
                    self._stats['device_name'] = device_name
                    return  # Success - this is video socket
                else:
                    raise Exception("Invalid codec data received")
                    
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"[FAIL] Video codec validation failed: {e}")
                raise Exception("Connected to non-video socket or incompatible scrcpy version")
            
        except Exception as e:
            raise Exception(f"Protocol validation failed: {e}")
    
    async def _read_next_frame(self) -> Optional[NALUnit]:
        """Read next video frame from scrcpy socket with proper protocol"""
        try:
            # Read 12-byte frame header
            header_data = await self._read_exact(12)
            if not header_data:
                return None
            
            # Parse frame header according to scrcpy protocol
            # Bytes 0-7: PTS with flags in MSBs
            # Bytes 8-11: packet size
            pts_with_flags = struct.unpack('>Q', header_data[:8])[0]
            packet_size = struct.unpack('>I', header_data[8:12])[0]
            
            # Extract flags from PTS
            config_packet = bool(pts_with_flags & (1 << 63))
            key_frame = bool(pts_with_flags & (1 << 62))
            pts = pts_with_flags & ((1 << 62) - 1)  # Clear flag bits
            
            # Read packet data
            packet_data = await self._read_exact(packet_size)
            if not packet_data:
                logger.warning(f"Failed to read packet data ({packet_size} bytes)")
                return None
            
            self._stats['bytes_received'] += len(header_data) + len(packet_data)
            self._stats['frames_received'] += 1
            
            if key_frame:
                self._stats['keyframes'] += 1
            
            # Create NAL unit
            nal_unit = NALUnit(
                data=packet_data,
                timestamp=pts / 1000000.0,  # Convert to seconds
                is_keyframe=key_frame,
                frame_type="IDR" if key_frame else "P"
            )
            
            logger.debug(f"Frame: size={packet_size}, pts={pts}, keyframe={key_frame}, config={config_packet}")
            return nal_unit
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
    
    async def _read_exact_with_buffer(self, size: int, timeout: float = 5.0) -> Optional[bytes]:
        """Read exact number of bytes from socket, using internal buffer first"""
        data = b''
        
        # Use buffered data first
        if self._initial_buffer:
            buffer_use = min(size, len(self._initial_buffer))
            data = self._initial_buffer[:buffer_use]
            self._initial_buffer = self._initial_buffer[buffer_use:]
        
        # Read remaining data from socket
        while len(data) < size:
            try:
                chunk = await asyncio.wait_for(
                    self.reader.read(size - len(data)), 
                    timeout=timeout
                )
                if not chunk:
                    logger.debug(f"Socket closed while reading {size} bytes (got {len(data)} so far)")
                    return None
                data += chunk
            except asyncio.TimeoutError:
                logger.debug(f"Timeout reading {size} bytes (got {len(data)} so far, timeout={timeout}s)")
                return None
        return data
    
    async def _read_exact(self, size: int, timeout: float = 5.0) -> Optional[bytes]:
        """Read exact number of bytes from socket (legacy method)"""
        return await self._read_exact_with_buffer(size, timeout)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        stats = self._stats.copy()
        if stats['start_time'] > 0:
            elapsed = time.time() - stats['start_time']
            stats['elapsed_time'] = elapsed
            if elapsed > 0:
                stats['fps'] = stats['frames_received'] / elapsed
                stats['bitrate_kbps'] = (stats['bytes_received'] * 8) / (elapsed * 1000)
        return stats
    
    async def disconnect(self):
        """Disconnect from scrcpy socket"""
        self._connected = False
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.reader = None
        self.writer = None
        logger.info("Disconnected from scrcpy socket")
    
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
