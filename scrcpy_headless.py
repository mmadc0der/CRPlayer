#!/usr/bin/env python3
"""
Scrcpy Headless Manager - Start scrcpy without display for exclusive video access
"""

import asyncio
import subprocess
import logging
import time
import signal
import os
from typing import Optional

logger = logging.getLogger(__name__)

class ScrcpyHeadlessManager:
    """Manage headless scrcpy instance for exclusive video streaming"""
    
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.process: Optional[subprocess.Popen] = None
        self.scid = None  # Will be detected from running process
        
    async def start(self) -> bool:
        """Start headless scrcpy instance using standalone server mode"""
        try:
            # Kill any existing scrcpy processes
            await self._cleanup_existing()
            
            # Generate random SCID for this session
            import random
            self.scid = format(random.randint(0, 0x7FFFFFFF), '08x')
            
            # Use standalone server mode for proper protocol support
            # This ensures device metadata and frame headers are sent correctly
            cmd = [
                'adb', 'shell',
                'CLASSPATH=/data/local/tmp/scrcpy-server.jar',
                'app_process', '/', 'com.genymobile.scrcpy.Server', '3.3.1',
                f'scid={self.scid}',
                'log_level=info',
                'tunnel_forward=true',
                'audio=false',
                'control=false',
                'cleanup=false',
                'send_device_meta=true',    # Enable device metadata
                'send_frame_meta=true',     # Enable frame headers
                'send_codec_meta=true',     # Enable codec metadata
                'max_size=1600',
                'video_bit_rate=20000000',
                'max_fps=60'
            ]
            
            if self.device_id:
                # Insert device selection before shell command
                cmd.insert(1, '-s')
                cmd.insert(2, self.device_id)
            
            logger.info(f"Starting headless scrcpy server: {' '.join(cmd)}")
            
            # Start process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for startup
            await asyncio.sleep(3)
            
            # Check if process is still running
            if self.process.poll() is None:
                logger.info(f"Headless scrcpy server started successfully with SCID: {self.scid}")
                return True
            else:
                stdout, stderr = self.process.communicate()
                logger.error(f"Scrcpy server failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start headless scrcpy: {e}")
            return False
    
    async def stop(self):
        """Stop headless scrcpy instance"""
        if self.process:
            try:
                if os.name == 'nt':
                    self.process.terminate()
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        self.process.kill()
                    else:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                
                logger.info("Headless scrcpy stopped")
                
            except Exception as e:
                logger.error(f"Error stopping scrcpy: {e}")
            
            self.process = None
    
    async def _cleanup_existing(self):
        """Kill existing scrcpy processes"""
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/f', '/im', 'scrcpy.exe'], 
                             capture_output=True, check=False)
            else:
                subprocess.run(['pkill', '-f', 'scrcpy'], 
                             capture_output=True, check=False)
            
            # Wait for cleanup
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.warning(f"Error cleaning up existing scrcpy: {e}")
    
    
    def get_socket_name(self) -> str:
        """Get the socket name for this headless instance (scrcpy 2.0+ format)"""
        if self.scid:
            # SCID is already in hex format
            return f"localabstract:scrcpy_{self.scid}"
        else:
            return "localabstract:scrcpy"
    
    def is_running(self) -> bool:
        """Check if headless scrcpy is running"""
        return self.process is not None and self.process.poll() is None

# Test the headless manager
async def test_headless():
    manager = ScrcpyHeadlessManager()
    
    try:
        if await manager.start():
            logger.info("Headless scrcpy is running")
            
            # Test connection
            from inference.scrcpy_socket import ScrcpySocketDemux
            demux = ScrcpySocketDemux()
            
            # Override the SCID detection to use our fixed SCID
            demux._get_scrcpy_scid = lambda: asyncio.create_task(
                asyncio.coroutine(lambda: manager.scid)()
            )
            
            if await demux.connect():
                logger.info("Successfully connected to headless scrcpy!")
                
                # Read a few frames
                frame_count = 0
                async for nal_unit in demux.stream_nal_units():
                    frame_count += 1
                    logger.info(f"Frame {frame_count}: {len(nal_unit.data)} bytes, "
                               f"keyframe={nal_unit.is_keyframe}")
                    if frame_count >= 5:
                        break
                
                await demux.disconnect()
            else:
                logger.error("Failed to connect to headless scrcpy")
        
    finally:
        await manager.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_headless())
