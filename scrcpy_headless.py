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
        """Start headless scrcpy instance"""
        try:
            # Kill any existing scrcpy processes
            await self._cleanup_existing()
            
            # Build scrcpy command for headless operation
            cmd = [
                'scrcpy',
                '--no-audio',     # No audio
                '--no-control',   # No input control
                '--max-fps=60',
                '--max-size=1600', 
                '--video-bit-rate=20M'
            ]
            
            if self.device_id:
                cmd.extend(['-s', self.device_id])
            
            logger.info(f"Starting headless scrcpy: {' '.join(cmd)}")
            
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
                logger.info("Headless scrcpy started successfully")
                
                # Detect the SCID from the running process
                await self._detect_scid()
                
                return True
            else:
                stdout, stderr = self.process.communicate()
                logger.error(f"Scrcpy failed to start: {stderr.decode()}")
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
    
    async def _detect_scid(self):
        """Detect SCID from running scrcpy process"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'scrcpy-server.jar' in line and 'scid=' in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith('scid='):
                            self.scid = part.split('=')[1]
                            logger.info(f"Detected SCID: {self.scid}")
                            return
            logger.warning("Could not detect SCID from process")
        except Exception as e:
            logger.error(f"Error detecting SCID: {e}")
    
    def get_socket_name(self) -> str:
        """Get the socket name for this headless instance"""
        if self.scid:
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
