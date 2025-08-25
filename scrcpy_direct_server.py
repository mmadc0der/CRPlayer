#!/usr/bin/env python3
"""
Direct Scrcpy Server Manager - Deploy and manage scrcpy-server.jar directly
Based on py-scrcpy-client approach for better socket control
"""

import asyncio
import subprocess
import logging
import time
import struct
import socket
from typing import Optional, Tuple
from adbutils import adb, AdbDevice, AdbConnection, Network

logger = logging.getLogger(__name__)

class DirectScrcpyServer:
    """Deploy and manage scrcpy-server.jar directly on device"""
    
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.device: Optional[AdbDevice] = None
        self.server_process: Optional[AdbConnection] = None
        self.video_socket: Optional[socket.socket] = None
        self.control_socket: Optional[socket.socket] = None
        self.device_name: Optional[str] = None
        self.resolution: Optional[Tuple[int, int]] = None
        
    async def start(self) -> bool:
        """Deploy server and establish connections"""
        try:
            # Connect to device
            if not await self._connect_device():
                return False
            
            # Deploy server
            if not await self._deploy_server():
                return False
            
            # Start server process
            if not await self._start_server():
                return False
            
            # Connect to sockets
            if not await self._connect_sockets():
                return False
            
            logger.info(f"Direct scrcpy server started successfully")
            logger.info(f"Device: {self.device_name}")
            logger.info(f"Resolution: {self.resolution}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start direct scrcpy server: {e}")
            await self.stop()
            return False
    
    async def stop(self):
        """Stop server and cleanup connections"""
        if self.video_socket:
            try:
                self.video_socket.close()
            except:
                pass
            self.video_socket = None
            
        if self.control_socket:
            try:
                self.control_socket.close()
            except:
                pass
            self.control_socket = None
            
        if self.server_process:
            try:
                self.server_process.close()
            except:
                pass
            self.server_process = None
            
        logger.info("Direct scrcpy server stopped")
    
    async def _connect_device(self) -> bool:
        """Connect to ADB device"""
        try:
            if self.device_id:
                self.device = adb.device(serial=self.device_id)
            else:
                devices = adb.device_list()
                if not devices:
                    logger.error("No ADB devices found")
                    return False
                self.device = devices[0]
            
            logger.info(f"Connected to device: {self.device.serial}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to device: {e}")
            return False
    
    async def _deploy_server(self) -> bool:
        """Deploy scrcpy-server.jar to device"""
        try:
            local_jar_path = "inference/scrcpy-server.jar"
            device_jar_path = "/data/local/tmp/scrcpy-server.jar"
            
            # Check if local jar exists
            import os
            if not os.path.exists(local_jar_path):
                logger.error(f"Local scrcpy-server.jar not found at {local_jar_path}")
                return False
            
            logger.info(f"Pushing {local_jar_path} to device...")
            
            # Push jar to device
            self.device.sync.push(local_jar_path, device_jar_path)
            
            # Verify deployment
            result = self.device.shell(f"ls -la {device_jar_path}", timeout=5)
            if "scrcpy-server.jar" not in result:
                logger.error("Failed to deploy scrcpy-server.jar to device")
                return False
            
            logger.info("scrcpy-server.jar deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy server: {e}")
            return False
    
    async def _start_server(self) -> bool:
        """Start scrcpy server process on device"""
        try:
            # Server command - back to app_process with proper parameter format
            # Based on py-scrcpy-client working implementation
            command = (
                "CLASSPATH=/data/local/tmp/scrcpy-server.jar "
                "app_process / com.genymobile.scrcpy.Server "
                "2.4 "
                "log_level=info "
                "max_size=1600 "
                "max_fps=60 "
                "video_bit_rate=20000000 "
                "video_encoder=OMX.google.h264.encoder "
                "video_codec=h264 "
                "tunnel_forward=false "
                "send_frame_meta=false "
                "control=true "
                "audio=false "
                "show_touches=false "
                "stay_awake=false "
                "power_off_on_close=false "
                "clipboard_autosync=false"
            )
            
            logger.info("Starting scrcpy server process in background...")
            
            # Start server process in background using nohup to prevent blocking
            # Need to use sh -c to properly handle environment variables with nohup
            background_command = f"nohup sh -c '{command}' > /dev/null 2>&1 &"
            result = self.device.shell(background_command, timeout=5)
            logger.info(f"Server start command result: {result}")
            
            # Wait for server to initialize
            logger.info("Waiting for server to initialize...")
            await asyncio.sleep(5)
            
            # Check if server process is running
            ps_result = self.device.shell("ps | grep scrcpy", timeout=5)
            if "scrcpy" in ps_result:
                logger.info("Scrcpy server process is running")
                return True
            else:
                logger.warning("Scrcpy server process not found in ps output")
                # Try alternative approach - check for java process
                java_result = self.device.shell("ps | grep java", timeout=5)
                if "com.genymobile.scrcpy.Server" in java_result:
                    logger.info("Found scrcpy server java process")
                    return True
                else:
                    logger.error("Scrcpy server process not found")
                    return False
            
        except Exception as e:
            logger.error(f"Failed to start server process: {e}")
            return False
    
    async def _connect_sockets(self) -> bool:
        """Connect to video and control sockets"""
        try:
            # Connect to video socket with retry
            for attempt in range(10):
                try:
                    self.video_socket = self.device.create_connection(
                        Network.LOCAL_ABSTRACT, "scrcpy"
                    )
                    break
                except Exception as e:
                    if attempt < 9:
                        logger.info(f"Socket connection attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(1)
                    else:
                        raise e
            
            # Read dummy byte
            dummy_byte = self.video_socket.recv(1)
            if not dummy_byte or dummy_byte != b"\x00":
                raise ConnectionError(f"Expected dummy byte \\x00, got: {dummy_byte}")
            
            logger.info("Received dummy byte successfully")
            
            # Connect control socket
            self.control_socket = self.device.create_connection(
                Network.LOCAL_ABSTRACT, "scrcpy"
            )
            
            # Read device name (64 bytes)
            device_name_bytes = self.video_socket.recv(64)
            self.device_name = device_name_bytes.decode("utf-8").rstrip("\x00")
            
            if not self.device_name:
                raise ConnectionError("Failed to receive device name")
            
            # Read resolution (4 bytes: width, height as uint16 big-endian)
            resolution_bytes = self.video_socket.recv(4)
            if len(resolution_bytes) != 4:
                raise ConnectionError("Failed to receive resolution data")
            
            width, height = struct.unpack(">HH", resolution_bytes)
            self.resolution = (width, height)
            
            # Set video socket to non-blocking for streaming
            self.video_socket.setblocking(False)
            
            logger.info("Socket connections established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect sockets: {e}")
            return False
    
    def read_video_data(self, size: int = 0x10000) -> bytes:
        """Read video data from socket (non-blocking)"""
        if not self.video_socket:
            return b""
        
        try:
            return self.video_socket.recv(size)
        except BlockingIOError:
            return b""
        except Exception as e:
            logger.error(f"Error reading video data: {e}")
            return b""
    
    def is_connected(self) -> bool:
        """Check if server is connected and ready"""
        return (self.video_socket is not None and 
                self.control_socket is not None and
                self.device_name is not None and
                self.resolution is not None)

# Test the direct server
async def test_direct_server():
    server = DirectScrcpyServer()
    
    try:
        if await server.start():
            logger.info("✓ Direct scrcpy server started successfully")
            
            # Test reading some video data
            data_count = 0
            for i in range(50):  # Try for 5 seconds
                data = server.read_video_data()
                if data:
                    data_count += 1
                    logger.info(f"Received video data: {len(data)} bytes")
                
                await asyncio.sleep(0.1)
            
            if data_count > 0:
                logger.info(f"✓ Successfully received {data_count} video data packets")
            else:
                logger.warning("✗ No video data received")
        else:
            logger.error("✗ Failed to start direct scrcpy server")
        
    finally:
        await server.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
                       datefmt='%H:%M:%S')
    asyncio.run(test_direct_server())
