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
            
            # With tunnel_forward=true, we need to establish socket connections FIRST
            # then start the server process
            if not await self._setup_sockets():
                return False
            
            # Start server process (it will connect to our established sockets)
            if not await self._start_server():
                return False
            
            # Complete the socket handshake
            if not await self._complete_socket_setup():
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
            scrcpy_version = "2.4"
            command = (
                f"export CLASSPATH=/data/local/tmp/scrcpy-server.jar && "
                f"app_process / com.genymobile.scrcpy.Server "
                f"{scrcpy_version} "
                f"log_level=info "
                f"max_size=1600 "
                f"max_fps=60 "
                f"video_bit_rate=20000000 "
                f"video_encoder=OMX.google.h264.encoder "
                f"video_codec=h264 "
                f"tunnel_forward=true "
                f"send_frame_meta=false "
                f"control=true "
                f"audio=false "
                f"show_touches=false "
                f"stay_awake=false "
                f"power_off_on_close=false "
                f"clipboard_autosync=false"
            )
            
            logger.info("Starting scrcpy server process in background...")
            
            # Start server in background immediately - it will wait for our connection
            logger.info("Starting server in background (it will wait for connection)...")
            background_command = f"nohup sh -c '{command}' > /data/local/tmp/scrcpy.log 2>&1 &"
            result = self.device.shell(background_command, timeout=5)
            logger.info(f"Background command result: {result}")
            
            # Give server a moment to start
            await asyncio.sleep(2)
            
            # Wait for server to initialize
            logger.info("Waiting for server to initialize...")
            await asyncio.sleep(10)
            
            # Read the log file to see what happened
            try:
                log_output = self.device.shell("cat /data/local/tmp/scrcpy.log", timeout=5)
                if log_output.strip():
                    logger.info(f"Server log output:\n{log_output}")
                else:
                    logger.info("No log output yet, waiting longer...")
                    await asyncio.sleep(10)
                    log_output = self.device.shell("cat /data/local/tmp/scrcpy.log", timeout=5)
                    if log_output.strip():
                        logger.info(f"Server log output (after delay):\n{log_output}")
            except Exception as e:
                logger.warning(f"Could not read server log: {e}")
            
            # Check if server process is running with multiple methods
            logger.info("Checking if server process is running...")
            
            # Method 1: Check for scrcpy processes
            ps_scrcpy = self.device.shell("ps | grep scrcpy", timeout=5)
            logger.info(f"ps | grep scrcpy: {ps_scrcpy}")
            
            # Method 2: Check for app_process with our jar
            ps_app = self.device.shell("ps | grep app_process", timeout=5)
            logger.info(f"ps | grep app_process: {ps_app}")
            
            # Method 3: Check for any process using our jar file
            ps_jar = self.device.shell("ps | grep scrcpy-server.jar", timeout=5)
            logger.info(f"ps | grep scrcpy-server.jar: {ps_jar}")
            
            # Method 4: Test if we can execute the command directly (without nohup)
            logger.info("Testing direct command execution (will timeout, but shows if command works)...")
            try:
                direct_test = self.device.shell(f"timeout 3 sh -c '{command}'", timeout=5)
                logger.info(f"Direct command (3s timeout): {direct_test}")
            except Exception as e:
                logger.info(f"Direct command test exception (expected): {e}")
            
            # Method 5: Check if nohup is available and working
            nohup_test = self.device.shell("which nohup", timeout=5)
            logger.info(f"nohup availability: {nohup_test}")
            
            # Method 6: Test a simple nohup command
            simple_nohup = self.device.shell("nohup echo 'test' > /data/local/tmp/nohup_test.log 2>&1 &", timeout=5)
            logger.info(f"Simple nohup test result: {simple_nohup}")
            
            await asyncio.sleep(1)
            nohup_log = self.device.shell("cat /data/local/tmp/nohup_test.log", timeout=5)
            logger.info(f"Simple nohup log: {nohup_log}")
            
            # Method 7: Test just the CLASSPATH export
            logger.info("Testing CLASSPATH export...")
            classpath_test = self.device.shell("export CLASSPATH=/data/local/tmp/scrcpy-server.jar && echo $CLASSPATH", timeout=5)
            logger.info(f"CLASSPATH test: {classpath_test}")
            
            # Method 8: Test app_process without scrcpy parameters
            logger.info("Testing basic app_process...")
            try:
                app_test = self.device.shell("timeout 2 sh -c 'export CLASSPATH=/data/local/tmp/scrcpy-server.jar && app_process / com.genymobile.scrcpy.Server --help'", timeout=5)
                logger.info(f"App process test: {app_test}")
            except Exception as e:
                logger.info(f"App process test exception: {e}")
            
            # Method 9: Test if the jar file is valid
            jar_test = self.device.shell("file /data/local/tmp/scrcpy-server.jar", timeout=5)
            logger.info(f"JAR file check: {jar_test}")
            
            if "scrcpy" in ps_scrcpy or "app_process" in ps_app or "scrcpy-server.jar" in ps_jar:
                logger.info("Server process found!")
                return True
            else:
                logger.error("Server process not found with any method")
                return False
            
        except Exception as e:
            logger.error(f"Failed to start server process: {e}")
            return False
    
    async def _setup_sockets(self) -> bool:
        """Setup ADB port forwarding for tunnel_forward=true"""
        try:
            logger.info("Setting up ADB port forwarding...")
            
            # Set up port forwarding for video socket
            video_port = 27183  # Default scrcpy video port
            control_port = 27184  # Default scrcpy control port
            
            # Forward local ports to device abstract sockets
            self.device.forward(f"tcp:{video_port}", "localabstract:scrcpy")
            logger.info(f"Video port forwarding: tcp:{video_port} -> localabstract:scrcpy")
            
            self.device.forward(f"tcp:{control_port}", "localabstract:scrcpy") 
            logger.info(f"Control port forwarding: tcp:{control_port} -> localabstract:scrcpy")
            
            # Store ports for later use
            self.video_port = video_port
            self.control_port = control_port
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup port forwarding: {e}")
            return False
    
    async def _complete_socket_setup(self) -> bool:
        """Complete socket setup after server connects"""
        try:
            import socket
            
            # Connect to the forwarded ports
            logger.info("Connecting to forwarded ports...")
            
            # Connect to video socket via TCP
            self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.video_socket.connect(("127.0.0.1", self.video_port))
            logger.info("Connected to video socket")
            
            # Connect to control socket via TCP
            self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_socket.connect(("127.0.0.1", self.control_port))
            logger.info("Connected to control socket")
            
            # Read dummy byte
            dummy_byte = self.video_socket.recv(1)
            if not dummy_byte or dummy_byte != b"\x00":
                raise ConnectionError(f"Expected dummy byte \\x00, got: {dummy_byte}")
            
            logger.info("Received dummy byte successfully")
            
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
            
            logger.info("Socket setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete socket setup: {e}")
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
