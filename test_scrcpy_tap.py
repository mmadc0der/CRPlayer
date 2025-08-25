#!/usr/bin/env python3
"""
Test tapping into existing scrcpy video stream using different approaches
"""

import asyncio
import logging
import subprocess
import socket
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_scrcpy_headless():
    """Test starting scrcpy in headless mode for our exclusive use"""
    logger.info("Testing headless scrcpy approach...")
    
    # Kill existing scrcpy
    try:
        subprocess.run(['pkill', '-f', 'scrcpy'], check=False)
        await asyncio.sleep(2)
    except:
        pass
    
    # Start scrcpy in headless mode (no display window)
    logger.info("Starting headless scrcpy...")
    try:
        # Use --no-playback to disable video display but keep streaming
        process = subprocess.Popen([
            'scrcpy', 
            '--no-audio',
            '--no-control', 
            '--no-playback',  # This disables the display window
            '--max-fps=30',
            '--max-size=1280',
            '--video-bit-rate=8M'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        await asyncio.sleep(3)
        
        # Check if process is running
        if process.poll() is None:
            logger.info("Headless scrcpy started successfully")
            
            # Now try to connect
            from inference.scrcpy_socket import ScrcpySocketDemux
            demux = ScrcpySocketDemux()
            
            try:
                if await demux.connect():
                    logger.info("Successfully connected to headless scrcpy!")
                    
                    # Try to read a few frames
                    frame_count = 0
                    async for nal_unit in demux.stream_nal_units():
                        frame_count += 1
                        logger.info(f"Frame {frame_count}: {len(nal_unit.data)} bytes")
                        if frame_count >= 5:
                            break
                    
                    await demux.disconnect()
                    return True
                else:
                    logger.error("Failed to connect to headless scrcpy")
                    
            except Exception as e:
                logger.error(f"Error with headless scrcpy: {e}")
            finally:
                await demux.disconnect()
        
        # Clean up
        process.terminate()
        process.wait()
        
    except Exception as e:
        logger.error(f"Failed to start headless scrcpy: {e}")
    
    return False

async def test_scrcpy_server_only():
    """Test connecting directly to scrcpy-server without client"""
    logger.info("Testing server-only approach...")
    
    # Kill existing scrcpy
    try:
        subprocess.run(['pkill', '-f', 'scrcpy'], check=False)
        await asyncio.sleep(2)
    except:
        pass
    
    try:
        # Push server and start it manually
        logger.info("Starting scrcpy-server directly...")
        
        # Create a custom server command
        server_cmd = [
            'adb', 'shell',
            'CLASSPATH=/data/local/tmp/scrcpy-server.jar',
            'app_process', '/', 'com.genymobile.scrcpy.Server',
            '3.3.1',  # version
            'scid=12345678',  # custom SCID
            'log_level=info',
            'video_bit_rate=8000000',
            'audio=false',
            'max_size=1280',
            'max_fps=30'
        ]
        
        # Start server
        process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Create forward for our custom SCID
        await asyncio.sleep(2)
        subprocess.run(['adb', 'forward', 'tcp:27185', 'localabstract:scrcpy_12345678'], check=True)
        
        # Wait for server to be ready
        await asyncio.sleep(3)
        
        # Try to connect
        logger.info("Attempting to connect to server-only scrcpy...")
        
        try:
            reader, writer = await asyncio.open_connection('localhost', 27185)
            logger.info("Connected to server-only scrcpy!")
            
            # Try to read some data
            data = await asyncio.wait_for(reader.read(100), timeout=5.0)
            if data:
                logger.info(f"Received data: {len(data)} bytes: {data.hex()}")
                return True
            else:
                logger.warning("No data received")
                
        except Exception as e:
            logger.error(f"Connection failed: {e}")
        
        # Cleanup
        process.terminate()
        subprocess.run(['adb', 'forward', '--remove', 'tcp:27185'], check=False)
        
    except Exception as e:
        logger.error(f"Server-only test failed: {e}")
    
    return False

async def main():
    """Test different approaches to get scrcpy video data"""
    
    logger.info("=== Testing Scrcpy Video Access Approaches ===")
    
    # Test 1: Headless scrcpy
    if await test_scrcpy_headless():
        logger.info("SUCCESS: Headless scrcpy approach works!")
        return
    
    # Test 2: Server-only approach
    if await test_scrcpy_server_only():
        logger.info("SUCCESS: Server-only approach works!")
        return
    
    logger.error("All approaches failed - scrcpy video access not working")

if __name__ == "__main__":
    asyncio.run(main())
