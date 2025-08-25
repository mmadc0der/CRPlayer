#!/usr/bin/env python3
"""
Test the updated scrcpy socket implementation with proper protocol handling
"""

import asyncio
import logging
from inference.scrcpy_socket import ScrcpySocketDemux

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_scrcpy_connection():
    """Test scrcpy socket connection with proper protocol"""
    demux = ScrcpySocketDemux()
    
    try:
        # Connect to scrcpy
        logger.info("Connecting to scrcpy...")
        if not await demux.connect():
            logger.error("Failed to connect to scrcpy")
            return False
        
        logger.info("✅ Connected successfully!")
        
        # Try to read a few frames
        logger.info("Reading frames...")
        frame_count = 0
        
        async for nal_unit in demux.stream_nal_units():
            frame_count += 1
            logger.info(f"Frame {frame_count}: {len(nal_unit.data)} bytes, "
                       f"keyframe={nal_unit.is_keyframe}, "
                       f"timestamp={nal_unit.timestamp:.3f}s")
            
            # Stop after 10 frames for testing
            if frame_count >= 10:
                break
        
        # Print statistics
        stats = demux.get_stats()
        logger.info(f"Statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    finally:
        await demux.disconnect()

if __name__ == "__main__":
    asyncio.run(test_scrcpy_connection())
