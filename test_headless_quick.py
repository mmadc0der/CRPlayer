#!/usr/bin/env python3
"""
Quick test for headless scrcpy with improved protocol handling
"""

import asyncio
import logging
import sys
from scrcpy_headless import ScrcpyHeadlessManager
from inference.scrcpy_socket import ScrcpySocketDemux

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

async def test_headless():
    """Test headless scrcpy with improved protocol"""
    print("Testing headless scrcpy connection...")
    
    manager = ScrcpyHeadlessManager()
    demux = ScrcpySocketDemux()
    
    try:
        # Start headless scrcpy
        print("Starting headless scrcpy...")
        if not await manager.start():
            print("✗ Failed to start headless scrcpy")
            return
        
        print(f"✓ Headless scrcpy started with SCID: {manager.scid}")
        
        # Wait longer for scrcpy to be fully ready
        print("Waiting for scrcpy to be fully ready...")
        await asyncio.sleep(5)
        
        # Override SCID detection
        async def get_scid():
            return manager.scid
        demux._get_scrcpy_scid = get_scid
        
        # Try connection
        print("Attempting socket connection...")
        success = await demux.connect(max_retries=1)
        
        if success:
            print("✓ Connection successful!")
            
            # Try to read one frame
            print("Testing frame reading...")
            async for nal_unit in demux.stream_nal_units():
                print(f"✓ First frame: {len(nal_unit.data)} bytes, keyframe={nal_unit.is_keyframe}")
                break
            
            stats = demux.get_stats()
            print(f"Stats: {stats}")
            
        else:
            print("✗ Connection failed")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        logging.exception("Detailed error:")
    finally:
        await demux.disconnect()
        await manager.stop()

if __name__ == "__main__":
    asyncio.run(test_headless())
