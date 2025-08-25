#!/usr/bin/env python3
"""
Quick test script to test socket connection with improved detection
"""

import asyncio
import logging
import sys
from inference.scrcpy_socket import ScrcpySocketDemux

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

async def test_connection():
    """Quick connection test"""
    print("Testing scrcpy socket connection...")
    
    demux = ScrcpySocketDemux()
    
    try:
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

if __name__ == "__main__":
    asyncio.run(test_connection())
