#!/usr/bin/env python3
"""
Debug script to investigate scrcpy socket connection issues
Sets up comprehensive logging to trace all connection steps
"""

import asyncio
import logging
import sys
import subprocess
from inference.scrcpy_socket import ScrcpySocketDemux

def setup_debug_logging():
    """Setup comprehensive debug logging"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # Setup file handler for detailed logs
    file_handler = logging.FileHandler('scrcpy_debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Also configure specific loggers
    logging.getLogger('inference.scrcpy_socket').setLevel(logging.DEBUG)
    logging.getLogger('asyncio').setLevel(logging.INFO)  # Reduce asyncio noise
    
    print("Debug logging enabled - logs will be saved to scrcpy_debug.log")

def check_prerequisites():
    """Check if scrcpy and adb are available"""
    print("\n=== Checking Prerequisites ===")
    
    # Check ADB
    try:
        result = subprocess.run(['adb', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ ADB available: {result.stdout.split()[4]}")
        else:
            print("✗ ADB not working")
            return False
    except FileNotFoundError:
        print("✗ ADB not found in PATH")
        return False
    
    # Check scrcpy
    try:
        result = subprocess.run(['scrcpy', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip().split()[1] if len(result.stdout.split()) > 1 else "unknown"
            print(f"✓ Scrcpy available: {version}")
        else:
            print("✗ Scrcpy not working")
            return False
    except FileNotFoundError:
        print("✗ Scrcpy not found in PATH")
        return False
    
    # Check device connection
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        devices = [line for line in result.stdout.split('\n')[1:] if line.strip() and 'device' in line]
        if devices:
            print(f"✓ {len(devices)} device(s) connected:")
            for device in devices:
                print(f"  - {device}")
        else:
            print("✗ No devices connected")
            return False
    except Exception as e:
        print(f"✗ Error checking devices: {e}")
        return False
    
    return True

def check_running_scrcpy():
    """Check for running scrcpy processes"""
    print("\n=== Checking Running Scrcpy Processes ===")
    
    try:
        import os
        if os.name == 'nt':
            result = subprocess.run(['wmic', 'process', 'get', 'commandline'], 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        scrcpy_processes = []
        for line in result.stdout.split('\n'):
            if 'scrcpy' in line.lower():
                scrcpy_processes.append(line.strip())
        
        if scrcpy_processes:
            print(f"Found {len(scrcpy_processes)} scrcpy-related processes:")
            for i, proc in enumerate(scrcpy_processes):
                print(f"  {i+1}: {proc[:100]}..." if len(proc) > 100 else f"  {i+1}: {proc}")
        else:
            print("No scrcpy processes found")
            
        # Look specifically for scrcpy-server.jar
        server_processes = [line for line in result.stdout.split('\n') 
                          if 'scrcpy-server.jar' in line]
        
        if server_processes:
            print(f"\nFound {len(server_processes)} scrcpy-server processes:")
            for i, proc in enumerate(server_processes):
                print(f"  Server {i+1}: {proc}")
                # Extract SCID if present
                if 'scid=' in proc:
                    parts = proc.split()
                    for part in parts:
                        if part.startswith('scid='):
                            scid = part.split('=')[1]
                            print(f"    SCID: {scid}")
        else:
            print("No scrcpy-server.jar processes found")
            print("You may need to start scrcpy first!")
            
    except Exception as e:
        print(f"Error checking processes: {e}")

async def test_connection():
    """Test scrcpy socket connection with detailed logging"""
    print("\n=== Testing Scrcpy Socket Connection ===")
    
    demux = ScrcpySocketDemux()
    
    try:
        print("Attempting to connect to scrcpy socket...")
        success = await demux.connect(max_retries=1)
        
        if success:
            print("✓ Connection successful!")
            
            # Try to read a few frames
            print("Testing frame reading...")
            frame_count = 0
            async for nal_unit in demux.stream_nal_units():
                frame_count += 1
                print(f"Frame {frame_count}: {len(nal_unit.data)} bytes, "
                      f"keyframe={nal_unit.is_keyframe}")
                
                if frame_count >= 3:
                    break
            
            # Show stats
            stats = demux.get_stats()
            print(f"Stats: {stats}")
            
        else:
            print("✗ Connection failed")
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        logging.exception("Detailed error:")
    finally:
        await demux.disconnect()

async def main():
    """Main debug routine"""
    print("=== Scrcpy Connection Debug Tool ===")
    
    # Setup logging
    setup_debug_logging()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease fix the prerequisites before continuing.")
        return
    
    # Check running processes
    check_running_scrcpy()
    
    # Test connection
    await test_connection()
    
    print("\n=== Debug Complete ===")
    print("Check scrcpy_debug.log for detailed logs")

if __name__ == "__main__":
    asyncio.run(main())
