#!/usr/bin/env python3
"""
Raw socket test to debug scrcpy connection
"""

import socket
import subprocess
import time
import sys

def test_raw_socket_connection():
    """Test raw socket connection to scrcpy"""
    
    # Get scrcpy port
    result = subprocess.run(['adb', 'forward', '--list'], capture_output=True, text=True)
    scrcpy_port = None
    
    for line in result.stdout.strip().split('\n'):
        if 'localabstract:scrcpy' in line:
            parts = line.split()
            if len(parts) >= 2:
                scrcpy_port = int(parts[1].split(':')[-1])
                break
    
    if not scrcpy_port:
        print("❌ No scrcpy forward found")
        print("Make sure scrcpy is running with video enabled")
        return False
    
    print(f"✅ Found scrcpy forward on port: {scrcpy_port}")
    
    # Test socket connection
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        print(f"🔌 Connecting to localhost:{scrcpy_port}...")
        sock.connect(('localhost', scrcpy_port))
        print("✅ Socket connected successfully")
        
        # Try to read some data
        sock.settimeout(3.0)
        print("📡 Waiting for data...")
        
        try:
            data = sock.recv(1024)
            if data:
                print(f"✅ Received {len(data)} bytes!")
                print(f"📊 First 32 bytes: {data[:32].hex()}")
                print(f"📊 Raw data preview: {data[:64]}")
                return True
            else:
                print("❌ No data received (empty read)")
                return False
                
        except socket.timeout:
            print("❌ Timeout waiting for data")
            return False
            
    except Exception as e:
        print(f"❌ Socket error: {e}")
        return False
    finally:
        sock.close()

def check_scrcpy_status():
    """Check if scrcpy is running properly"""
    print("\n=== Checking scrcpy status ===")
    
    # Check processes
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    scrcpy_processes = [line for line in result.stdout.split('\n') if 'scrcpy' in line.lower()]
    
    if scrcpy_processes:
        print("✅ Found scrcpy processes:")
        for proc in scrcpy_processes:
            print(f"  {proc}")
    else:
        print("❌ No scrcpy processes found")
    
    # Check ADB forwards
    result = subprocess.run(['adb', 'forward', '--list'], capture_output=True, text=True)
    print(f"\n📋 ADB forwards:")
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
    else:
        print("  None")
    
    # Check listening ports
    result = subprocess.run(['netstat', '-tlpn'], capture_output=True, text=True)
    scrcpy_ports = [line for line in result.stdout.split('\n') if '271' in line and 'LISTEN' in line]
    
    if scrcpy_ports:
        print(f"\n🔌 Listening ports (271xx range):")
        for port in scrcpy_ports:
            print(f"  {port}")
    else:
        print("\n❌ No scrcpy ports listening")

def main():
    print("🔍 Testing scrcpy socket connection...")
    
    check_scrcpy_status()
    
    print("\n=== Testing socket connection ===")
    success = test_raw_socket_connection()
    
    if success:
        print("\n✅ Socket test PASSED - scrcpy is streaming data")
    else:
        print("\n❌ Socket test FAILED - no data from scrcpy")
        print("\n💡 Troubleshooting:")
        print("1. Make sure scrcpy is running WITHOUT --no-playback")
        print("2. Try: scrcpy --no-audio --max-fps=60 --max-size=1600")
        print("3. Check if device screen is on and unlocked")

if __name__ == "__main__":
    main()
