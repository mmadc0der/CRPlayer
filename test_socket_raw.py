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
    
    # Get all ADB forwards
    result = subprocess.run(['adb', 'forward', '--list'], capture_output=True, text=True)
    
    # Try to find video socket - scrcpy might use different naming
    video_ports = []
    control_ports = []
    
    for line in result.stdout.strip().split('\n'):
        if 'localabstract:scrcpy' in line:
            parts = line.split()
            if len(parts) >= 2:
                port = int(parts[1].split(':')[-1])
                if 'video' in line:
                    video_ports.append(port)
                else:
                    control_ports.append(port)
    
    # Test control socket first (to confirm it's control-only)
    if control_ports:
        print(f"🎮 Testing control socket on port {control_ports[0]}...")
        if test_single_port(control_ports[0], "control"):
            print("✅ Control socket has data (unexpected!)")
        else:
            print("❌ Control socket has no video data (expected)")
    
    # Look for video socket - scrcpy 3.x might use different approach
    print(f"\n🎥 Looking for video socket...")
    
    # Try common video socket patterns
    video_socket_names = [
        'localabstract:scrcpy_video',
        'localabstract:scrcpy-video', 
        'localabstract:scrcpy_stream',
        'localfilesystem:/tmp/scrcpy_video'
    ]
    
    for socket_name in video_socket_names:
        print(f"🔍 Trying to create forward for {socket_name}...")
        try:
            # Try to create forward for video socket
            port = 27185  # Use different port
            result = subprocess.run([
                'adb', 'forward', f'tcp:{port}', socket_name
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Created forward for {socket_name} on port {port}")
                if test_single_port(port, "video"):
                    return True
                # Clean up
                subprocess.run(['adb', 'forward', '--remove', f'tcp:{port}'], capture_output=True)
            else:
                print(f"❌ Failed to forward {socket_name}: {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ Error testing {socket_name}: {e}")
    
    print("❌ No video socket found with standard names")
    return False

def test_single_port(port, socket_type):
    """Test a single port for data"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        sock.connect(('localhost', port))
        sock.settimeout(2.0)
        
        data = sock.recv(1024)
        if data:
            print(f"✅ {socket_type} socket port {port}: received {len(data)} bytes!")
            print(f"📊 First 32 bytes: {data[:32].hex()}")
            return True
        else:
            print(f"❌ {socket_type} socket port {port}: no data")
            return False
            
    except Exception as e:
        print(f"❌ {socket_type} socket port {port}: {e}")
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
    
    # Check ALL ADB forwards (not just scrcpy)
    result = subprocess.run(['adb', 'forward', '--list'], capture_output=True, text=True)
    print(f"\n📋 ALL ADB forwards:")
    if result.stdout.strip():
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")
            # Look for video socket patterns
            if 'scrcpy' in line and 'video' in line:
                print("    ^ This might be the video socket!")
    else:
        print("  None")
    
    # Check device-side sockets
    print(f"\n📱 Checking device sockets...")
    result = subprocess.run(['adb', 'shell', 'ls', '/data/local/tmp/'], capture_output=True, text=True)
    if 'scrcpy' in result.stdout:
        print("✅ scrcpy server files found on device")
    
    # Check listening ports
    result = subprocess.run(['netstat', '-tlpn'], capture_output=True, text=True)
    scrcpy_ports = [line for line in result.stdout.split('\n') if '271' in line and 'LISTEN' in line]
    
    if scrcpy_ports:
        print(f"\n🔌 Listening ports (271xx range):")
        for port in scrcpy_ports:
            print(f"  {port}")
    else:
        print("\n❌ No scrcpy ports listening")
    
    # Try to find video socket
    print(f"\n🎥 Looking for video socket...")
    # scrcpy typically uses scrcpy_video for video stream
    result = subprocess.run(['adb', 'shell', 'netstat', '-l'], capture_output=True, text=True)
    if result.returncode == 0:
        video_sockets = [line for line in result.stdout.split('\n') if 'scrcpy' in line]
        if video_sockets:
            print("📱 Device-side scrcpy sockets:")
            for sock in video_sockets:
                print(f"  {sock}")
        else:
            print("❌ No scrcpy sockets found on device")

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
