#!/usr/bin/env python3
"""
Debug scrcpy socket connections to understand the actual protocol
"""

import subprocess
import socket
import time

def check_scrcpy_processes():
    """Check running scrcpy processes"""
    print("=== Scrcpy Processes ===")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'scrcpy' in line.lower():
                print(f"Process: {line.strip()}")
    except:
        print("Cannot check processes on Windows")

def check_adb_forwards():
    """Check ADB forwards"""
    print("\n=== ADB Forwards ===")
    result = subprocess.run(['adb', 'forward', '--list'], capture_output=True, text=True)
    print(f"ADB forwards:\n{result.stdout}")
    
    return result.stdout

def check_device_sockets():
    """Check sockets on device"""
    print("\n=== Device Sockets ===")
    try:
        result = subprocess.run(['adb', 'shell', 'netstat', '-l'], capture_output=True, text=True)
        print("Device listening sockets:")
        for line in result.stdout.split('\n'):
            if 'scrcpy' in line:
                print(f"  {line.strip()}")
    except:
        print("Cannot check device sockets")

def test_socket_connection(port):
    """Test basic socket connection"""
    print(f"\n=== Testing Socket Connection to Port {port} ===")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        print(f"Connecting to localhost:{port}...")
        sock.connect(('localhost', port))
        print("Connected successfully")
        
        # Try to read immediately
        sock.settimeout(2.0)
        try:
            data = sock.recv(1024)
            if data:
                print(f"Received {len(data)} bytes immediately: {data.hex()}")
            else:
                print("No immediate data")
        except socket.timeout:
            print("No immediate data (timeout)")
        
        # Keep connection open and try again
        print("Waiting 3 seconds...")
        time.sleep(3)
        
        try:
            data = sock.recv(1024)
            if data:
                print(f"Received {len(data)} bytes after wait: {data.hex()}")
            else:
                print("Socket closed by remote")
        except socket.timeout:
            print("Still no data after wait")
        
        sock.close()
        
    except Exception as e:
        print(f"Connection failed: {e}")

def main():
    check_scrcpy_processes()
    forwards = check_adb_forwards()
    check_device_sockets()
    
    # Extract port from forwards
    for line in forwards.split('\n'):
        if 'localabstract:scrcpy' in line:
            parts = line.split()
            if len(parts) >= 2:
                port_part = parts[1] if len(parts) == 3 else parts[0]
                if 'tcp:' in port_part:
                    port = int(port_part.split(':')[1])
                    test_socket_connection(port)
                    break

if __name__ == "__main__":
    main()
