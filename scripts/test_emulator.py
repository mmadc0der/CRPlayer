#!/usr/bin/env python3
"""
Test script for Android Emulator in Docker
Проверяет подключение к эмулятору, захват экрана и базовые операции
"""

import time
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, timeout=30):
    """Execute command and return result"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_adb_connection(host="localhost", port=5555):
    """Test ADB connection to emulator"""
    print(f"🔌 Testing ADB connection to {host}:{port}...")
    
    # Connect to emulator
    success, stdout, stderr = run_command(f"adb connect {host}:{port}")
    if not success:
        print(f"❌ Failed to connect: {stderr}")
        return False
    
    # List devices
    success, stdout, stderr = run_command("adb devices")
    if not success or f"{host}:{port}" not in stdout:
        print(f"❌ Device not found in ADB devices list")
        return False
    
    print(f"✅ ADB connection successful")
    return True

def test_emulator_boot(timeout=300):
    """Test if emulator has fully booted"""
    print("🚀 Checking emulator boot status...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        success, stdout, stderr = run_command("adb shell getprop sys.boot_completed")
        if success and "1" in stdout.strip():
            print("✅ Emulator fully booted")
            return True
        
        print(f"⏳ Waiting for boot completion... ({int(time.time() - start_time)}s)")
        time.sleep(10)
    
    print("❌ Emulator boot timeout")
    return False

def test_screen_capture():
    """Test screen capture functionality"""
    print("📱 Testing screen capture...")
    
    # Take screenshot on device
    success, stdout, stderr = run_command("adb shell screencap -p /sdcard/test_capture.png")
    if not success:
        print(f"❌ Failed to capture screen: {stderr}")
        return False
    
    # Pull screenshot to local machine
    os.makedirs("test_output", exist_ok=True)
    success, stdout, stderr = run_command("adb pull /sdcard/test_capture.png test_output/")
    if not success:
        print(f"❌ Failed to pull screenshot: {stderr}")
        return False
    
    # Check if file exists and has reasonable size
    screenshot_path = Path("test_output/test_capture.png")
    if not screenshot_path.exists():
        print("❌ Screenshot file not found")
        return False
    
    file_size = screenshot_path.stat().st_size
    if file_size < 1000:  # Less than 1KB is probably an error
        print(f"❌ Screenshot file too small: {file_size} bytes")
        return False
    
    print(f"✅ Screen capture successful ({file_size} bytes)")
    return True

def test_input_simulation():
    """Test input simulation (tap, swipe)"""
    print("👆 Testing input simulation...")
    
    # Get screen size
    success, stdout, stderr = run_command("adb shell wm size")
    if not success:
        print(f"❌ Failed to get screen size: {stderr}")
        return False
    
    # Parse screen size (format: "Physical size: 1080x1920")
    try:
        size_line = stdout.strip().split(": ")[-1]
        width, height = map(int, size_line.split("x"))
        print(f"📏 Screen size: {width}x{height}")
    except:
        print("❌ Failed to parse screen size")
        return False
    
    # Test tap in center
    center_x, center_y = width // 2, height // 2
    success, stdout, stderr = run_command(f"adb shell input tap {center_x} {center_y}")
    if not success:
        print(f"❌ Failed to simulate tap: {stderr}")
        return False
    
    # Test swipe
    success, stdout, stderr = run_command(f"adb shell input swipe {center_x} {center_y} {center_x} {center_y-200} 500")
    if not success:
        print(f"❌ Failed to simulate swipe: {stderr}")
        return False
    
    print("✅ Input simulation successful")
    return True

def test_vnc_connection(host="localhost", port=5900):
    """Test VNC connection availability"""
    print(f"🖥️  Testing VNC connection to {host}:{port}...")
    
    # Try to connect to VNC port
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("✅ VNC port is accessible")
            print(f"💡 Connect with VNC client to: {host}:{port}")
            return True
        else:
            print("❌ VNC port not accessible")
            return False
    except Exception as e:
        print(f"❌ VNC connection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("🧪 CRPlayer Emulator Test Suite")
    print("=" * 50)
    
    tests = [
        ("ADB Connection", lambda: test_adb_connection()),
        ("Emulator Boot", lambda: test_emulator_boot()),
        ("Screen Capture", lambda: test_screen_capture()),
        ("Input Simulation", lambda: test_input_simulation()),
        ("VNC Connection", lambda: test_vnc_connection()),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Emulator is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
