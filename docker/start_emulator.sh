#!/bin/bash

# Start Emulator Script for CRPlayer (без KVM зависимости)
set -e

echo "Starting CRPlayer Android Emulator (Software Mode)..."

# Set display for headless operation
export DISPLAY=:99

# Start Xvfb (virtual framebuffer)
echo "Starting X11 virtual framebuffer..."
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for X server to start
sleep 5

# Test X server
echo "Testing X11 server..."
DISPLAY=:99 xdpyinfo >/dev/null 2>&1 && echo "X11 server is running" || echo "X11 server failed to start"

# Start window manager
echo "Starting window manager..."
DISPLAY=:99 fluxbox &
WM_PID=$!

# Wait for window manager
sleep 3

# Start VNC server for remote access (bind to all interfaces for external access)
echo "Starting VNC server..."
x11vnc -display :99 -nopw -listen 0.0.0.0 -xkb -noxdamage -forever -shared -bg -rfbport 5900 -o /tmp/x11vnc.log

# Verify VNC is running
sleep 2
if pgrep x11vnc > /dev/null; then
    echo "VNC server started successfully"
else
    echo "VNC server failed to start"
    cat /tmp/x11vnc.log 2>/dev/null || echo "No VNC log available"
fi

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $EMULATOR_PID 2>/dev/null || true
    kill $WM_PID 2>/dev/null || true
    kill $XVFB_PID 2>/dev/null || true
    exit 0
}

# Set trap for cleanup
trap cleanup SIGTERM SIGINT

# Kill any existing ADB server to avoid conflicts
echo "Stopping any existing ADB server..."
adb kill-server 2>/dev/null || true
sleep 2

# Start ADB server listening on all interfaces (for host to reach container ADB server)
echo "Starting ADB server (0.0.0.0:5037)..."
unset ADB_SERVER_SOCKET
adb kill-server 2>/dev/null || true
nohup adb -a -P 5037 server nodaemon >/tmp/adb_server.log 2>&1 &
sleep 3

# Debug: Check emulator availability
echo "=== Debug Information ==="
echo "PATH: $PATH"
echo "ANDROID_SDK_ROOT: $ANDROID_SDK_ROOT"
echo "Checking emulator command..."
which emulator || echo "emulator not found in PATH"
ls -la $ANDROID_SDK_ROOT/emulator/emulator 2>/dev/null || echo "emulator binary not found"
$ANDROID_SDK_ROOT/emulator/emulator -version 2>/dev/null || echo "emulator version check failed"

# Check if KVM is available
KVM_AVAILABLE=false
if [ -e /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
    echo "KVM device found and accessible, enabling hardware acceleration"
    KVM_AVAILABLE=true
else
    echo "KVM device not accessible, using software acceleration"
    echo "KVM status: $(ls -la /dev/kvm 2>/dev/null || echo 'not found')"
    KVM_AVAILABLE=false
fi

# Start Android emulator
echo "Starting Android emulator..."
if [ "$KVM_AVAILABLE" = true ]; then
    # With hardware acceleration
    echo "Starting emulator with KVM acceleration..."
    DISPLAY=:99 $ANDROID_SDK_ROOT/emulator/emulator -avd ClashRoyale_AVD \
        -no-audio \
        -gpu angle_indirect \
        -memory ${EMULATOR_RAM:-4096} \
        -partition-size ${EMULATOR_PARTITION:-8192} \
        -no-snapshot-save \
        -no-snapshot-load \
        -wipe-data \
        -port 5554 \
        -accel on \
        -feature -Vulkan \
        -camera-back webcam0 \
        -camera-front webcam0 \
        -skip-adb-auth \
        -verbose &
else
    # Software only mode
    echo "Starting emulator in software mode..."
    DISPLAY=:99 $ANDROID_SDK_ROOT/emulator/emulator -avd ClashRoyale_AVD \
        -no-audio \
        -gpu swiftshader_indirect \
        -memory ${EMULATOR_RAM:-3072} \
        -partition-size ${EMULATOR_PARTITION:-6144} \
        -no-snapshot-save \
        -no-snapshot-load \
        -wipe-data \
        -port 5554 \
        -accel off \
        -skip-adb-auth \
        -verbose &
fi

EMULATOR_PID=$!
echo "Emulator PID: $EMULATOR_PID"

# Check if emulator process is running
sleep 5
if kill -0 $EMULATOR_PID 2>/dev/null; then
    echo "Emulator process is running"
else
    echo "ERROR: Emulator process died immediately!"
    exit 1
fi

# Wait for emulator to boot and establish ADB connection
echo "Waiting for emulator to boot and establishing ADB connection..."

# First, wait for emulator to be responsive
timeout=300
echo "Phase 1: Waiting for emulator to become responsive..."
while [ $timeout -gt 0 ]; do
    # Restart ADB server if needed
    if ! pgrep -f "adb.*server" >/dev/null; then
        echo "Restarting ADB server..."
        adb kill-server 2>/dev/null || true
        sleep 2
        adb start-server
        sleep 3
    fi
    
    # Try to detect emulator
    if adb devices | grep -q "emulator-5554"; then
        echo "✅ Emulator detected by ADB"
        break
    fi
    
    echo "Waiting for emulator detection... ($timeout seconds remaining)"
    sleep 5
    timeout=$((timeout - 5))
done

if [ $timeout -le 0 ]; then
    echo "❌ Emulator detection timeout"
    echo "Available ADB devices:"
    adb devices -l
    exit 1
fi

# Phase 2: Wait for Android boot completion
echo "Phase 2: Waiting for Android system boot..."
timeout=300
while [ $timeout -gt 0 ]; do
    if adb -s emulator-5554 shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; then
        echo "✅ Android system booted successfully"
        break
    fi
    echo "Waiting for Android boot... ($timeout seconds remaining)"
    sleep 10
    timeout=$((timeout - 10))
done

if [ $timeout -le 0 ]; then
    echo "⚠️ Android boot timeout, but continuing..."
fi

# Phase 3: Enable network ADB mode (do not auto-connect from inside container)
echo "Phase 3: Enabling network ADB mode on port 5555 (host should connect)..."
adb -s emulator-5554 tcpip 5555
sleep 3

# Ensure container ADB server does not hold a connection to localhost:5555
echo "Disconnecting any local ADB TCP connections inside container..."
adb disconnect localhost:5555 2>/dev/null || true
adb disconnect :5555 2>/dev/null || true
sleep 1

# Stop container ADB server so only host connects over mapped port
echo "Stopping container ADB server to free TCP device for host..."
adb kill-server 2>/dev/null || true

# Show current ADB devices inside container for reference
echo "Current ADB connection status (inside container):"
adb devices -l

echo "Configuring emulator settings..."

# Network connection already established, skip duplicate setup

# Configure emulator settings
echo "Applying Android settings..."

# Disable animations for faster operation
adb -s emulator-5554 shell settings put global window_animation_scale 0 2>/dev/null || true
adb -s emulator-5554 shell settings put global transition_animation_scale 0 2>/dev/null || true
adb -s emulator-5554 shell settings put global animator_duration_scale 0 2>/dev/null || true

# Set screen timeout to never
adb -s emulator-5554 shell settings put system screen_off_timeout 2147483647 2>/dev/null || true

# Disable screen lock
adb -s emulator-5554 shell locksettings set-disabled true 2>/dev/null || true

# Set screen orientation to portrait
adb -s emulator-5554 shell settings put system user_rotation 0 2>/dev/null || true

echo "Emulator setup complete!"
echo "ADB available on port 5555"
echo "VNC available on port 5900"
echo "Container IP: $(hostname -I | awk '{print $1}')"

# Create a test screenshot to verify functionality
echo "Taking test screenshot..."
sleep 5
adb -s emulator-5554 shell screencap -p /sdcard/test_screen.png 2>/dev/null || echo "Screenshot test skipped"
adb -s emulator-5554 pull /sdcard/test_screen.png /data/screenshots/test_screen.png 2>/dev/null || echo "Screenshot pull failed"

# Display system information
echo "=== System Information ==="
echo "Emulator PID: $EMULATOR_PID"
echo "Available memory: $(free -h | grep Mem)"
echo "CPU info: $(nproc) cores"
echo "KVM available: $(ls -la /dev/kvm 2>/dev/null || echo 'No KVM - using software mode')"
if [ "$KVM_AVAILABLE" = true ]; then
    echo "Acceleration mode: KVM (hardware)"
else
    echo "Acceleration mode: Software only"
fi

# Monitor emulator process
monitor_emulator() {
    while kill -0 $EMULATOR_PID 2>/dev/null; do
        sleep 60
        echo "$(date): Emulator still running (PID: $EMULATOR_PID)"
    done
    echo "$(date): Emulator process died!"
}

monitor_emulator &

# Keep container running
wait $EMULATOR_PID
