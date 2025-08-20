#!/bin/bash

# Start Emulator Script for CRPlayer (без KVM зависимости)
set -e

echo "Starting CRPlayer Android Emulator..."

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

# Ensure a local ADB server is running (default localhost binding)
echo "Starting local ADB server..."
adb start-server >/tmp/adb_server.log 2>&1 || true
sleep 1

# Debug: Check emulator availability
echo "=== Debug Information ==="
echo "PATH: $PATH"
echo "ANDROID_SDK_ROOT: $ANDROID_SDK_ROOT"
echo "Checking emulator command..."
which emulator || echo "emulator not found in PATH"
ls -la $ANDROID_SDK_ROOT/emulator/emulator 2>/dev/null || echo "emulator binary not found"
$ANDROID_SDK_ROOT/emulator/emulator -version 2>/dev/null || echo "emulator version check failed"
echo "Skipping accel-check (forcing software emulation)..."

echo "Starting Android emulator (software-only)..."

# Normalize memory/partition envs for clarity in logs
MEMORY_MB=${EMULATOR_RAM:-8192}
PARTITION_MB=${EMULATOR_PARTITION:-16384}
echo "Emulator memory: ${MEMORY_MB} MB | System partition: ${PARTITION_MB} MB"

GPU_MODE=${EMULATOR_GPU_MODE:-swiftshader_indirect}
echo "Emulator GPU mode: ${GPU_MODE}"
DISPLAY=:99 $ANDROID_SDK_ROOT/emulator/emulator -avd ClashRoyale_AVD \
    -no-audio \
    -gpu ${GPU_MODE} \
    -memory ${MEMORY_MB} \
    -partition-size ${PARTITION_MB} \
    -no-snapshot-save \
    -no-snapshot-load \
    -wipe-data \
    -accel off \
    -verbose &

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

echo "Skipping ARM translation (Houdini) steps entirely. Using native ARM64 system image."

echo "Keeping ADB local-only (no TCP/IP exposure)."
echo "Current ADB devices:"
adb devices -l || true

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
echo "Acceleration mode: Software only"

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
