#!/bin/bash

# Start Emulator Script for CRPlayer (без KVM зависимости)
set -e

echo "Starting CRPlayer Android Emulator (Software Mode)..."

# Set display for headless operation
export DISPLAY=:99

# Start Xvfb (virtual framebuffer)
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Start window manager
fluxbox -display :99 &
WM_PID=$!

# Wait for X server to start
sleep 5

# Start VNC server for remote access (bind to all interfaces for external access)
x11vnc -display :99 -nopw -listen 0.0.0.0 -xkb -forever -shared -bg -rfbport 5900

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

# Start ADB server
adb start-server

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
if [ -e /dev/kvm ]; then
    echo "KVM device found, enabling hardware acceleration"
    KVM_AVAILABLE=true
else
    echo "KVM device not found, using software acceleration"
    KVM_AVAILABLE=false
fi

# Start Android emulator
echo "Starting Android emulator..."
if [ "$KVM_AVAILABLE" = true ]; then
    # With hardware acceleration
    $ANDROID_SDK_ROOT/emulator/emulator -avd ClashRoyale_AVD \
        -no-audio \
        -no-window \
        -gpu ${EMULATOR_GPU:-swiftshader_indirect} \
        -memory ${EMULATOR_RAM:-2048} \
        -partition-size ${EMULATOR_PARTITION:-4096} \
        -no-snapshot-save \
        -no-snapshot-load \
        -wipe-data \
        -port 5554 \
        -accel on \
        -verbose &
else
    # Software only mode
    $ANDROID_SDK_ROOT/emulator/emulator -avd ClashRoyale_AVD \
        -no-audio \
        -no-window \
        -gpu ${EMULATOR_GPU:-swiftshader_indirect} \
        -memory ${EMULATOR_RAM:-2048} \
        -partition-size ${EMULATOR_PARTITION:-4096} \
        -no-snapshot-save \
        -no-snapshot-load \
        -wipe-data \
        -port 5554 \
        -accel ${EMULATOR_ACCEL:-off} \
        -verbose &
fi

EMULATOR_PID=$!
echo "Emulator PID: $EMULATOR_PID"

# Wait for emulator to boot
echo "Waiting for emulator to boot..."
adb wait-for-device

# Wait additional time for full boot (увеличено время для software mode)
echo "Waiting for Android system to fully boot..."
BOOT_TIMEOUT=600  # 10 minutes for software emulation
BOOT_START=$(date +%s)

while [ $(($(date +%s) - BOOT_START)) -lt $BOOT_TIMEOUT ]; do
    BOOT_STATUS=$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || echo "0")
    if [ "$BOOT_STATUS" = "1" ]; then
        echo "Android system fully booted!"
        break
    fi
    echo "Waiting for boot completion... ($(($(($(date +%s) - BOOT_START)) / 60))m $(($(date +%s) - BOOT_START))s elapsed)"
    sleep 15
done

# Final boot check
BOOT_STATUS=$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || echo "0")
if [ "$BOOT_STATUS" != "1" ]; then
    echo "Warning: Boot may not be complete, but continuing..."
fi

echo "Configuring emulator settings..."

# Enable ADB over network
adb tcpip 5555
sleep 2
adb connect localhost:5555

# Configure emulator settings
echo "Applying Android settings..."

# Disable animations for faster operation
adb shell settings put global window_animation_scale 0 2>/dev/null || true
adb shell settings put global transition_animation_scale 0 2>/dev/null || true
adb shell settings put global animator_duration_scale 0 2>/dev/null || true

# Set screen timeout to never
adb shell settings put system screen_off_timeout 2147483647 2>/dev/null || true

# Disable screen lock
adb shell locksettings set-disabled true 2>/dev/null || true

# Set screen orientation to portrait
adb shell settings put system user_rotation 0 2>/dev/null || true

echo "Emulator setup complete!"
echo "ADB available on port 5555"
echo "VNC available on port 5900"
echo "Container IP: $(hostname -I | awk '{print $1}')"

# Create a test screenshot to verify functionality
echo "Taking test screenshot..."
sleep 5
adb shell screencap -p /sdcard/test_screen.png 2>/dev/null || echo "Screenshot test skipped"
adb pull /sdcard/test_screen.png /data/screenshots/test_screen.png 2>/dev/null || echo "Screenshot pull failed"

# Display system information
echo "=== System Information ==="
echo "Emulator PID: $EMULATOR_PID"
echo "Available memory: $(free -h | grep Mem)"
echo "CPU info: $(nproc) cores"
echo "KVM available: $(ls -la /dev/kvm 2>/dev/null || echo 'No KVM - using software mode')"
echo "Acceleration mode: ${EMULATOR_ACCEL:-off}"

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
