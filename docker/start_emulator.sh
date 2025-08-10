#!/bin/bash

# Start Emulator Script for CRPlayer
set -e

echo "Starting CRPlayer Android Emulator..."

# Set display for headless operation
export DISPLAY=:99

# Start Xvfb (virtual framebuffer)
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Start window manager
fluxbox -display :99 &
WM_PID=$!

# Wait for X server to start
sleep 3

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

# Start Android emulator
echo "Starting Android emulator..."
emulator -avd ClashRoyale_AVD \
    -no-audio \
    -no-window \
    -gpu swiftshader_indirect \
    -memory ${EMULATOR_RAM:-3072} \
    -partition-size ${EMULATOR_PARTITION:-6144} \
    -no-snapshot-save \
    -no-snapshot-load \
    -wipe-data \
    -port 5554 \
    -verbose &

EMULATOR_PID=$!
echo "Emulator PID: $EMULATOR_PID"

# Wait for emulator to boot
echo "Waiting for emulator to boot..."
adb wait-for-device

# Wait additional time for full boot
sleep 30

# Check if emulator is ready
while [ "`adb shell getprop sys.boot_completed | tr -d '\r'`" != "1" ]; do
    echo "Waiting for emulator to complete boot..."
    sleep 5
done

echo "Emulator is ready!"

# Enable ADB over network
adb tcpip 5555
sleep 2
adb connect localhost:5555

# Configure emulator settings
echo "Configuring emulator settings..."

# Disable animations for faster operation
adb shell settings put global window_animation_scale 0
adb shell settings put global transition_animation_scale 0
adb shell settings put global animator_duration_scale 0

# Set screen timeout to never
adb shell settings put system screen_off_timeout 2147483647

# Disable screen lock
adb shell locksettings set-disabled true

# Set screen orientation to portrait
adb shell settings put system user_rotation 0

echo "Emulator setup complete!"
echo "ADB available on port 5555"
echo "VNC available on port 5900"
echo "Container IP: $(hostname -I | awk '{print $1}')"

# Create a test screenshot to verify functionality
echo "Taking test screenshot..."
adb shell screencap -p /sdcard/test_screen.png
adb pull /sdcard/test_screen.png /data/screenshots/test_screen.png 2>/dev/null || echo "Screenshot test failed"

# Display system information
echo "=== System Information ==="
echo "Emulator PID: $EMULATOR_PID"
echo "Available memory: $(free -h | grep Mem)"
echo "CPU info: $(nproc) cores"
echo "KVM available: $(ls -la /dev/kvm 2>/dev/null || echo 'No KVM')"

# Monitor emulator process
monitor_emulator() {
    while kill -0 $EMULATOR_PID 2>/dev/null; do
        sleep 30
        echo "$(date): Emulator still running (PID: $EMULATOR_PID)"
    done
    echo "$(date): Emulator process died!"
}

monitor_emulator &

# Keep container running
wait $EMULATOR_PID
