#!/bin/bash

# KVM Setup Script for LXC Container
# Проверяет и настраивает KVM для Android эмулятора

set -e

echo "=== KVM Setup for Android Emulator in LXC ==="

# Check if running in LXC
if [ ! -f /proc/1/environ ] || ! grep -q container=lxc /proc/1/environ 2>/dev/null; then
    echo "Warning: This script is designed for LXC containers"
fi

# Function to check KVM availability
check_kvm() {
    echo "Checking KVM availability..."
    
    if [ -e /dev/kvm ]; then
        echo "✅ /dev/kvm device exists"
        
        # Check permissions
        if [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
            echo "✅ KVM device has correct permissions"
            return 0
        else
            echo "❌ KVM device permissions incorrect"
            ls -la /dev/kvm
            return 1
        fi
    else
        echo "❌ /dev/kvm device not found"
        return 1
    fi
}

# Function to test KVM functionality
test_kvm() {
    echo "Testing KVM functionality..."
    
    if command -v kvm-ok >/dev/null 2>&1; then
        kvm-ok
    else
        echo "kvm-ok not available, installing cpu-checker..."
        apt-get update && apt-get install -y cpu-checker
        kvm-ok
    fi
}

# Function to setup alternative without KVM
setup_software_mode() {
    echo "Setting up software emulation mode..."
    
    # Create environment file for Docker Compose
    cat > /tmp/emulator.env << EOF
EMULATOR_ACCEL=off
EMULATOR_GPU=swiftshader_indirect
EMULATOR_RAM=2048
EMULATOR_PARTITION=4096
EOF
    
    echo "✅ Software emulation mode configured"
    echo "Environment variables saved to /tmp/emulator.env"
}

# Function to fix LXC KVM setup
fix_lxc_kvm() {
    echo "Attempting to fix KVM in LXC..."
    
    # Check if we're in privileged container
    if [ "$(cat /proc/1/attr/current 2>/dev/null)" != "unconfined" ]; then
        echo "❌ Container is not privileged. KVM requires privileged LXC container."
        echo "Please recreate container with --privileged 1"
        return 1
    fi
    
    # Try to create KVM device node
    if [ ! -e /dev/kvm ]; then
        echo "Attempting to create /dev/kvm device node..."
        mknod /dev/kvm c 10 232 2>/dev/null || echo "Failed to create device node"
        chmod 666 /dev/kvm 2>/dev/null || echo "Failed to set permissions"
    fi
    
    # Check if it worked
    if [ -e /dev/kvm ]; then
        echo "✅ KVM device node created"
        return 0
    else
        echo "❌ Failed to create KVM device node"
        return 1
    fi
}

# Main execution
main() {
    echo "Starting KVM setup process..."
    
    # First, try to check existing KVM
    if check_kvm; then
        echo "🎉 KVM is available and working!"
        
        if test_kvm; then
            echo "✅ KVM functionality confirmed"
            
            # Create environment for hardware acceleration
            cat > /tmp/emulator.env << EOF
EMULATOR_ACCEL=on
EMULATOR_GPU=host
EMULATOR_RAM=3072
EMULATOR_PARTITION=6144
EOF
            echo "Hardware acceleration mode configured"
            return 0
        fi
    fi
    
    echo "KVM not available, attempting fixes..."
    
    # Try to fix KVM setup
    if fix_lxc_kvm && check_kvm; then
        echo "🎉 KVM fixed and working!"
        return 0
    fi
    
    echo "⚠️  KVM setup failed, falling back to software mode"
    setup_software_mode
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Use docker-compose.emulator-fixed.yml for emulator without KVM"
    echo "2. Expect slower performance but better compatibility"
    echo "3. Consider using Proxmox host KVM passthrough if needed"
    
    return 0
}

# Run main function
main

echo ""
echo "=== KVM Setup Complete ==="
echo "Check the output above for next steps"
