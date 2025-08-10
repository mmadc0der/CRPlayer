#!/bin/bash

# App Compatibility Fix Script for Android Emulator
# Устраняет проблемы с запуском игр и приложений

set -e

echo "=== Android App Compatibility Fixer ==="

# Function to run ADB command with device specification
adb_cmd() {
    adb -s emulator-5554 "$@"
}

# Function to check if app is installed
check_app_installed() {
    local package_name="$1"
    if adb_cmd shell pm list packages | grep -q "$package_name"; then
        echo "✅ $package_name is installed"
        return 0
    else
        echo "❌ $package_name is not installed"
        return 1
    fi
}

# Function to enable developer options and USB debugging
enable_developer_mode() {
    echo "Enabling developer options..."
    
    # Enable developer options
    adb_cmd shell settings put global development_settings_enabled 1
    
    # Enable USB debugging
    adb_cmd shell settings put global adb_enabled 1
    
    # Enable mock locations
    adb_cmd shell settings put secure mock_location 1
    
    echo "✅ Developer options enabled"
}

# Function to install Google Play Services if missing
install_google_services() {
    echo "Checking Google Play Services..."
    
    if ! check_app_installed "com.google.android.gms"; then
        echo "Installing Google Play Services..."
        # This would require manual installation in most cases
        echo "⚠️  Google Play Services missing - install manually from Play Store"
    fi
}

# Function to fix graphics and performance settings
fix_graphics_settings() {
    echo "Optimizing graphics settings..."
    
    # Force GPU rendering
    adb_cmd shell setprop debug.egl.hw 1
    adb_cmd shell setprop ro.kernel.qemu.gles 1
    adb_cmd shell setprop ro.opengles.version 196610
    
    # Disable hardware overlays (can cause black screens)
    adb_cmd shell service call SurfaceFlinger 1008 i32 1
    
    # Enable hardware acceleration
    adb_cmd shell setprop debug.sf.hw 1
    
    echo "✅ Graphics settings optimized"
}

# Function to fix network and VPN issues
fix_network_settings() {
    echo "Fixing network settings..."
    
    # Reset network settings
    adb_cmd shell settings put global airplane_mode_on 0
    adb_cmd shell am broadcast -a android.intent.action.AIRPLANE_MODE --ez state false
    
    # Enable mobile data
    adb_cmd shell svc data enable
    
    # Fix DNS settings
    adb_cmd shell setprop net.dns1 8.8.8.8
    adb_cmd shell setprop net.dns2 8.8.4.4
    
    echo "✅ Network settings fixed"
}

# Function to increase app permissions
fix_app_permissions() {
    local package_name="$1"
    
    if [ -z "$package_name" ]; then
        echo "Usage: fix_app_permissions <package_name>"
        return 1
    fi
    
    echo "Fixing permissions for $package_name..."
    
    # Grant common permissions
    local permissions=(
        "android.permission.INTERNET"
        "android.permission.ACCESS_NETWORK_STATE"
        "android.permission.ACCESS_WIFI_STATE"
        "android.permission.WRITE_EXTERNAL_STORAGE"
        "android.permission.READ_EXTERNAL_STORAGE"
        "android.permission.CAMERA"
        "android.permission.RECORD_AUDIO"
        "android.permission.ACCESS_FINE_LOCATION"
        "android.permission.ACCESS_COARSE_LOCATION"
    )
    
    for permission in "${permissions[@]}"; do
        adb_cmd shell pm grant "$package_name" "$permission" 2>/dev/null || true
    done
    
    echo "✅ Permissions granted for $package_name"
}

# Function to clear app data and cache
reset_app_data() {
    local package_name="$1"
    
    if [ -z "$package_name" ]; then
        echo "Usage: reset_app_data <package_name>"
        return 1
    fi
    
    echo "Resetting data for $package_name..."
    
    # Stop the app
    adb_cmd shell am force-stop "$package_name"
    
    # Clear app data and cache
    adb_cmd shell pm clear "$package_name"
    
    echo "✅ App data cleared for $package_name"
}

# Function to fix Clash Royale specifically
fix_clash_royale() {
    local cr_package="com.supercell.clashroyale"
    
    echo "Applying Clash Royale specific fixes..."
    
    if check_app_installed "$cr_package"; then
        # Reset app data
        reset_app_data "$cr_package"
        
        # Fix permissions
        fix_app_permissions "$cr_package"
        
        # Disable battery optimization
        adb_cmd shell dumpsys deviceidle whitelist +"$cr_package"
        
        # Set app to not be killed
        adb_cmd shell am set-inactive "$cr_package" false
        
        echo "✅ Clash Royale fixes applied"
        echo "💡 Try launching Clash Royale now"
    else
        echo "❌ Clash Royale not found. Install it first from Play Store"
    fi
}

# Function to show system information
show_system_info() {
    echo "=== System Information ==="
    
    echo "Android version: $(adb_cmd shell getprop ro.build.version.release)"
    echo "API level: $(adb_cmd shell getprop ro.build.version.sdk)"
    echo "Architecture: $(adb_cmd shell getprop ro.product.cpu.abi)"
    echo "GPU: $(adb_cmd shell getprop ro.hardware.egl)"
    echo "OpenGL: $(adb_cmd shell getprop ro.opengles.version)"
    
    # Check available memory
    echo "Memory info:"
    adb_cmd shell cat /proc/meminfo | head -3
    
    # Check running services
    echo "Google Play Services status:"
    adb_cmd shell dumpsys package com.google.android.gms | grep -E "(versionName|enabled)" | head -2 || echo "Not installed"
}

# Main menu
main_menu() {
    echo ""
    echo "Select an option:"
    echo "1. Show system information"
    echo "2. Enable developer mode"
    echo "3. Fix graphics settings"
    echo "4. Fix network settings"
    echo "5. Fix Clash Royale specifically"
    echo "6. Fix app permissions (specify package)"
    echo "7. Reset app data (specify package)"
    echo "8. Run all fixes"
    echo "9. Exit"
    
    read -p "Enter choice [1-9]: " choice
    
    case $choice in
        1) show_system_info ;;
        2) enable_developer_mode ;;
        3) fix_graphics_settings ;;
        4) fix_network_settings ;;
        5) fix_clash_royale ;;
        6) 
            read -p "Enter package name: " pkg
            fix_app_permissions "$pkg"
            ;;
        7)
            read -p "Enter package name: " pkg
            reset_app_data "$pkg"
            ;;
        8)
            echo "Running all fixes..."
            enable_developer_mode
            fix_graphics_settings
            fix_network_settings
            install_google_services
            fix_clash_royale
            echo "✅ All fixes applied"
            ;;
        9) echo "Exiting..."; exit 0 ;;
        *) echo "Invalid option" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    main_menu
}

# Check if ADB is connected
if ! adb_cmd shell echo "test" >/dev/null 2>&1; then
    echo "❌ ADB not connected to emulator"
    echo "Make sure emulator is running and try: adb connect localhost:5555"
    exit 1
fi

echo "✅ Connected to Android emulator"

# Run main menu
main_menu
