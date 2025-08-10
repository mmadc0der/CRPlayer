#!/bin/bash

# ADB Connection Diagnostic Script
# Диагностирует и исправляет проблемы с ADB подключением к эмулятору

set -e

echo "=== ADB Connection Diagnostic ==="

# Function to check if emulator container is running
check_container() {
    echo "Checking Docker container status..."
    if docker ps | grep -q "cr_emulator_test"; then
        echo "✅ Emulator container is running"
        return 0
    else
        echo "❌ Emulator container is not running"
        echo "Start it with: docker-compose -f docker/docker-compose.emulator-test.yml up -d"
        return 1
    fi
}

# Function to check port accessibility
check_ports() {
    echo "Checking port accessibility..."
    
    # Check ADB port (5555)
    if nc -z localhost 5555 2>/dev/null; then
        echo "✅ Port 5555 (ADB) is accessible"
    else
        echo "❌ Port 5555 (ADB) is not accessible"
    fi
    
    # Check VNC port (5900)
    if nc -z localhost 5900 2>/dev/null; then
        echo "✅ Port 5900 (VNC) is accessible"
    else
        echo "❌ Port 5900 (VNC) is not accessible"
    fi
}

# Function to check emulator process inside container
check_emulator_process() {
    echo "Checking emulator process inside container..."
    
    if docker exec cr_emulator_test ps aux | grep -q "[e]mulator"; then
        echo "✅ Emulator process is running inside container"
        
        # Show emulator process details
        echo "Emulator process details:"
        docker exec cr_emulator_test ps aux | grep "[e]mulator" | head -1
    else
        echo "❌ Emulator process is not running inside container"
    fi
}

# Function to check ADB server inside container
check_adb_server() {
    echo "Checking ADB server inside container..."
    
    # Check if ADB server is running
    if docker exec cr_emulator_test pgrep adb >/dev/null 2>&1; then
        echo "✅ ADB server is running inside container"
        
        # Show ADB server processes
        echo "ADB processes:"
        docker exec cr_emulator_test ps aux | grep "[a]db" || echo "No ADB processes found"
    else
        echo "❌ ADB server is not running inside container"
    fi
    
    # Check ADB devices inside container
    echo "ADB devices inside container:"
    docker exec cr_emulator_test adb devices -l || echo "Failed to get ADB devices"
}

# Function to restart ADB connection
restart_adb() {
    echo "Restarting ADB connection..."
    
    # Kill local ADB server
    echo "Killing local ADB server..."
    adb kill-server 2>/dev/null || true
    
    # Restart ADB server inside container
    echo "Restarting ADB server inside container..."
    docker exec cr_emulator_test bash -c "
        adb kill-server 2>/dev/null || true
        sleep 2
        adb start-server
        sleep 2
        adb connect localhost:5555
        adb devices -l
    "
    
    # Start local ADB server and connect
    echo "Starting local ADB server and connecting..."
    adb start-server
    sleep 2
    adb connect localhost:5555
    
    echo "Final ADB status:"
    adb devices -l
}

# Function to show container logs
show_container_logs() {
    echo "Recent container logs (last 50 lines):"
    docker logs --tail 50 cr_emulator_test
}

# Function to test ADB commands
test_adb_commands() {
    echo "Testing ADB commands..."
    
    # Test basic shell command
    echo "Testing basic shell command..."
    if adb -s localhost:5555 shell echo "test" 2>/dev/null; then
        echo "✅ Basic shell command works"
    else
        echo "❌ Basic shell command failed"
    fi
    
    # Test getting Android version
    echo "Testing Android version query..."
    if android_version=$(adb -s localhost:5555 shell getprop ro.build.version.release 2>/dev/null); then
        echo "✅ Android version: $android_version"
    else
        echo "❌ Failed to get Android version"
    fi
    
    # Test boot completion status
    echo "Testing boot completion status..."
    if boot_status=$(adb -s localhost:5555 shell getprop sys.boot_completed 2>/dev/null); then
        if [ "$boot_status" = "1" ]; then
            echo "✅ Android boot completed"
        else
            echo "⚠️  Android boot not completed (status: $boot_status)"
        fi
    else
        echo "❌ Failed to get boot status"
    fi
}

# Main diagnostic function
run_diagnostics() {
    echo "Running full ADB diagnostics..."
    echo "================================"
    
    check_container || return 1
    check_ports
    check_emulator_process
    check_adb_server
    
    echo ""
    echo "Current ADB status:"
    adb devices -l
    
    echo ""
    test_adb_commands
}

# Interactive menu
main_menu() {
    echo ""
    echo "ADB Diagnostic Options:"
    echo "1. Run full diagnostics"
    echo "2. Check container status"
    echo "3. Check ports"
    echo "4. Restart ADB connection"
    echo "5. Show container logs"
    echo "6. Test ADB commands"
    echo "7. Exit"
    
    read -p "Enter choice [1-7]: " choice
    
    case $choice in
        1) run_diagnostics ;;
        2) check_container ;;
        3) check_ports ;;
        4) restart_adb ;;
        5) show_container_logs ;;
        6) test_adb_commands ;;
        7) echo "Exiting..."; exit 0 ;;
        *) echo "Invalid option" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    main_menu
}

# Check if required tools are available
if ! command -v docker >/dev/null 2>&1; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "❌ ADB is not installed or not in PATH"
    exit 1
fi

if ! command -v nc >/dev/null 2>&1; then
    echo "⚠️  netcat (nc) is not available - port checks will be skipped"
fi

# Run main menu
main_menu
