#!/bin/bash

# Simple ADB Connection Test
# Тестирует базовое ADB соединение без сетевых подключений

echo "=== Simple ADB Test ==="

# Function to test basic ADB functionality
test_basic_adb() {
    echo "Testing basic ADB functionality..."
    
    # Check if container is running
    if ! docker ps | grep -q "cr_emulator_test"; then
        echo "❌ Container not running"
        return 1
    fi
    
    # Test ADB inside container
    echo "Testing ADB inside container..."
    docker exec cr_emulator_test bash -c "
        echo 'ADB devices inside container:'
        adb devices -l
        echo ''
        echo 'Testing shell command:'
        adb -s emulator-5554 shell echo 'Hello from Android'
        echo ''
        echo 'Android version:'
        adb -s emulator-5554 shell getprop ro.build.version.release
        echo ''
        echo 'Boot status:'
        adb -s emulator-5554 shell getprop sys.boot_completed
    "
}

# Function to test from host
test_host_adb() {
    echo "Testing ADB from host..."
    
    echo "Current ADB devices from host:"
    adb devices -l
    echo ""
    
    # Try to connect if offline
    if adb devices | grep -q "offline"; then
        echo "Attempting to reconnect offline devices..."
        adb reconnect
        sleep 3
        adb devices -l
    fi
    
    # Test commands from host
    echo "Testing commands from host:"
    
    # Try with localhost:5555
    if adb devices | grep -q "localhost:5555.*device"; then
        echo "Testing with localhost:5555:"
        adb -s localhost:5555 shell echo "Host to emulator via network" || echo "Failed"
    fi
    
    # Try direct connection
    echo "Attempting direct connection:"
    adb connect localhost:5555
    sleep 2
    adb devices -l
}

# Function to fix connection issues
fix_connection() {
    echo "Attempting to fix ADB connection..."
    
    # Kill all ADB servers
    echo "Killing all ADB servers..."
    adb kill-server 2>/dev/null || true
    docker exec cr_emulator_test adb kill-server 2>/dev/null || true
    sleep 3
    
    # Restart ADB server on host
    echo "Starting ADB server on host..."
    adb start-server
    sleep 3
    
    # Restart ADB server in container
    echo "Starting ADB server in container..."
    docker exec cr_emulator_test bash -c "
        export ADB_SERVER_SOCKET=tcp:0.0.0.0:5037
        adb start-server
        sleep 3
        adb -s emulator-5554 tcpip 5555
        sleep 2
    "
    
    # Connect from host
    echo "Connecting from host..."
    adb connect localhost:5555
    sleep 3
    
    echo "Final status:"
    adb devices -l
}

# Function to show detailed network info
show_network_info() {
    echo "=== Network Diagnostic ==="
    
    # Container IP
    echo "Container IP:"
    docker inspect cr_emulator_test | grep -A 5 "IPAddress" | head -10
    
    # Port mapping
    echo "Port mapping:"
    docker port cr_emulator_test
    
    # Network connectivity test
    echo "Testing port connectivity:"
    nc -zv localhost 5555 2>&1 || echo "Port 5555 not accessible"
    nc -zv localhost 5900 2>&1 || echo "Port 5900 not accessible"
    
    # Check what's listening inside container
    echo "Listening ports inside container:"
    docker exec cr_emulator_test netstat -tlnp | grep -E "(5037|5554|5555)" || echo "No ADB ports found"
}

# Interactive menu
echo "Select test option:"
echo "1. Test basic ADB functionality"
echo "2. Test ADB from host"
echo "3. Fix connection issues"
echo "4. Show network information"
echo "5. Run all tests"

read -p "Enter choice [1-5]: " choice

case $choice in
    1) test_basic_adb ;;
    2) test_host_adb ;;
    3) fix_connection ;;
    4) show_network_info ;;
    5) 
        echo "Running all tests..."
        test_basic_adb
        echo ""
        test_host_adb
        echo ""
        show_network_info
        ;;
    *) echo "Invalid option" ;;
esac

echo ""
echo "Test completed!"
