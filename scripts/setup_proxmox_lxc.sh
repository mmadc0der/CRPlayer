#!/bin/bash

# Proxmox LXC Setup Script for CRPlayer
# Run this script on Proxmox host

set -e

# Configuration
LXC_ID=200
LXC_NAME="crplayer"
LXC_CORES=6
LXC_MEMORY=12288
LXC_SWAP=4096
LXC_STORAGE="local-lvm:100"
LXC_TEMPLATE="ubuntu-22.04-standard_22.04-1_amd64.tar.xz"
BRIDGE="vmbr0"

echo "=== CRPlayer Proxmox LXC Setup ==="

# Check if running on Proxmox
if ! command -v pct &> /dev/null; then
    echo "Error: This script must be run on a Proxmox host"
    exit 1
fi

# Check if LXC ID is available
if pct status $LXC_ID &> /dev/null; then
    echo "Warning: LXC $LXC_ID already exists"
    read -p "Do you want to destroy and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping and destroying LXC $LXC_ID..."
        pct stop $LXC_ID || true
        pct destroy $LXC_ID || true
    else
        echo "Exiting..."
        exit 1
    fi
fi

# Create LXC container
echo "Creating LXC container $LXC_ID..."
pct create $LXC_ID $LXC_TEMPLATE \
    --hostname $LXC_NAME \
    --cores $LXC_CORES \
    --memory $LXC_MEMORY \
    --swap $LXC_SWAP \
    --storage $LXC_STORAGE \
    --net0 name=eth0,bridge=$BRIDGE,ip=dhcp \
    --privileged 1 \
    --features nesting=1,keyctl=1 \
    --onboot 1

echo "LXC container created successfully!"

# Configure GPU passthrough
echo "Configuring GPU passthrough..."

# Add GPU device mappings to LXC config
LXC_CONFIG="/etc/pve/lxc/${LXC_ID}.conf"

cat >> $LXC_CONFIG << EOF

# GPU Passthrough for NVIDIA
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file

# KVM for Android emulator hardware acceleration
lxc.cgroup2.devices.allow: c 10:232 rwm
lxc.mount.entry: /dev/kvm dev/kvm none bind,optional,create=file
EOF

echo "GPU passthrough configured!"

# Start LXC container
echo "Starting LXC container..."
pct start $LXC_ID

# Wait for container to boot
echo "Waiting for container to boot..."
sleep 10

# Install basic packages
echo "Installing basic packages in container..."
pct exec $LXC_ID -- bash -c "
    apt-get update
    apt-get install -y curl wget git nano htop
    apt-get install -y software-properties-common apt-transport-https ca-certificates gnupg lsb-release
"

# Install Docker
echo "Installing Docker in container..."
pct exec $LXC_ID -- bash -c "
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu jammy stable' | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable docker
    systemctl start docker
"

# Install NVIDIA Docker runtime
echo "Installing NVIDIA Docker runtime..."
pct exec $LXC_ID -- bash -c "
    distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
    apt-get update
    apt-get install -y nvidia-docker2
    systemctl restart docker
"

# Test GPU access
echo "Testing GPU access in container..."
pct exec $LXC_ID -- bash -c "
    if command -v nvidia-smi &> /dev/null; then
        echo 'GPU access test:'
        nvidia-smi
    else
        echo 'Warning: nvidia-smi not found, installing NVIDIA drivers...'
        apt-get install -y nvidia-driver-525
        echo 'Please reboot the container after this script completes'
    fi
"

# Get container IP
LXC_IP=$(pct exec $LXC_ID -- hostname -I | awk '{print $1}')

echo "=== Setup Complete ==="
echo "LXC ID: $LXC_ID"
echo "LXC Name: $LXC_NAME"
echo "LXC IP: $LXC_IP"
echo ""
echo "Next steps:"
echo "1. SSH into container: ssh root@$LXC_IP"
echo "2. Clone CRPlayer repository"
echo "3. Run: cd CRPlayer && docker-compose up -d"
echo ""
echo "Access points:"
echo "- Emulator VNC: $LXC_IP:5900"
echo "- ML Server API: $LXC_IP:8080"
echo "- Grafana Dashboard: $LXC_IP:3000"
