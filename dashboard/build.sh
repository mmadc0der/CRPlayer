#!/bin/bash
# Build script for CRPlayer Dashboard with Annotation Tool (GPU-enabled)

set -e

echo "Building CRPlayer Dashboard with GPU support..."

# Ensure we're in the dashboard directory
cd "$(dirname "$0")"

# Auto-detect host user's UID and GID for proper file permissions
echo "Detecting host user permissions..."
if [ -z "${PUID}" ]; then
    export PUID=$(id -u)
    echo "Using host UID: ${PUID}"
fi
if [ -z "${PGID}" ]; then
    export PGID=$(id -g)
    echo "Using host GID: ${PGID}"
fi
echo ""

# Check if research models.py exists
if [ ! -f "../research/screen-page-classification/models.py" ]; then
    echo "Warning: Research models.py not found. Autolabel feature may not work."
    echo "Expected at: ../research/screen-page-classification/models.py"
fi

# Check if model checkpoint exists
if [ ! -f "../data/models/SingleLabelClassification/model.pth" ]; then
    echo "Warning: Model checkpoint not found. Autolabel will not work until a model is trained."
    echo "Expected at: ../data/models/SingleLabelClassification/model.pth"
fi

# Check GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: NVIDIA drivers not found. GPU acceleration will not be available."
    echo "To enable GPU support:"
    echo "1. Install NVIDIA drivers on your host system"
    echo "2. Install nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "3. Configure Docker daemon with GPU support (see daemon.json below)"
    echo ""
fi

# Build and start the containers
echo "Building Docker containers with GPU support..."
docker-compose build --no-cache

echo "Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Check GPU status in container
echo "Checking GPU status in annotation container..."
if docker-compose exec -T annotation-app curl -s http://localhost:5000/api/gpu/status | grep -q "true"; then
    echo "✅ GPU is available in the annotation container!"
else
    echo "⚠️  GPU not detected in container. Running on CPU."
fi

# Check health
docker-compose ps

echo ""
echo "Dashboard is available at: http://localhost:8080"
echo "Annotation tool is available at: http://localhost:8080/annotation/"
echo ""
echo "To check GPU status: curl http://localhost:5000/api/gpu/status"
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"

echo ""
echo "GPU Setup Instructions:"
echo "========================"
echo "If GPU is not working, ensure your Docker daemon.json includes:"
echo '{'
echo '  "runtimes": {'
echo '    "nvidia": {'
echo '      "path": "nvidia-container-runtime",'
echo '      "runtimeArgs": []'
echo '    }'
echo '  },'
echo '  "default-runtime": "nvidia"'
echo '}'
echo ""
echo "Save this as /etc/docker/daemon.json and restart Docker service."
