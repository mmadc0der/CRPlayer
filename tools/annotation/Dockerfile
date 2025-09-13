# Use NVIDIA CUDA base image for GPU support
FROM  nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set non-interactive frontend for apt to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies (non-interactive)
RUN apt-get update && apt-get install -y \
  python3.11 \
  python3.11-dev \
  python3-full \
  python3-pip \
  python3-venv \
  curl \
  git \
  build-essential \
  tzdata \
  && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install Python dependencies
COPY requirements.txt .
RUN python3.11 -m venv /opt/venv
RUN ln -s /opt/venv/bin/python3.11 /usr/bin/python
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data volumes and models
RUN mkdir -p /app/production_data /app/collected_data /app/markup_data /app/sparse_data /app/data/models /app/research/screen-page-classification

# Copy research models if available (for autolabel)
# The research code should be mounted or copied separately
# We'll rely on the volume mount in docker-compose for this

# --- Create non-root user (use high UID to avoid system user conflicts) ---
ARG UID=10000
ARG GID=10000
RUN set -eux; \
  groupadd -g ${GID} appgroup 2>/dev/null || true; \
  useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser 2>/dev/null || \
  (echo "Failed to create appuser with UID ${UID}, trying different approach" && \
  useradd -m -g ${GID} -s /bin/bash appuser 2>/dev/null || \
  useradd -m -s /bin/bash appuser); \
  chown -R appuser:appgroup /app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/sessions || exit 1

# Run the application
USER appuser
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
