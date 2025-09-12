# Use NVIDIA CUDA base image for GPU support
FROM  nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3.12 \
  python3.12-dev \
  python3-pip \
  curl \
  git \
  && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data volumes and models
RUN mkdir -p /app/production_data /app/collected_data /app/markup_data /app/sparse_data /app/data/models /app/research/screen-page-classification

# Copy research models if available (for autolabel)
# The research code should be mounted or copied separately
# We'll rely on the volume mount in docker-compose for this

# --- Create non-root user (configurable UID/GID) and set ownership ---
ARG UID=1000
ARG GID=1000
RUN set -eux; \
  groupadd -g ${GID} appgroup || true; \
  id -u appuser >/dev/null 2>&1 || useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser; \
  chown -R appuser:appgroup /app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/sessions || exit 1

# Run the application
USER appuser

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
