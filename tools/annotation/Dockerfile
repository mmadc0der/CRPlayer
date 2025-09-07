FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data volumes
RUN mkdir -p /app/production_data /app/collected_data /app/markup_data /app/sparse_data

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
