FROM python:3.12-slim

WORKDIR /app

# Install system deps for PIL and networking tools (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy tool code
COPY . /app

EXPOSE 8765

CMD ["python", "-m", "tools.replayer.cli", "replay", "--help"]

