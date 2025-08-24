# CRPlayer Dashboard

Web-based dashboard for monitoring device screen and pipeline metrics.

## Quick Start

1. **Start the dashboard container:**
```bash
cd dashboard
docker-compose up -d
```

2. **Access the dashboard:**
Open http://localhost:8080 in your browser

3. **Start the pipeline with WebSocket server (coming next):**
```bash
python pipeline.py /tmp/scrcpy_stream --dashboard
```

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   scrcpy FIFO   │───▶│   Pipeline   │───▶│ WebSocket Server│
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌──────────────┐    ┌─────────────────┐
                       │ Other        │    │  Web Dashboard  │
                       │ Subscribers  │    │  (nginx:8080)   │
                       └──────────────┘    └─────────────────┘
```

## Features

### Current (Frontend Only)
- ✅ Modern responsive UI with glassmorphism design
- ✅ Connection status indicator
- ✅ Device screen display area
- ✅ Real-time metrics cards (data rate, frame rate, total data)
- ✅ Activity log with timestamps
- ✅ Touch/click coordinate mapping for device interaction
- ✅ Screenshot functionality
- ✅ Docker container with nginx serving

### Coming Next (Server Integration)
- 🔄 WebSocket server subscriber
- 🔄 MKV frame decoding and streaming
- 🔄 Real-time metrics from pipeline
- 🔄 ADB touch command integration

## Dashboard Components

- **`static/index.html`** - Main dashboard UI
- **`static/dashboard.js`** - WebSocket client and interaction logic
- **`docker-compose.yml`** - Container orchestration
- **`nginx.conf`** - Web server configuration with WebSocket proxy

## Development

The dashboard shows "Disconnected" state until the WebSocket server is implemented. All UI components are functional and ready for data integration.
