# CRPlayer Dashboard

Docker-compose setup for the CRPlayer web dashboard with nginx reverse proxy and Flask annotation tool.

## Quick Start

```bash
cd dashboard && cp .env.example .env

# Stop existing containers
docker stop crplayer-dashboard

# Start the dashboard stack
cd dashboard
docker-compose up -d

# Check status
docker-compose ps
```

## Access Points

- **Main Dashboard:** http://localhost:8080
- **Annotation Tool:** http://localhost:8080/annotation/
- **Health Check:** http://localhost:8080/health

## Services

### Nginx (Port 8080)
- Serves dashboard static files
- Reverse proxy to Flask annotation app
- Load balancing and caching

### Flask Annotation App (Internal Port 5000)
- Web-based data annotation interface
- Automatic session discovery
- Real-time progress tracking

## Data Volumes

The annotation app has access to:
- `production_data/` (read-only)
- `collected_data/` (read-only) 
- `markup_data/` (read-write)
- `sparse_data/` (read-only)

## Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild annotation app
docker-compose build annotation-app
docker-compose up -d

# Scale services (if needed)
docker-compose up -d --scale annotation-app=2
```

## Configuration

- **docker-compose.yml** - Service definitions
- **nginx-compose.conf** - Nginx configuration for container networking
- **.env** - Environment variables
- **Dockerfile** - Flask app container build

## Troubleshooting

**Check service health:**
```bash
docker-compose ps
curl http://localhost:8080/health
```

**View logs:**
```bash
docker-compose logs nginx
docker-compose logs annotation-app
```

**Restart services:**
```bash
docker-compose restart
```
