# Docker Setup for CRPlayer Dashboard with Autolabel

## Prerequisites

1. Docker and Docker Compose installed
2. Trained model checkpoint at `data/models/SingleLabelClassification/model.pth`
3. Research code available at `research/screen-page-classification/`

## Quick Start

### Linux/Mac:
```bash
cd dashboard
./build.sh
```

### Windows:
```cmd
cd dashboard
build.bat
```

## Manual Setup

1. **Build the containers:**
   ```bash
   docker-compose build
   ```

2. **Start the services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - Dashboard: http://localhost:8080
   - Annotation Tool: http://localhost:8080/annotation/

## Volume Mounts

The docker-compose configuration mounts these directories:

- `../data:/app/data` - Data directory with models and sessions
- `../research/screen-page-classification:/app/research/screen-page-classification` - Research code for autolabel

## Autolabel Requirements

For the autolabel feature to work in Docker:

1. **Model checkpoint** must exist at:
   - Host: `data/models/SingleLabelClassification/model.pth`
   - Container: `/app/data/models/SingleLabelClassification/model.pth`

2. **Research code** must be available at:
   - Host: `research/screen-page-classification/models.py`
   - Container: `/app/research/screen-page-classification/models.py`

3. **Python dependencies** are installed via requirements.txt:
   - torch, torchvision (for inference)
   - timm (for model architectures)
   - transformers (for some model types)

## Environment Variables

- `PUID`: User ID for file permissions (default: 1000)
- `PGID`: Group ID for file permissions (default: 1000)

## Troubleshooting

### Autolabel not working

1. Check model exists:
   ```bash
   docker exec crplayer-annotation ls -la /app/data/models/SingleLabelClassification/
   ```

2. Check research code is mounted:
   ```bash
   docker exec crplayer-annotation ls -la /app/research/screen-page-classification/
   ```

3. Test autolabel service:
   ```bash
   docker exec crplayer-annotation python -c "
   from services.autolabel_service import AutoLabelService
   s = AutoLabelService()
   print('Service initialized successfully')
   "
   ```

4. Check logs:
   ```bash
   docker-compose logs annotation-app
   ```

### Permission issues

If you get permission errors, ensure the UID/GID match your host user:

```bash
export PUID=$(id -u)
export PGID=$(id -g)
docker-compose up -d
```

## API Endpoints

The autolabel endpoints are available at:

- Single frame: `POST /annotation/api/annotations/autolabel`
- Full session: `POST /annotation/api/annotations/autolabel_session`

Example curl test:
```bash
curl -X POST http://localhost:8080/annotation/api/annotations/autolabel \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id",
    "dataset_id": 1,
    "frame_idx": 0,
    "confidence_threshold": 0.9
  }'
```

## Stopping Services

```bash
docker-compose down
```

To also remove volumes:
```bash
docker-compose down -v
```
