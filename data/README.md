# CRPlayer Data Directory Structure

Centralized data management for the CRPlayer AI training pipeline.

## Directory Structure

```
data/
├── raw/                    # Raw captured game sessions
│   ├── session_001/
│   │   ├── frames/         # Raw frame images
│   │   ├── metadata.json   # Session metadata
│   │   └── status.json     # Processing status
│   └── session_002/
├── annotated/              # Annotated sessions ready for training
│   ├── session_001/
│   │   ├── frames/
│   │   ├── annotations.json
│   │   ├── metadata.json
│   │   └── status.json
├── datasets/               # Organized training datasets
│   ├── clash_royale_v1/
│   │   ├── train/
│   │   │   ├── menu/
│   │   │   ├── loading/
│   │   │   ├── battle/
│   │   │   └── final/
│   │   ├── val/
│   │   └── dataset_info.json
├── models/                 # Trained model artifacts
│   ├── game_state_classifier/
│   └── rl_agents/
└── temp/                   # Temporary processing files
```

## Workflow States

Each session has a `status.json` file tracking its processing state:

### Raw Sessions
```json
{
  "status": "captured",
  "created_at": "2025-08-26T20:00:00Z",
  "frame_count": 1500,
  "game_name": "clash_royale",
  "resolution": "800x360",
  "fps": 30,
  "next_step": "annotation"
}
```

### Annotated Sessions
```json
{
  "status": "annotated",
  "created_at": "2025-08-26T20:00:00Z",
  "annotated_at": "2025-08-26T21:30:00Z",
  "frame_count": 1500,
  "annotated_count": 1200,
  "annotation_progress": 80.0,
  "game_states": {
    "menu": 300,
    "loading": 150,
    "battle": 600,
    "final": 150
  },
  "next_step": "dataset_creation"
}
```

### Dataset Ready
```json
{
  "status": "dataset_ready",
  "created_at": "2025-08-26T20:00:00Z",
  "dataset_created_at": "2025-08-26T22:00:00Z",
  "included_in_dataset": "clash_royale_v1",
  "train_split": 0.8,
  "val_split": 0.2
}
```

## Status Values

- **`captured`** - Raw session data collected
- **`processing`** - Currently being processed
- **`annotation_ready`** - Ready for manual annotation
- **`annotating`** - Currently being annotated
- **`annotated`** - Annotation complete
- **`dataset_ready`** - Included in training dataset
- **`archived`** - Moved to long-term storage

## Usage

### Data Collection
```bash
# Collect new session
python pipeline_script.py scripts/prod.yaml
# Creates: data/raw/session_YYYYMMDD_HHMMSS/
```

### Annotation
```bash
# Start annotation tool
cd dashboard && docker-compose up -d
# Access: http://localhost:8080/annotation/
```

### Dataset Creation
```bash
# Export annotated sessions to training dataset
python tools/create_dataset.py --input data/annotated/ --output data/datasets/clash_royale_v2/
```
