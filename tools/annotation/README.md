# Web-based Annotation Tool

A Flask web application for annotating collected game frame data in-place via browser.

## Features

- **Web Interface** - Clean, responsive web UI accessible from any browser
- **In-place Annotation** - No file transfers needed, works directly with collected data
- **Session Management** - Automatically discovers and loads annotation sessions
- **Real-time Progress** - Live statistics and progress tracking
- **Keyboard Shortcuts** - Efficient annotation workflow
- **DB-backed storage** - Robust SQLite schema with integrity checks
- **Dataset Export** - Export labeled dataset as ZIP with manifest (optionally include images)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install flask
   ```

2. **Start the annotation server:**
   ```bash
   cd tools/annotation
   python app.py
   ```

3. **Open in browser:**
   ```
   http://127.0.0.1:5000
   ```

## Usage

### Starting the Server

```bash
# Default (localhost:5000)
python app.py

# Custom host/port
python app.py --host 0.0.0.0 --port 8080

# Debug mode
python app.py --debug
```

### Logging

- The application uses a centralized logging system with request correlation and structured output.
- Environment variables:
  - `ANNOTATION_LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR (default INFO; DEBUG when `--debug`)
  - `ANNOTATION_LOG_JSON`: set to `1` to enable JSON logs (requires `python-json-logger`)
  - `ANNOTATION_LOG_FILE`: path to write logs (defaults to stdout)

- Log format (text):
  `2025-01-01T12:00:00Z INFO annotation.api.annotation_api:api_save_multilabel [req=... ] - message`
  Includes logger name and method via `%(name)s:%(funcName)s` but omits file/line positions per policy.

- Request logs:
  - Access lines at INFO with method, path and status code
  - Detailed debug lines per endpoint when `DEBUG` is enabled

### Annotation Workflow

1. **Select Session** - Choose from automatically discovered sessions
2. **Navigate Frames** - Use arrow keys or navigation buttons
3. **Annotate** - Select game state, importance, add notes
4. **Save Progress** - Press Space to save and move to next frame
5. **Export Dataset** - Download ZIP with manifest and labels

### Keyboard Shortcuts

- **←/→** - Navigate frames
- **Space** - Save annotation and go to next frame
- **1-4** - Select game state (menu/loading/battle/final)
- **Q/W/E** - Set importance level (low/medium/high)

## API Endpoints

- See `API.md` for full details.
- Notable new endpoint:
  - `GET /api/datasets/{dataset_id}/export?include_images=true|false` – downloads a ZIP with `manifest.json` and optional images.

### Export Format

The ZIP contains a `manifest.json` with the following structure:

```
{
  "version": 1,
  "dataset": { "id": 1, "name": "My Dataset", "target_type": "SingleLabelClassification", "created_at": "..." },
  "classes": [ { "id": 10, "name": "cat", "idx": 0 } ],
  "samples": [
    {
      "session_id": "sess1",
      "frame_id": "frame_001",
      "image": { "included": true, "path": "images/sess1/frame_001.png", "frame_path_rel": "../../raw/sess1/frame_001.png" },
      "target": { "type": "single_label", "class_id": 10, "class_name": "cat" }
    }
  ]
}
```

Targets supported:
- Regression: `{ "type": "regression", "value": 0.42 }`
- Single label: `{ "type": "single_label", "class_id": 10, "class_name": "cat" }`
- Multilabel: `{ "type": "multilabel", "class_ids": [10,11], "class_names": ["cat","dog"] }`

## File Structure

```
tools/annotation/
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web interface
└── README.md           # This file
```

## Session Discovery

The tool automatically searches for annotation sessions in:
- `production_data/`
- `collected_data/`
- `markup_data/`
- `sparse_data/`
- Current directory

Sessions are identified by the presence of `metadata.json` files.

## Running Outside Docker

The app and export work without Docker. The SQLite DB path is determined by:
- `ANNOTATION_DB_PATH` environment variable, or
- default `data/annotated/annotated.db` under the repository root.

Example (in a virtualenv):
```bash
export ANNOTATION_DB_PATH=$(pwd)/data/annotated/annotated.db
python tools/annotation/app.py --host 127.0.0.1 --port 5000
curl -f -o dataset_1.zip "http://127.0.0.1:5000/api/datasets/1/export?include_images=1"
```
