# Web-based Annotation Tool

A Flask web application for annotating collected game frame data in-place via browser.

## Features

- **Web Interface** - Clean, responsive web UI accessible from any browser
- **In-place Annotation** - No file transfers needed, works directly with collected data
- **Session Management** - Automatically discovers and loads annotation sessions
- **Real-time Progress** - Live statistics and progress tracking
- **Keyboard Shortcuts** - Efficient annotation workflow
- **DB-backed storage** - Annotations stored in SQLite with views and triggers
- **Dataset Download** - Download labeled items as CSV or JSONL

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
5. **Download Labeled Data** - Use API to fetch CSV/JSONL for training

### Keyboard Shortcuts

- **←/→** - Navigate frames
- **Space** - Save annotation and go to next frame
- **1-4** - Select game state (menu/loading/battle/final)
- **Q/W/E** - Set importance level (low/medium/high)

## API Endpoints

- `GET /` - Main annotation interface
- `GET /api/sessions` - List available sessions
- `POST /api/load_session` - Load specific session
- `GET /api/frame/<idx>` - Get frame data and annotation
- `GET /api/image/<idx>` - Serve frame image (now served with long-lived cache headers)
- `POST /api/annotate` - Save frame annotation
- `POST /api/export` - Export annotated dataset
- `GET /api/stats` - Get annotation statistics

### DB-backed (current)

- `GET /api/datasets/{dataset_id}/progress` - Dataset progress summary
- `GET /api/datasets/{dataset_id}/labeled` - Labeled items view
- `GET /api/datasets/{dataset_id}/sessions/{session_id}/unlabeled_indices` - Indices of unlabeled frames for filtering
- `GET /api/datasets/{dataset_id}/download?format=csv|jsonl` - Download labeled items as file

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

## Download Formats

- CSV columns (stable order): `session_id, frame_id, value_real, single_label_class_id, multilabel_class_ids_csv, frame_path_rel`
- JSONL: one JSON object per labeled item with the same fields as `/api/datasets/{dataset_id}/labeled` plus `frame_path_rel` when resolvable.
