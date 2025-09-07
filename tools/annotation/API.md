# CRPlayer Annotation API (DB-backed)

This document describes the current DB-backed API endpoints, request/response schemas, and data model expectations used by the CRPlayer annotation tool.

Base path: all routes are mounted by `tools/annotation/api.py`.

## Sessions and Indexing

- GET /api/sessions

  - List discovered sessions from `data/raw/` with metadata.

- POST /api/reindex
  - Scans `data/raw/` to populate SQLite `sessions` and `frames` tables.
  - Also prunes DB sessions whose `root_path` is missing on disk.
  - Response: `{ ok: true, summary: { sessions_indexed, sessions_removed, frames_indexed, details } }`

## Projects & Datasets

- GET /api/projects
- POST /api/projects

  - Body: `{ name: string, description?: string }`

- GET /api/projects/{project_id}/datasets
- POST /api/projects/{project_id}/datasets

  - Body: `{ name: string, description?: string, target_type_id: number }`
    - target_type_id: 1=Regression, 2=SingleLabelClassification, 3=MultiLabelClassification

- GET /api/datasets/{dataset_id}/progress

  - Returns aggregate counts of labeled/unlabeled for the dataset.

- GET /api/datasets/{dataset_id}/labeled

  - Lists labeled items using `dataset_labeled_view`.

- GET /api/datasets/{dataset_id}/sessions/{session_id}/unlabeled_indices
  - Returns indices of unlabeled frames for the given dataset-session pair.
  - Response: `{ indices: number[], total: number, labeled: number, unlabeled: number }`

## Enrollment

- POST /api/datasets/{dataset_id}/enroll_session
  - Body: `{ session_id: string, settings?: object }`
  - Behavior:
    - Creates membership rows in `annotations` for all frames of the session with status `unlabeled`.
    - If `settings` provided, stores once in `dataset_session_settings` for the pair (dataset_id, session_id).

## Frames & Images

- GET /api/frame?session_id=...&project_name=...&idx=number

  - Returns `{ frame: {...} }` by index within session.
  - Note: legacy filesystem annotations are removed.

- GET /api/image?session_id=...&idx=number
  - Sends the frame image file for the given session/index.
  - Caching: `Cache-Control: public, max-age=31536000, immutable` with conditional responses enabled.

## Annotations (DB-backed)

All annotation write endpoints support resolving the target frame via `frame_id` or `frame_idx`:

- If `frame_id` omitted and `frame_idx` provided, the server resolves `frame_id` using session metadata.
- On write, membership in `annotations` is ensured; status is updated accordingly by repository logic.

- POST /api/annotations/regression

  - Body:
    ```json
    {
      "session_id": "string",
      "dataset_id": 1,
      "frame_id": "optional string",
      "frame_idx": 12,
      "value": 0.73,
      "override_settings": { "hotkeys": { "A": 1 } }
    }
    ```
  - Response: unified frame annotation object including `status`, payloads, and `effective_settings`.

- POST /api/annotations/single_label

  - Body:
    ```json
    {
      "session_id": "string",
      "dataset_id": 1,
      "frame_id": "optional string",
      "frame_idx": 12,
      "class_id": 3,
      "override_settings": { "categories": ["cat", "dog"] }
    }
    ```
  - Response: unified frame annotation object including `status`, payloads, and `effective_settings`.

- POST /api/annotations/multilabel
  - Body:
    ```json
    {
      "session_id": "string",
      "dataset_id": 1,
      "frame_id": "optional string",
      "frame_idx": 12,
      "class_ids": [1, 4, 5],
      "override_settings": { "multi": true }
    }
    ```
  - Response: unified frame annotation object including `status`, payloads, and `effective_settings`.

## Dataset-Session Settings

- PUT /api/datasets/{dataset_id}/sessions/{session_id}/settings

  - Body: `{ "settings": object }`
  - Upserts baseline settings for the dataset-session pair in `dataset_session_settings`.

- GET /api/datasets/{dataset_id}/sessions/{session_id}/settings

  - Response: `{ settings: object | null }`

- DELETE /api/datasets/{dataset_id}/sessions/{session_id}/settings
  - Deletes the dataset-session baseline settings row.

## Data Model Notes

- `annotations` table holds membership per (dataset_id, frame_id) with `status` and optional per-frame `settings_json` (override). Baseline settings live in `dataset_session_settings`.
- Effective settings are a shallow merge of baseline and per-frame override.
- Referential integrity is enforced using composite FKs (e.g., `dataset_classes` belongs to its dataset).
- Deterministic multilabel aggregation is provided via `annotations_view` ordering by `dataset_classes.idx`.

## Deprecations

- Filesystem-based annotation storage and `POST /api/save_annotation` are removed.
- Any code relying on per-frame JSON under `data/annotated/` should migrate to the DB-backed endpoints above.
