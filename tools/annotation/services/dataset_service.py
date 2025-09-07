from __future__ import annotations

from typing import List, Optional, Dict, Any
import json
import tempfile
import zipfile
from pathlib import Path
import os

import warnings

from core.session_manager import SessionManager
from core.dataset_builder import DatasetBuilder
from core.path_resolver import resolve_session_dir, resolve_frame_relative_path


class DatasetService:
  """Orchestrates dataset build using DatasetBuilder and resolvers."""

  def __init__(self, session_manager: SessionManager):
    self.sm = session_manager
    self.builder = DatasetBuilder(session_manager)

  def fetch_labeled(self, dataset_id: int) -> List[Dict[str, Any]]:
    """Return labeled items for a dataset (DB-backed)."""
    return self.builder.fetch_labeled(dataset_id)

  def build_from_sessions(self, project_name: str, session_ids: Optional[List[str]] = None):
    """DEPRECATED: Legacy filesystem dataset builds are not supported."""
    warnings.warn(
      "build_from_sessions is deprecated. Use DB-backed dataset endpoints or DatasetService.fetch_labeled.",
      DeprecationWarning,
      stacklevel=2,
    )
    raise NotImplementedError("Use DB-backed dataset flows")

  def export_zip(self, dataset_id: int, include_images: bool = False) -> Path:
    """Build a ZIP export for the given dataset and return the temp file path.

    Manifest format (manifest.json):
      {
        "version": 1,
        "dataset": { id, name, target_type, created_at },
        "classes": [ { id, name, idx } ],
        "samples": [
          {
            "session_id": str,
            "frame_id": str,
            "image": {
              "included": bool,
              "path": str | null,            # path inside ZIP when included
              "frame_path_rel": str | null   # best-effort external relative path for reference
            },
            "target": {
              "type": "regression" | "single_label" | "multilabel",
              "value"?: float,
              "class_id"?: int,
              "class_name"?: str,
              "class_ids"?: [int],
              "class_names"?: [str]
            }
          }, ...
        ]
      }
    """
    from db.connection import get_connection
    from db.schema import init_db
    from db.projects import get_dataset as db_get_dataset
    from db.classes import list_dataset_classes as db_list_dataset_classes
    from core.path_resolver import resolve_session_dir, resolve_frame_absolute_path

    conn = get_connection()
    init_db(conn)

    meta = db_get_dataset(conn, int(dataset_id))
    if not meta:
      raise ValueError(f"Dataset not found: {dataset_id}")

    classes = db_list_dataset_classes(conn, int(dataset_id))
    class_id_to_name: Dict[int, str] = {int(c["id"]): str(c["name"]) for c in classes}

    labeled_rows = self.builder.fetch_labeled(int(dataset_id))

    tmp = tempfile.NamedTemporaryFile(prefix=f"dataset_{dataset_id}_", suffix=".zip", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
      manifest: Dict[str, Any] = {
        "version": 1,
        "dataset": {
          "id": int(meta["id"]),
          "name": meta.get("name"),
          "target_type": meta.get("target_type_name"),
          "created_at": meta.get("created_at"),
        },
        "classes": classes,
        "samples": [],
      }

      # Build samples and optionally embed images
      for r in labeled_rows:
        session_id = str(r.get("session_id"))
        frame_id = str(r.get("frame_id"))
        frame_path_rel = r.get("frame_path_rel")

        image_info: Dict[str, Any] = {"included": False, "path": None, "frame_path_rel": frame_path_rel}

        # Attempt to resolve absolute path via metadata filename if embedding images
        abs_image: Optional[Path] = None
        if include_images:
          try:
            session_info = self.sm.find_session_by_id(session_id)
            if session_info and session_info.get("metadata"):
              filename = None
              for fr in session_info["metadata"].get("frames", []) or []:
                if str(fr.get("frame_id")) == frame_id:
                  filename = fr.get("filename")
                  break
              if filename:
                session_dir = Path(session_info["session_dir"]) if session_info.get("session_dir") else None
                if session_dir and Path(session_dir).exists():
                  abs_image = resolve_frame_absolute_path(self.sm, Path(session_dir), session_id, str(filename))
          except Exception:
            abs_image = None

        if include_images and abs_image and Path(abs_image).exists():
          # Store under images/<session_id>/<basename>
          basename = Path(abs_image).name
          arcname = os.path.join("images", session_id, basename)
          try:
            zf.write(str(abs_image), arcname)
            image_info["included"] = True
            image_info["path"] = arcname
          except Exception:
            # If embedding fails, keep as external reference only
            image_info["included"] = False
            image_info["path"] = None

        # Build target object depending on available payloads
        target: Dict[str, Any] = {}
        if r.get("value_real") is not None:
          target = {"type": "regression", "value": float(r["value_real"])}
        elif r.get("single_label_class_id") is not None:
          cid = int(r["single_label_class_id"])  # type: ignore
          target = {
            "type": "single_label",
            "class_id": cid,
            "class_name": class_id_to_name.get(cid),
          }
        else:
          csv_ids = r.get("multilabel_class_ids_csv")
          ids: List[int] = []
          if isinstance(csv_ids, str) and csv_ids.strip():
            try:
              ids = [int(x) for x in csv_ids.split(",") if x]
            except Exception:
              ids = []
          target = {
            "type": "multilabel",
            "class_ids": ids,
            "class_names": [class_id_to_name.get(i) for i in ids],
          }

        manifest["samples"].append({
          "session_id": session_id,
          "frame_id": frame_id,
          "image": image_info,
          "target": target,
        })

      # Write manifest
      zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return tmp_path
