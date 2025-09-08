"""
Unit tests for DatasetService.export_zip
"""

import os
import json
import zipfile
from pathlib import Path

from core.session_manager import SessionManager
from services.dataset_service import DatasetService


def _populate_db_for_export(db_conn, temp_data_dir: Path, target_type_id: int = 1):
  from db.projects import create_project, create_dataset
  from db.repository import (
    upsert_session,
    upsert_frame,
    get_session_db_id,
    get_frame_db_id,
    upsert_regression,
    upsert_single_label,
    replace_multilabel_set,
    ensure_membership,
  )

  project_id = create_project(db_conn, "P", None)
  dataset_id = create_dataset(db_conn, project_id, "D", None, target_type_id)

  # Session and frames with metadata containing filenames
  session_id = "sessA"
  session_dir = temp_data_dir / "raw" / session_id
  session_dir.mkdir(parents=True, exist_ok=True)
  frames_md = []
  for i in range(1, 3):
    fid = f"f{i}"
    fname = f"{fid}.png"
    (session_dir / fname).write_bytes(b"x")
    frames_md.append({"frame_id": fid, "filename": fname})

  sid_db = upsert_session(db_conn, session_id, str(session_dir), {"frames": frames_md})
  for i in range(1, 3):
    upsert_frame(db_conn, sid_db, f"f{i}", i * 10)
  db_conn.commit()

  # Add annotations depending on target_type
  # Resolve ids for direct repository inserts
  sid = get_session_db_id(db_conn, session_id)
  assert sid is not None
  fid = get_frame_db_id(db_conn, sid, "f1")
  assert fid is not None

  if target_type_id == 0:
    ensure_membership(db_conn, dataset_id, fid)
    upsert_regression(db_conn, dataset_id, fid, 3.14)
  elif target_type_id == 1:
    # Single label
    from db.classes import get_or_create_dataset_class
    cls = get_or_create_dataset_class(db_conn, dataset_id, "cat")
    db_conn.commit()
    ensure_membership(db_conn, dataset_id, fid)
    upsert_single_label(db_conn, dataset_id, fid, int(cls["id"]))
  else:
    # Multilabel
    from db.classes import get_or_create_dataset_class
    c1 = get_or_create_dataset_class(db_conn, dataset_id, "a")
    c2 = get_or_create_dataset_class(db_conn, dataset_id, "b")
    db_conn.commit()
    ensure_membership(db_conn, dataset_id, fid)
    replace_multilabel_set(db_conn, dataset_id, fid, [int(c1["id"]), int(c2["id"])])

  return dataset_id


def test_export_zip_manifest_regression(temp_data_dir: Path):
  # Use a file-backed DB so service connections see the same data
  from db.connection import get_connection
  from db.schema import init_db
  db_path = temp_data_dir / "unit_ds_export.db"
  os.environ["ANNOTATION_DB_PATH"] = str(db_path)
  conn = get_connection()
  init_db(conn)
  sm = SessionManager(data_root=str(temp_data_dir))
  svc = DatasetService(sm)
  dsid = _populate_db_for_export(conn, temp_data_dir, target_type_id=0)

  zip_path = svc.export_zip(dsid, include_images=False)
  try:
    with zipfile.ZipFile(zip_path, "r") as zf:
      manifest = json.loads(zf.read("manifest.json"))
      assert manifest["dataset"]["id"] == dsid
      assert manifest["version"] == 1
      assert any(s["target"]["type"] == "regression" for s in manifest["samples"])  # type: ignore
  finally:
    Path(zip_path).unlink(missing_ok=True)


def test_export_zip_includes_images(temp_data_dir: Path):
  from db.connection import get_connection
  from db.schema import init_db
  db_path = temp_data_dir / "unit_ds_export_img.db"
  os.environ["ANNOTATION_DB_PATH"] = str(db_path)
  conn = get_connection()
  init_db(conn)
  sm = SessionManager(data_root=str(temp_data_dir))
  svc = DatasetService(sm)
  dsid = _populate_db_for_export(conn, temp_data_dir, target_type_id=0)

  zip_path = svc.export_zip(dsid, include_images=True)
  try:
    with zipfile.ZipFile(zip_path, "r") as zf:
      names = zf.namelist()
      assert any(n.startswith("images/") for n in names)
  finally:
    Path(zip_path).unlink(missing_ok=True)

