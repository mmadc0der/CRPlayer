"""
Unit tests for DatasetService.export_zip
"""

import json
import zipfile
from pathlib import Path

from core.session_manager import SessionManager
from services.dataset_service import DatasetService


def _populate_db_for_export(temp_db, temp_data_dir: Path, target_type_id: int = 1):
  from db.projects import create_project, create_dataset
  from db.repository import upsert_session, upsert_frame
  from services.annotation_service import AnnotationService

  project_id = create_project(temp_db, "P", None)
  dataset_id = create_dataset(temp_db, project_id, "D", None, target_type_id)

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

  sid_db = upsert_session(temp_db, session_id, str(session_dir), {"frames": frames_md})
  for i in range(1, 3):
    upsert_frame(temp_db, sid_db, f"f{i}", i * 10)
  temp_db.commit()

  # Add annotations depending on target_type
  ann = AnnotationService(SessionManager(data_root=str(temp_data_dir), conn=temp_db))
  if target_type_id == 0:
    ann.save_regression(session_id, dataset_id, "f1", 3.14)
  elif target_type_id == 1:
    # Single label
    from db.classes import get_or_create_dataset_class
    cls = get_or_create_dataset_class(temp_db, dataset_id, "cat")
    temp_db.commit()
    ann.save_single_label(session_id, dataset_id, "f1", cls["id"])  # type: ignore
  else:
    # Multilabel
    from db.classes import get_or_create_dataset_class
    c1 = get_or_create_dataset_class(temp_db, dataset_id, "a")
    c2 = get_or_create_dataset_class(temp_db, dataset_id, "b")
    temp_db.commit()
    ann.save_multilabel(session_id, dataset_id, "f1", [c1["id"], c2["id"]])  # type: ignore

  return dataset_id


def test_export_zip_manifest_regression(temp_db, temp_data_dir: Path):
  sm = SessionManager(data_root=str(temp_data_dir), conn=temp_db)
  svc = DatasetService(sm)
  dsid = _populate_db_for_export(temp_db, temp_data_dir, target_type_id=0)

  zip_path = svc.export_zip(dsid, include_images=False)
  try:
    with zipfile.ZipFile(zip_path, "r") as zf:
      manifest = json.loads(zf.read("manifest.json"))
      assert manifest["dataset"]["id"] == dsid
      assert manifest["version"] == 1
      assert any(s["target"]["type"] == "regression" for s in manifest["samples"])  # type: ignore
  finally:
    Path(zip_path).unlink(missing_ok=True)


def test_export_zip_includes_images(temp_db, temp_data_dir: Path):
  sm = SessionManager(data_root=str(temp_data_dir), conn=temp_db)
  svc = DatasetService(sm)
  dsid = _populate_db_for_export(temp_db, temp_data_dir, target_type_id=0)

  zip_path = svc.export_zip(dsid, include_images=True)
  try:
    with zipfile.ZipFile(zip_path, "r") as zf:
      names = zf.namelist()
      assert any(n.startswith("images/") for n in names)
  finally:
    Path(zip_path).unlink(missing_ok=True)

