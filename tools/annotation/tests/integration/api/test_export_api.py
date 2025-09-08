"""
Integration tests for dataset export API endpoint.
"""

import io
import json
import zipfile
from pathlib import Path

from flask.testing import FlaskClient


def _setup_regression_dataset_with_labels(temp_data_dir: Path) -> int:
  """Create project/dataset/session/frames and add regression labels in the DB.

  Returns created dataset_id.
  """
  from db.connection import get_connection
  from db.schema import init_db
  from db.projects import create_project, create_dataset
  from db.repository import upsert_session, upsert_frame
  from services.annotation_service import AnnotationService
  from core.session_manager import SessionManager

  conn = get_connection()
  init_db(conn)

  # Create project and Regression dataset
  project_id = create_project(conn, "Test Project", "Export tests")
  dataset_id = create_dataset(conn, project_id, "Export DS", "Export dataset", 0)  # Regression

  # Prepare session with frames and metadata containing filenames
  session_id = "test_session_001"
  session_root = temp_data_dir / "raw" / session_id
  session_root.mkdir(parents=True, exist_ok=True)

  # Create a couple of image files and matching metadata
  frames_md = []
  for i in range(1, 3):
    fid = f"frame_{i:03d}"
    fname = f"{fid}.png"
    (session_root / fname).write_bytes(b"img")
    frames_md.append({"frame_id": fid, "filename": fname})

  md = {"game_name": "TestGame", "frames": frames_md}
  sid_db = upsert_session(conn, session_id, str(session_root), md)
  for i in range(1, 3):
    fid = f"frame_{i:03d}"
    upsert_frame(conn, sid_db, fid, i * 1000)
  conn.commit()
  conn.close()

  # Add regression labels via service
  svc = AnnotationService(SessionManager(data_root=str(temp_data_dir)))
  svc.save_regression(session_id=session_id, dataset_id=dataset_id, frame_id="frame_001", value=1.23)
  svc.save_regression(session_id=session_id, dataset_id=dataset_id, frame_id="frame_002", value=4.56)

  return dataset_id


class TestExportAPI:

  def test_export_zip_without_images(self, client: FlaskClient, temp_data_dir: Path):
    dataset_id = _setup_regression_dataset_with_labels(temp_data_dir)

    resp = client.get(f"/api/datasets/{dataset_id}/export")
    assert resp.status_code == 200
    ctype = resp.headers.get("Content-Type", "")
    assert ("zip" in ctype) or ("octet-stream" in ctype)

    # Inspect zip content from bytes
    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
      names = set(zf.namelist())
      assert "manifest.json" in names
      # No images should be present
      assert not any(n.startswith("images/") for n in names)
      manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
      assert manifest["dataset"]["id"] == dataset_id
      assert manifest["version"] == 1
      assert len(manifest["samples"]) >= 2
      # Check target for regression
      assert all(s["target"]["type"] == "regression" for s in manifest["samples"])  # type: ignore

  def test_export_zip_with_images(self, client: FlaskClient, temp_data_dir: Path):
    dataset_id = _setup_regression_dataset_with_labels(temp_data_dir)

    resp = client.get(f"/api/datasets/{dataset_id}/export?include_images=1")
    assert resp.status_code == 200

    with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
      names = set(zf.namelist())
      assert "manifest.json" in names
      # Images should be included
      assert any(n.startswith("images/") for n in names)
      manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
      # Verify image info flags
      included = [s["image"]["included"] for s in manifest["samples"]]
      assert any(included)
