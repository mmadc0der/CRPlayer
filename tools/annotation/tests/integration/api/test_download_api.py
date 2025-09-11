"""
Integration tests for dataset download endpoint.
"""

import json
from flask.testing import FlaskClient
from unittest.mock import patch

from tests.fixtures.factories import ProjectDataFactory, DatasetDataFactory


class TestDatasetDownloadAPI:

  def _create_project_and_dataset(self, client: FlaskClient) -> int:
    project_data = ProjectDataFactory()
    pr = client.post("/api/projects", data=json.dumps(project_data), content_type="application/json")
    project_id = pr.get_json()["id"]

    dataset_data = DatasetDataFactory()
    dr = client.post(f"/api/projects/{project_id}/datasets",
                     data=json.dumps(dataset_data),
                     content_type="application/json")
    return dr.get_json()["id"]

  def test_download_csv_success(self, client: FlaskClient):
    dataset_id = self._create_project_and_dataset(client)
    sample_rows = [
      {
        "dataset_id": dataset_id,
        "session_id": "test_session_000",
        "frame_id": "frame_001",
        "value_real": None,
        "single_label_class_id": 2,
        "multilabel_class_ids_csv": None,
        "frame_path_rel": "raw/test_session_000/frame_001.png",
      },
      {
        "dataset_id": dataset_id,
        "session_id": "test_session_000",
        "frame_id": "frame_002",
        "value_real": 0.75,
        "single_label_class_id": None,
        "multilabel_class_ids_csv": "1,3",
        "frame_path_rel": "raw/test_session_000/frame_002.png",
      },
    ]

    with patch("services.dataset_service.DatasetService.fetch_labeled", return_value=sample_rows):
      resp = client.get(f"/api/datasets/{dataset_id}/download?format=csv&filename=my_export")
      assert resp.status_code == 200
      assert resp.headers.get("Content-Type", "").startswith("text/csv")
      cd = resp.headers.get("Content-Disposition", "")
      assert "attachment" in cd and "my_export.csv" in cd

      body = resp.data.decode("utf-8").splitlines()
      assert body[0].split(",") == [
        "session_id",
        "frame_id",
        "value_real",
        "single_label_class_id",
        "multilabel_class_ids_csv",
        "frame_path_rel",
      ]
      assert len(body) == 1 + len(sample_rows)
      assert "frame_001" in body[1]
      assert "frame_002" in body[2]

  def test_download_jsonl_success(self, client: FlaskClient):
    dataset_id = self._create_project_and_dataset(client)
    sample_rows = [
      {
        "dataset_id": dataset_id,
        "session_id": "s1",
        "frame_id": "f1",
        "value_real": 1.23,
        "single_label_class_id": None,
        "multilabel_class_ids_csv": None,
        "frame_path_rel": None,
      },
      {
        "dataset_id": dataset_id,
        "session_id": "s1",
        "frame_id": "f2",
        "value_real": None,
        "single_label_class_id": 5,
        "multilabel_class_ids_csv": None,
        "frame_path_rel": None,
      },
    ]

    with patch("services.dataset_service.DatasetService.fetch_labeled", return_value=sample_rows):
      resp = client.get(f"/api/datasets/{dataset_id}/download?format=jsonl")
      assert resp.status_code == 200
      ctype = resp.headers.get("Content-Type", "")
      assert ctype.startswith("application/x-ndjson") or ctype.startswith("application/octet-stream")
      cd = resp.headers.get("Content-Disposition", "")
      assert "attachment" in cd and ".jsonl" in cd

      lines = resp.data.decode("utf-8").splitlines()
      assert len(lines) == len(sample_rows)
      parsed = [json.loads(line) for line in lines]
      assert parsed[0]["frame_id"] == "f1"
      assert parsed[1]["single_label_class_id"] == 5

  def test_download_bad_format(self, client: FlaskClient):
    dataset_id = self._create_project_and_dataset(client)
    resp = client.get(f"/api/datasets/{dataset_id}/download?format=xml")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "bad_request"

  def test_download_dataset_not_found(self, client: FlaskClient):
    resp = client.get("/api/datasets/999999/download")
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["code"] == "not_found"
