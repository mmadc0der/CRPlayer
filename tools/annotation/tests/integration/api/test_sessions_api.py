"""
Integration tests for sessions API endpoints.
"""

import pytest
import json
from flask.testing import FlaskClient
from unittest.mock import patch

from tests.fixtures.factories import SessionMetadataFactory, create_session_with_frames


class TestSessionsAPI:
  """Test sessions API endpoints integration."""

  def test_discover_sessions_empty(self, client: FlaskClient):
    """Test discovering sessions when none exist."""
    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = []

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()
      assert data == []

  def test_discover_sessions_with_data(self, client: FlaskClient):
    """Test discovering sessions with existing data."""
    mock_sessions = [
      {
        "session_id": "test_session_001",
        "path": "/test/path/session_001",
        "frames_count": 10,
        "game_name": "TestGame",
        "start_time": "2023-01-01T00:00:00Z",
        "collection_status": "completed",
        "annotation_status": "in_progress",
      },
      {
        "session_id": "test_session_002",
        "path": "/test/path/session_002",
        "frames_count": 5,
        "game_name": "TestGame",
        "start_time": "2023-01-01T01:00:00Z",
        "collection_status": "completed",
        "annotation_status": "not_started",
      },
    ]

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()

      assert len(data) == 2
      assert data[0]["session_id"] == "test_session_001"
      assert data[0]["frames_count"] == 10
      assert data[1]["session_id"] == "test_session_002"
      assert data[1]["frames_count"] == 5

  def test_discover_sessions_error_handling(self, client: FlaskClient):
    """Test error handling in session discovery."""
    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.side_effect = Exception("Database error")

      response = client.get("/api/sessions")

      assert response.status_code == 500
      data = response.get_json()
      assert data["code"] == "sessions_error"
      assert "Database error" in data["details"]["error"]

  def test_session_metadata_fields(self, client: FlaskClient):
    """Test that all expected session metadata fields are returned."""
    session_data = create_session_with_frames("test_session", frame_count=3)
    mock_sessions = [{
      "session_id": session_data["session_id"],
      "path": session_data["root_path"],
      "frames_count": len(session_data["frames"]),
      "game_name": session_data["metadata"]["game_name"],
      "start_time": session_data["metadata"]["start_time"],
      "version": session_data["metadata"]["version"],
      "resolution": session_data["metadata"]["resolution"],
      "fps": session_data["metadata"]["fps"],
      "collection_status": "completed",
      "annotation_status": "not_started",
    }]

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()

      session = data[0]
      assert "session_id" in session
      assert "path" in session
      assert "frames_count" in session
      assert "game_name" in session
      assert "start_time" in session
      assert "version" in session
      assert "resolution" in session
      assert "fps" in session
      assert "collection_status" in session
      assert "annotation_status" in session

  def test_session_ordering(self, client: FlaskClient):
    """Test that sessions are returned in consistent order."""
    mock_sessions = [
      {
        "session_id": "session_c",
        "path": "/c",
        "frames_count": 1
      },
      {
        "session_id": "session_a",
        "path": "/a",
        "frames_count": 1
      },
      {
        "session_id": "session_b",
        "path": "/b",
        "frames_count": 1
      },
    ]

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()

      # Sessions should be returned in the order provided by SessionManager
      assert len(data) == 3
      assert data[0]["session_id"] == "session_c"
      assert data[1]["session_id"] == "session_a"
      assert data[2]["session_id"] == "session_b"

  def test_session_with_missing_metadata(self, client: FlaskClient):
    """Test handling of sessions with missing or incomplete metadata."""
    mock_sessions = [{
      "session_id": "incomplete_session",
      "path": "/incomplete/path",
      "frames_count": 5,
      # Missing game_name, start_time, etc.
      "collection_status": "unknown",
      "annotation_status": "not_started",
    }]

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()

      session = data[0]
      assert session["session_id"] == "incomplete_session"
      assert session["frames_count"] == 5
      # Should handle missing metadata gracefully
      assert "collection_status" in session
      assert "annotation_status" in session

  def test_large_session_count(self, client: FlaskClient):
    """Test handling of large numbers of sessions."""
    # Create many mock sessions
    mock_sessions = []
    for i in range(100):
      mock_sessions.append({
        "session_id": f"session_{i:03d}",
        "path": f"/test/session_{i:03d}",
        "frames_count": i + 1,
        "game_name": "TestGame",
        "start_time": "2023-01-01T00:00:00Z",
      })

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()

      assert len(data) == 100
      # Verify first and last sessions
      assert data[0]["session_id"] == "session_000"
      assert data[99]["session_id"] == "session_099"

  def test_session_path_formats(self, client: FlaskClient):
    """Test different session path formats are handled correctly."""
    mock_sessions = [
      {
        "session_id": "unix_path_session",
        "path": "/unix/style/path",
        "frames_count": 1
      },
      {
        "session_id": "relative_path_session",
        "path": "relative/path",
        "frames_count": 1
      },
      {
        "session_id": "deep_path_session",
        "path": "/very/deep/nested/directory/structure/session",
        "frames_count": 1,
      },
    ]

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()

      assert len(data) == 3
      paths = [session["path"] for session in data]
      assert "/unix/style/path" in paths
      assert "relative/path" in paths
      assert "/very/deep/nested/directory/structure/session" in paths

  def test_session_frame_counts(self, client: FlaskClient):
    """Test that frame counts are correctly reported."""
    mock_sessions = [
      {
        "session_id": "empty_session",
        "path": "/empty",
        "frames_count": 0
      },
      {
        "session_id": "small_session",
        "path": "/small",
        "frames_count": 5
      },
      {
        "session_id": "large_session",
        "path": "/large",
        "frames_count": 1000
      },
    ]

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      response = client.get("/api/sessions")

      assert response.status_code == 200
      data = response.get_json()

      frame_counts = {session["session_id"]: session["frames_count"] for session in data}
      assert frame_counts["empty_session"] == 0
      assert frame_counts["small_session"] == 5
      assert frame_counts["large_session"] == 1000

  def test_reindex_prunes_missing_sessions(self, client: FlaskClient, tmp_path):
    """Sessions removed from disk should be pruned from DB on reindex."""
    import os
    from db.connection import get_connection
    from db.repository import upsert_session, upsert_frame

    # Create a session pointing to a temp dir, then remove the dir
    sess_dir = tmp_path / "raw" / "ghost_session"
    sess_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    try:
      sid = upsert_session(conn, "ghost_session", str(sess_dir), {"frames": []})
      upsert_frame(conn, sid, "frame_001", 1000)
      conn.commit()
    finally:
      conn.close()

    # Remove directory to simulate missing session
    os.rmdir(sess_dir)

    # Run reindex
    resp = client.post("/api/reindex")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "sessions_removed" in data["summary"]
    assert data["summary"]["sessions_removed"] >= 1

  def test_concurrent_session_requests(self, client: FlaskClient):
    """Test that concurrent session discovery requests work correctly."""
    mock_sessions = [{"session_id": "test_session", "path": "/test", "frames_count": 1}]

    with patch("core.session_manager.SessionManager.discover_sessions") as mock_discover:
      mock_discover.return_value = mock_sessions

      # Make multiple concurrent requests (simulated by rapid succession)
      responses = []
      for _ in range(5):
        response = client.get("/api/sessions")
        responses.append(response)

      # All requests should succeed
      for response in responses:
        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 1
        assert data[0]["session_id"] == "test_session"
