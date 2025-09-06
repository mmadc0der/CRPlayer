"""
Tests for core session manager.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.session_manager import SessionManager
from tests.fixtures.factories import SessionMetadataFactory, create_session_with_frames


class TestSessionManager:
  """Test SessionManager functionality."""

  def test_init_with_defaults(self):
    """Test SessionManager initialization with default values."""
    sm = SessionManager()

    assert sm.data_root == Path("data")
    assert sm.raw_dir == Path("data/raw")
    assert sm._external_conn is None

  def test_init_with_custom_data_root(self, temp_data_dir):
    """Test SessionManager initialization with custom data root."""
    sm = SessionManager(data_root=str(temp_data_dir))

    assert sm.data_root == temp_data_dir
    assert sm.raw_dir == temp_data_dir / "raw"

  def test_init_with_external_connection(self, temp_db):
    """Test SessionManager initialization with external connection."""
    sm = SessionManager(conn=temp_db)

    assert sm._external_conn == temp_db

  def test_conn_uses_external_connection(self, temp_db):
    """Test that _conn() uses external connection when provided."""
    sm = SessionManager(conn=temp_db)

    conn = sm._conn()
    assert conn == temp_db

  @patch("core.session_manager.get_connection")
  @patch("core.session_manager.init_db")
  def test_conn_creates_new_connection(self, mock_init_db, mock_get_connection, temp_db):
    """Test that _conn() creates new connection when none provided."""
    mock_get_connection.return_value = temp_db

    sm = SessionManager()
    conn = sm._conn()

    mock_get_connection.assert_called_once()
    mock_init_db.assert_called_once_with(temp_db)
    assert conn == temp_db

  def test_discover_sessions_empty_database(self, temp_db):
    """Test discover_sessions with empty database."""
    sm = SessionManager(conn=temp_db)

    sessions = sm.discover_sessions()
    assert sessions == []

  def test_discover_sessions_with_data(self, populated_db, temp_data_dir):
    """Test discover_sessions with populated database."""
    sm = SessionManager(data_root=str(temp_data_dir), conn=populated_db)

    sessions = sm.discover_sessions()

    assert len(sessions) > 0
    session = sessions[0]

    # Check required fields
    assert "session_id" in session
    assert "path" in session
    assert "frames_count" in session
    assert "game_name" in session
    assert "start_time" in session

  def test_find_session_by_id_existing(self, populated_db, temp_data_dir):
    """Test find_session_by_id with existing session."""
    sm = SessionManager(data_root=str(temp_data_dir), conn=populated_db)

    session = sm.find_session_by_id("test_session_001")

    assert session is not None
    assert session["session_id"] == "test_session_001"

  def test_find_session_by_id_nonexistent(self, temp_db):
    """Test find_session_by_id with nonexistent session."""
    sm = SessionManager(conn=temp_db)

    session = sm.find_session_by_id("nonexistent_session")
    assert session is None

  def test_get_session_path_by_id_existing(self, populated_db, temp_data_dir):
    """Test get_session_path_by_id with existing session."""
    sm = SessionManager(data_root=str(temp_data_dir), conn=populated_db)

    path = sm.get_session_path_by_id("test_session_001")
    assert path is not None
    assert isinstance(path, str)

  def test_get_session_path_by_id_nonexistent(self, temp_db):
    """Test get_session_path_by_id with nonexistent session."""
    sm = SessionManager(conn=temp_db)

    path = sm.get_session_path_by_id("nonexistent_session")
    assert path is None

  def test_get_session_db_id_existing(self, populated_db):
    """Test get_session_db_id with existing session."""
    sm = SessionManager(conn=populated_db)

    db_id = sm.get_session_db_id("test_session_001")
    assert db_id is not None
    assert isinstance(db_id, int)
    assert db_id > 0

  def test_get_session_db_id_nonexistent(self, temp_db):
    """Test get_session_db_id with nonexistent session."""
    sm = SessionManager(conn=temp_db)

    db_id = sm.get_session_db_id("nonexistent_session")
    assert db_id is None

  def test_get_frames_for_session_existing(self, populated_db):
    """Test get_frames_for_session with existing session."""
    sm = SessionManager(conn=populated_db)

    frames = sm.get_frames_for_session("test_session_001")

    assert len(frames) > 0
    frame = frames[0]
    assert "frame_id" in frame
    assert "ts_ms" in frame

  def test_get_frames_for_session_nonexistent(self, temp_db):
    """Test get_frames_for_session with nonexistent session."""
    sm = SessionManager(conn=temp_db)

    frames = sm.get_frames_for_session("nonexistent_session")
    assert frames == []

  def test_status_json_reading(self, temp_data_dir, populated_db):
    """Test reading status.json files from session directories."""
    sm = SessionManager(data_root=str(temp_data_dir), conn=populated_db)

    # Update the session's root_path to match our test directory
    session_dir = temp_data_dir / "raw" / "test_session_001"
    session_dir.mkdir(parents=True, exist_ok=True)
    populated_db.execute("UPDATE sessions SET root_path = ? WHERE session_id = ?",
                         (str(session_dir), "test_session_001"))
    populated_db.commit()

    status_data = {"collection_status": "completed", "annotation_status": "in_progress", "notes": "Test session"}

    status_file = session_dir / "status.json"
    status_file.write_text(json.dumps(status_data))

    sessions = sm.discover_sessions()
    session = next((s for s in sessions if s["session_id"] == "test_session_001"), None)

    assert session is not None
    assert session["collection_status"] == "completed"
    assert session["annotation_status"] == "in_progress"
    assert session["notes"] == "Test session"

  def test_metadata_parsing_error_handling(self, temp_db):
    """Test handling of invalid metadata JSON."""
    from db.repository import upsert_session

    # Insert session with invalid JSON
    upsert_session(temp_db, "test_session", "/path", {})
    temp_db.execute("UPDATE sessions SET metadata_json = ? WHERE session_id = ?", ("invalid json {", "test_session"))
    temp_db.commit()

    sm = SessionManager(conn=temp_db)
    sessions = sm.discover_sessions()

    # Should handle the error gracefully
    assert len(sessions) == 1
    session = sessions[0]
    assert session["session_id"] == "test_session"
    # Should have default values when JSON parsing fails
    assert session.get("game_name") == "Unknown"

  def test_frame_count_calculation(self, temp_db):
    """Test that frame counts are calculated correctly."""
    from db.repository import upsert_session, upsert_frame

    # Create session with frames
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})
    for i in range(5):
      upsert_frame(temp_db, session_db_id, f"frame_{i:03d}", i * 1000)
    temp_db.commit()

    sm = SessionManager(conn=temp_db)
    sessions = sm.discover_sessions()

    assert len(sessions) == 1
    session = sessions[0]
    assert session["frames_count"] == 5

  @patch("core.session_manager.Path.exists")
  def test_status_file_not_found_handling(self, mock_exists, populated_db, temp_data_dir):
    """Test handling when status.json doesn't exist."""
    mock_exists.return_value = False

    sm = SessionManager(data_root=str(temp_data_dir), conn=populated_db)
    sessions = sm.discover_sessions()

    # Should work without status.json
    assert len(sessions) > 0
    session = sessions[0]
    # Should have default status values
    assert session.get("collection_status") == "unknown"
    assert session.get("annotation_status") == "not_started"
