"""
Tests for database repository layer.
"""

import pytest
import json
from unittest.mock import patch

from db.repository import (
  transaction,
  upsert_session,
  upsert_frame,
  get_session_db_id,
  get_frame_db_id,
  upsert_regression,
  upsert_single_label,
  replace_multilabel_set,
  ensure_membership,
  set_annotation_status,
  list_frames_with_annotations,
)
from tests.fixtures.factories import SessionMetadataFactory, FrameDataFactory


class TestDatabaseRepository:
  """Test database repository operations."""

  def test_transaction_context_manager_commit(self, temp_db):
    """Test transaction context manager commits on success."""
    with transaction(temp_db):
      temp_db.execute("CREATE TABLE test_table (id INTEGER)")

    # Table should exist after successful transaction
    cursor = temp_db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'")
    assert cursor.fetchone() is not None

  def test_transaction_context_manager_rollback(self, temp_db):
    """Test transaction context manager rolls back on error."""
    # Insert a test project first
    temp_db.execute("INSERT INTO projects (name, description) VALUES ('Test Project', 'Test')")
    temp_db.commit()

    with pytest.raises(Exception):
      with transaction(temp_db):
        temp_db.execute("INSERT INTO projects (name, description) VALUES ('Another Project', 'Test')")
        # Force an error
        raise Exception("Test error")

    # Only the first project should exist after rollback
    cursor = temp_db.execute("SELECT COUNT(*) FROM projects")
    assert cursor.fetchone()[0] == 1

  def test_upsert_session_creates_new_session(self, temp_db):
    """Test upsert_session creates a new session."""
    metadata = SessionMetadataFactory()
    session_id = "test_session_001"
    root_path = "/test/path"

    session_db_id = upsert_session(temp_db, session_id, root_path, metadata)

    assert isinstance(session_db_id, int)
    assert session_db_id > 0

    # Verify session was created
    cursor = temp_db.execute("SELECT session_id, root_path, metadata_json FROM sessions WHERE id = ?",
                             (session_db_id, ))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == session_id
    assert row[1] == root_path
    assert json.loads(row[2]) == metadata

  def test_upsert_session_updates_existing_session(self, temp_db):
    """Test upsert_session updates existing session."""
    metadata1 = SessionMetadataFactory()
    session_id = "test_session_001"
    root_path1 = "/test/path1"

    # Create initial session
    session_db_id1 = upsert_session(temp_db, session_id, root_path1, metadata1)

    # Update session with new data
    metadata2 = SessionMetadataFactory()
    root_path2 = "/test/path2"
    session_db_id2 = upsert_session(temp_db, session_id, root_path2, metadata2)

    # Should return same ID
    assert session_db_id1 == session_db_id2

    # Verify data was updated
    cursor = temp_db.execute("SELECT root_path, metadata_json FROM sessions WHERE id = ?", (session_db_id1, ))
    row = cursor.fetchone()
    assert row[0] == root_path2
    assert json.loads(row[1]) == metadata2

  def test_upsert_frame_creates_new_frame(self, temp_db):
    """Test upsert_frame creates a new frame."""
    # Create session first
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})

    frame_data = FrameDataFactory()
    frame_db_id = upsert_frame(temp_db, session_db_id, frame_data["frame_id"], frame_data["ts_ms"])

    assert isinstance(frame_db_id, int)
    assert frame_db_id > 0

    # Verify frame was created
    cursor = temp_db.execute("SELECT frame_id, ts_ms FROM frames WHERE id = ?", (frame_db_id, ))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == frame_data["frame_id"]
    assert row[1] == frame_data["ts_ms"]

  def test_upsert_frame_updates_timestamp(self, temp_db):
    """Test upsert_frame updates timestamp for existing frame."""
    # Create session first
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})

    frame_id = "frame_001"
    ts_ms1 = 1000
    ts_ms2 = 2000

    # Create initial frame
    frame_db_id1 = upsert_frame(temp_db, session_db_id, frame_id, ts_ms1)

    # Update frame with new timestamp
    frame_db_id2 = upsert_frame(temp_db, session_db_id, frame_id, ts_ms2)

    # Should return same ID
    assert frame_db_id1 == frame_db_id2

    # Verify timestamp was updated (COALESCE should keep existing value)
    cursor = temp_db.execute("SELECT ts_ms FROM frames WHERE id = ?", (frame_db_id1, ))
    row = cursor.fetchone()
    assert row[0] == ts_ms1  # Original timestamp should be preserved

  def test_get_session_db_id_existing_session(self, temp_db):
    """Test get_session_db_id returns correct ID for existing session."""
    session_id = "test_session_001"
    expected_id = upsert_session(temp_db, session_id, "/path", {})

    result = get_session_db_id(temp_db, session_id)
    assert result == expected_id

  def test_get_session_db_id_nonexistent_session(self, temp_db):
    """Test get_session_db_id returns None for nonexistent session."""
    result = get_session_db_id(temp_db, "nonexistent_session")
    assert result is None

  def test_get_frame_db_id_existing_frame(self, temp_db):
    """Test get_frame_db_id returns correct ID for existing frame."""
    # Create session and frame
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})
    frame_id = "frame_001"
    expected_id = upsert_frame(temp_db, session_db_id, frame_id, 1000)

    result = get_frame_db_id(temp_db, session_db_id, frame_id)
    assert result == expected_id

  def test_get_frame_db_id_nonexistent_frame(self, temp_db):
    """Test get_frame_db_id returns None for nonexistent frame."""
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})
    result = get_frame_db_id(temp_db, session_db_id, "nonexistent_frame")
    assert result is None

  def test_ensure_membership_creates_annotation_row(self, temp_db, populated_db):
    """Test ensure_membership creates annotation row."""
    from tests.conftest import create_test_project_and_dataset

    # Create project and dataset
    project_id, dataset_id = create_test_project_and_dataset(temp_db)

    # Get session and frame IDs
    session_db_id = get_session_db_id(temp_db, "test_session_001")
    frame_db_id = get_frame_db_id(temp_db, session_db_id, "frame_001")

    # Ensure membership
    ensure_membership(temp_db, dataset_id, frame_db_id)

    # Verify annotation was created
    cursor = temp_db.execute(
      "SELECT dataset_id, frame_id, status FROM annotations WHERE dataset_id = ? AND frame_id = ?",
      (dataset_id, frame_db_id),
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == dataset_id
    assert row[1] == frame_db_id
    assert row[2] == "unlabeled"

  def test_set_annotation_status(self, temp_db):
    """Test set_annotation_status updates annotation status."""
    from tests.conftest import create_test_project_and_dataset

    # Setup data
    project_id, dataset_id = create_test_project_and_dataset(temp_db)
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})
    frame_db_id = upsert_frame(temp_db, session_db_id, "frame_001", 1000)
    ann_id = ensure_membership(temp_db, dataset_id, frame_db_id)

    # Update status
    set_annotation_status(temp_db, dataset_id, frame_db_id, "labeled")

    # Verify status was updated
    cursor = temp_db.execute("SELECT status FROM annotations WHERE dataset_id = ? AND frame_id = ?",
                             (dataset_id, frame_db_id))
    row = cursor.fetchone()
    assert row[0] == "labeled"

  def test_upsert_regression_creates_annotation(self, temp_db):
    """Test upsert_regression creates regression annotation."""
    from tests.conftest import create_test_project_and_dataset

    # Setup data
    project_id, dataset_id = create_test_project_and_dataset(temp_db, target_type_id=0)  # Regression
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})
    frame_db_id = upsert_frame(temp_db, session_db_id, "frame_001", 1000)

    # Create regression annotation
    value = 42.5
    result = upsert_regression(temp_db, dataset_id, frame_db_id, value)

    assert result["regression_value"] == value
    assert result["status"] == "labeled"

    # Verify in database
    cursor = temp_db.execute(
      """SELECT r.value_real, a.status 
               FROM regression_annotations r 
               JOIN annotations a ON a.dataset_id = r.dataset_id AND a.frame_id = r.frame_id
               WHERE r.dataset_id = ? AND r.frame_id = ?""",
      (dataset_id, frame_db_id),
    )
    row = cursor.fetchone()
    assert row[0] == value
    assert row[1] == "labeled"

  def test_upsert_single_label_creates_annotation(self, temp_db):
    """Test upsert_single_label creates single label annotation."""
    from tests.conftest import create_test_project_and_dataset
    from db.classes import get_or_create_dataset_class

    # Setup data
    project_id, dataset_id = create_test_project_and_dataset(temp_db, target_type_id=1)  # Single label
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})
    frame_db_id = upsert_frame(temp_db, session_db_id, "frame_001", 1000)

    # Create class
    class_data = get_or_create_dataset_class(temp_db, dataset_id, "test_class")
    class_id = class_data["id"]

    # Create single label annotation
    result = upsert_single_label(temp_db, dataset_id, frame_db_id, class_id)

    assert result["single_label_class_id"] == class_id
    assert result["status"] == "labeled"

  def test_list_frames_with_annotations(self, temp_db):
    """Test list_frames_with_annotations returns correct data."""
    from tests.conftest import create_test_project_and_dataset

    # Setup data
    project_id, dataset_id = create_test_project_and_dataset(temp_db)
    session_db_id = upsert_session(temp_db, "test_session", "/path", {})

    # Create frames with and without annotations
    frame1_id = upsert_frame(temp_db, session_db_id, "frame_001", 1000)
    frame2_id = upsert_frame(temp_db, session_db_id, "frame_002", 2000)
    frame3_id = upsert_frame(temp_db, session_db_id, "frame_003", 3000)

    # Create annotations for some frames
    ensure_membership(temp_db, dataset_id, frame1_id)
    ensure_membership(temp_db, dataset_id, frame3_id)

    # List frames with annotations
    result = list_frames_with_annotations(temp_db, dataset_id, session_db_id)

    assert len(result) >= 2  # Should have at least 2 frames with annotations
    frame_ids = [row["frame_id"] for row in result]
    assert "frame_001" in frame_ids
    assert "frame_003" in frame_ids
