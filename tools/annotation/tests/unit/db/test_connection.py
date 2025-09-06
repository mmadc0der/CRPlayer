"""
Tests for database connection management.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from db.connection import get_db_path, get_connection, _detect_repo_root


class TestDatabaseConnection:
  """Test database connection functionality."""

  def test_detect_repo_root_with_data_dir(self, temp_data_dir):
    """Test repository root detection when data directory exists."""
    # Create a mock data directory
    data_dir = temp_data_dir / "data"
    data_dir.mkdir(exist_ok=True)

    with patch("db.connection.Path") as mock_path:
      mock_path_instance = MagicMock()
      mock_path_instance.parents = [None, None, None, temp_data_dir]
      mock_path_instance.resolve.return_value = mock_path_instance
      mock_path.__file__ = str(temp_data_dir / "db" / "connection.py")
      mock_path.return_value = mock_path_instance

      # Mock the exists check
      def exists_side_effect():
        return (temp_data_dir / "data").exists()

      mock_path_instance.__truediv__.return_value.exists = exists_side_effect

      result = _detect_repo_root()
      # The function should find a valid base directory

  def test_get_db_path_with_custom_path(self, temp_data_dir):
    """Test get_db_path with custom path."""
    custom_path = temp_data_dir / "custom_db.sqlite"
    result = get_db_path(custom_path)
    assert result == custom_path
    assert result.parent.exists()  # Should create parent directories

  def test_get_db_path_with_env_variable(self, temp_data_dir):
    """Test get_db_path with environment variable."""
    env_path = temp_data_dir / "env_db.sqlite"
    with patch.dict("os.environ", {"ANNOTATION_DB_PATH": str(env_path)}):
      result = get_db_path()
      assert result == env_path

  def test_get_connection_creates_valid_connection(self, temp_data_dir):
    """Test that get_connection creates a valid SQLite connection."""
    db_path = temp_data_dir / "test.db"
    conn = get_connection(db_path)

    try:
      # Test that connection is valid
      assert isinstance(conn, sqlite3.Connection)
      assert conn.row_factory == sqlite3.Row

      # Test that pragmas are set correctly
      cursor = conn.execute("PRAGMA foreign_keys")
      foreign_keys = cursor.fetchone()[0]
      assert foreign_keys == 1  # Foreign keys should be enabled

      cursor = conn.execute("PRAGMA journal_mode")
      journal_mode = cursor.fetchone()[0]
      assert journal_mode == "wal"  # WAL mode should be enabled

    finally:
      conn.close()

  def test_get_connection_with_invalid_path(self):
    """Test get_connection with path that causes sqlite3 to fail."""
    # Use a path with invalid characters that sqlite3 cannot handle
    invalid_path = Path("///invalid:::path???.db")

    # The function should either succeed (creating directories) or fail gracefully
    try:
      conn = get_connection(invalid_path)
      # If successful, verify it's a valid connection
      assert conn is not None
      conn.close()
    except RuntimeError as e:
      # If it fails, ensure it's a proper error message
      assert "Failed to connect to database" in str(e)

  def test_connection_timeout_and_settings(self, temp_data_dir):
    """Test that connection is created with proper timeout and settings."""
    db_path = temp_data_dir / "timeout_test.db"
    conn = get_connection(db_path)

    try:
      # Test that we can execute queries (connection is working)
      conn.execute("CREATE TABLE test (id INTEGER)")
      conn.execute("INSERT INTO test (id) VALUES (1)")
      result = conn.execute("SELECT id FROM test").fetchone()
      assert result[0] == 1

      # Test cache size setting
      cursor = conn.execute("PRAGMA cache_size")
      cache_size = cursor.fetchone()[0]
      assert cache_size == -64000  # 64MB cache

    finally:
      conn.close()

  def test_db_path_creation_with_nested_directories(self, temp_data_dir):
    """Test that deeply nested database paths are created correctly."""
    nested_path = temp_data_dir / "level1" / "level2" / "level3" / "nested.db"
    result = get_db_path(nested_path)

    assert result == nested_path
    assert result.parent.exists()
    assert result.parent.is_dir()

  def test_concurrent_connections(self, temp_data_dir):
    """Test that multiple connections can be created to the same database."""
    db_path = temp_data_dir / "concurrent.db"

    conn1 = get_connection(db_path)
    conn2 = get_connection(db_path)

    try:
      # Both connections should be valid
      conn1.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
      conn1.commit()

      conn2.execute("INSERT INTO test (id) VALUES (1)")
      conn2.commit()

      result = conn1.execute("SELECT COUNT(*) FROM test").fetchone()
      assert result[0] == 1

    finally:
      conn1.close()
      conn2.close()
