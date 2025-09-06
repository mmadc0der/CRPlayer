"""
Enhanced test configuration for 100% success rate.
This module provides improved fixtures and utilities for perfect test isolation.
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, Generator

from core.session_manager import SessionManager
from services.annotation_service import AnnotationService
from services.session_service import SessionService
from services.settings_service import SettingsService
from db.schema import init_db


@pytest.fixture
def isolated_db() -> Generator[sqlite3.Connection, None, None]:
  """Create a completely isolated database for each test."""
  conn = sqlite3.connect(":memory:")
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON;")
  init_db(conn)
  try:
    yield conn
  finally:
    conn.close()


@pytest.fixture
def mock_session_manager_complete(isolated_db):
  """Create a properly mocked SessionManager for service tests."""
  sm = SessionManager(conn=isolated_db)

  # Add mock methods that some tests expect
  sm.get_session_path_by_id = Mock(return_value="/test/path")
  sm.find_session_by_id = Mock(return_value={
    "session_id": "test_session",
    "path": "/test/path",
    "frames_count": 5,
    "game_name": "TestGame"
  })
  sm.get_session_db_id = Mock(return_value=123)

  return sm


@pytest.fixture
def settings_service_isolated(isolated_db):
  """Create a SettingsService with isolated database."""
  return SettingsService(conn=isolated_db)


@pytest.fixture
def session_service_isolated(mock_session_manager_complete):
  """Create a SessionService with properly mocked dependencies."""
  return SessionService(mock_session_manager_complete)


@pytest.fixture
def annotation_service_isolated(mock_session_manager_complete):
  """Create an AnnotationService with properly mocked dependencies."""
  return AnnotationService(mock_session_manager_complete)


def setup_test_data(conn: sqlite3.Connection) -> Dict[str, Any]:
  """Set up common test data in database."""
  from db.projects import create_project, create_dataset
  from db.repository import upsert_session, upsert_frame

  # Create test project and dataset
  project_id = create_project(conn, "Test Project", "Test Description")
  dataset_id = create_dataset(conn, project_id, "Test Dataset", "Test Dataset", 1)

  # Create test session and frames
  session_db_id = upsert_session(conn, "test_session", "/test/path", {"game_name": "TestGame"})
  frame_db_id = upsert_frame(conn, session_db_id, "frame_001", 1000)

  conn.commit()

  return {
    "project_id": project_id,
    "dataset_id": dataset_id,
    "session_db_id": session_db_id,
    "frame_db_id": frame_db_id,
  }
