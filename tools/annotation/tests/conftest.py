"""
Pytest configuration and shared fixtures for annotation tool tests.
"""

# Ensure the annotation tool root is on sys.path so imports like `from core...` work
import sys
from pathlib import Path as _Path

_THIS_DIR = _Path(__file__).resolve().parent
_TOOL_ROOT = _THIS_DIR.parent
if str(_TOOL_ROOT) not in sys.path:
  sys.path.insert(0, str(_TOOL_ROOT))

import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch

from flask import Flask
from flask.testing import FlaskClient

from core.session_manager import SessionManager
from db.connection import get_connection, get_db_path
from db.schema import init_db
from services.session_service import SessionService
from services.annotation_service import AnnotationService
from services.settings_service import SettingsService
from services.dataset_service import DatasetService


@pytest.fixture(scope="session")
def temp_data_dir() -> Generator[Path, None, None]:
  """Create a temporary directory for test data that persists for the session."""
  temp_dir = Path(tempfile.mkdtemp(prefix="annotation_test_"))
  try:
    yield temp_dir
  finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _force_test_db_env(temp_data_dir: Path) -> Generator[None, None, None]:
  """Force all DB connections during tests to use an isolated temp file.

  This guards against accidental writes/cleans on the real database when any
  code path calls get_connection() before per-test fixtures set env vars.
  """
  import os
  import sqlite3

  original_db_path = os.environ.get("ANNOTATION_DB_PATH")
  session_db_path = temp_data_dir / "session_default.db"

  # Ensure the file exists and has schema so early imports are safe
  conn = sqlite3.connect(str(session_db_path))
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON;")
  init_db(conn)
  conn.close()

  os.environ["ANNOTATION_DB_PATH"] = str(session_db_path)
  try:
    yield
  finally:
    if original_db_path is not None:
      os.environ["ANNOTATION_DB_PATH"] = original_db_path
    else:
      os.environ.pop("ANNOTATION_DB_PATH", None)


@pytest.fixture
def temp_db() -> Generator[sqlite3.Connection, None, None]:
  """Create a temporary in-memory SQLite database for testing."""
  conn = sqlite3.connect(":memory:")
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON;")
  init_db(conn)
  try:
    yield conn
  finally:
    conn.close()


@pytest.fixture
def temp_db_file(temp_data_dir: Path) -> Generator[Path, None, None]:
  """Create a temporary SQLite database file for testing."""
  db_path = temp_data_dir / "test_annotation.db"
  db_path.parent.mkdir(parents=True, exist_ok=True)

  # Initialize the database
  conn = sqlite3.connect(str(db_path))
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON;")
  init_db(conn)
  conn.close()

  yield db_path


@pytest.fixture
def mock_session_manager(temp_data_dir: Path, temp_db: sqlite3.Connection) -> SessionManager:
  """Create a SessionManager instance for testing."""
  return SessionManager(data_root=str(temp_data_dir), conn=temp_db)


@pytest.fixture
def session_service(mock_session_manager: SessionManager) -> SessionService:
  """Create a SessionService instance for testing."""
  return SessionService(mock_session_manager)


@pytest.fixture
def annotation_service(mock_session_manager: SessionManager) -> AnnotationService:
  """Create an AnnotationService instance for testing."""
  return AnnotationService(mock_session_manager)


@pytest.fixture
def settings_service() -> SettingsService:
  """Create a SettingsService instance for testing."""
  return SettingsService()


@pytest.fixture
def dataset_service(mock_session_manager: SessionManager) -> DatasetService:
  """Create a DatasetService instance for testing."""
  return DatasetService(mock_session_manager)


@pytest.fixture(scope="function")
def app(temp_data_dir: Path) -> Generator[Flask, None, None]:
  """Create a Flask application configured for testing with isolated in-memory database."""
  import tempfile
  import os
  from db.schema import init_db
  import sqlite3

  app = Flask(__name__)
  app.config.update({
    "TESTING": True,
    "WTF_CSRF_ENABLED": False,
    "SECRET_KEY": "test-secret-key",
    "APPLICATION_ROOT": "/annotation"
  })

  # Create a unique temporary database file for each test
  import uuid

  test_db_path = temp_data_dir / f"test_{uuid.uuid4().hex}.db"

  # Set environment variable to use test database
  original_db_path = os.environ.get("ANNOTATION_DB_PATH")
  os.environ["ANNOTATION_DB_PATH"] = str(test_db_path)

  # Initialize the test database
  conn = sqlite3.connect(str(test_db_path))
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON;")
  init_db(conn)
  conn.close()

  # Create a real SessionManager with test database connection
  mock_sm = SessionManager(data_root=str(temp_data_dir))

  # Import and register blueprint
  from api import create_annotation_api

  app.register_blueprint(create_annotation_api(mock_sm))

  # Add the main route
  @app.route("/")
  def index():
    return "Test annotation interface"

  try:
    yield app
  finally:
    # Restore original environment variable
    if original_db_path is not None:
      os.environ["ANNOTATION_DB_PATH"] = original_db_path
    else:
      os.environ.pop("ANNOTATION_DB_PATH", None)


@pytest.fixture
def client(app: Flask, temp_data_dir: Path) -> Generator[FlaskClient, None, None]:
  """Create a test client for the Flask application with clean database."""
  import os
  import uuid

  test_db_path = temp_data_dir / f"client_test_{uuid.uuid4().hex}.db"
  original_db_path = os.environ.get("ANNOTATION_DB_PATH")
  os.environ["ANNOTATION_DB_PATH"] = str(test_db_path)

  # Initialize the test database
  conn = sqlite3.connect(str(test_db_path))
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON;")
  init_db(conn)
  conn.close()

  test_client = app.test_client()

  # Clean up database before each test
  with app.app_context():
    try:
      from db.connection import get_connection

      conn = get_connection()
      # Clear all tables to ensure clean state (in reverse dependency order)
      for stmt in (
          "DELETE FROM annotations",
          "DELETE FROM frames",
          "DELETE FROM sessions",
          "DELETE FROM datasets",
          "DELETE FROM projects",
      ):
        try:
          conn.execute(stmt)
        except Exception:
          pass
      conn.commit()
      conn.close()
    except Exception:
      # If cleanup fails, continue with test
      pass

  try:
    yield test_client
  finally:
    # Restore original env var after client use to avoid leaking to other tests
    if original_db_path is not None:
      os.environ["ANNOTATION_DB_PATH"] = original_db_path
    else:
      os.environ.pop("ANNOTATION_DB_PATH", None)


@pytest.fixture
def sample_session_metadata() -> Dict[str, Any]:
  """Sample session metadata for testing."""
  return {
    "game_name": "TestGame",
    "start_time": "2023-01-01T00:00:00Z",
    "version": "1.0.0",
    "resolution": "1920x1080",
    "fps": 30,
  }


@pytest.fixture
def sample_project_data() -> Dict[str, Any]:
  """Sample project data for testing."""
  return {"name": "Test Project", "description": "A test project for annotation"}


@pytest.fixture
def sample_dataset_data() -> Dict[str, Any]:
  """Sample dataset data for testing."""
  return {
    "name": "Test Dataset",
    "description": "A test dataset for classification",
    "target_type_id": 1,  # SingleLabelClassification
  }


@pytest.fixture
def sample_frames_data() -> list[Dict[str, Any]]:
  """Sample frame data for testing."""
  return [
    {
      "frame_id": "frame_001",
      "ts_ms": 1000
    },
    {
      "frame_id": "frame_002",
      "ts_ms": 2000
    },
    {
      "frame_id": "frame_003",
      "ts_ms": 3000
    },
    {
      "frame_id": "frame_004",
      "ts_ms": 4000
    },
    {
      "frame_id": "frame_005",
      "ts_ms": 5000
    },
  ]


@pytest.fixture
def populated_db(temp_db: sqlite3.Connection, sample_session_metadata: Dict[str, Any],
                 sample_frames_data: list[Dict[str, Any]]) -> sqlite3.Connection:
  """Create a database populated with test data."""
  from db.repository import upsert_session, upsert_frame
  from db.projects import create_project, create_dataset

  # Create a test project
  project_id = create_project(temp_db, "Test Project", "Test project description")

  # Create a test dataset
  dataset_id = create_dataset(temp_db, project_id, "Test Dataset", "Test dataset", 1)

  # Create a test session
  session_db_id = upsert_session(temp_db, "test_session_001", "/test/path", sample_session_metadata)

  # Create test frames
  for frame_data in sample_frames_data:
    upsert_frame(temp_db, session_db_id, frame_data["frame_id"], frame_data["ts_ms"])

  temp_db.commit()
  return temp_db


@pytest.fixture
def mock_frame_files(temp_data_dir: Path) -> Dict[str, Path]:
  """Create mock frame image files for testing."""
  session_dir = temp_data_dir / "raw" / "test_session_001"
  session_dir.mkdir(parents=True, exist_ok=True)

  # Create mock image files (empty files for testing)
  frame_files = {}
  for i in range(1, 6):
    frame_id = f"frame_{i:03d}"
    frame_file = session_dir / f"{frame_id}.png"
    frame_file.write_bytes(b"mock_image_data")
    frame_files[frame_id] = frame_file

  return frame_files


# Utility functions for tests
def create_test_session(conn: sqlite3.Connection,
                        session_id: str,
                        metadata: Dict[str, Any] = None,
                        frame_count: int = 5) -> int:
  """Helper function to create a test session with frames."""
  from db.repository import upsert_session, upsert_frame

  if metadata is None:
    metadata = {"game_name": "TestGame", "version": "1.0.0"}

  session_db_id = upsert_session(conn, session_id, f"/test/{session_id}", metadata)

  # Create test frames
  for i in range(1, frame_count + 1):
    frame_id = f"frame_{i:03d}"
    ts_ms = i * 1000  # 1 second intervals
    upsert_frame(conn, session_db_id, frame_id, ts_ms)

  conn.commit()
  return session_db_id


def create_test_project_and_dataset(conn: sqlite3.Connection,
                                    project_name: str = None,
                                    dataset_name: str = None,
                                    target_type_id: int = 1) -> tuple[int, int]:
  """Helper function to create a test project and dataset with unique names."""
  import uuid
  from db.projects import create_project, create_dataset

  # Generate unique names if not provided
  if project_name is None:
    project_name = f"Test Project {uuid.uuid4().hex[:8]}"
  if dataset_name is None:
    dataset_name = f"Test Dataset {uuid.uuid4().hex[:8]}"

  project_id = create_project(conn, project_name, f"Description for {project_name}")
  dataset_id = create_dataset(conn, project_id, dataset_name, f"Description for {dataset_name}", target_type_id)

  conn.commit()
  return project_id, dataset_id
