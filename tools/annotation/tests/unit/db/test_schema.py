"""
Tests for database schema initialization and management.
"""

import pytest
import sqlite3
from db.schema import init_db, SCHEMA_SQL


class TestDatabaseSchema:
    """Test database schema functionality."""

    def test_init_db_creates_all_tables(self, temp_db):
        """Test that init_db creates all required tables."""
        # init_db is called in the temp_db fixture, so tables should exist
        cursor = temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'annotation_frame_settings',
            'annotations',
            'dataset_classes',
            'dataset_session_settings',
            'datasets',
            'frames',
            'projects',
            'sessions',
            'target_types'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} was not created"

    def test_init_db_creates_target_types(self, temp_db):
        """Test that init_db populates target_types with default values."""
        cursor = temp_db.execute("SELECT id, name FROM target_types ORDER BY id")
        target_types = cursor.fetchall()
        
        expected_types = [
            (0, 'Regression'),
            (1, 'SingleLabelClassification'),
            (2, 'MultiLabelClassification')
        ]
        
        assert len(target_types) == 3
        for i, (expected_id, expected_name) in enumerate(expected_types):
            assert target_types[i][0] == expected_id
            assert target_types[i][1] == expected_name

    def test_init_db_idempotent(self, temp_db):
        """Test that init_db can be called multiple times without errors."""
        # Call init_db again
        init_db(temp_db)
        
        # Check that target_types still has correct data (no duplicates)
        cursor = temp_db.execute("SELECT COUNT(*) FROM target_types")
        count = cursor.fetchone()[0]
        assert count == 3

    def test_foreign_key_constraints_enabled(self, temp_db):
        """Test that foreign key constraints are enabled."""
        cursor = temp_db.execute("PRAGMA foreign_keys")
        foreign_keys_enabled = cursor.fetchone()[0]
        assert foreign_keys_enabled == 1

    def test_projects_table_structure(self, temp_db):
        """Test the projects table structure."""
        cursor = temp_db.execute("PRAGMA table_info(projects)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}  # name: type
        
        expected_columns = {
            'id': 'INTEGER',
            'name': 'TEXT',
            'description': 'TEXT',
            'created_at': 'DATETIME'
        }
        
        for col_name, col_type in expected_columns.items():
            assert col_name in columns
            assert columns[col_name] == col_type

    def test_datasets_table_structure(self, temp_db):
        """Test the datasets table structure."""
        cursor = temp_db.execute("PRAGMA table_info(datasets)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        expected_columns = {
            'id': 'INTEGER',
            'project_id': 'INTEGER',
            'name': 'TEXT',
            'description': 'TEXT',
            'target_type_id': 'INTEGER',
            'created_at': 'DATETIME'
        }
        
        for col_name, col_type in expected_columns.items():
            assert col_name in columns
            assert columns[col_name] == col_type

    def test_sessions_table_structure(self, temp_db):
        """Test the sessions table structure."""
        cursor = temp_db.execute("PRAGMA table_info(sessions)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        expected_columns = {
            'id': 'INTEGER',
            'session_id': 'TEXT',
            'root_path': 'TEXT',
            'metadata_json': 'TEXT'
        }
        
        for col_name, col_type in expected_columns.items():
            assert col_name in columns
            assert columns[col_name] == col_type

    def test_frames_table_structure(self, temp_db):
        """Test the frames table structure."""
        cursor = temp_db.execute("PRAGMA table_info(frames)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        expected_columns = {
            'id': 'INTEGER',
            'session_id': 'INTEGER',
            'frame_id': 'TEXT',
            'ts_ms': 'INTEGER'
        }
        
        for col_name, col_type in expected_columns.items():
            assert col_name in columns
            assert columns[col_name] == col_type

    def test_annotations_table_structure(self, temp_db):
        """Test the annotations table structure."""
        cursor = temp_db.execute("PRAGMA table_info(annotations)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        
        expected_columns = {
            'id': 'INTEGER',
            'dataset_id': 'INTEGER',
            'frame_id': 'INTEGER',
            'status': 'TEXT',
            'regression_value': 'REAL',
            'single_label_class_id': 'INTEGER',
            'created_at': 'DATETIME',
            'updated_at': 'DATETIME'
        }
        
        for col_name, col_type in expected_columns.items():
            assert col_name in columns
            assert columns[col_name] == col_type

    def test_foreign_key_relationships(self, temp_db):
        """Test that foreign key relationships are properly defined."""
        # Test datasets -> projects relationship
        cursor = temp_db.execute("PRAGMA foreign_key_list(datasets)")
        fk_info = cursor.fetchall()
        
        # Should have foreign key to projects table
        assert any(fk[2] == 'projects' for fk in fk_info)
        
        # Test frames -> sessions relationship
        cursor = temp_db.execute("PRAGMA foreign_key_list(frames)")
        fk_info = cursor.fetchall()
        
        # Should have foreign key to sessions table
        assert any(fk[2] == 'sessions' for fk in fk_info)

    def test_unique_constraints(self, temp_db):
        """Test unique constraints are working."""
        from db.projects import create_project
        
        # Create a project
        project_id = create_project(temp_db, "Test Project", "Description")
        
        # Try to create another project with the same name
        with pytest.raises(sqlite3.IntegrityError):
            create_project(temp_db, "Test Project", "Another description")

    def test_check_constraints(self, temp_db):
        """Test check constraints are working."""
        from db.repository import upsert_session, upsert_frame
        
        # Create a session
        session_db_id = upsert_session(temp_db, "test_session", "/path", {})
        
        # Try to insert frame with negative timestamp (should fail)
        with pytest.raises(sqlite3.IntegrityError):
            upsert_frame(temp_db, session_db_id, "frame_001", -1000)

    def test_indexes_created(self, temp_db):
        """Test that necessary indexes are created."""
        cursor = temp_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
        )
        indexes = [row[0] for row in cursor.fetchall()]
        
        # Check for specific indexes mentioned in schema
        expected_indexes = [
            'idx_frames_session_frame',
            'idx_annotations_dataset_frame',
            'idx_annotations_dataset_status'
        ]
        
        for index in expected_indexes:
            assert index in indexes, f"Index {index} was not created"