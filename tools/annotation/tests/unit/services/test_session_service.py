"""
Tests for session service.
"""

import pytest
from unittest.mock import Mock, patch

from services.session_service import SessionService


class TestSessionService:
    """Test SessionService functionality."""

    def test_init(self, mock_session_manager):
        """Test SessionService initialization."""
        service = SessionService(mock_session_manager)
        assert service.sm == mock_session_manager

    def test_resolve_session_dir_by_id_success(self, session_service, mock_session_manager):
        """Test resolve_session_dir_by_id with existing session."""
        with patch.object(mock_session_manager, 'get_session_path_by_id', return_value="/test/path") as mock_method:
            result = session_service.resolve_session_dir_by_id("test_session")
            assert str(result) == "/test/path"
            mock_method.assert_called_once_with("test_session")

    def test_resolve_session_dir_by_id_not_found(self, session_service, mock_session_manager):
        """Test resolve_session_dir_by_id with nonexistent session."""
        with patch.object(mock_session_manager, 'get_session_path_by_id', return_value=None):
            with pytest.raises(FileNotFoundError, match="Session not found by id: nonexistent"):
                session_service.resolve_session_dir_by_id("nonexistent")

    def test_get_session_info_success(self, session_service, mock_session_manager):
        """Test get_session_info with existing session."""
        expected_info = {
            "session_id": "test_session",
            "path": "/test/path",
            "frames_count": 5,
            "game_name": "TestGame"
        }
        
        with patch.object(mock_session_manager, 'find_session_by_id', return_value=expected_info) as mock_method:
            result = session_service.get_session_info("test_session")
            assert result == expected_info
            mock_method.assert_called_once_with("test_session")

    def test_get_session_info_not_found(self, session_service, mock_session_manager):
        """Test get_session_info with nonexistent session."""
        with patch.object(mock_session_manager, 'find_session_by_id', return_value=None):
            with pytest.raises(FileNotFoundError, match="Session not found by id: nonexistent"):
                session_service.get_session_info("nonexistent")

    @patch('services.session_service.get_connection')
    @patch('services.session_service.init_db')
    def test_get_frame_row_by_idx_success(self, mock_init_db, mock_get_connection, 
                                         session_service, mock_session_manager, temp_db):
        """Test _get_frame_row_by_idx with valid index."""
        mock_get_connection.return_value = temp_db
        # Mock get_session_db_id method
        
        # Mock database query result
        temp_db.execute = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("frame_001", 1000)
        temp_db.execute.return_value = mock_cursor
        
        frame_id, ts_ms = session_service._get_frame_row_by_idx("test_session", 0)
        
        assert frame_id == "frame_001"
        assert ts_ms == 1000
        mock_session_manager.get_session_db_id.assert_called_once_with("test_session")

    def test_get_frame_row_by_idx_negative_index(self, session_service):
        """Test _get_frame_row_by_idx with negative index."""
        with pytest.raises(IndexError, match="frame_idx must be >= 0"):
            session_service._get_frame_row_by_idx("test_session", -1)

    def test_get_frame_row_by_idx_session_not_found(self, session_service, mock_session_manager):
        """Test _get_frame_row_by_idx with nonexistent session."""
        # Mock get_session_db_id method to return None
        
        with pytest.raises(FileNotFoundError, match="Session not found by id: nonexistent"):
            session_service._get_frame_row_by_idx("nonexistent", 0)

    @patch('services.session_service.get_connection')
    @patch('services.session_service.init_db')
    def test_get_frame_row_by_idx_frame_not_found(self, mock_init_db, mock_get_connection,
                                                 session_service, mock_session_manager, temp_db):
        """Test _get_frame_row_by_idx with index out of range."""
        mock_get_connection.return_value = temp_db
        # Mock get_session_db_id method
        
        # Mock database queries
        temp_db.execute = Mock()
        
        # First call returns None (no frame found)
        # Second call returns total count
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [None, (5,)]  # No frame at index, total count is 5
        temp_db.execute.return_value = mock_cursor
        
        with pytest.raises(IndexError, match="frame_idx 10 out of range"):
            session_service._get_frame_row_by_idx("test_session", 10)

    def test_get_frame_by_idx_success(self, session_service):
        """Test get_frame_by_idx with valid data."""
        with patch.object(session_service, '_get_frame_row_by_idx', return_value=("frame_001", 1000)):
            result = session_service.get_frame_by_idx("test_session", 0)
            
            assert result == {"frame_id": "frame_001", "ts_ms": 1000}

    def test_get_frame_by_idx_no_timestamp(self, session_service):
        """Test get_frame_by_idx with no timestamp."""
        with patch.object(session_service, '_get_frame_row_by_idx', return_value=("frame_001", None)):
            result = session_service.get_frame_by_idx("test_session", 0)
            
            assert result == {"frame_id": "frame_001", "ts_ms": None}

    def test_get_frame_by_idx_error_propagation(self, session_service):
        """Test that get_frame_by_idx propagates errors from _get_frame_row_by_idx."""
        with patch.object(session_service, '_get_frame_row_by_idx', side_effect=FileNotFoundError("Session not found")):
            with pytest.raises(FileNotFoundError, match="Session not found"):
                session_service.get_frame_by_idx("test_session", 0)

    @patch('services.session_service.get_connection')
    @patch('services.session_service.init_db')
    def test_database_connection_handling(self, mock_init_db, mock_get_connection, 
                                        session_service, mock_session_manager, temp_db):
        """Test that database connections are properly handled."""
        mock_get_connection.return_value = temp_db
        # Mock get_session_db_id method
        
        # Mock successful query
        temp_db.execute = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("frame_001", 1000)
        temp_db.execute.return_value = mock_cursor
        
        session_service._get_frame_row_by_idx("test_session", 0)
        
        # Verify database connection was established and initialized
        mock_get_connection.assert_called_once()
        mock_init_db.assert_called_once_with(temp_db)

    def test_frame_ordering_query(self, session_service, mock_session_manager):
        """Test that frames are ordered correctly by timestamp and frame_id."""
        with patch('services.session_service.get_connection') as mock_get_connection:
            with patch('services.session_service.init_db'):
                mock_conn = Mock()
                mock_get_connection.return_value = mock_conn
                # Mock get_session_db_id method
                
                # Mock query execution
                mock_cursor = Mock()
                mock_cursor.fetchone.return_value = ("frame_001", 1000)
                mock_conn.execute.return_value = mock_cursor
                
                session_service._get_frame_row_by_idx("test_session", 0)
                
                # Verify the SQL query includes proper ordering
                call_args = mock_conn.execute.call_args[0]
                sql_query = call_args[0]
                assert "ORDER BY COALESCE(ts_ms, frame_id), frame_id" in sql_query
                assert "LIMIT 1 OFFSET ?" in sql_query