"""
Tests for settings service.
"""

import pytest
from unittest.mock import Mock, patch

from services.settings_service import SettingsService
from tests.fixtures.factories import DatasetSessionSettingsFactory


class TestSettingsServiceBase:
    """Base class for settings service tests with common mocking setup."""
    
    def setup_mocks(self, settings_service, return_session_id=123):
        """Set up common mocks for settings service tests."""
        mock_conn = Mock()
        mock_conn.return_value = Mock()
        
        # Mock all the dependencies
        patches = [
            patch.object(settings_service, '_conn', return_value=mock_conn.return_value),
            patch('services.settings_service.get_session_db_id', return_value=return_session_id),
            patch('services.settings_service.repo_upsert_dataset_session_settings'),
            patch('services.settings_service.repo_get_dataset_session_settings'),
            patch('services.settings_service.repo_delete_dataset_session_settings'),
        ]
        
        return patches, mock_conn.return_value


class TestSettingsService(TestSettingsServiceBase):
    """Test SettingsService functionality."""

    def test_init(self):
        """Test SettingsService initialization."""
        service = SettingsService()
        assert service is not None

    @patch('services.settings_service.get_connection')
    @patch('services.settings_service.init_db')
    def test_conn_method(self, mock_init_db, mock_get_connection, temp_db):
        """Test _conn method creates and initializes connection."""
        mock_get_connection.return_value = temp_db
        service = SettingsService()
        
        conn = service._conn()
        
        mock_get_connection.assert_called_once()
        mock_init_db.assert_called_once_with(temp_db)
        assert conn == temp_db

    def test_upsert_dataset_session_settings_success(self, settings_service):
        """Test upsert_dataset_session_settings with valid data."""
        settings_data = DatasetSessionSettingsFactory()
        
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_upsert_dataset_session_settings') as mock_upsert:
                with patch('services.settings_service.get_session_db_id', return_value=123) as mock_get_id:
                    settings_service.upsert_dataset_session_settings(
                        dataset_id=1,
                        session_id_str="test_session",
                        settings=settings_data
                    )
                    
                    mock_get_id.assert_called_once_with(mock_connection, "test_session")
                    mock_upsert.assert_called_once_with(mock_connection, 1, 123, settings_data)
                mock_connection.commit.assert_called_once()

    def test_upsert_dataset_session_settings_error_handling(self, settings_service):
        """Test upsert_dataset_session_settings error handling."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.upsert_dataset_session_settings', 
                      side_effect=Exception("Database error")):
                with pytest.raises(Exception, match="Database error"):
                    settings_service.upsert_dataset_session_settings(
                        dataset_id=1,
                        session_id_str="test_session",
                        settings={"key": "value"}
                    )

    def test_get_dataset_session_settings_success(self, settings_service):
        """Test get_dataset_session_settings with existing settings."""
        expected_settings = DatasetSessionSettingsFactory()
        
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_get_dataset_session_settings') as mock_get:
                with patch('services.settings_service.get_session_db_id', return_value=123) as mock_get_id:
                    mock_get.return_value = expected_settings
                    
                    result = settings_service.get_dataset_session_settings(
                        dataset_id=1,
                        session_id_str="test_session"
                    )
                    
                    assert result == expected_settings
                    mock_get_id.assert_called_once_with(mock_connection, "test_session")
                    mock_get.assert_called_once_with(mock_connection, 1, 123)

    def test_get_dataset_session_settings_not_found(self, settings_service):
        """Test get_dataset_session_settings with nonexistent settings."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_get_dataset_session_settings') as mock_get:
                mock_get.return_value = {}
                
                result = settings_service.get_dataset_session_settings(
                    dataset_id=1,
                    session_id="nonexistent_session"
                )
                
                assert result == {}

    def test_clear_dataset_session_settings_success(self, settings_service):
        """Test clear_dataset_session_settings."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_delete_dataset_session_settings') as mock_clear:
                mock_clear.return_value = 1  # One row affected
                
                result = settings_service.clear_dataset_session_settings(
                    dataset_id=1,
                    session_id_str="test_session"
                )
                
                assert result == 1
                mock_clear.assert_called_once_with(mock_connection, 1, "test_session")
                mock_connection.commit.assert_called_once()

    def test_clear_dataset_session_settings_not_found(self, settings_service):
        """Test clear_dataset_session_settings with nonexistent settings."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_delete_dataset_session_settings') as mock_clear:
                mock_clear.return_value = 0  # No rows affected
                
                result = settings_service.clear_dataset_session_settings(
                    dataset_id=1,
                    session_id="nonexistent_session"
                )
                
                assert result == 0

    def test_connection_management(self, settings_service):
        """Test that connections are properly managed."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_get_dataset_session_settings'):
                settings_service.get_dataset_session_settings(1, "test_session")
                
                # Verify connection was created and closed
                mock_conn.assert_called_once()
                mock_connection.close.assert_called_once()

    def test_transaction_handling_in_upsert(self, settings_service):
        """Test transaction handling in upsert operations."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            # Mock successful upsert
            with patch('services.settings_service.repo_upsert_dataset_session_settings'):
                settings_service.upsert_dataset_session_settings(
                    dataset_id=1,
                    session_id_str="test_session",
                    settings={"key": "value"}
                )
                
                # Verify commit was called
                mock_connection.commit.assert_called_once()

    def test_transaction_rollback_on_error(self, settings_service):
        """Test transaction rollback on error."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            # Mock database error
            with patch('services.settings_service.upsert_dataset_session_settings', 
                      side_effect=Exception("Database error")):
                with pytest.raises(Exception):
                    settings_service.upsert_dataset_session_settings(
                        dataset_id=1,
                        session_id_str="test_session",
                        settings={"key": "value"}
                    )
                
                # Connection should still be closed even on error
                mock_connection.close.assert_called_once()

    def test_settings_validation(self, settings_service):
        """Test that settings are properly validated/handled."""
        # Test with None settings
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_upsert_dataset_session_settings') as mock_upsert:
                settings_service.upsert_dataset_session_settings(
                    dataset_id=1,
                    session_id_str="test_session",
                    settings=None
                )
                
                # Should pass None to the repository layer
                mock_upsert.assert_called_once_with(mock_connection, 1, "test_session", None)

    def test_empty_settings_handling(self, settings_service):
        """Test handling of empty settings."""
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_upsert_dataset_session_settings') as mock_upsert:
                settings_service.upsert_dataset_session_settings(
                    dataset_id=1,
                    session_id_str="test_session",
                    settings={}
                )
                
                mock_upsert.assert_called_once_with(mock_connection, 1, "test_session", {})

    def test_complex_settings_handling(self, settings_service):
        """Test handling of complex nested settings."""
        complex_settings = {
            "ui_preferences": {
                "theme": "dark",
                "shortcuts": {"save": "Ctrl+S", "next": "Right"}
            },
            "annotation_defaults": {
                "confidence": 0.8,
                "auto_advance": True
            },
            "custom_fields": ["field1", "field2", "field3"]
        }
        
        with patch.object(settings_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.settings_service.repo_upsert_dataset_session_settings') as mock_upsert:
                settings_service.upsert_dataset_session_settings(
                    dataset_id=1,
                    session_id_str="test_session",
                    settings=complex_settings
                )
                
                mock_upsert.assert_called_once_with(mock_connection, 1, "test_session", complex_settings)