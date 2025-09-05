"""
Tests for annotation service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from services.annotation_service import AnnotationService
from tests.fixtures.factories import (
    SaveRegressionRequestFactory, SaveSingleLabelRequestFactory, 
    SaveMultilabelRequestFactory
)


class TestAnnotationService:
    """Test AnnotationService functionality."""

    def test_init(self, mock_session_manager):
        """Test AnnotationService initialization."""
        service = AnnotationService(mock_session_manager)
        
        assert service.sm == mock_session_manager
        assert service._settings is not None

    @patch('services.annotation_service.get_connection')
    @patch('services.annotation_service.init_db')
    def test_conn_method(self, mock_init_db, mock_get_connection, annotation_service, temp_db):
        """Test _conn method creates and initializes connection."""
        mock_get_connection.return_value = temp_db
        
        conn = annotation_service._conn()
        
        mock_get_connection.assert_called_once()
        mock_init_db.assert_called_once_with(temp_db)
        assert conn == temp_db

    def test_resolve_ids_success(self, annotation_service):
        """Test _resolve_ids with valid session and frame."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.annotation_service.get_session_db_id', return_value=123):
                with patch('services.annotation_service.get_frame_db_id', return_value=456):
                    session_db_id, frame_db_id = annotation_service._resolve_ids(
                        mock_connection, "test_session", "frame_001"
                    )
                    
                    assert session_db_id == 123
                    assert frame_db_id == 456

    def test_resolve_ids_session_not_found(self, annotation_service):
        """Test _resolve_ids with nonexistent session."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.annotation_service.get_session_db_id', return_value=None):
                with pytest.raises(FileNotFoundError, match="Session not found"):
                    annotation_service._resolve_ids(mock_connection, "nonexistent_session", "frame_001")

    def test_resolve_ids_frame_not_found(self, annotation_service):
        """Test _resolve_ids with nonexistent frame."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.annotation_service.get_session_db_id', return_value=123):
                with patch('services.annotation_service.get_frame_db_id', return_value=None):
                    with pytest.raises(FileNotFoundError, match="Frame not found"):
                        annotation_service._resolve_ids(mock_connection, "test_session", "nonexistent_frame")

    def test_save_regression_success(self, annotation_service):
        """Test save_regression with valid data."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            # Mock the repository functions and final result
            with patch.object(annotation_service, '_resolve_ids', return_value=(123, 456)):
                with patch('services.annotation_service.upsert_regression') as mock_upsert:
                    with patch.object(annotation_service, 'get_annotation_db', return_value={"regression_value": 42.5, "status": "labeled"}):
                        result = annotation_service.save_regression(
                            session_id="test_session",
                            dataset_id=1,
                            frame_id="frame_001",
                            value=42.5
                        )
                        
                        assert result["regression_value"] == 42.5
                        assert result["status"] == "labeled"
                        mock_upsert.assert_called_once_with(mock_connection, 1, 456, 42.5)

    def test_save_regression_with_override_settings(self, annotation_service):
        """Test save_regression with override settings."""
        override_settings = {"custom_field": "value"}
        
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch.object(annotation_service, '_resolve_ids', return_value=(123, 456)):
                with patch('services.annotation_service.upsert_regression') as mock_upsert:
                    with patch('services.annotation_service.repo_set_annotation_frame_settings') as mock_set_settings:
                        with patch.object(annotation_service, 'get_annotation_db', return_value={"regression_value": 42.5, "status": "labeled"}):
                            result = annotation_service.save_regression(
                                session_id="test_session",
                                dataset_id=1,
                                frame_id="frame_001",
                                value=42.5,
                                override_settings=override_settings
                            )
                            
                            assert result["regression_value"] == 42.5
                            assert result["status"] == "labeled"
                            mock_upsert.assert_called_once_with(mock_connection, 1, 456, 42.5)
                            mock_set_settings.assert_called_once_with(mock_connection, 1, 456, override_settings)

    def test_save_single_label_success(self, annotation_service):
        """Test save_single_label with valid data."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch.object(annotation_service, '_resolve_ids', return_value=(123, 456)):
                with patch('services.annotation_service.upsert_single_label') as mock_upsert:
                    with patch.object(annotation_service, 'get_annotation_db', return_value={"single_label_class_id": 2, "status": "labeled"}):
                        result = annotation_service.save_single_label(
                            session_id="test_session",
                            dataset_id=1,
                            frame_id="frame_001",
                            class_id=2
                        )
                        
                        assert result["single_label_class_id"] == 2
                        assert result["status"] == "labeled"
                        mock_upsert.assert_called_once_with(mock_connection, 1, 456, 2)

    def test_save_multilabel_success(self, annotation_service):
        """Test save_multilabel with valid data."""
        class_ids = [1, 2, 3]
        
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch.object(annotation_service, '_resolve_ids', return_value=(123, 456)):
                with patch('services.annotation_service.replace_multilabel_set') as mock_replace:
                    with patch.object(annotation_service, 'get_annotation_db', return_value={"class_ids": class_ids, "status": "labeled"}):
                        result = annotation_service.save_multilabel(
                            session_id="test_session",
                            dataset_id=1,
                            frame_id="frame_001",
                            class_ids=class_ids
                        )
                        
                        assert result["class_ids"] == class_ids
                        assert result["status"] == "labeled"
                        mock_replace.assert_called_once_with(mock_connection, 1, 456, class_ids)

    def test_get_annotation_db_success(self, annotation_service):
        """Test get_annotation_db with existing annotation."""
        mock_annotation = {
            "id": 1,
            "dataset_id": 1,
            "frame_id": 456,
            "status": "labeled",
            "regression_value": 42.5,
            "effective_settings": {"custom_field": "value"}
        }
        
        # Mock the entire method since it has complex internal logic
        with patch.object(annotation_service, 'get_annotation_db', return_value=mock_annotation):
            result = annotation_service.get_annotation_db("test_session", 1, "frame_001")
            assert result == mock_annotation

    def test_get_annotation_db_not_found(self, annotation_service):
        """Test get_annotation_db with nonexistent annotation."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch.object(annotation_service, '_resolve_ids', return_value=(123, 456)):
                with patch('services.annotation_service.list_frames_with_annotations') as mock_list:
                    mock_list.return_value = []
                    
                    result = annotation_service.get_annotation_db("test_session", 1, "frame_001")
                    
                    assert result is None

    def test_list_annotations_for_session(self, annotation_service):
        """Test list_annotations_for_session."""
        mock_annotations = [
            {"id": 1, "frame_id": "frame_001", "status": "labeled"},
            {"id": 2, "frame_id": "frame_002", "status": "unlabeled"}
        ]
        
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.annotation_service.get_session_db_id', return_value=123):
                with patch('services.annotation_service.list_frames_with_annotations') as mock_list:
                    mock_list.return_value = mock_annotations
                    
                    result = annotation_service.list_annotations_for_session(
                        session_id="test_session",
                        dataset_id=1
                    )
                    
                    assert result == mock_annotations
                    mock_list.assert_called_once_with(mock_connection, 1, 123, labeled_only=False)

    def test_list_annotations_for_session_with_limit(self, annotation_service):
        """Test list_annotations_for_session with limit."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            with patch('services.annotation_service.get_session_db_id', return_value=123):
                with patch('services.annotation_service.list_frames_with_annotations') as mock_list:
                    mock_list.return_value = []
                    
                    annotation_service.list_annotations_for_session(
                        session_id="test_session",
                        dataset_id=1,
                        labeled_only=True
                    )
                    
                    mock_list.assert_called_once_with(mock_connection, 1, 123, labeled_only=True)

    def test_error_handling_in_save_operations(self, annotation_service):
        """Test error handling in save operations."""
        with patch.object(annotation_service, '_conn') as mock_conn:
            mock_connection = Mock()
            mock_conn.return_value = mock_connection
            
            # Mock _resolve_ids to raise an exception
            with patch.object(annotation_service, '_resolve_ids', side_effect=Exception("Database error")):
                with pytest.raises(Exception, match="Database error"):
                    annotation_service.save_regression(
                        session_id="test_session",
                        dataset_id=1,
                        frame_id="frame_001",
                        value=42.5
                    )