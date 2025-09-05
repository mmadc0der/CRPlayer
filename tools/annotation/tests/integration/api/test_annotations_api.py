"""
Integration tests for annotations API endpoints.
"""

import pytest
import json
from flask.testing import FlaskClient
from unittest.mock import patch

from tests.fixtures.factories import (
    ProjectDataFactory, DatasetDataFactory, SaveRegressionRequestFactory,
    SaveSingleLabelRequestFactory, SaveMultilabelRequestFactory
)


class TestAnnotationsAPI:
    """Test annotations API endpoints integration."""

    def setup_project_and_dataset(self, client: FlaskClient, target_type_id: int = 1):
        """Helper method to set up project and dataset for testing."""
        # Create project
        project_data = ProjectDataFactory()
        project_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = project_response.get_json()['id']

        # Create dataset
        dataset_data = DatasetDataFactory(target_type_id=target_type_id)
        dataset_response = client.post(
            f'/api/projects/{project_id}/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        dataset_response_data = dataset_response.get_json()
        print(f"Dataset creation response: {dataset_response.status_code} - {dataset_response_data}")
        dataset_id = dataset_response_data['id']

        # Create a test session and frame directly in the database
        from db.connection import get_connection
        from tests.conftest import create_test_session

        conn = get_connection()
        try:
            session_db_id = create_test_session(conn, "test_session_000", frame_count=5)
            print(f"Created test session with ID: {session_db_id}")
        finally:
            conn.close()

        return project_id, dataset_id

    def test_get_annotation_for_frame_not_found(self, client: FlaskClient):
        """Test getting annotation for nonexistent frame."""
        response = client.get('/api/annotations/frame?session_id=nonexistent&dataset_id=1&idx=0')
        
        assert response.status_code == 500  # Session not found
        data = response.get_json()
        assert 'code' in data

    def test_get_annotation_for_frame_invalid_params(self, client: FlaskClient):
        """Test getting annotation with invalid parameters."""
        # Missing required parameters
        response = client.get('/api/annotations/frame')
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['code'] == 'bad_request'

    def test_save_regression_success(self, client: FlaskClient):
        """Test saving regression annotation."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=0)  # Regression
        
        request_data = SaveRegressionRequestFactory(
            dataset_id=dataset_id,
            frame_idx=0,
            value=42.5
        )

        with patch('services.annotation_service.AnnotationService.save_regression') as mock_save, \
             patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame:
            mock_save.return_value = {"regression_value": 42.5, "status": "labeled"}
            mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}
            
            response = client.post(
                '/api/annotations/regression',
                data=json.dumps(request_data),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert data['annotation']['regression_value'] == 42.5

    def test_save_regression_invalid_data(self, client: FlaskClient):
        """Test saving regression with invalid data."""
        response = client.post(
            '/api/annotations/regression',
            data=json.dumps({}),  # Missing required fields
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['code'] == 'bad_request'

    def test_save_single_label_with_class_id(self, client: FlaskClient):
        """Test saving single label annotation with class_id."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=1)  # Single label
        
        request_data = SaveSingleLabelRequestFactory(
            dataset_id=dataset_id,
            frame_idx=0,
            class_id=1
        )
        
        with patch('services.annotation_service.AnnotationService.save_single_label') as mock_save, \
             patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame:
            mock_save.return_value = {"single_label_class_id": 1, "status": "labeled"}
            mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}
            
            response = client.post(
                '/api/annotations/single_label',
                data=json.dumps(request_data),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert data['annotation']['single_label_class_id'] == 1

    def test_save_single_label_with_category_name(self, client: FlaskClient):
        """Test saving single label annotation with category_name."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=1)
        
        request_data = {
            "session_id": "test_session_000",
            "dataset_id": dataset_id,
            "frame_idx": 0,
            "category_name": "battle"
        }

        with patch('services.annotation_service.AnnotationService.save_single_label') as mock_save, \
             patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame, \
             patch('db.classes.get_or_create_dataset_class') as mock_get_class:
            mock_get_class.return_value = {"id": 1, "name": "battle"}
            mock_save.return_value = {"single_label_class_id": 1, "status": "labeled"}
            mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}

            response = client.post(
                '/api/annotations/single_label',
                data=json.dumps(request_data),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True

    def test_save_multilabel_success(self, client: FlaskClient):
        """Test saving multilabel annotation."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=2)  # Multi label
        
        request_data = SaveMultilabelRequestFactory(
            dataset_id=dataset_id,
            frame_idx=0,
            class_ids=[1, 2, 3]
        )
        
        with patch('services.annotation_service.AnnotationService.save_multilabel') as mock_save, \
             patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame:
            mock_save.return_value = {"class_ids": [1, 2, 3], "status": "labeled"}
            mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}
            
            response = client.post(
                '/api/annotations/multilabel',
                data=json.dumps(request_data),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['class_ids'] == [1, 2, 3]

    def test_save_multilabel_with_category_names(self, client: FlaskClient):
        """Test saving multilabel annotation with category names."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=2)
        
        request_data = {
            "session_id": "test_session_001",
            "dataset_id": dataset_id,
            "frame_idx": 0,
            "class_ids": [],
            "category_names": ["battle", "boss", "critical"]
        }
        
        with patch('services.annotation_service.AnnotationService.save_multilabel') as mock_save, \
             patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame, \
             patch('db.classes.get_or_create_dataset_class') as mock_get_class:
                mock_get_class.side_effect = [
                    {"id": 1, "name": "battle"},
                    {"id": 2, "name": "boss"},
                    {"id": 3, "name": "critical"}
                ]
                mock_save.return_value = {"class_ids": [1, 2, 3], "status": "labeled"}
                mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}

                response = client.post(
                    '/api/annotations/multilabel',
                    data=json.dumps(request_data),
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                data = response.get_json()
                assert data['class_ids'] == [1, 2, 3]

    def test_save_annotation_with_override_settings(self, client: FlaskClient):
        """Test saving annotation with override settings."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=0)
        
        override_settings = {"confidence": 0.95, "custom_field": "value"}
        request_data = SaveRegressionRequestFactory(
            dataset_id=dataset_id,
            frame_idx=0,
            value=42.5,
            override_settings=override_settings
        )
        
        with patch('services.annotation_service.AnnotationService.save_regression') as mock_save, \
             patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame:
            mock_save.return_value = {"regression_value": 42.5, "status": "labeled"}
            mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}
            
            response = client.post(
                '/api/annotations/regression',
                data=json.dumps(request_data),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            # Verify override_settings were passed to the service
            mock_save.assert_called_once()
            call_kwargs = mock_save.call_args[1]
            assert call_kwargs['override_settings'] == override_settings

    def test_frame_id_vs_frame_idx_resolution(self, client: FlaskClient):
        """Test that frame_idx is properly resolved to frame_id."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=0)
        
        request_data = SaveRegressionRequestFactory(
            dataset_id=dataset_id,
            frame_idx=0,  # Using frame_idx instead of frame_id
            value=42.5
        )
        
        with patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame:
            with patch('services.annotation_service.AnnotationService.save_regression') as mock_save:
                mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}
                mock_save.return_value = {"regression_value": 42.5, "status": "labeled"}
                
                response = client.post(
                    '/api/annotations/regression',
                    data=json.dumps(request_data),
                    content_type='application/json'
                )
                
                assert response.status_code == 200
                # Verify frame_idx was resolved to frame_id
                mock_get_frame.assert_called_once_with(request_data["session_id"], 0)
                mock_save.assert_called_once()
                call_kwargs = mock_save.call_args[1]
                assert call_kwargs['frame_id'] == "frame_001"

    def test_annotation_error_handling(self, client: FlaskClient):
        """Test error handling in annotation endpoints."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=0)
        
        request_data = SaveRegressionRequestFactory(
            dataset_id=dataset_id,
            frame_idx=0,
            value=42.5
        )
        
        with patch('services.annotation_service.AnnotationService.save_regression') as mock_save:
            mock_save.side_effect = FileNotFoundError("Session not found")
            
            response = client.post(
                '/api/annotations/regression',
                data=json.dumps(request_data),
                content_type='application/json'
            )
            
            assert response.status_code == 404
            data = response.get_json()
            assert data['code'] == 'not_found'

    def test_validation_error_handling(self, client: FlaskClient):
        """Test validation error handling."""
        project_id, dataset_id = self.setup_project_and_dataset(client, target_type_id=1)
        
        # Invalid request - missing both class_id and category_name
        request_data = {
            "session_id": "test_session_001",
            "dataset_id": dataset_id,
            "frame_idx": 0
            # Missing class_id and category_name
        }
        
        response = client.post(
            '/api/annotations/single_label',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['code'] == 'bad_request'

    def test_malformed_json_handling(self, client: FlaskClient):
        """Test handling of malformed JSON."""
        response = client.post(
            '/api/annotations/regression',
            data='{"invalid": json}',  # Malformed JSON
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'code' in data

    def test_missing_content_type(self, client: FlaskClient):
        """Test handling of missing content type."""
        request_data = SaveRegressionRequestFactory()
        
        response = client.post(
            '/api/annotations/regression',
            data=json.dumps(request_data)
            # Missing content_type
        )
        
        # Flask should still handle this, but may behave differently
        assert response.status_code in [400, 415]  # Bad request or unsupported media type


class TestAnnotationRetrievalAPI:
    """Test annotation retrieval endpoints."""

    def test_get_annotation_success(self, client: FlaskClient):
        """Test getting annotation for existing frame."""
        with patch('services.annotation_service.AnnotationService.get_annotation_db') as mock_get:
            mock_annotation = {
                "id": 1,
                "dataset_id": 1,
                "frame_id": 456,
                "status": "labeled",
                "regression_value": 42.5
            }
            mock_get.return_value = mock_annotation
            
            with patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame:
                mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}
                
                response = client.get('/api/annotations/frame?session_id=test_session&dataset_id=1&idx=0')
                
                assert response.status_code == 200
                data = response.get_json()
                assert data['annotation'] == mock_annotation

    def test_get_annotation_not_found(self, client: FlaskClient):
        """Test getting annotation for frame without annotation."""
        with patch('services.annotation_service.AnnotationService.get_annotation_db') as mock_get:
            mock_get.return_value = None
            
            with patch('services.session_service.SessionService.get_frame_by_idx') as mock_get_frame:
                mock_get_frame.return_value = {"frame_id": "frame_001", "ts_ms": 1000}
                
                response = client.get('/api/annotations/frame?session_id=test_session&dataset_id=1&idx=0')
                
                assert response.status_code == 200
                data = response.get_json()
                assert data['annotation'] is None