"""
Integration tests for projects API endpoints.
"""

import pytest
import json
from flask.testing import FlaskClient

from tests.fixtures.factories import ProjectDataFactory, DatasetDataFactory


class TestProjectsAPI:
    """Test projects API endpoints integration."""

    def test_list_projects_empty(self, client: FlaskClient):
        """Test listing projects when none exist."""
        response = client.get('/api/projects')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data == []

    def test_create_project_success(self, client: FlaskClient):
        """Test creating a new project."""
        project_data = ProjectDataFactory()
        
        response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 201]  # 200 if exists, 201 if created
        data = response.get_json()
        
        assert 'id' in data
        assert data['name'] == project_data['name']
        assert data['description'] == project_data['description']
        assert 'created_at' in data

    def test_create_project_invalid_data(self, client: FlaskClient):
        """Test creating project with invalid data."""
        response = client.post(
            '/api/projects',
            data=json.dumps({}),  # Missing required fields
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'code' in data
        assert data['code'] == 'bad_request'

    def test_create_project_duplicate_name(self, client: FlaskClient):
        """Test creating project with duplicate name."""
        project_data = ProjectDataFactory()
        
        # Create first project
        response1 = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        assert response1.status_code in [200, 201]  # 200 if exists, 201 if created
        
        # Try to create second project with same name
        response2 = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        
        assert response2.status_code == 200  # Returns existing project
        data = response2.get_json()
        # Should return the existing project data, not an error
        assert 'id' in data
        assert data['name'] == project_data['name']

    def test_list_projects_with_data(self, client: FlaskClient):
        """Test listing projects with existing data."""
        # Create multiple projects
        project1_data = ProjectDataFactory()
        project2_data = ProjectDataFactory()
        
        client.post('/api/projects', data=json.dumps(project1_data), content_type='application/json')
        client.post('/api/projects', data=json.dumps(project2_data), content_type='application/json')
        
        response = client.get('/api/projects')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert len(data) == 2
        # Should be ordered by id DESC (newest first)
        assert data[0]['name'] == project2_data['name']
        assert data[1]['name'] == project1_data['name']

    def test_update_project_success(self, client: FlaskClient):
        """Test updating an existing project."""
        # Create project first
        project_data = ProjectDataFactory()
        create_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = create_response.get_json()['id']
        
        # Update project
        update_data = {
            'name': 'Updated Project Name',
            'description': 'Updated description'
        }
        
        response = client.put(
            f'/api/projects/{project_id}',
            data=json.dumps(update_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['id'] == project_id
        assert data['name'] == update_data['name']
        assert data['description'] == update_data['description']

    def test_update_project_partial(self, client: FlaskClient):
        """Test updating project with partial data."""
        # Create project first
        project_data = ProjectDataFactory()
        create_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = create_response.get_json()['id']
        
        # Update only name
        update_data = {'name': 'Updated Name Only'}
        
        response = client.put(
            f'/api/projects/{project_id}',
            data=json.dumps(update_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['name'] == update_data['name']
        assert data['description'] == project_data['description']  # Should remain unchanged

    def test_update_nonexistent_project(self, client: FlaskClient):
        """Test updating nonexistent project."""
        update_data = {'name': 'Updated Name'}
        
        response = client.put(
            '/api/projects/999',
            data=json.dumps(update_data),
            content_type='application/json'
        )
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['code'] == 'not_found'

    def test_delete_project_success(self, client: FlaskClient):
        """Test deleting an existing project."""
        # Create project first
        project_data = ProjectDataFactory()
        create_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = create_response.get_json()['id']
        
        # Delete project
        response = client.delete(f'/api/projects/{project_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['deleted'] == 1

    def test_delete_project_with_datasets_fails(self, client: FlaskClient):
        """Test deleting project with datasets fails without force flag."""
        # Create project and dataset
        project_data = ProjectDataFactory()
        create_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = create_response.get_json()['id']
        
        # Create dataset under project
        dataset_data = DatasetDataFactory()
        client.post(
            f'/api/projects/{project_id}/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        
        # Try to delete project without force flag
        response = client.delete(f'/api/projects/{project_id}')
        
        assert response.status_code == 409
        data = response.get_json()
        assert data['code'] == 'conflict'

    def test_delete_project_with_force_flag(self, client: FlaskClient):
        """Test deleting project with datasets using force flag."""
        # Create project and dataset
        project_data = ProjectDataFactory()
        create_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = create_response.get_json()['id']
        
        # Create dataset under project
        dataset_data = DatasetDataFactory()
        client.post(
            f'/api/projects/{project_id}/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        
        # Delete project with force flag
        response = client.delete(f'/api/projects/{project_id}?force=true')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['deleted'] == 1

    def test_delete_nonexistent_project(self, client: FlaskClient):
        """Test deleting nonexistent project."""
        response = client.delete('/api/projects/999')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['deleted'] == 0


class TestDatasetsAPI:
    """Test datasets API endpoints integration."""

    def test_create_dataset_success(self, client: FlaskClient):
        """Test creating a new dataset."""
        # Create project first
        project_data = ProjectDataFactory()
        project_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = project_response.get_json()['id']
        
        # Create dataset
        dataset_data = DatasetDataFactory()
        response = client.post(
            f'/api/projects/{project_id}/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert 'id' in data
        assert data['name'] == dataset_data['name']
        assert data['description'] == dataset_data['description']
        assert data['target_type_id'] == dataset_data['target_type_id']
        assert data['project_id'] == project_id

    def test_create_dataset_invalid_project(self, client: FlaskClient):
        """Test creating dataset under nonexistent project."""
        dataset_data = DatasetDataFactory()
        response = client.post(
            '/api/projects/999/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'code' in data

    def test_list_datasets_empty(self, client: FlaskClient):
        """Test listing datasets when none exist."""
        # Create project first
        project_data = ProjectDataFactory()
        project_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = project_response.get_json()['id']
        
        response = client.get(f'/api/projects/{project_id}/datasets')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data == []

    def test_list_datasets_with_data(self, client: FlaskClient):
        """Test listing datasets with existing data."""
        # Create project
        project_data = ProjectDataFactory()
        project_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = project_response.get_json()['id']
        
        # Create multiple datasets
        dataset1_data = DatasetDataFactory()
        dataset2_data = DatasetDataFactory()
        
        client.post(f'/api/projects/{project_id}/datasets', 
                   data=json.dumps(dataset1_data), content_type='application/json')
        client.post(f'/api/projects/{project_id}/datasets', 
                   data=json.dumps(dataset2_data), content_type='application/json')
        
        response = client.get(f'/api/projects/{project_id}/datasets')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert len(data) == 2
        # Should be ordered by id DESC (newest first)
        assert data[0]['name'] == dataset2_data['name']
        assert data[1]['name'] == dataset1_data['name']

    def test_get_dataset_success(self, client: FlaskClient):
        """Test getting an existing dataset."""
        # Create project and dataset
        project_data = ProjectDataFactory()
        project_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = project_response.get_json()['id']
        
        dataset_data = DatasetDataFactory()
        dataset_response = client.post(
            f'/api/projects/{project_id}/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        dataset_id = dataset_response.get_json()['id']
        
        # Get dataset
        response = client.get(f'/api/datasets/{dataset_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['id'] == dataset_id
        assert data['name'] == dataset_data['name']

    def test_get_nonexistent_dataset(self, client: FlaskClient):
        """Test getting nonexistent dataset."""
        response = client.get('/api/datasets/999')
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['code'] == 'not_found'

    def test_update_dataset_success(self, client: FlaskClient):
        """Test updating an existing dataset."""
        # Create project and dataset
        project_data = ProjectDataFactory()
        project_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = project_response.get_json()['id']
        
        dataset_data = DatasetDataFactory()
        dataset_response = client.post(
            f'/api/projects/{project_id}/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        dataset_id = dataset_response.get_json()['id']
        
        # Update dataset
        update_data = {
            'name': 'Updated Dataset Name',
            'description': 'Updated description'
        }
        
        response = client.put(
            f'/api/datasets/{dataset_id}',
            data=json.dumps(update_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['name'] == update_data['name']
        assert data['description'] == update_data['description']

    def test_delete_dataset_success(self, client: FlaskClient):
        """Test deleting an existing dataset."""
        # Create project and dataset
        project_data = ProjectDataFactory()
        project_response = client.post(
            '/api/projects',
            data=json.dumps(project_data),
            content_type='application/json'
        )
        project_id = project_response.get_json()['id']
        
        dataset_data = DatasetDataFactory()
        dataset_response = client.post(
            f'/api/projects/{project_id}/datasets',
            data=json.dumps(dataset_data),
            content_type='application/json'
        )
        dataset_id = dataset_response.get_json()['id']
        
        # Delete dataset
        response = client.delete(f'/api/datasets/{dataset_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['deleted'] == 1