"""
Tests for database projects operations.
"""

import pytest
import sqlite3
from db.projects import (
  list_projects,
  create_project,
  update_project,
  delete_project,
  list_datasets,
  create_dataset,
  get_dataset,
  update_dataset,
  delete_dataset,
  dataset_progress,
  get_dataset_by_name,
)
from tests.fixtures.factories import ProjectDataFactory, DatasetDataFactory


class TestProjectsDatabase:
  """Test database operations for projects."""

  def test_create_project_success(self, temp_db):
    """Test creating a new project."""
    project_data = ProjectDataFactory()

    project_id = create_project(temp_db, project_data["name"], project_data["description"])

    assert isinstance(project_id, int)
    assert project_id > 0

    # Verify project was created
    cursor = temp_db.execute("SELECT name, description FROM projects WHERE id = ?", (project_id, ))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == project_data["name"]
    assert row[1] == project_data["description"]

  def test_create_project_duplicate_name_fails(self, temp_db):
    """Test creating project with duplicate name fails."""
    project_data = ProjectDataFactory()

    # Create first project
    create_project(temp_db, project_data["name"], project_data["description"])

    # Try to create second project with same name
    with pytest.raises(sqlite3.IntegrityError):
      create_project(temp_db, project_data["name"], "Different description")

  def test_list_projects_empty(self, temp_db):
    """Test listing projects when none exist."""
    projects = list_projects(temp_db)
    assert projects == []

  def test_list_projects_with_data(self, temp_db):
    """Test listing projects with existing data."""
    # Create multiple projects
    project1_data = ProjectDataFactory()
    project2_data = ProjectDataFactory()

    id1 = create_project(temp_db, project1_data["name"], project1_data["description"])
    id2 = create_project(temp_db, project2_data["name"], project2_data["description"])

    projects = list_projects(temp_db)

    assert len(projects) == 2
    # Should be ordered by id DESC (newest first)
    assert projects[0]["id"] == id2
    assert projects[1]["id"] == id1
    assert projects[0]["name"] == project2_data["name"]
    assert projects[1]["name"] == project1_data["name"]

  def test_update_project_name_only(self, temp_db):
    """Test updating only project name."""
    project_data = ProjectDataFactory()
    project_id = create_project(temp_db, project_data["name"], project_data["description"])

    new_name = "Updated Project Name"
    rows_affected = update_project(temp_db, project_id, new_name, None)

    assert rows_affected == 1

    # Verify update
    cursor = temp_db.execute("SELECT name, description FROM projects WHERE id = ?", (project_id, ))
    row = cursor.fetchone()
    assert row[0] == new_name
    assert row[1] == project_data["description"]  # Should remain unchanged

  def test_update_project_description_only(self, temp_db):
    """Test updating only project description."""
    project_data = ProjectDataFactory()
    project_id = create_project(temp_db, project_data["name"], project_data["description"])

    new_description = "Updated project description"
    rows_affected = update_project(temp_db, project_id, None, new_description)

    assert rows_affected == 1

    # Verify update
    cursor = temp_db.execute("SELECT name, description FROM projects WHERE id = ?", (project_id, ))
    row = cursor.fetchone()
    assert row[0] == project_data["name"]  # Should remain unchanged
    assert row[1] == new_description

  def test_update_project_both_fields(self, temp_db):
    """Test updating both name and description."""
    project_data = ProjectDataFactory()
    project_id = create_project(temp_db, project_data["name"], project_data["description"])

    new_name = "Updated Name"
    new_description = "Updated Description"
    rows_affected = update_project(temp_db, project_id, new_name, new_description)

    assert rows_affected == 1

    # Verify update
    cursor = temp_db.execute("SELECT name, description FROM projects WHERE id = ?", (project_id, ))
    row = cursor.fetchone()
    assert row[0] == new_name
    assert row[1] == new_description

  def test_update_project_no_fields(self, temp_db):
    """Test updating project with no fields returns 0."""
    project_data = ProjectDataFactory()
    project_id = create_project(temp_db, project_data["name"], project_data["description"])

    rows_affected = update_project(temp_db, project_id, None, None)
    assert rows_affected == 0

  def test_update_nonexistent_project(self, temp_db):
    """Test updating nonexistent project returns 0."""
    rows_affected = update_project(temp_db, 999, "New Name", "New Description")
    assert rows_affected == 0

  def test_delete_project_success(self, temp_db):
    """Test deleting an existing project."""
    project_data = ProjectDataFactory()
    project_id = create_project(temp_db, project_data["name"], project_data["description"])

    rows_affected = delete_project(temp_db, project_id)
    assert rows_affected == 1

    # Verify deletion
    cursor = temp_db.execute("SELECT COUNT(*) FROM projects WHERE id = ?", (project_id, ))
    count = cursor.fetchone()[0]
    assert count == 0

  def test_delete_nonexistent_project(self, temp_db):
    """Test deleting nonexistent project returns 0."""
    rows_affected = delete_project(temp_db, 999)
    assert rows_affected == 0


class TestDatasetsDatabase:
  """Test database operations for datasets."""

  def test_create_dataset_success(self, temp_db):
    """Test creating a new dataset."""
    # Create project first
    project_id = create_project(temp_db, "Test Project", "Description")

    dataset_data = DatasetDataFactory()
    dataset_id = create_dataset(temp_db, project_id, dataset_data["name"], dataset_data["description"],
                                dataset_data["target_type_id"])

    assert isinstance(dataset_id, int)
    assert dataset_id > 0

    # Verify dataset was created
    cursor = temp_db.execute("SELECT project_id, name, description, target_type_id FROM datasets WHERE id = ?",
                             (dataset_id, ))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == project_id
    assert row[1] == dataset_data["name"]
    assert row[2] == dataset_data["description"]
    assert row[3] == dataset_data["target_type_id"]

  def test_create_dataset_duplicate_name_in_project_fails(self, temp_db):
    """Test creating dataset with duplicate name in same project fails."""
    project_id = create_project(temp_db, "Test Project", "Description")
    dataset_data = DatasetDataFactory()

    # Create first dataset
    create_dataset(temp_db, project_id, dataset_data["name"], dataset_data["description"],
                   dataset_data["target_type_id"])

    # Try to create second dataset with same name in same project
    with pytest.raises(sqlite3.IntegrityError):
      create_dataset(temp_db, project_id, dataset_data["name"], "Different description", dataset_data["target_type_id"])

  def test_create_dataset_same_name_different_projects_success(self, temp_db):
    """Test creating datasets with same name in different projects succeeds."""
    project1_id = create_project(temp_db, "Project 1", "Description 1")
    project2_id = create_project(temp_db, "Project 2", "Description 2")

    dataset_name = "Same Dataset Name"

    # Should succeed for both projects
    dataset1_id = create_dataset(temp_db, project1_id, dataset_name, "Description 1", 1)
    dataset2_id = create_dataset(temp_db, project2_id, dataset_name, "Description 2", 1)

    assert dataset1_id != dataset2_id

  def test_list_datasets_empty(self, temp_db):
    """Test listing datasets when none exist for project."""
    project_id = create_project(temp_db, "Test Project", "Description")
    datasets = list_datasets(temp_db, project_id)
    assert datasets == []

  def test_list_datasets_with_data(self, temp_db):
    """Test listing datasets with existing data."""
    project_id = create_project(temp_db, "Test Project", "Description")

    # Create multiple datasets
    dataset1_data = DatasetDataFactory()
    dataset2_data = DatasetDataFactory()

    id1 = create_dataset(temp_db, project_id, dataset1_data["name"], dataset1_data["description"],
                         dataset1_data["target_type_id"])
    id2 = create_dataset(temp_db, project_id, dataset2_data["name"], dataset2_data["description"],
                         dataset2_data["target_type_id"])

    datasets = list_datasets(temp_db, project_id)

    assert len(datasets) == 2
    # Should be ordered by id DESC (newest first)
    assert datasets[0]["id"] == id2
    assert datasets[1]["id"] == id1
    assert datasets[0]["name"] == dataset2_data["name"]
    assert datasets[1]["name"] == dataset1_data["name"]

  def test_get_dataset_existing(self, temp_db):
    """Test getting an existing dataset."""
    project_id = create_project(temp_db, "Test Project", "Description")
    dataset_data = DatasetDataFactory()

    dataset_id = create_dataset(temp_db, project_id, dataset_data["name"], dataset_data["description"],
                                dataset_data["target_type_id"])

    dataset = get_dataset(temp_db, dataset_id)

    assert dataset is not None
    assert dataset["id"] == dataset_id
    assert dataset["name"] == dataset_data["name"]
    assert dataset["description"] == dataset_data["description"]
    assert dataset["target_type_id"] == dataset_data["target_type_id"]

  def test_get_dataset_nonexistent(self, temp_db):
    """Test getting nonexistent dataset returns None."""
    dataset = get_dataset(temp_db, 999)
    assert dataset is None

  def test_get_dataset_by_name_existing(self, temp_db):
    """Test getting dataset by name."""
    project_id = create_project(temp_db, "Test Project", "Description")
    dataset_data = DatasetDataFactory()

    dataset_id = create_dataset(temp_db, project_id, dataset_data["name"], dataset_data["description"],
                                dataset_data["target_type_id"])

    dataset = get_dataset_by_name(temp_db, project_id, dataset_data["name"])

    assert dataset is not None
    assert dataset["id"] == dataset_id
    assert dataset["name"] == dataset_data["name"]

  def test_get_dataset_by_name_nonexistent(self, temp_db):
    """Test getting nonexistent dataset by name returns None."""
    project_id = create_project(temp_db, "Test Project", "Description")
    dataset = get_dataset_by_name(temp_db, project_id, "Nonexistent Dataset")
    assert dataset is None

  def test_update_dataset_success(self, temp_db):
    """Test updating dataset."""
    project_id = create_project(temp_db, "Test Project", "Description")
    dataset_data = DatasetDataFactory()

    dataset_id = create_dataset(temp_db, project_id, dataset_data["name"], dataset_data["description"],
                                dataset_data["target_type_id"])

    new_name = "Updated Dataset Name"
    new_description = "Updated description"

    rows_affected = update_dataset(temp_db, dataset_id, new_name, new_description)
    assert rows_affected == 1

    # Verify update
    dataset = get_dataset(temp_db, dataset_id)
    assert dataset["name"] == new_name
    assert dataset["description"] == new_description

  def test_delete_dataset_success(self, temp_db):
    """Test deleting dataset."""
    project_id = create_project(temp_db, "Test Project", "Description")
    dataset_data = DatasetDataFactory()

    dataset_id = create_dataset(temp_db, project_id, dataset_data["name"], dataset_data["description"],
                                dataset_data["target_type_id"])

    rows_affected = delete_dataset(temp_db, dataset_id)
    assert rows_affected == 1

    # Verify deletion
    dataset = get_dataset(temp_db, dataset_id)
    assert dataset is None

  def test_dataset_progress_empty(self, temp_db):
    """Test dataset progress with no annotations."""
    project_id = create_project(temp_db, "Test Project", "Description")
    dataset_id = create_dataset(temp_db, project_id, "Test Dataset", "Description", 1)

    progress = dataset_progress(temp_db, dataset_id)

    assert progress["total"] == 0
    assert progress["labeled"] == 0
    assert progress["annotated"] == 0  # Legacy field
