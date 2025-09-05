# Testing Guide for Annotation Tool

This document provides comprehensive information about testing the annotation web application.

## Overview

The test suite is designed to thoroughly test the annotation tool while preserving its existing architecture. Tests are organized into multiple layers:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test API endpoints and component interactions  
- **Database Tests**: Test database operations and schema
- **Fixtures**: Reusable test data and utilities

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── fixtures/                  # Test data factories and utilities
│   ├── __init__.py
│   └── factories.py           # Factory classes for test data generation
├── unit/                      # Unit tests
│   ├── core/                  # Core layer tests
│   │   └── test_session_manager.py
│   ├── services/              # Service layer tests
│   │   ├── test_annotation_service.py
│   │   ├── test_session_service.py
│   │   └── test_settings_service.py
│   ├── db/                    # Database layer tests
│   │   ├── test_connection.py
│   │   ├── test_schema.py
│   │   ├── test_repository.py
│   │   └── test_projects.py
│   ├── models/                # Model tests
│   │   └── __init__.py
│   └── test_dto.py           # Data Transfer Object tests
└── integration/               # Integration tests
    └── api/                   # API endpoint tests
        ├── test_projects_api.py
        ├── test_annotations_api.py
        └── test_sessions_api.py
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
make install-test

# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-db
make test-api
```

### Using the Test Runner

The `run_tests.py` script provides flexible test execution:

```bash
# Run all tests with coverage
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run tests in parallel (faster)
python run_tests.py --parallel 4

# Run tests without coverage (faster)
python run_tests.py --fast

# Run with verbose output
python run_tests.py --verbose

# Run tests matching a pattern
python run_tests.py --pattern "test_create"

# Run tests with specific markers
python run_tests.py --markers "not slow"
```

### Using pytest Directly

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-branch --cov-report=html

# Run specific test file
pytest tests/unit/services/test_annotation_service.py

# Run specific test method
pytest tests/unit/services/test_annotation_service.py::TestAnnotationService::test_save_regression_success

# Run tests with markers
pytest -m "unit"
pytest -m "not slow"
```

## Test Configuration

### pytest.ini

The `pytest.ini` file configures:
- Test discovery patterns
- Coverage settings
- Markers for test categorization
- Warning filters

### Test Markers

Tests are categorized with markers:
- `unit`: Unit tests
- `integration`: Integration tests  
- `db`: Database tests
- `slow`: Slow running tests
- `api`: API endpoint tests

## Test Fixtures

### Core Fixtures (conftest.py)

- `temp_data_dir`: Temporary directory for test data
- `temp_db`: In-memory SQLite database
- `temp_db_file`: Temporary database file
- `mock_session_manager`: SessionManager with test database
- `app`: Flask application configured for testing
- `client`: Flask test client
- `populated_db`: Database with test data

### Test Data Factories

Located in `tests/fixtures/factories.py`:

```python
# Create test session metadata
metadata = SessionMetadataFactory()

# Create test project data
project = ProjectDataFactory()

# Create test dataset data
dataset = DatasetDataFactory()

# Create complex test scenarios
scenario = create_annotation_scenario(session_count=3, frames_per_session=10)
```

## Writing Tests

### Unit Test Example

```python
def test_save_regression_success(self, annotation_service):
    """Test save_regression with valid data."""
    with patch.object(annotation_service, '_conn') as mock_conn:
        mock_connection = Mock()
        mock_conn.return_value = mock_connection
        
        with patch.object(annotation_service, '_resolve_ids', return_value=(123, 456)):
            with patch('services.annotation_service.upsert_regression') as mock_upsert:
                mock_upsert.return_value = {"regression_value": 42.5, "status": "labeled"}
                
                result = annotation_service.save_regression(
                    session_id="test_session",
                    dataset_id=1,
                    frame_id="frame_001",
                    value=42.5
                )
                
                assert result["regression_value"] == 42.5
                assert result["status"] == "labeled"
```

### Integration Test Example

```python
def test_create_project_success(self, client: FlaskClient):
    """Test creating a new project."""
    project_data = ProjectDataFactory()
    
    response = client.post(
        '/api/projects',
        data=json.dumps(project_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'id' in data
    assert data['name'] == project_data['name']
```

## Coverage Reports

### Generating Coverage

```bash
# Generate HTML coverage report
make test-coverage

# View coverage report
open htmlcov/index.html
```

### Coverage Targets

The test suite aims for:
- **Overall Coverage**: >80%
- **Critical Paths**: >95% (database operations, API endpoints)
- **Business Logic**: >90% (services, core functionality)

## Best Practices

### Test Organization

1. **One test class per module/class being tested**
2. **Descriptive test method names** that explain what is being tested
3. **Group related tests** using test classes
4. **Use fixtures** for common setup/teardown

### Test Data

1. **Use factories** for generating test data instead of hardcoded values
2. **Isolate tests** - each test should create its own data
3. **Clean up** temporary data after tests
4. **Use realistic data** that represents actual usage

### Mocking

1. **Mock external dependencies** (database, file system, network)
2. **Test business logic** not implementation details
3. **Verify interactions** with mocked dependencies
4. **Keep mocks simple** and focused

### Assertions

1. **Test one thing per test** method
2. **Use specific assertions** instead of generic ones
3. **Test both positive and negative cases**
4. **Include edge cases** and error conditions

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        cd tools/annotation
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run tests
      run: |
        cd tools/annotation
        make ci
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./tools/annotation/coverage.xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure PYTHONPATH includes the project root
2. **Database Errors**: Tests use in-memory databases; check fixture setup
3. **Mock Errors**: Verify patch targets match actual import paths
4. **Fixture Errors**: Check fixture dependencies and scope

### Debugging Tests

```bash
# Run with pdb on failures
pytest --pdb

# Run with verbose output
pytest -v -s

# Run single test with print statements
pytest -s tests/path/to/test.py::test_method
```

### Performance Issues

```bash
# Run tests in parallel
pytest -n 4

# Skip slow tests
pytest -m "not slow"

# Profile test execution
pytest --durations=10
```

## Architecture Compliance

The test suite is designed to respect the existing architecture:

1. **No breaking changes** to existing code
2. **Tests use dependency injection** where possible
3. **Database tests use transactions** for isolation
4. **API tests use Flask test client** not external HTTP calls
5. **Mocks preserve interface contracts**

## Maintenance

### Adding New Tests

1. Follow existing patterns and naming conventions
2. Add appropriate markers for test categorization
3. Update this documentation if adding new test types
4. Ensure new tests run in CI pipeline

### Updating Tests

1. Keep tests in sync with code changes
2. Update fixtures when data models change
3. Refactor common test patterns into utilities
4. Remove obsolete tests when features are removed