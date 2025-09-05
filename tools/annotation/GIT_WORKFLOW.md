# Git Workflow and Testing Guide

This document outlines the git workflow and testing procedures for the annotation tool.

## 🚀 Quick Start

### Initial Setup

```bash
cd tools/annotation

# Set up development environment
make setup-dev

# This will:
# - Install Python dependencies
# - Install test dependencies  
# - Install git hooks for validation
```

### Daily Development Workflow

```bash
# Make your changes
git add tools/annotation/

# Commit (hooks will run automatically)
git commit -m "feat: your changes"

# Push (comprehensive tests will run)
git push origin your-branch
```

## 🔧 Git Hooks

The annotation tool uses git hooks to ensure code quality:

### Pre-commit Hook
Runs automatically before each commit:
- ✅ Python syntax validation
- ✅ Critical import checks  
- ✅ Fast test suite (< 30 seconds)
- ❌ Blocks commit if any check fails

### Pre-push Hook  
Runs automatically before each push:
- ✅ Comprehensive test suite
- ✅ All unit and integration tests
- ❌ Blocks push if tests fail

### Skip Hooks (Emergency Only)
```bash
git commit --no-verify    # Skip pre-commit
git push --no-verify      # Skip pre-push
```

## 🧪 Testing Commands

### Quick Testing
```bash
make test-fast           # Fast tests only
make pre-commit          # Same as pre-commit hook
make validate            # Lint + fast tests
```

### Comprehensive Testing
```bash
make test               # All tests with coverage
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make ci                 # Full CI pipeline
```

### Debugging Tests
```bash
# Run specific test
pytest tests/unit/db/test_schema.py::TestDatabaseSchema::test_init_db_creates_all_tables -v

# Run with detailed output
pytest tests/ -v --tb=long

# Run with pdb on failure
pytest tests/ --pdb
```

## 🏗️ CI/CD Pipeline

### GitHub Actions Workflows

#### 1. `annotation-tool-ci.yml` (Full CI)
Triggers on: Push to `main`/`develop` branches
- ✅ Multi-Python version testing (3.9-3.12)
- ✅ Code quality checks (Black, Flake8)
- ✅ Security scanning (Bandit, Safety)
- ✅ Full test suite with coverage
- ✅ Import validation
- ✅ Flask app startup test

#### 2. `annotation-tool-pr.yml` (PR Validation)  
Triggers on: Pull requests to `main`/`develop`
- ✅ Fast validation checks
- ✅ Code formatting verification
- ✅ Critical import validation
- ✅ Database schema validation
- ✅ API endpoint validation
- ✅ PR summary generation

### Merge Requirements

All pull requests must pass:
1. ✅ Code formatting (Black)
2. ✅ Linting (Flake8)  
3. ✅ Import validation
4. ✅ Database schema validation
5. ✅ API endpoint validation
6. ✅ Fast test suite

## 📁 Project Structure

```
tools/annotation/
├── .pre-commit-config.yaml    # Pre-commit configuration
├── install-git-hooks.sh       # Git hooks installer
├── pre-commit-setup.sh        # Pre-commit setup (alternative)
├── Makefile                   # Development commands
├── run_tests.py              # Test runner script
├── pytest.ini               # Pytest configuration
├── requirements.txt          # Runtime dependencies
├── requirements-test.txt     # Test dependencies
├── api.py                    # Flask API routes
├── app.py                    # Flask application
├── dto.py                    # Data transfer objects
├── core/                     # Core business logic
├── services/                 # Service layer
├── db/                       # Database layer
├── models/                   # Data models
├── tests/                    # Test suite
│   ├── conftest.py          # Pytest fixtures
│   ├── fixtures/            # Test data factories
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
└── static/                   # Static web assets
```

## 🔍 Code Quality Standards

### Formatting
- **Black**: Line length 120, automatic formatting
- **Flake8**: Linting with max line length 120

### Testing
- **Coverage**: Minimum 80% overall
- **Critical paths**: 95%+ coverage (database, API)
- **Business logic**: 90%+ coverage (services, core)

### Import Organization
```python
# Standard library
import json
import sqlite3
from typing import Optional

# Third-party packages  
from flask import Flask
import pytest

# Local imports
from core.session_manager import SessionManager
from services.annotation_service import AnnotationService
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Git Hooks Not Running
```bash
# Reinstall hooks
cd tools/annotation
./install-git-hooks.sh
```

#### 2. Import Errors in Tests
```bash
# Set PYTHONPATH
export PYTHONPATH=/workspace/tools/annotation:$PYTHONPATH
cd tools/annotation
pytest tests/
```

#### 3. Database Test Failures
```bash
# Clean test database
rm -f tests/test_*.db
pytest tests/unit/db/ -v
```

#### 4. Slow Tests
```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Show slowest tests
pytest tests/ --durations=10
```

### Getting Help

1. **Check test output**: `make test`
2. **Run validation**: `make validate`  
3. **Check code quality**: `make lint`
4. **Format code**: `make format`
5. **View coverage**: `make test-coverage`

## 📝 Commit Message Format

Use conventional commits format:

```
type(scope): description

feat(api): add new regression endpoint
fix(db): resolve transaction rollback issue  
docs(readme): update installation instructions
test(unit): add session manager tests
refactor(services): simplify annotation logic
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## 🎯 Development Best Practices

### Before Making Changes
1. ✅ Pull latest changes
2. ✅ Run `make validate` to ensure clean state
3. ✅ Create feature branch

### During Development  
1. ✅ Make small, focused commits
2. ✅ Write tests for new functionality
3. ✅ Run `make test-fast` frequently
4. ✅ Keep coverage above 80%

### Before Submitting PR
1. ✅ Run `make check` (lint + test)
2. ✅ Ensure all tests pass
3. ✅ Write descriptive commit messages
4. ✅ Update documentation if needed

### Code Review Checklist
- [ ] Tests cover new functionality
- [ ] No decrease in code coverage
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Breaking changes documented
- [ ] Performance impact considered

## 🔄 Release Process

1. **Feature Development**: Work in feature branches
2. **Testing**: All tests must pass in CI
3. **Code Review**: Peer review required
4. **Merge**: Squash and merge to `develop`
5. **Release**: Merge `develop` to `main` for release

This workflow ensures high code quality and prevents regressions in the annotation tool.
