# Makefile for annotation tool testing

.PHONY: help install-test test test-unit test-integration test-db test-api test-coverage test-fast clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-test: ## Install test dependencies
	pip install -r requirements-test.txt

test: ## Run all tests with coverage
	python run_tests.py --type all --coverage-html

test-unit: ## Run unit tests only
	python run_tests.py --type unit

test-integration: ## Run integration tests only
	python run_tests.py --type integration

test-db: ## Run database layer tests only
	python run_tests.py --type db

test-api: ## Run API integration tests only
	python run_tests.py --type api

test-coverage: ## Generate detailed coverage report
	python run_tests.py --type coverage

test-fast: ## Run all tests without coverage (faster)
	python run_tests.py --type all --fast

test-parallel: ## Run tests in parallel (faster on multi-core systems)
	python run_tests.py --type all --parallel 4

test-verbose: ## Run tests with verbose output
	python run_tests.py --type all --verbose

test-markers: ## Run tests with specific markers (e.g., make test-markers MARKERS="not slow")
	python run_tests.py --type all --markers "$(MARKERS)"

test-pattern: ## Run tests matching pattern (e.g., make test-pattern PATTERN="test_create")
	python run_tests.py --type all --pattern "$(PATTERN)"

clean: ## Clean test artifacts
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint: ## Run code linting (if flake8 is installed)
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 --max-line-length=120 --exclude=tests/fixtures .; \
	else \
		echo "flake8 not installed. Run: pip install flake8"; \
	fi

format: ## Format code (if black is installed)
	@if command -v black >/dev/null 2>&1; then \
		black --line-length=120 .; \
	else \
		echo "black not installed. Run: pip install black"; \
	fi

check: ## Run all quality checks
	$(MAKE) lint
	$(MAKE) test

# Development workflow targets
dev-setup: ## Set up development environment
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	@echo "Development environment set up successfully!"

dev-test: ## Quick development test run
	python run_tests.py --type all --fast --parallel 2

ci: ## Run CI pipeline (what CI/CD should run)
	python run_tests.py --type all --coverage-html --parallel 4