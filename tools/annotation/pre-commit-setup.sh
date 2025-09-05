#!/bin/bash
# Pre-commit setup script for annotation tool

set -e

echo "Setting up pre-commit hooks for annotation tool..."

# Install pre-commit if not available
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip3 install --user pre-commit
fi

# Install the git hook scripts
echo "Installing pre-commit hooks..."
pre-commit install

# Install additional dependencies
echo "Installing code formatting tools..."
pip3 install --user black flake8

echo "✅ Pre-commit hooks installed successfully!"
echo ""
echo "The following checks will run before each commit:"
echo "  - Python syntax check"
echo "  - Python import check"
echo "  - Code formatting with Black"
echo "  - Linting with Flake8"
echo "  - Test suite execution"
echo ""
echo "To run checks manually: pre-commit run --all-files"
echo "To skip hooks for a commit: git commit --no-verify"
