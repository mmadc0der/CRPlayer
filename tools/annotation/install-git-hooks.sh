#!/bin/bash
# Install git hooks for annotation tool validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DIR="$(git rev-parse --git-dir 2>/dev/null || echo "")"

if [ -z "$GIT_DIR" ]; then
    echo "❌ Not in a git repository"
    exit 1
fi

HOOKS_DIR="$GIT_DIR/hooks"

echo "🔧 Installing git hooks for annotation tool..."

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook for annotation tool

set -e

# Check if we're in the annotation tool directory or if annotation tool files are being committed
ANNOTATION_FILES=$(git diff --cached --name-only | grep "^tools/annotation/" || true)

if [ -z "$ANNOTATION_FILES" ]; then
    echo "ℹ️  No annotation tool files in this commit, skipping validation"
    exit 0
fi

echo "🔍 Validating annotation tool changes..."

cd tools/annotation

# Export PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Check Python syntax
echo "🐍 Checking Python syntax..."
python3 -m py_compile $(find . -name "*.py" | grep -v __pycache__ | head -10) || {
    echo "❌ Python syntax errors found"
    exit 1
}

# Check imports for core modules
echo "📦 Checking critical imports..."
python3 -c "
import sys
try:
    import api, app, dto
    print('✅ Critical imports OK')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || exit 1

# Run fast tests
echo "🧪 Running fast tests..."
python3 -m pytest tests/ -x --tb=short -q --maxfail=3 \
    --durations=5 \
    -m "not slow" \
    2>/dev/null || {
    echo "❌ Tests failed. Fix tests before committing."
    echo "💡 Run 'make test' for detailed output"
    exit 1
}

echo "✅ Pre-commit validation passed"
EOF

chmod +x "$HOOKS_DIR/pre-commit"

# Create pre-push hook
cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/bin/bash
# Pre-push hook for annotation tool

set -e

# Check if annotation tool files are being pushed
ANNOTATION_FILES=$(git diff --name-only HEAD~1..HEAD | grep "^tools/annotation/" || true)

if [ -z "$ANNOTATION_FILES" ]; then
    echo "ℹ️  No annotation tool files in this push, skipping validation"
    exit 0
fi

echo "🚀 Running pre-push validation for annotation tool..."

cd tools/annotation
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run more comprehensive tests before push
echo "🧪 Running comprehensive test suite..."
python3 -m pytest tests/ --tb=short -q --maxfail=10 || {
    echo "❌ Comprehensive tests failed"
    echo "💡 Run 'make test' to see detailed results"
    exit 1
}

echo "✅ Pre-push validation passed"
EOF

chmod +x "$HOOKS_DIR/pre-push"

echo "✅ Git hooks installed successfully!"
echo ""
echo "Installed hooks:"
echo "  📝 pre-commit: Syntax, imports, and fast tests"
echo "  🚀 pre-push: Comprehensive test suite"
echo ""
echo "To skip hooks temporarily:"
echo "  git commit --no-verify"
echo "  git push --no-verify"
