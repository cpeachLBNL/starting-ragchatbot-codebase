#!/bin/bash

# Complete code quality check script
# Runs all formatters and linters in sequence

echo "🚀 Running complete code quality check..."

# Format code first
echo "Step 1: Formatting code..."
./format.sh

# Then run quality checks
echo "Step 2: Running quality checks..."
./lint.sh

# Run tests if they exist
if [ -f "backend/tests/test_*.py" ]; then
    echo "Step 3: Running tests..."
    cd backend && uv run python -m pytest tests/ -v
else
    echo "Step 3: No tests found, skipping..."
fi

echo "✅ All quality checks complete!"