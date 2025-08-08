#!/bin/bash

# Code quality linting script
# Runs flake8 for code style and mypy for type checking

echo "🔍 Running code quality checks..."

echo "Running flake8..."
uv run flake8 backend/ main.py

echo "Running mypy..."
uv run mypy backend/ main.py

echo "✅ Code quality checks complete!"