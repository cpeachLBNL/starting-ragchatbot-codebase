#!/bin/bash

# Code formatting script
# Runs black and isort to format Python code consistently

echo "🔧 Running code formatters..."

echo "Running black..."
uv run black backend/ main.py

echo "Running isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"