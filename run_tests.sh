#!/bin/bash
# Script to run tests with coverage

set -e

echo "Installing test dependencies..."
pip install -q pytest pytest-cov coverage

echo "Running tests with coverage (excluding Ollama/model tests)..."
pytest \
    --cov=core \
    --cov=scenarios \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    --cov-fail-under=80 \
    --cov-config=pytest.ini \
    -v \
    -m "not requires_ollama and not requires_model" \
    --ignore=core/model.py

echo "Tests completed successfully!"

