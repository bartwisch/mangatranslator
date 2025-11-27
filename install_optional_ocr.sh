#!/bin/bash

# One-time installer for optional OCR dependencies (Manga-OCR, PaddleOCR, EasyOCR)
# Run this script ONCE after creating your virtual environment.

# Project root (adjust only if you move this script elsewhere)
PROJECT_DIR="/Users/christoph/Dokumente/entwicklung/mangatranslator"

cd "$PROJECT_DIR" || {
  echo "Failed to cd into $PROJECT_DIR" >&2
  exit 1
}

if [ ! -d "venv" ]; then
  echo "Virtual environment 'venv' not found in $PROJECT_DIR" >&2
  echo "Create it first, e.g.: python -m venv venv" >&2
  exit 1
fi

# Activate venv
# shellcheck disable=SC1091
source "venv/bin/activate" || {
  echo "Failed to activate venv at venv/bin/activate" >&2
  exit 1
}

echo "Installing optional OCR dependencies into the venv..."

pip install -r requirements-optional.txt

STATUS=$?

deactivate || true

if [ $STATUS -eq 0 ]; then
  echo "✅ Optional OCR dependencies installed successfully."
  echo "You can now start the app with: ./run.sh"
else
  echo "❌ Installation failed (exit code $STATUS). Check the pip output above." >&2
  exit $STATUS
fi
