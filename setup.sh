#!/bin/bash

# Manga Translator Setup Script
# ==============================

echo "📚 Manga Translator Setup"
echo "========================="

# Check OS (currently no extra system packages are required)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"
fi

# Create virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo ""
echo "📦 Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Then open: http://localhost:8501"
echo ""
echo "OCR Config Page: http://localhost:8501/config"
