@echo off
REM Manga Translator Setup Script for Windows
REM ==========================================

echo.
echo 📚 Manga Translator Setup (Windows)
echo ====================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Check for Tesseract
where tesseract >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  Tesseract OCR not found!
    echo.
    echo Please install manually:
    echo 1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo 2. Run installer and add to PATH
    echo 3. Re-run this script
    echo.
    pause
    exit /b 1
)

echo ✅ Tesseract found

REM Create virtual environment
echo.
echo 🐍 Creating Python virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch first (CPU version for compatibility)
echo.
echo 📦 Installing PyTorch (CPU)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM Install other dependencies
echo.
echo 📦 Installing remaining packages (this may take a few minutes)...
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo To start the app:
echo   venv\Scripts\activate.bat
echo   streamlit run app.py
echo.
echo Then open: http://localhost:8501
echo.
pause
