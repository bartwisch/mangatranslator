---
title: Manga Translator
emoji: 📚
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: gradio_app.py
pinned: false
license: mit
---

# 📚 Manga Translator

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/bartwisch/mangatranslator)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

An AI-powered application to translate Manga/Comic PDFs from English to German. It preserves the original layout by detecting speech bubbles, removing the original text, and overlaying the translated text.

**Repository:** [github.com/bartwisch/mangatranslator](https://github.com/bartwisch/mangatranslator/)

## ✨ Features

*   **Multiple OCR Engines** (Lazy Loading):
    *   **Magi** ⭐ (The Manga Whisperer) - Default, best for manga, detects speech bubbles automatically
    *   **manga-ocr** - Specialized for manga fonts (optional)
    *   **PaddleOCR** - Good general purpose (optional)
    *   **EasyOCR** - Multi-language support (optional)
*   **Speech Bubble Grouping**: Automatically groups text lines within speech bubbles for context-aware translation
*   **Multiple Translation Engines**:
    *   **Google Translate** (Free)
    *   **DeepL** (High Quality, requires API Key)
    *   **OpenAI GPT-4o-mini** (Context-aware, requires API Key)
    *   **xAI Grok** (Context-aware, requires API Key)
    *   **xAI Grok Vision** (No OCR needed, uses vision model)
*   **Smart Layout**: Automatically cleans speech bubbles and fits translated text (dynamically resizing fonts).
*   **Interactive Preview**: Select specific pages to translate visually.
*   **OCR Config Page**: Live preview to tune OCR parameters and bubble grouping.
*   **Cost Estimation**: Shows token usage and estimated costs for AI models.

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

Best option – provides **16 GB RAM** for free, which is needed for the OCR models.

1.  Go to **[huggingface.co/spaces](https://huggingface.co/spaces)**
2.  Click **"Create new Space"**
3.  Select **Gradio** as SDK
4.  Clone this repo or link your GitHub repo
5.  The app will auto-deploy using the YAML header in this README

**Live Demo:** [huggingface.co/spaces/bartwisch/mangatranslator](https://huggingface.co/spaces/bartwisch/mangatranslator)

### Streamlit Cloud (Limited RAM)

Alternative with less RAM (~1 GB), may struggle with large PDFs.

1.  Go to **[share.streamlit.io](https://share.streamlit.io/)**
2.  Click **"New App"**
3.  Select "Use existing repo" and enter: `bartwisch/mangatranslator`
4.  Set **Main file path** to `app.py`
5.  Click **Deploy!** 🎈

*Note: First deployment takes 3-5 minutes to install PyTorch and OCR models.*

## 🛠️ Local Installation

### Quick Setup (Recommended)

**macOS / Linux:**
```bash
git clone https://github.com/bartwisch/mangatranslator.git
cd mangatranslator
./setup.sh
```

**Windows:**
```cmd
git clone https://github.com/bartwisch/mangatranslator.git
cd mangatranslator
setup.bat
```

### Manual Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/bartwisch/mangatranslator.git
    cd mangatranslator
    ```

2.  **Set up Python environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate.bat
    
    # Install base requirements (includes Magi OCR)
    pip install -r requirements.txt
    ```

3.  **Optional: Install additional OCR engines**:
    ```bash
    # Install all optional engines
    pip install -r requirements-optional.txt
    
    # Or install individually:
    pip install manga-ocr paddlepaddle paddleocr  # Manga-OCR
    pip install paddlepaddle paddleocr            # PaddleOCR only
    pip install easyocr                           # EasyOCR only
    ```

4.  **Run the app**:
    ```bash
    streamlit run app.py
    # Or use: ./run.sh
    ```
    
5.  **Open in browser**: http://localhost:8501

### OCR Configuration Page

Navigate to the **Configuration** page in the app to:
- Select your preferred OCR engine (Magi is default)
- Choose OCR preprocessing mode
- Upload a PDF and preview OCR detection
- Adjust bubble grouping threshold
- Compare different OCR engines

## 🔑 API Keys

The app requires API Keys for **DeepL**, **OpenAI**, or **xAI** if you choose to use those services.
*   Keys are entered securely in the Configuration page.
*   Keys are **NOT** stored in the repository.
*   Google Translate is available as a free fallback.

## 📋 Requirements

*   Python 3.10+
*   See `requirements.txt` for base Python packages (includes Magi OCR).
*   See `requirements-optional.txt` for optional OCR engines.
*   See `packages.txt` for system dependencies (required for Linux/Cloud deployment).

## 🎯 OCR Engine Comparison

| Engine | Best For | Speed | Quality | Installation |
|--------|----------|-------|---------|--------------|
| **Magi** ⭐ | Manga (auto bubble detection) | Medium | Excellent | Default ✅ |
| Manga-OCR | Manga/Comic fonts | Fast | Very Good | Optional |
| PaddleOCR | General purpose | Fast | Good | Optional |
| EasyOCR | Multi-language | Slow | Good | Optional |

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

### Third-Party OCR Libraries

This application uses or optionally supports several third-party OCR engines and libraries, including but not limited to:

- `magi-ocr` (custom model stack based on PyTorch and Transformers)
- `manga-ocr` (MIT License)
- `PaddleOCR` (Apache-2.0 License)
- `EasyOCR` (Apache-2.0 License)

These components are subject to their respective licenses as provided by their authors.
