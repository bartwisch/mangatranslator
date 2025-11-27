import os
import tempfile
from typing import List, Optional

import certifi
import gradio as gr
from PIL import Image

from src.pdf_handler import PDFHandler
from src.ocr_handler import OCRHandler
from src.translator import TranslatorService
from src.image_processor import ImageProcessor

try:
    import spaces
except Exception:
    class _GPUDecorator:
        def __call__(self, fn):
            return fn

    class _SpacesFallback:
        GPU = _GPUDecorator()

    spaces = _SpacesFallback()

# Fix SSL issues for HTTPS APIs (DeepL / OpenAI / xAI)
os.environ["SSL_CERT_FILE"] = certifi.where()

pdf_handler = PDFHandler()
image_processor = ImageProcessor()


def parse_page_range(range_str: str) -> List[int]:
    """Parse a page range string (e.g., "1-3, 5, 7-9") into a list of 0-indexed integers."""
    if not range_str or not range_str.strip():
        return []

    pages: List[int] = []
    parts = [p.strip() for p in range_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            pages.extend(list(range(start, end + 1)))
        else:
            try:
                pages.append(int(part))
            except ValueError:
                continue

    unique_pages = sorted(set(p for p in pages if p > 0))
    return [p - 1 for p in unique_pages]


def _build_translator(service_label: str, deepl_key: str, openai_key: str, xai_key: str) -> TranslatorService:
    label_map = {
        "Google Translate": "google",
        "DeepL": "deepl",
        "OpenAI GPT-4o-mini": "openai",
        "xAI Grok": "xai",
    }
    service_type = label_map.get(service_label, "google")

    api_key: Optional[str] = None
    if service_type == "deepl":
        api_key = deepl_key or None
    elif service_type == "openai":
        api_key = openai_key or None
    elif service_type == "xai":
        api_key = xai_key or None

    if service_type in ["deepl", "openai", "xai"] and not api_key:
        raise ValueError(f"Missing API key for {service_type} service.")

    return TranslatorService(source="en", target="de", service_type=service_type, api_key=api_key)


@spaces.GPU
def translate_manga(
    pdf_path: str,
    page_range: str,
    ocr_engine: str,
    bubble_threshold: float,
    preprocess_mode: str,
    translator_label: str,
    deepl_key: str,
    openai_key: str,
    xai_key: str,
    show_boxes: bool,
    use_vision: bool,
):
    if not pdf_path:
        return None, [], "Bitte eine PDF-Datei hochladen."

    try:
        pages = parse_page_range(page_range)
    except Exception:
        pages = []

    if not pages:
        pages = None

    translator = _build_translator(translator_label, deepl_key, openai_key, xai_key)

    ocr_handler: Optional[OCRHandler] = None
    if not use_vision:
        # On ZeroGPU we can safely enable GPU acceleration
        ocr_handler = OCRHandler(lang_list=["en"], gpu=True, ocr_engine=ocr_engine)

    progress = gr.Progress(track_tqdm=True)

    progress(0.0, desc="PDF wird geladen und in Bilder umgewandelt...")
    images: List[Image.Image] = pdf_handler.extract_images_from_pdf(pdf_path, zoom=1.5, pages=pages)
    total = len(images)
    if total == 0:
        return None, [], "Keine Seiten im PDF gefunden."

    processed_images: List[Image.Image] = []

    for idx, img in enumerate(images):
        progress((idx / total), desc=f"Verarbeite Seite {idx + 1} von {total}...")

        text_regions = []

        if use_vision:
            vision_results = translator.translate_image_with_vision(img)
            for item in vision_results:
                bbox = item["bbox"]
                original = item.get("original", "")
                translated = item.get("translated", "")
                text_regions.append((bbox, original, translated))
        else:
            ocr_results = ocr_handler.detect_and_group_text(
                img,
                distance_threshold=bubble_threshold,
                preprocess_mode=preprocess_mode,
            )
            for bbox, text in ocr_results:
                if len(text.strip()) < 2:
                    continue
                translated_text = translator.translate_text(text)
                text_regions.append((bbox, text, translated_text))

        if show_boxes:
            processed = image_processor.draw_boxes_only(img.copy(), text_regions)
        else:
            processed = image_processor.overlay_text(img.copy(), text_regions)

        processed_images.append(processed)

    tmp_dir = tempfile.mkdtemp(prefix="mangatranslator_")
    output_pdf_path = os.path.join(tmp_dir, "translated_manga.pdf")

    pdf_handler.save_images_as_pdf(processed_images, output_pdf_path)

    progress(1.0, desc="Fertig!")

    return output_pdf_path, processed_images, "Fertig! Du kannst das übersetzte PDF herunterladen."


with gr.Blocks() as demo:
    gr.Markdown("# 📚 Manga Translator (Gradio + ZeroGPU)")
    gr.Markdown(
        "Lädt ein Manga-PDF, erkennt Sprechblasen per OCR und legt die deutsche Übersetzung in die Sprechblasen."
    )

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(
                label="Manga PDF hochladen",
                file_types=[".pdf"],
                type="filepath",
            )
            page_range = gr.Textbox(
                label="Seitenbereich (optional)",
                placeholder="z.B. 1-5, 7, 10-12 (leer = alle Seiten)",
            )

            ocr_engine = gr.Radio(
                label="OCR-Engine",
                choices=["magi", "manga-ocr", "paddleocr", "easyocr"],
                value="magi",
            )
            preprocess_mode = gr.Radio(
                label="OCR Preprocessing",
                choices=["gentle", "none", "aggressive"],
                value="gentle",
            )
            bubble_threshold = gr.Slider(
                label="Bubble-Gruppierung (Pixel)",
                minimum=20,
                maximum=200,
                value=80,
                step=5,
            )
            show_boxes = gr.Checkbox(
                label="Nur Boxen zeichnen (Debug)",
                value=False,
            )

        with gr.Column():
            translator_label = gr.Radio(
                label="Übersetzungsdienst",
                choices=[
                    "Google Translate",
                    "DeepL",
                    "OpenAI GPT-4o-mini",
                    "xAI Grok",
                ],
                value="Google Translate",
            )
            use_vision = gr.Checkbox(
                label="Vision-Modus (OpenAI/xAI Vision)",
                value=False,
            )
            deepl_key = gr.Textbox(
                label="DeepL API Key",
                type="password",
                visible=True,
            )
            openai_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                visible=True,
            )
            xai_key = gr.Textbox(
                label="xAI API Key",
                type="password",
                visible=True,
            )

            run_btn = gr.Button("🚀 Übersetzen")

    output_pdf = gr.File(label="Übersetztes PDF")
    preview_gallery = gr.Gallery(
        label="Vorschau der verarbeiteten Seiten",
        columns=3,
        height="auto",
    )
    status_box = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=translate_manga,
        inputs=[
            pdf_input,
            page_range,
            ocr_engine,
            bubble_threshold,
            preprocess_mode,
            translator_label,
            deepl_key,
            openai_key,
            xai_key,
            show_boxes,
            use_vision,
        ],
        outputs=[output_pdf, preview_gallery, status_box],
    )


if __name__ == "__main__":
    demo.launch()
