import streamlit as st
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.pdf_handler import PDFHandler

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")

st.title("⚙️ Configuration")

# Initialize PDF handler globally
pdf_handler = PDFHandler()

# Session state initialization
if 'config_pdf_path' not in st.session_state:
    st.session_state.config_pdf_path = None
if 'config_previews' not in st.session_state:
    st.session_state.config_previews = []
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = None
if 'ocr_cache' not in st.session_state:
    st.session_state.ocr_cache = {}
if 'high_res_images' not in st.session_state:
    st.session_state.high_res_images = {}

# --- Global Settings (Session State) ---
if 'stored_deepl_key' not in st.session_state:
    st.session_state.stored_deepl_key = ""
if 'stored_openai_key' not in st.session_state:
    st.session_state.stored_openai_key = ""
if 'stored_xai_key' not in st.session_state:
    st.session_state.stored_xai_key = ""
if 'translation_service_selection' not in st.session_state:
    st.session_state.translation_service_selection = "OpenAI GPT-4o-mini (API Key - Recommended)"
if 'debug_mode_checkbox' not in st.session_state:
    st.session_state.debug_mode_checkbox = False
if 'show_boxes_checkbox' not in st.session_state:
    st.session_state.show_boxes_checkbox = False
if 'use_ellipse_bubbles' not in st.session_state:
    st.session_state.use_ellipse_bubbles = True
if 'ellipse_padding_x' not in st.session_state:
    st.session_state.ellipse_padding_x = 15
if 'ellipse_padding_y' not in st.session_state:
    st.session_state.ellipse_padding_y = 15
if 'bubble_threshold_setting' not in st.session_state:
    st.session_state.bubble_threshold_setting = 160
if 'ocr_engine_selection' not in st.session_state:
    st.session_state.ocr_engine_selection = "magi"
if 'ocr_preprocess_mode' not in st.session_state:
    st.session_state.ocr_preprocess_mode = "gentle"
if 'pdf_zoom_factor' not in st.session_state:
    st.session_state.pdf_zoom_factor = 2.0
if 'ocr_confidence_threshold' not in st.session_state:
    st.session_state.ocr_confidence_threshold = 0.4
if 'box_padding_x' not in st.session_state:
    st.session_state.box_padding_x = 30
if 'box_padding_y' not in st.session_state:
    st.session_state.box_padding_y = 10

# Ensure config page starts with known defaults the first time it's opened
if 'config_initialized' not in st.session_state:
    st.session_state.box_padding_x = 30
    st.session_state.box_padding_y = 10
    st.session_state.config_initialized = True

# Create tabs for different configuration sections
tab_general, tab_ocr_tool = st.tabs(["🌍 General Settings", "🔧 OCR Tool"])

with tab_general:
    st.header("Global Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Translation Service")
        service_options = [
            "OpenAI GPT-4o-mini (API Key - Recommended)", 
            "Google Translate (Free - Experimental)", 
            "DeepL (API Key - Experimental)", 
            "xAI Grok (API Key - Experimental)", 
            "xAI Grok Vision (No OCR - Experimental)"
        ]
        
        st.selectbox(
            "Select Translation Service",
            options=service_options,
            key='translation_service_selection'
        )
        
        service_choice = st.session_state.translation_service_selection
        
        if "DeepL" in service_choice:
            st.text_input("DeepL API Key", type="password", key="stored_deepl_key", help="Paste your DeepL API Key here.")
        elif "OpenAI" in service_choice:
            st.text_input(
                "OpenAI API Key",
                type="password",
                key="stored_openai_key",
                value=st.session_state.stored_openai_key,
                help="Paste your OpenAI API Key here.",
            )
        elif "xAI" in service_choice:
            st.text_input("xAI API Key", type="password", key="stored_xai_key", help="Paste your xAI API Key here.")

    with col2:
        st.subheader("Debug & Display Options")
        st.info("Debug & Display Options sind im 'OCR Tool'-Tab verfügbar.")

with tab_ocr_tool:
    st.header("OCR Configuration & Testing Tool")
    
    # OCR Settings at the top
    st.subheader("🔧 Global OCR Settings")
    col_ocr1, col_ocr2 = st.columns(2)
    
    with col_ocr1:
        st.selectbox(
            "OCR Engine",
            options=['magi', 'manga-ocr', 'paddleocr', 'easyocr'],
            key='ocr_engine_selection',
            help="'magi' = best for manga (detects speech bubbles) [DEFAULT], 'manga-ocr' = specialized for manga fonts (experimental, optional), 'paddleocr' = fast and general purpose, 'easyocr' = multi-language support"
        )
    
    with col_ocr2:
        st.selectbox(
            "OCR Preprocessing",
            options=['gentle', 'none', 'raw', 'aggressive'],
            key='ocr_preprocess_mode',
            help="'gentle' = recommended for manga, 'none' = original image (3x scaled), 'raw' = no scaling, 'aggressive' = strong binarization"
        )
    
    
    st.divider()
    
    # Current Settings Display (always visible)
    st.subheader("📊 Current OCR Settings")
    col_settings1, col_settings2, col_settings3, col_settings4 = st.columns(4)
    
    with col_settings1:
        st.metric("OCR Engine", st.session_state.ocr_engine_selection.upper())
    
    with col_settings2:
        st.metric("Preprocessing", st.session_state.ocr_preprocess_mode.capitalize())
    
    with col_settings3:
        st.metric("Confidence Threshold", f"{st.session_state.ocr_confidence_threshold:.2f}")
    
    with col_settings4:
        st.metric("Box Padding", f"X:{st.session_state.box_padding_x}px Y:{st.session_state.box_padding_y}px")
    
    st.divider()

    # Debug & Display Options (moved here from General tab)
    st.subheader("Debug & Display Options")
    dbg_col1, dbg_col2 = st.columns(2)
    
    with dbg_col1:
        st.checkbox("Debug Mode", help="Show OCR text vs. Translation table.", key="debug_mode_checkbox")
        st.checkbox("Show OCR Boxes", help="Zeigt nur die erkannten Textbereiche als Rahmen.", key="show_boxes_checkbox")
        st.checkbox("Elliptical Bubbles", help="Nutze Ellipsen statt Rechtecken für übersetzte Textblasen.", key="use_ellipse_bubbles")
        st.info("💡 **Bubble-Erkennung:** Die Textboxen werden jetzt automatisch auf die tatsächliche Sprechblasen-Größe begrenzt.")
    
    with dbg_col2:
        st.slider(
            "Max Ellipse Padding X",
            min_value=0,
            max_value=80,
            step=1,
            key="ellipse_padding_x",
            help="Maximales horizontales Padding (wird durch Bubble-Erkennung begrenzt).",
        )
        st.slider(
            "Max Ellipse Padding Y",
            min_value=0,
            max_value=80,
            step=1,
            key="ellipse_padding_y",
            help="Maximales vertikales Padding (wird durch Bubble-Erkennung begrenzt).",
        )
        st.slider(
            "Bubble Grouping Distance (Global)", 
            min_value=30, 
            max_value=300, 
            step=10,
            key="bubble_threshold_setting",
            help="Maximaler Abstand (Pixel) um Textzeilen zu einer Sprechblase zusammenzufassen. Höher = mehr Gruppierung."
        )
        st.slider(
            "PDF Zoom Factor (Global)",
            min_value=1.0,
            max_value=3.0,
            step=0.5,
            key="pdf_zoom_factor",
            help="Auflösung beim Importieren von PDFs. 2.0 = Standard. 1.0 = Schneller/Weicher. 3.0 = Schärfer."
        )
        st.slider(
            "OCR Confidence Threshold (Global)",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="ocr_confidence_threshold",
            help="Minimale Sicherheit für Textboxen. Boxen unter diesem Wert werden ignoriert."
        )
        st.slider(
            "Box Padding X (Global)",
            min_value=-10,
            max_value=50,
            step=1,
            key="box_padding_x",
            help="Vergrößert (>0) oder verkleinert (<0) die erkannten Boxen horizontal (links/rechts)."
        )
        st.slider(
            "Box Padding Y (Global)",
            min_value=-10,
            max_value=50,
            step=1,
            key="box_padding_y",
            help="Vergrößert (>0) oder verkleinert (<0) die erkannten Boxen vertikal (oben/unten)."
        )

    st.divider()
    st.subheader("📄 Test OCR on PDF Pages")
    st.markdown("Lade ein PDF hoch, klicke auf eine Seite, und passe den Threshold an um die Sprechblasen-Erkennung zu optimieren.")

    def draw_boxes(image: Image.Image, text_results):
        """Zeichnet farbige Boxen auf das Bild"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Lade Font
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            except:
                font = ImageFont.load_default()
        
        colors = ["#FF0000", "#0066FF", "#00CC00", "#FF9900", "#9900FF", "#00CCCC", "#FF00FF", "#FFCC00"]
        
        for i, item in enumerate(text_results):
            bbox = item[0]
            text = item[1] if len(item) > 1 else ""
            
            pts = np.array(bbox)
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            
            box_color = colors[i % len(colors)]
            
            # Dicken Rahmen zeichnen
            for offset in range(4):
                draw.rectangle(
                    [x_min - offset, y_min - offset, x_max + offset, y_max + offset], 
                    outline=box_color
                )
            
            # Hintergrund für Label
            label = f"[{i+1}]"
            draw.rectangle([x_min, y_min - 20, x_min + 30, y_min], fill=box_color)
            draw.text((x_min + 2, y_min - 18), label, fill="white", font=font)
        
        return img_copy

    def select_page(page_num):
        st.session_state.selected_page = page_num

    # PDF Upload
    uploaded_pdf = st.file_uploader("📄 PDF hochladen (für OCR Config)", type=["pdf"])

    if uploaded_pdf:
        # Check if new file
        file_id = f"{uploaded_pdf.name}_{uploaded_pdf.size}"
        
        if st.session_state.config_pdf_path is None or not os.path.exists(st.session_state.config_pdf_path):
            # Save temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                st.session_state.config_pdf_path = tmp.name
            
            # Generate previews
            with st.spinner("Lade Seiten-Vorschau..."):
                st.session_state.config_previews = pdf_handler.extract_images_from_pdf(
                    st.session_state.config_pdf_path, zoom=0.8
                )
            st.session_state.selected_page = None
            st.session_state.ocr_cache = {}
            st.session_state.high_res_images = {}
        
        if st.session_state.config_previews:
            if st.session_state.selected_page is None:
                # Show page grid
                st.subheader("📖 Seite auswählen")
                
                num_cols = 5
                cols = st.columns(num_cols)
                
                for i, preview in enumerate(st.session_state.config_previews):
                    with cols[i % num_cols]:
                        st.markdown(f"Seite {i+1}")
                        
                        st.image(preview, width="stretch")
                        
                        st.button(
                            f"Auswählen", 
                            key=f"select_page_{i}",
                            on_click=select_page,
                            args=(i,),
                            type="secondary"
                        )
            
            # If page selected, show OCR config
            if st.session_state.selected_page is not None:
                if st.button("← Zurück zur Übersicht"):
                    st.session_state.selected_page = None
                    st.rerun()
                
                st.divider()
                
                page_idx = st.session_state.selected_page
                
                # Sidebar controls for OCR Tool
                st.sidebar.header("🔧 OCR Tool Einstellungen")
                
                # Local threshold for this tool, defaulting to global setting
                tool_bubble_threshold = st.sidebar.slider(
                    "Bubble Grouping Distance (Test)", 
                    min_value=30, 
                    max_value=400, 
                    value=st.session_state.bubble_threshold_setting, 
                    step=10,
                    help="Test-Wert für diesen Viewer. Ändert nicht die globale Einstellung."
                )
                
                ocr_engine = st.sidebar.selectbox(
                    "OCR Engine",
                    options=['magi', 'manga-ocr', 'paddleocr', 'easyocr'],
                    index=0,
                    help="'magi' = beste für Manga (erkennt Sprechblasen) [STANDARD], 'manga-ocr' = experimentell für Manga-Fonts (optional), 'paddleocr' = schnell"
                )
                
                preprocess_mode = st.sidebar.selectbox(
                    "OCR Preprocessing",
                    options=['gentle', 'none', 'raw', 'aggressive'],
                    index=0,
                    help="'gentle' = empfohlen für Manga, 'none' = Originalbild (3x), 'raw' = Keine Skalierung, 'aggressive' = starke Binarisierung"
                )
                
                show_raw = st.sidebar.checkbox("Zeige Roh-OCR zum Vergleich", value=False)
                show_translation_preview = st.sidebar.checkbox("Zeige Übersetzungs-Vorschau", value=False, help="Zeigt wie der übersetzte Text aussehen würde (mit weißem Hintergrund)")
                
                # New Test Controls
                st.sidebar.markdown("---")
                test_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, st.session_state.ocr_confidence_threshold, 0.05)
                test_padding_x = st.sidebar.slider("Box Padding X", -10, 50, st.session_state.box_padding_x, 1, help="Horizontal (links/rechts)")
                test_padding_y = st.sidebar.slider("Box Padding Y", -10, 50, st.session_state.box_padding_y, 1, help="Vertikal (oben/unten)")
                
                st.sidebar.divider()
                st.sidebar.info("💡 **Tipps:**\n- Magi ist am besten für Manga\n- Gentle preprocessing empfohlen")
                
                # Load high-res image for selected page
                cache_key = f"page_{page_idx}"
                
                if cache_key not in st.session_state.high_res_images:
                    with st.spinner(f"Lade Seite {page_idx + 1} in hoher Auflösung..."):
                        high_res = pdf_handler.extract_images_from_pdf(
                            st.session_state.config_pdf_path, 
                            pages=[page_idx], 
                            zoom=2
                        )
                        if high_res:
                            st.session_state.high_res_images[cache_key] = high_res[0]
                
                if cache_key in st.session_state.high_res_images:
                    image = st.session_state.high_res_images[cache_key]
                    
                    # Run OCR (cached per page, engine, preprocess mode)
                    # Run OCR (cached per page, engine, preprocess mode) - RAW RESULTS ONLY
                    ocr_key = f"ocr_{page_idx}_{ocr_engine}_{preprocess_mode}_raw"
                    if ocr_key not in st.session_state.ocr_cache:
                        with st.spinner(f"🔍 Analysiere Text mit {ocr_engine.upper()}..."):
                            # Lazy load OCR handler here to avoid circular imports if any
                            from src.ocr_handler import OCRHandler
                            ocr_handler_tool = OCRHandler(lang_list=['en'], gpu=False, ocr_engine=ocr_engine)
                            
                            # Get RAW results (no filtering/padding yet)
                            raw_results = ocr_handler_tool.detect_text(
                                image, 
                                paragraph=False, 
                                preprocess_mode=preprocess_mode,
                                confidence_threshold=0.0,
                                box_padding_x=0,
                                box_padding_y=0
                            )
                            st.session_state.ocr_cache[ocr_key] = raw_results
                    
                    # Get cached raw results
                    cached_raw_results = st.session_state.ocr_cache[ocr_key]
                    
                    # Apply dynamic filtering/padding
                    from src.ocr_handler import OCRHandler
                    # We need an instance to call the method, or make it static. It's an instance method.
                    # Re-instantiating is cheap if model is lazy loaded or we reuse.
                    # Ideally we reuse ocr_handler_tool but it's inside the if block.
                    ocr_handler_tool = OCRHandler(lang_list=['en'], gpu=False, ocr_engine=ocr_engine)
                    
                    # Convert PIL Image to numpy array to get shape
                    import numpy as np
                    img_array = np.array(image)
                    
                    filtered_results = ocr_handler_tool.filter_results(
                        cached_raw_results, 
                        confidence_threshold=test_conf, 
                        box_padding_x=test_padding_x,
                        box_padding_y=test_padding_y,
                        image_shape=img_array.shape[:2]
                    )
                    
                    # Use filtered results for grouping and display
                    raw_results = filtered_results
                    
                    # Group with current threshold
                    from src.ocr_handler import OCRHandler
                    ocr_handler_tool = OCRHandler(lang_list=['en'], gpu=False, ocr_engine=ocr_engine)
                    grouped_results = ocr_handler_tool.group_text_into_bubbles(raw_results, distance_threshold=tool_bubble_threshold)
                    
                    # Display
                    st.subheader(f"📄 Seite {page_idx + 1} - OCR Ergebnis")
                    
                    # Translation Preview Mode
                    if show_translation_preview:
                        st.markdown(f"**🎨 Übersetzungs-Vorschau: {len(grouped_results)} Boxen** (Padding: X={test_padding_x}px, Y={test_padding_y}px)")
                        
                        # Import translator and image processor
                        from src.translator import TranslatorService
                        from src.image_processor import ImageProcessor
                        
                        # Get API key from session state
                        api_key = st.session_state.get('stored_openai_key', '')
                        
                        if not api_key:
                            st.warning("⚠️ Kein OpenAI API Key gefunden. Bitte in den Einstellungen eingeben.")
                            # Show boxes instead
                            preview_image = draw_boxes(image, grouped_results)
                            st.image(preview_image, width="stretch")
                        else:
                            with st.spinner("Übersetze Text..."):
                                try:
                                    translator = TranslatorService(source='en', target='de', service_type='openai', api_key=api_key)
                                    image_processor = ImageProcessor()
                                    
                                    # Prepare text regions for translation
                                    text_regions = []
                                    for bbox, text in grouped_results:
                                        if len(text.strip()) >= 2:
                                            translated = translator.translate_text(text)
                                            text_regions.append((bbox, text, translated))
                                    
                                    # Create preview with translation overlay
                                    use_ellipse = st.session_state.get('use_ellipse_bubbles', True)
                                    ellipse_padding_x = st.session_state.get('ellipse_padding_x', 28)
                                    ellipse_padding_y = st.session_state.get('ellipse_padding_y', 28)
                                    preview_image = image_processor.overlay_text(
                                        image.copy(),
                                        text_regions,
                                        use_ellipse=use_ellipse,
                                        ellipse_padding_x=ellipse_padding_x,
                                        ellipse_padding_y=ellipse_padding_y,
                                    )
                                    st.image(preview_image, width="stretch")
                                    
                                except Exception as e:
                                    st.error(f"Übersetzungsfehler: {e}")
                                    # Fallback to boxes
                                    preview_image = draw_boxes(image, grouped_results)
                                    st.image(preview_image, width="stretch")
                    
                    elif show_raw:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**🔴 Roh-OCR: {len(raw_results)} Boxen**")
                            raw_image = draw_boxes(image, raw_results)
                            st.image(raw_image, width="stretch")
                        
                        with col2:
                            st.markdown(f"**🟢 Gruppiert: {len(grouped_results)} Boxen** (Threshold: {tool_bubble_threshold}px)")
                            grouped_image = draw_boxes(image, grouped_results)
                            st.image(grouped_image, width="stretch")
                    else:
                        st.markdown(f"**🟢 Gruppiert: {len(grouped_results)} Boxen** (Threshold: {tool_bubble_threshold}px)")
                        grouped_image = draw_boxes(image, grouped_results)
                        st.image(grouped_image, width="stretch")
                    
                    # Show detected texts
                    with st.expander(f"📝 Erkannte Texte ({len(grouped_results)} Gruppen)", expanded=True):
                        for i, item in enumerate(grouped_results):
                            text = item[1] if len(item) > 1 else ""
                            colors = ["🔴", "🔵", "🟢", "🟠", "🟣", "🩵", "🩷", "🟡"]
                            color = colors[i % len(colors)]
                            st.markdown(f"{color} **[{i+1}]** {text}")
                    
                    # Stats
                    st.divider()
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    col_stat1.metric("Roh-Boxen", len(raw_results))
                    col_stat2.metric("Gruppierte Boxen", len(grouped_results))
                    reduction = 100 - (len(grouped_results) / max(len(raw_results), 1) * 100)
                    col_stat3.metric("Reduktion", f"{reduction:.0f}%")

    else:
        st.info("👆 Lade ein PDF hoch um die OCR-Boxen zu konfigurieren.")
