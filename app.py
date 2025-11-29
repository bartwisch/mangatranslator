import streamlit as st
import os
import tempfile
import certifi
import json
from PIL import Image
from src.pdf_handler import PDFHandler
from src.ocr_handler import OCRHandler
from src.translator import TranslatorService
from src.image_processor import ImageProcessor
from src.ui_state import should_display_thumbnails

# Fix SSL issue permanently for this session
os.environ['SSL_CERT_FILE'] = certifi.where()

st.set_page_config(page_title="Manga Translator", page_icon="logo.png")

st.title("📚 Manga Translator (English -> DEUTSCH!)")

@st.cache_resource
def load_ocr(ocr_engine: str = 'magi'):
    # Magi is best for manga (detects speech bubbles + OCR)
    # manga-ocr is specialized for manga/comic fonts (experimental, optional)
    # PaddleOCR is good general purpose
    return OCRHandler(lang_list=['en'], gpu=False, ocr_engine=ocr_engine)

def load_translator(service_type: str, api_key: str = None):
    # We don't cache this resource anymore if the key can change dynamically
    return TranslatorService(source='en', target='de', service_type=service_type, api_key=api_key)

def parse_page_range(range_str: str) -> list[int]:
    """Parse a page range string (e.g., "1-3, 5, 7-9") into a list of 0-indexed integers."""
    if not range_str.strip():
        return None
    
    pages = set()
    parts = range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                # Convert to 0-indexed, inclusive
                pages.update(range(start - 1, end))
            except ValueError:
                continue
        else:
            try:
                # Convert to 0-indexed
                pages.add(int(part) - 1)
            except ValueError:
                continue
                
    return sorted(list(pages))

from src.utils import load_local_secrets, save_local_secrets

def main():
    # Session State Initialization
    if 'preview_images' not in st.session_state:
        st.session_state.preview_images = []
    if 'temp_pdf_path' not in st.session_state:
        st.session_state.temp_pdf_path = None
    if 'last_uploaded_file_id' not in st.session_state:
        st.session_state.last_uploaded_file_id = None
        
    # Persistent API Keys & Settings (Defaults if not set in Config)
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
        st.session_state.ocr_preprocess_mode = "raw"
    if 'pdf_zoom_factor' not in st.session_state:
        st.session_state.pdf_zoom_factor = 2.0
    if 'ocr_confidence_threshold' not in st.session_state:
        st.session_state.ocr_confidence_threshold = 0.4
    if 'box_padding_x' not in st.session_state:
        st.session_state.box_padding_x = 30
    if 'box_padding_y' not in st.session_state:
        st.session_state.box_padding_y = 10
    if 'stop_translation' not in st.session_state:
        st.session_state.stop_translation = False

    if 'local_secrets_loaded' not in st.session_state:
        secrets = load_local_secrets()
        if not st.session_state.get('stored_openai_key') and isinstance(secrets, dict):
            key = secrets.get('openai_api_key') or secrets.get('openai_key')
            if key:
                st.session_state.stored_openai_key = key
        st.session_state.local_secrets_loaded = True

    # Read settings from Session State
    service_choice = st.session_state.translation_service_selection
    debug_mode = st.session_state.debug_mode_checkbox
    show_boxes = st.session_state.show_boxes_checkbox
    use_ellipse_bubbles = st.session_state.get('use_ellipse_bubbles', True)
    ellipse_padding_x = st.session_state.get('ellipse_padding_x', 15)
    ellipse_padding_y = st.session_state.get('ellipse_padding_y', 15)
    bubble_threshold = st.session_state.bubble_threshold_setting
    ocr_engine = st.session_state.ocr_engine_selection
    ocr_preprocess = st.session_state.ocr_preprocess_mode
    pdf_zoom = st.session_state.pdf_zoom_factor
    ocr_confidence = st.session_state.ocr_confidence_threshold
    box_padding_x = st.session_state.box_padding_x
    box_padding_y = st.session_state.box_padding_y

    pdf_handler = PDFHandler()
    image_processor = ImageProcessor()

    # --- Quick Settings Panel ---
    with st.expander("⚙️ Quick Settings", expanded=not st.session_state.stored_openai_key):
        st.markdown("**OpenAI API Key** (Recommended for best translation quality)")
        col_key_input, col_key_test = st.columns([3, 1])

        with col_key_input:
            st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                key="stored_openai_key",
                value=st.session_state.stored_openai_key,
                help="Your API key is stored securely in this session only.",
                label_visibility="collapsed"
            )

        with col_key_test:
            if st.button("Test Key", key="test_openai_key"):
                if not st.session_state.stored_openai_key:
                    st.error("Please enter an OpenAI API Key before testing.")
                else:
                    try:
                        tester = TranslatorService(
                            source='en',
                            target='de',
                            service_type='openai',
                            api_key=st.session_state.stored_openai_key,
                        )
                        if hasattr(tester, "validate_api_key"):
                            tester.validate_api_key()
                    except Exception as e:
                        st.error(f"❌ API Key Error: {e}")
                    else:
                        st.success("✓ OpenAI API Key is valid.")

        if st.session_state.stored_openai_key:
            st.success("✓ API Key configured")
            if st.button("Save Key on this Device", key="save_openai_key"):
                secrets = load_local_secrets()
                if not isinstance(secrets, dict):
                    secrets = {}
                secrets["openai_api_key"] = st.session_state.stored_openai_key
                try:
                    save_local_secrets(secrets)
                except Exception as e:
                    st.error(f"Could not save key locally: {e}")
                else:
                    st.success("OpenAI API Key saved locally on this machine.")
        else:
            st.info("💡 Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)")
        
        st.divider()
        st.markdown("🔧 **[Visit Configuration Page](/config)** for advanced settings (OCR engine, translation service, debug options, etc.)")

    # --- Service Configuration Logic ---
    api_key = None
    service_type = 'google'
    use_vision = False
    
    if "DeepL" in service_choice:
        service_type = 'deepl'
        api_key = st.session_state.stored_deepl_key
        
    elif "OpenAI" in service_choice:
        service_type = 'openai'
        api_key = st.session_state.stored_openai_key
        
    elif "xAI" in service_choice: # Covers both Grok and Vision
        service_type = 'xai'
        if "Vision" in service_choice:
            use_vision = True
        api_key = st.session_state.stored_xai_key
    
    # Show API key warning at the top if needed
    if (service_type in ['deepl', 'openai', 'xai']) and not api_key:
        st.error(f"⚠️ **Missing API Key:** Please enter your {service_type.capitalize()} API Key in the Quick Settings above or visit the **[Configuration](/config)** page.")
    
    uploaded_file = st.file_uploader("Upload a Manga PDF (English)", type=["pdf"])
    
    # Only load OCR if NOT using vision mode
    if not use_vision:
        ocr_handler = load_ocr(ocr_engine=ocr_engine)
    else:
        ocr_handler = None
    
    # Initialize translator
    if (service_type in ['deepl', 'openai', 'xai']) and not api_key:
        translator = None
    else:
        try:
            translator = load_translator(service_type=service_type, api_key=api_key)
        except Exception as e:
            st.error(f"❌ **Translation Error:** Failed to initialize translator: {e}")
            translator = None

    if uploaded_file is not None:
        # Check for new file upload
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if st.session_state.last_uploaded_file_id != current_file_id:
            # New file detected! Reset state.
            st.session_state.last_uploaded_file_id = current_file_id
            
            # Reset translation state flags
            st.session_state.translation_in_progress = False
            st.session_state.trigger_translation = False
            st.session_state.stop_translation = False
            
            # Cleanup old temp file
            if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
                try:
                    os.remove(st.session_state.temp_pdf_path)
                except:
                    pass

            # Save new temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.temp_pdf_path = tmp_file.name
            
            # Generate Previews (Low Res)
            with st.spinner("Generating page previews..."):
                st.session_state.preview_images = pdf_handler.extract_images_from_pdf(st.session_state.temp_pdf_path, zoom=1)
                
            # Initialize all pages as selected
            for i in range(len(st.session_state.preview_images)):
                st.session_state[f"page_select_{i}"] = True

        st.success("File uploaded successfully!")
        
        # --- Page Selection UI ---
        st.subheader("Select Pages to Translate")

        should_show_thumbnails = should_display_thumbnails(
            st.session_state.get('translation_in_progress', False)
        )
        start_translation = False

        if should_show_thumbnails:
            # Selection Buttons
            col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1, 1, 2, 2])
            if col_sel1.button("Select All"):
                for i in range(len(st.session_state.preview_images)):
                    st.session_state[f"page_select_{i}"] = True
                st.rerun()
                
            if col_sel2.button("Deselect All"):
                for i in range(len(st.session_state.preview_images)):
                    st.session_state[f"page_select_{i}"] = False
                st.rerun()

            # Top Start Translation Button
            if col_sel4.button("🚀 Start Translation", type="primary", key="start_translate_top"):
                # We'll use session state to trigger translation
                st.session_state.trigger_translation = True

            # Grid Display
            num_cols = 4
            cols = st.columns(num_cols)

            # CSS for compact checkboxes
            st.markdown("""
                <style>
                /* Prevent checkbox label wrapping */
                div[data-testid="stCheckbox"] label span {
                    white-space: nowrap;
                }
                </style>
            """, unsafe_allow_html=True)

            # Calculate target aspect ratio (tallest image in the batch)
            target_ratio = 1000.0
            if st.session_state.preview_images:
                for img in st.session_state.preview_images:
                    w, h = img.size
                    ratio = w / h
                    if ratio < target_ratio:
                        target_ratio = ratio
                
                # Limit extreme ratios
                target_ratio = max(target_ratio, 0.5)

            for i, img in enumerate(st.session_state.preview_images):
                # Create new columns for every new row
                if i > 0 and i % num_cols == 0:
                    cols = st.columns(num_cols)

                with cols[i % num_cols]:
                    # Create a bordered container for the "card" look
                    with st.container(border=True):
                        # Check selection state
                        key = f"page_select_{i}"
                        # Use False as default to prevent accidental selection
                        # The key should always exist after upload (initialized above)
                        is_selected = st.session_state.get(key, False)
                        
                        # Toggle Button (acts as header)
                        btn_label = f"✅ Page {i+1}" if is_selected else f"⬜ Page {i+1}"
                        btn_type = "primary" if is_selected else "secondary"
                        
                        if st.button(
                            btn_label,
                            key=f"btn_{i}",
                            type=btn_type,
                            width="stretch",
                        ):
                            # Toggle state
                            st.session_state[key] = not is_selected
                            st.rerun()
                        
                        # Image with negative margin to pull it up closer
                        st.markdown('<div style="margin-top: -10px;"></div>', unsafe_allow_html=True)
                        
                        # Normalize image to target ratio
                        w, h = img.size
                        current_ratio = w / h
                        
                        if abs(current_ratio - target_ratio) > 0.01:
                            # Pad to match target ratio (which is taller/narrower)
                            new_h = int(w / target_ratio)
                            new_w = w
                            
                            # Create background (dark gray for dark mode compatibility)
                            norm_img = Image.new("RGB", (new_w, new_h), (30, 30, 30))
                            offset_y = (new_h - h) // 2
                            norm_img.paste(img, (0, offset_y))
                            st.image(norm_img, width="stretch")
                        else:
                            st.image(img, width="stretch")
                        
                        # Second toggle button below image (for clicking on image area)
                        # Using a minimal icon-only button
                        toggle_icon = "✓" if is_selected else "○"
                        if st.button(
                            toggle_icon,
                            key=f"img_btn_{i}",
                            help="Click to toggle selection",
                            width="stretch",
                        ):
                            # Toggle state
                            st.session_state[key] = not is_selected
                            st.rerun()
        else:
            st.info("Translation has started. Thumbnails are hidden while processing is underway.")

        selected_indices = [
            i for i in range(len(st.session_state.preview_images))
            if st.session_state.get(f"page_select_{i}", True)
        ]
        st.write(f"Selected {len(selected_indices)} pages.")

        # --- Translation Trigger (Bottom Button or Top Button) ---
        if should_show_thumbnails:
            # Disable start button if translation is already in progress
            start_translation = st.button("Start Translation", type="primary", key="start_translate_bottom", disabled=st.session_state.get('translation_in_progress', False)) or st.session_state.get('trigger_translation', False)
        else:
            start_translation = st.session_state.get('trigger_translation', False)
        
        # Reset trigger
        if st.session_state.get('trigger_translation', False):
            st.session_state.trigger_translation = False
        
        if start_translation:
            if not selected_indices:
                st.error("Please select at least one page.")
            elif translator is None:
                st.error("Translator not initialized. Please check your API Key in Configuration.")
            else:
                # Validate API key before starting translation to surface clear errors early
                try:
                    if hasattr(translator, "validate_api_key"):
                        translator.validate_api_key()
                except Exception as e:
                    st.error(f" API Key Error: {e}")
                else:
                    # Set translation in progress flag
                    st.session_state.translation_in_progress = True
                    st.rerun()
        
        # Check if translation is in progress
        if st.session_state.get('translation_in_progress', False):
            tmp_path = st.session_state.temp_pdf_path

            # Hide thumbnails and show progress
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("🔄 Translation in Progress")
            with col2:
                if st.button("⏹️ Stop", type="secondary", help="Stop translation and save completed pages"):
                    st.session_state.stop_translation = True
                    st.rerun()
            st.info(f"Translating {len(selected_indices)} selected pages...")
            
            try:
                # 1. Extract Images (High Res for processing)
                # Only extract the pages we actually need
                status_placeholder = st.empty()
                status_placeholder.info(f"📄 Extracting {len(selected_indices)} pages in high resolution (Zoom: {pdf_zoom}x)...")
                
                images = pdf_handler.extract_images_from_pdf(tmp_path, pages=selected_indices, zoom=pdf_zoom)
                
                processed_images = []
                all_text_data = [] # For debug mode
                
                # Progress tracking
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for i, img in enumerate(images):
                    # Check if user requested to stop
                    if st.session_state.get('stop_translation', False):
                        st.warning("⏹️ Translation stopped by user. Saving completed pages...")
                        break
                        
                    original_page_num = selected_indices[i] + 1
                    progress_text.text(f"🔍 Processing page {original_page_num} ({i+1}/{len(images)})...")
                    
                    text_regions = []
                    
                    if use_vision:
                        # VISION MODE
                        st.info(f"Analyzing page {original_page_num} with Grok Vision...")
                        vision_results = translator.translate_image_with_vision(img)
                        
                        for item in vision_results:
                            bbox = item['bbox']
                            original = item['original']
                            translated = item['translated']
                            text_regions.append((bbox, original, translated))
                            
                            if debug_mode:
                                all_text_data.append({"Page": original_page_num, "Original": original, "Translated": translated, "Type": "Vision"})
                                
                    else:
                        # CLASSIC OCR MODE
                        # 2. OCR mit Sprechblasen-Gruppierung
                        # Verwendet detect_and_group_text um nahe Textblöcke zusammenzufassen
                        ocr_results = ocr_handler.detect_and_group_text(
                            img, 
                            distance_threshold=bubble_threshold,
                            preprocess_mode=ocr_preprocess,
                            confidence_threshold=ocr_confidence,
                            box_padding_x=box_padding_x,
                            box_padding_y=box_padding_y
                        )
                        
                        for bbox, text in ocr_results:
                            # Überspringe leere oder sehr kurze Texte
                            if len(text.strip()) < 2:
                                continue
                                
                            # Übersetze den gesamten gruppierten Text
                            translated_text = translator.translate_text(text)
                            text_regions.append((bbox, text, translated_text))
                            
                            if debug_mode:
                                all_text_data.append({"Page": original_page_num, "Original": text, "Translated": translated_text, "Type": "OCR"})
                    
                    # 4. Image Processing (Common for both modes)
                    if show_boxes:
                        # Nur Rahmen zeichnen ohne Text zu ersetzen
                        processed_img = image_processor.draw_boxes_only(img.copy(), text_regions)
                    else:
                        processed_img = image_processor.overlay_text(
                            img.copy(),
                            text_regions,
                            use_ellipse=use_ellipse_bubbles,
                            ellipse_padding_x=ellipse_padding_x,
                            ellipse_padding_y=ellipse_padding_y,
                        )
                    processed_images.append(processed_img)
                    
                    progress_bar.progress((i + 1) / len(images))
                
                # 5. Save Result
                if not processed_images:
                    st.error("No pages were processed. No PDF generated.")
                    st.session_state.translation_in_progress = False
                    st.session_state.stop_translation = False
                    return

                # Build dynamic output filename: <title>-pageX-Y-german.pdf
                base_name = "translated_manga"
                if uploaded_file is not None and getattr(uploaded_file, "name", None):
                    base_name = os.path.splitext(uploaded_file.name)[0]

                # Sanitize base name for filesystem safety
                safe_base = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in base_name)

                if selected_indices:
                    first_page = selected_indices[0] + 1
                    last_page = selected_indices[-1] + 1
                    if first_page == last_page:
                        page_part = f"page{first_page}"
                    else:
                        page_part = f"page{first_page}-{last_page}"
                else:
                    page_part = "pages"

                language_part = "german"
                output_pdf_name = f"{safe_base}-{page_part}-{language_part}.pdf"

                # Always save to a fixed filename on disk so we don't accumulate many PDFs
                output_pdf_path = "translated_manga.pdf"
                pdf_handler.save_images_as_pdf(processed_images, output_pdf_path)
                
                # Check if translation was stopped
                if st.session_state.get('stop_translation', False):
                    st.warning(f"⏹️ Translation stopped. Saved {len(processed_images)} out of {len(selected_indices)} pages.")
                else:
                    st.success("Translation Complete!")

                # Download Button & Preview (show results as early as possible)
                with open(output_pdf_path, "rb") as f:
                    pdf_data = f.read()
                    st.download_button(
                        label="Download Translated PDF",
                        data=pdf_data,
                        file_name=output_pdf_name,
                        mime="application/pdf"
                    )
                    
                    # Show Preview Images (More reliable than PDF iframe)
                    st.divider()
                    st.markdown("###  Preview (Processed Pages)")
                    for i, p_img in enumerate(processed_images):
                        st.image(
                            p_img,
                            caption=f"Translated Page {selected_indices[i] + 1}",
                            width="stretch",
                        )

                # Display Cost Stats if available (below the main result)
                if hasattr(translator, 'get_usage_stats'):
                    stats = translator.get_usage_stats()
                    if stats['input_tokens'] > 0:
                        st.divider()
                        st.subheader(" Cost & Usage Estimate")
                        col_cost1, col_cost2, col_cost3 = st.columns(3)
                        col_cost1.metric("Input Tokens", f"{stats['input_tokens']:,}")
                        col_cost2.metric("Output Tokens", f"{stats['output_tokens']:,}")
                        
                        cost = translator.get_cost_estimate()
                        col_cost3.metric("Estimated Cost", f"${cost:.4f}")
                        st.caption("Note: Cost estimate based on GPT-4o-mini pricing ($0.15/$0.60 per 1M tokens).")

                if debug_mode and all_text_data:
                    st.divider()
                    st.subheader(" Debug: OCR & Translation Data")
                    st.dataframe(all_text_data, width="stretch")
                    
                    # Reset translation flags
                    st.session_state.translation_in_progress = False
                    st.session_state.stop_translation = False
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.translation_in_progress = False
                st.session_state.stop_translation = False
            finally:
                # Cleanup - don't remove temp file, we might need it for retries
                pass

if __name__ == "__main__":
    main()
