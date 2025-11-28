from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from typing import List, Tuple, Optional
import os

class ImageProcessor:
    def __init__(self):
        pass

    def _find_bubble_contour(self, img_array: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int, margin: int = 50) -> Optional[np.ndarray]:
        """
        Findet die Kontur der Sprechblase, die den erkannten Textbereich enthält.
        
        Die Methode sucht nach weißen/hellen Bereichen um die Textbox und findet
        die passende Sprechblasen-Kontur.
        
        Args:
            img_array: Bild als numpy array (RGB).
            x_min, y_min, x_max, y_max: Bounding Box des erkannten Textes.
            margin: Suchbereich um die Textbox in Pixeln.
            
        Returns:
            Kontur als numpy array oder None wenn keine gefunden.
        """
        h, w = img_array.shape[:2]
        
        # Erweitere den Suchbereich um die Textbox
        search_x1 = max(0, x_min - margin)
        search_y1 = max(0, y_min - margin)
        search_x2 = min(w, x_max + margin)
        search_y2 = min(h, y_max + margin)
        
        # Extrahiere den Suchbereich
        region = img_array[search_y1:search_y2, search_x1:search_x2]
        
        if region.size == 0:
            return None
        
        # Konvertiere zu Graustufen
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Binarisiere - Sprechblasen sind typischerweise weiß (>200)
        # Wir suchen nach hellen Bereichen
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Morphologische Operationen um kleine Lücken zu schließen
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Finde Konturen
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Berechne den Mittelpunkt der Textbox relativ zum Suchbereich
        text_center_x = (x_min + x_max) // 2 - search_x1
        text_center_y = (y_min + y_max) // 2 - search_y1
        text_width = x_max - x_min
        text_height = y_max - y_min
        text_area = text_width * text_height
        
        # Finde die Kontur, die den Textmittelpunkt enthält
        best_contour = None
        best_score = float('inf')
        
        for contour in contours:
            # Prüfe ob der Textmittelpunkt in der Kontur liegt
            if cv2.pointPolygonTest(contour, (text_center_x, text_center_y), False) >= 0:
                # Berechne die Fläche der Kontur
                area = cv2.contourArea(contour)
                
                # Die Kontur sollte mindestens so groß wie der Text sein,
                # aber nicht zu groß (max 20x Textfläche)
                if area >= text_area * 0.5 and area <= text_area * 20:
                    # Bevorzuge Konturen die näher an der Textgröße sind
                    # Score = Verhältnis von Konturfläche zu Textfläche
                    score = abs(area / text_area - 2.0)  # Ideale Blase ist ~2x Textfläche
                    
                    if score < best_score:
                        best_score = score
                        best_contour = contour
        
        if best_contour is not None:
            # Vereinfache die Kontur leicht für glattere Kanten
            epsilon = 0.01 * cv2.arcLength(best_contour, True)
            best_contour = cv2.approxPolyDP(best_contour, epsilon, True)
            
            # Verschiebe die Kontur zurück zu globalen Koordinaten
            best_contour = best_contour + np.array([search_x1, search_y1])
            return best_contour
        
        return None

    def _get_bubble_bounds(self, img_array: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int, padding_x: int = 0, padding_y: int = 0) -> Tuple[int, int, int, int]:
        """
        Ermittelt die Grenzen der Sprechblase, begrenzt auf die tatsächliche Bubble-Kontur.
        
        Falls keine Bubble erkannt wird, wird das ursprüngliche Padding verwendet,
        aber mit einer maximalen Grenze.
        
        Returns:
            Tuple (x_min, y_min, x_max, y_max) der Bubble-Grenzen.
        """
        h, w = img_array.shape[:2]
        
        # Versuche die Bubble-Kontur zu finden
        contour = self._find_bubble_contour(img_array, x_min, y_min, x_max, y_max)
        
        if contour is not None:
            # Berechne die Bounding-Box der Kontur
            bx, by, bw, bh = cv2.boundingRect(contour)
            bubble_x_min = bx
            bubble_y_min = by
            bubble_x_max = bx + bw
            bubble_y_max = by + bh
            
            # Füge ein kleines Padding hinzu (5% der Textbox-Größe),
            # aber bleibe innerhalb der Bubble-Grenzen
            text_w = x_max - x_min
            text_h = y_max - y_min
            small_pad_x = max(2, int(text_w * 0.05))
            small_pad_y = max(2, int(text_h * 0.05))
            
            final_x_min = max(bubble_x_min, x_min - small_pad_x)
            final_y_min = max(bubble_y_min, y_min - small_pad_y)
            final_x_max = min(bubble_x_max, x_max + small_pad_x)
            final_y_max = min(bubble_y_max, y_max + small_pad_y)
            
            return final_x_min, final_y_min, final_x_max, final_y_max
        
        # Fallback: Begrenze das Padding auf maximal 30% der Textbox-Größe
        text_w = x_max - x_min
        text_h = y_max - y_min
        max_pad_x = min(padding_x, int(text_w * 0.3))
        max_pad_y = min(padding_y, int(text_h * 0.3))
        
        final_x_min = max(0, x_min - max_pad_x)
        final_y_min = max(0, y_min - max_pad_y)
        final_x_max = min(w, x_max + max_pad_x)
        final_y_max = min(h, y_max + max_pad_y)
        
        return final_x_min, final_y_min, final_x_max, final_y_max

    def draw_boxes_only(self, image: Image.Image, text_regions: List[Tuple[List[List[int]], str, str]]) -> Image.Image:
        """
        Zeichnet nur rote Rahmen um die erkannten Textbereiche (ohne Text zu ersetzen).
        
        Args:
            image: The original PIL Image.
            text_regions: List of tuples (bbox, original_text, translated_text).
        
        Returns:
            Image with red boxes drawn around text regions.
        """
        draw = ImageDraw.Draw(image)
        
        for bbox, original, translated in text_regions:
            # Calculate bounding rectangle
            pts = np.array(bbox)
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            
            # Draw red rectangle outline (3px thick)
            for offset in range(3):
                draw.rectangle(
                    [x_min - offset, y_min - offset, x_max + offset, y_max + offset], 
                    outline="red"
                )
            
            # Draw text label above box
            try:
                font = self._load_font(12)
            except:
                font = ImageFont.load_default()
            
            # Truncate text if too long
            label = original[:50] + "..." if len(original) > 50 else original
            draw.text((x_min, y_min - 15), label, fill="red", font=font)
            
        return image

    def overlay_text(self, image: Image.Image, text_regions: List[Tuple[List[List[int]], str, str]], use_ellipse: bool = True, ellipse_padding_x: int = 0, ellipse_padding_y: int = 0) -> Image.Image:
        """
        Overlays translated text onto the image.
        
        Args:
            image: The original PIL Image.
            text_regions: List of tuples (bbox, original_text, translated_text).
                         bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
            use_ellipse: If True and no contour found, draw elliptical bubbles, otherwise rectangles.
            ellipse_padding_x: Horizontal padding (used as fallback if no bubble contour found).
            ellipse_padding_y: Vertical padding (used as fallback if no bubble contour found).
        
        Returns:
            Processed PIL Image.
        """
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size
        # Convert once to numpy for background color sampling and bubble detection
        img_array = np.array(image)
        
        for bbox, original, translated in text_regions:
            # Calculate bounding rectangle from OCR
            pts = np.array(bbox)
            ocr_x_min = int(np.min(pts[:, 0]))
            ocr_y_min = int(np.min(pts[:, 1]))
            ocr_x_max = int(np.max(pts[:, 0]))
            ocr_y_max = int(np.max(pts[:, 1]))
            
            # Versuche die tatsächliche Sprechblasen-Kontur zu finden
            bubble_contour = self._find_bubble_contour(
                img_array, 
                ocr_x_min, ocr_y_min, ocr_x_max, ocr_y_max
            )
            
            # Bestimme die Bounding-Box für den Text
            if bubble_contour is not None:
                # Benutze die Bounding-Box der Kontur für den Text
                bx, by, bw, bh = cv2.boundingRect(bubble_contour)
                x_min, y_min = bx, by
                x_max, y_max = bx + bw, by + bh
            else:
                # Fallback: Verwende die OCR-Box mit begrenztem Padding
                x_min, y_min, x_max, y_max = self._get_bubble_bounds(
                    img_array, 
                    ocr_x_min, ocr_y_min, ocr_x_max, ocr_y_max,
                    padding_x=ellipse_padding_x,
                    padding_y=ellipse_padding_y
                )

            # Sample background brightness inside the region to decide
            # whether the bubble is dark or light.
            bubble_fill = "white"
            text_color = "black"
            try:
                region = img_array[ocr_y_min:ocr_y_max, ocr_x_min:ocr_x_max]
                if region.size > 0:
                    # Use luminance (Y) from RGB
                    if len(region.shape) >= 3 and region.shape[-1] >= 3:
                        r = region[..., 0].astype(np.float32)
                        g = region[..., 1].astype(np.float32)
                        b = region[..., 2].astype(np.float32)
                        luma = 0.299 * r + 0.587 * g + 0.114 * b
                    else:
                        # Grayscale image
                        luma = region.astype(np.float32)

                    mean_luma = float(luma.mean()) / 255.0

                    # Threshold: <0.5 => dark bubble
                    if mean_luma < 0.5:
                        bubble_fill = "#111111"
                        text_color = "white"
            except Exception:
                # Fallback to default colors if anything goes wrong
                bubble_fill = "white"
                text_color = "black"
            
            # Zeichne den Hintergrund
            if bubble_contour is not None:
                # Zeichne die tatsächliche Sprechblasen-Kontur als Polygon
                # Konvertiere numpy Kontur zu Liste von Tupeln für PIL
                contour_points = bubble_contour.reshape(-1, 2)
                polygon_points = [(int(p[0]), int(p[1])) for p in contour_points]
                
                if len(polygon_points) >= 3:
                    draw.polygon(polygon_points, fill=bubble_fill, outline=bubble_fill)
            else:
                # Fallback: Ellipse oder Rechteck
                if use_ellipse:
                    draw.ellipse([x_min, y_min, x_max, y_max], fill=bubble_fill, outline=bubble_fill)
                else:
                    draw.rectangle([x_min, y_min, x_max, y_max], fill=bubble_fill, outline=bubble_fill)
            
            # Calculate box dimensions for text
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Draw text with chosen color
            self._draw_text_in_box(draw, translated, x_min, y_min, box_width, box_height, text_color=text_color)
            
        return image

    def _draw_text_in_box(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, w: int, h: int, text_color: str = "black"):
        """\n         Fits text inside a box by iteratively reducing font size and wrapping.
        """
        import textwrap
        
        # Skip drawing if text is None or empty/whitespace
        if text is None:
            return
        text = str(text)
        if not text.strip():
            return
        
        # Minimum legible font size
        min_fontsize = 8
        start_fontsize = 18 # Start ambitious
        
        padding = 4
        available_w = max(1, w - 2*padding)
        available_h = max(1, h - 2*padding)
        
        best_font = None
        best_wrapped_text = text
        
        # Iteratively try to fit text
        for fontsize in range(start_fontsize, min_fontsize - 1, -2):
            try:
                # Load font
                font = self._load_font(fontsize)
                
                # Estimate char width (heuristic: usually ~0.6 * fontsize for proportional fonts)
                # A better way is to measure 'x' or 'M'
                bbox = font.getbbox("M")
                char_w = bbox[2] - bbox[0] if bbox else fontsize * 0.6
                
                # Calculate max chars per line
                chars_per_line = max(1, int(available_w / char_w))
                
                # Wrap text
                # break_long_words=False ensures we don't split words like "Unbelievable" into "Unbelievab-le"
                # Instead, if a word is too long, the width check below will fail, and we'll try a smaller font.
                wrapped_text = textwrap.fill(text, width=chars_per_line, break_long_words=False)
                
                # Measure total height
                # getbbox returns (left, top, right, bottom)
                # For multiline, we need to rely on draw.multiline_textbbox if available (Pillow 8.0+)
                if hasattr(draw, 'multiline_textbbox'):
                    text_bbox = draw.multiline_textbbox((0,0), wrapped_text, font=font)
                    text_h = text_bbox[3] - text_bbox[1]
                    text_w = text_bbox[2] - text_bbox[0]
                else:
                    # Fallback for older Pillow
                    text_w, text_h = draw.textsize(wrapped_text, font=font)
                
                # Check if fits vertically and horizontally (roughly)
                if text_h <= available_h and text_w <= available_w * 1.1: # Allow slight overflow width-wise due to wrap inaccuracy
                    best_font = font
                    best_wrapped_text = wrapped_text
                    break # Found a fit!
                    
            except Exception as e:
                print(f"Font fitting error: {e}")
                continue
        
        # If loop finishes without break, we use the smallest font (last one tried)
        if best_font is None:
             best_font = self._load_font(min_fontsize)
             # Re-wrap for min font
             bbox = best_font.getbbox("M")
             char_w = bbox[2] - bbox[0] if bbox else min_fontsize * 0.6
             chars_per_line = max(1, int(available_w / char_w))
             best_wrapped_text = textwrap.fill(text, width=chars_per_line)

        # Center text vertically and horizontally
        if hasattr(draw, 'multiline_textbbox'):
            final_bbox = draw.multiline_textbbox((0,0), best_wrapped_text, font=best_font)
            final_w = final_bbox[2] - final_bbox[0]
            final_h = final_bbox[3] - final_bbox[1]
        else:
            final_w, final_h = draw.textsize(best_wrapped_text, font=best_font)
            
        center_y = y + (h - final_h) // 2
        center_y = max(y, center_y) # Don't go above box

        # Horizontal centering within box, respecting inner padding
        center_x = x + (w - final_w) // 2
        min_x = x + padding
        max_x = x + w - padding - final_w
        center_x = max(min_x, min(center_x, max_x))
        
        # Draw text in chosen color
        draw.multiline_text((center_x, center_y), best_wrapped_text, fill=text_color, font=best_font, align="center")

    def _load_font(self, fontsize: int):
        """Helper to load a font with fallback"""
        # Prioritize bundled font
        bundled_font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
        
        font_names = [
            bundled_font_path,
            "Arial.ttf",  # generic name (Windows/macOS dev)
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS supplemental
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # common Linux path (HF Spaces)
            "DejaVuSans.ttf",  # fallback by font name
        ]
        for name in font_names:
            try:
                return ImageFont.truetype(name, fontsize)
            except:
                continue
        return ImageFont.load_default()

