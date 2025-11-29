from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from typing import List, Tuple, Optional
import os

class ImageProcessor:
    def __init__(self):
        pass

    def _find_bubble_contour(self, img_array: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int, margin: int = 80) -> Optional[np.ndarray]:
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
        
        # Erweitere den Suchbereich um die Textbox (größerer Bereich für bessere Erkennung)
        text_w = x_max - x_min
        text_h = y_max - y_min
        # Dynamischer Margin basierend auf Textgröße, mindestens 80px
        dynamic_margin = max(margin, max(text_w, text_h))
        
        search_x1 = max(0, x_min - dynamic_margin)
        search_y1 = max(0, y_min - dynamic_margin)
        search_x2 = min(w, x_max + dynamic_margin)
        search_y2 = min(h, y_max + dynamic_margin)
        
        # Extrahiere den Suchbereich
        region = img_array[search_y1:search_y2, search_x1:search_x2]
        
        if region.size == 0:
            return None
        
        # Konvertiere zu Graustufen
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region.copy()
        
        # Verwende Otsu's Binarisierung für adaptive Schwellwertfindung
        # Das funktioniert besser bei verschiedenen Manga-Stilen
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Falls Otsu nicht gut funktioniert (zu wenig Kontrast), versuche festen Schwellwert
        # Prüfe ob das Ergebnis sinnvoll ist (nicht alles weiß oder schwarz)
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio < 0.1 or white_ratio > 0.9:
            # Otsu hat nicht gut funktioniert, verwende festen Schwellwert
            # Versuche verschiedene Schwellwerte
            for threshold in [180, 160, 200, 220]:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
                white_ratio = np.sum(binary == 255) / binary.size
                if 0.1 <= white_ratio <= 0.9:
                    break
        
        # Morphologische Operationen um kleine Lücken zu schließen und Rauschen zu entfernen
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Finde Konturen
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Berechne den Mittelpunkt der Textbox relativ zum Suchbereich
        text_center_x = (x_min + x_max) // 2 - search_x1
        text_center_y = (y_min + y_max) // 2 - search_y1
        text_area = text_w * text_h
        
        # Finde die Kontur, die den Textmittelpunkt enthält
        best_contour = None
        best_score = float('inf')
        
        for contour in contours:
            # Prüfe ob der Textmittelpunkt in der Kontur liegt
            if cv2.pointPolygonTest(contour, (text_center_x, text_center_y), False) >= 0:
                # Berechne die Fläche der Kontur
                area = cv2.contourArea(contour)
                
                # Die Kontur sollte mindestens so groß wie der Text sein,
                # aber nicht zu groß (max 25x Textfläche für größere Sprechblasen)
                if area >= text_area * 0.3 and area <= text_area * 25:
                    # Bevorzuge Konturen die näher an der Textgröße sind
                    # Score = Verhältnis von Konturfläche zu Textfläche
                    score = abs(area / text_area - 1.5)  # Ideale Blase ist ~1.5x Textfläche
                    
                    if score < best_score:
                        best_score = score
                        best_contour = contour
        
        if best_contour is not None:
            # Vereinfache die Kontur für glattere Kanten (aber nicht zu stark)
            epsilon = 0.005 * cv2.arcLength(best_contour, True)
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

    def overlay_text(self, image: Image.Image, text_regions: List[Tuple[List[List[int]], str, str]], use_ellipse: bool = True, ellipse_padding_x: int = 0, ellipse_padding_y: int = 0, debug: bool = False) -> Image.Image:
        """
        Overlays translated text onto the image.
        
        Füllt nur den Bereich den der übersetzte Text benötigt (plus Padding),
        nicht die gesamte Sprechblase.
        
        Args:
            image: The original PIL Image.
            text_regions: List of tuples (bbox, original_text, translated_text).
                         bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
            use_ellipse: If True, draw elliptical background, otherwise rectangle.
            ellipse_padding_x: Horizontal padding around the text.
            ellipse_padding_y: Vertical padding around the text.
            debug: If True, print debug info.
        
        Returns:
            Processed PIL Image.
        """
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size
        img_array = np.array(image)
        
        for bbox, original, translated in text_regions:
            # Calculate bounding rectangle from OCR (= Textbereich)
            pts = np.array(bbox)
            ocr_x_min = int(np.min(pts[:, 0]))
            ocr_y_min = int(np.min(pts[:, 1]))
            ocr_x_max = int(np.max(pts[:, 0]))
            ocr_y_max = int(np.max(pts[:, 1]))
            
            ocr_width = ocr_x_max - ocr_x_min
            ocr_height = ocr_y_max - ocr_y_min
            
            # Finde die Sprechblasen-Grenzen (größer als nur OCR-Bereich)
            bubble_x_min, bubble_y_min, bubble_x_max, bubble_y_max = self._get_bubble_bounds(
                img_array, ocr_x_min, ocr_y_min, ocr_x_max, ocr_y_max
            )
            bubble_width = bubble_x_max - bubble_x_min
            bubble_height = bubble_y_max - bubble_y_min
            
            if debug:
                print(f"OCR-Box: {ocr_width}x{ocr_height}, Bubble: {bubble_width}x{bubble_height}")

            # Sample background brightness to decide text color
            bubble_fill = "white"
            text_color = "black"
            try:
                region = img_array[ocr_y_min:ocr_y_max, ocr_x_min:ocr_x_max]
                if region.size > 0:
                    if len(region.shape) >= 3 and region.shape[-1] >= 3:
                        r = region[..., 0].astype(np.float32)
                        g = region[..., 1].astype(np.float32)
                        b = region[..., 2].astype(np.float32)
                        luma = 0.299 * r + 0.587 * g + 0.114 * b
                    else:
                        luma = region.astype(np.float32)
                    mean_luma = float(luma.mean()) / 255.0
                    if mean_luma < 0.5:
                        bubble_fill = "#111111"
                        text_color = "white"
            except Exception:
                pass
            
            # Nutze die Bubble-Größe für die Text-Berechnung (größerer Text!)
            # Nutze fast den gesamten Bubble-Bereich
            available_width = int(bubble_width * 0.92)
            available_height = int(bubble_height * 0.92)
            
            # Berechne die Text-Größe basierend auf verfügbarem Bubble-Platz
            text_info = self._calculate_text_size(draw, translated, available_width, available_height)
            
            if text_info is None:
                continue
                
            font, wrapped_text, text_w, text_h = text_info
            
            # Zentriere den Text in der BUBBLE (nicht nur OCR-Box) für maximale Größe
            bubble_center_x = bubble_x_min + bubble_width // 2
            bubble_center_y = bubble_y_min + bubble_height // 2
            center_x = bubble_center_x - text_w // 2
            center_y = bubble_center_y - text_h // 2
            
            # Der Füllbereich muss mindestens den OCR-Bereich (Original-Text) abdecken!
            # Plus etwas Padding für den neuen Text
            fill_x_min = min(ocr_x_min, center_x) - ellipse_padding_x
            fill_y_min = min(ocr_y_min, center_y) - ellipse_padding_y
            fill_x_max = max(ocr_x_max, center_x + text_w) + ellipse_padding_x
            fill_y_max = max(ocr_y_max, center_y + text_h) + ellipse_padding_y
            
            if use_ellipse:
                # Ensure ellipse covers the corners of the original OCR box
                # An ellipse inscribed in a box only touches the centers of the sides.
                # To cover the corners of a box of size WxH, the ellipse needs to be larger.
                # Using a factor of 1.25 for width (user feedback: 1.5 was too wide)
                # and 1.5 for height to ensure corner coverage.
                
                ocr_cx = (ocr_x_min + ocr_x_max) / 2
                ocr_cy = (ocr_y_min + ocr_y_max) / 2
                
                min_ellipse_w = ocr_width * 1.25
                min_ellipse_h = ocr_height * 1.5
                
                min_fill_x_min = ocr_cx - min_ellipse_w / 2
                min_fill_y_min = ocr_cy - min_ellipse_h / 2
                min_fill_x_max = ocr_cx + min_ellipse_w / 2
                min_fill_y_max = ocr_cy + min_ellipse_h / 2
                
                # Merge with existing fill bounds
                fill_x_min = min(fill_x_min, min_fill_x_min)
                fill_y_min = min(fill_y_min, min_fill_y_min)
                fill_x_max = max(fill_x_max, min_fill_x_max)
                fill_y_max = max(fill_y_max, min_fill_y_max)
            
            # Clamp to image bounds
            fill_x_min = max(0, fill_x_min)
            fill_y_min = max(0, fill_y_min)
            fill_x_max = min(img_w, fill_x_max)
            fill_y_max = min(img_h, fill_y_max)
            
            # Zeichne den Hintergrund (nur so groß wie der Text + Padding)
            if use_ellipse:
                draw.ellipse([fill_x_min, fill_y_min, fill_x_max, fill_y_max], 
                           fill=bubble_fill, outline=bubble_fill)
            else:
                draw.rectangle([fill_x_min, fill_y_min, fill_x_max, fill_y_max], 
                             fill=bubble_fill, outline=bubble_fill)
            
            # Zeichne den Text
            draw.multiline_text((center_x, center_y), wrapped_text, 
                              fill=text_color, font=font, align="center")
            
            if debug:
                print(f"  Text: '{translated[:30]}...' -> Fill: {fill_x_max-fill_x_min}x{fill_y_max-fill_y_min}px")
            
        return image

    def _smart_wrap_text(self, text: str, chars_per_line: int, font, draw) -> str:
        """
        Intelligenter Textumbruch der Wörter möglichst nicht trennt.
        Wenn ein Wort zu lang ist, wird es mit Bindestrich getrennt.
        
        Args:
            text: Der zu umbrechen Text
            chars_per_line: Maximale Zeichen pro Zeile (geschätzt)
            font: Die verwendete Schrift
            draw: ImageDraw Objekt für Textmessung
            
        Returns:
            Umgebrochener Text mit Zeilenumbrüchen
        """
        if not text or chars_per_line <= 0:
            return text
            
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            
            # Prüfe ob das Wort in die aktuelle Zeile passt
            space_needed = 1 if current_line else 0
            
            if current_length + space_needed + word_len <= chars_per_line:
                # Wort passt in die Zeile
                current_line.append(word)
                current_length += space_needed + word_len
            elif word_len <= chars_per_line:
                # Wort passt nicht, aber ist kurz genug für eine eigene Zeile
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_len
            else:
                # Wort ist zu lang - muss getrennt werden
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []
                    current_length = 0
                
                # Trenne das lange Wort mit Bindestrichen
                remaining = word
                while len(remaining) > chars_per_line:
                    # Trenne so dass mindestens 2 Zeichen übrig bleiben
                    split_at = max(2, chars_per_line - 1)  # -1 für den Bindestrich
                    lines.append(remaining[:split_at] + "-")
                    remaining = remaining[split_at:]
                
                if remaining:
                    current_line = [remaining]
                    current_length = len(remaining)
        
        # Letzte Zeile hinzufügen
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)

    def _calculate_text_size(self, draw: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int) -> Optional[Tuple]:
        """
        Berechnet die optimale Schriftgröße und den umgebrochenen Text.
        Maximiert die Schriftgröße so dass der Text in den verfügbaren Bereich passt.
        
        Returns:
            Tuple (font, wrapped_text, text_width, text_height) oder None
        """
        if text is None or not str(text).strip():
            return None
            
        text = str(text)
        
        min_fontsize = 10
        # Starte mit einer GROSSEN Schrift - so groß wie die Box es erlaubt
        # Für kurze Texte kann die Schrift sehr groß sein
        word_count = len(text.split())
        if word_count <= 3:
            # Kurzer Text: versuche sehr große Schrift
            start_fontsize = min(120, max(40, int(max_h * 0.8)))
        elif word_count <= 8:
            # Mittlerer Text
            start_fontsize = min(100, max(30, int(max_h * 0.6)))
        else:
            # Langer Text
            start_fontsize = min(80, max(24, int(max_h * 0.5)))
        
        padding = 4
        available_w = max(1, max_w - 2*padding)
        available_h = max(1, max_h - 2*padding)
        
        best_result = None
        
        for fontsize in range(start_fontsize, min_fontsize - 1, -1):
            try:
                font = self._load_font(fontsize)
                bbox = font.getbbox("M")
                char_w = bbox[2] - bbox[0] if bbox else fontsize * 0.6
                chars_per_line = max(1, int(available_w / char_w))
                
                # Intelligenter Textumbruch ohne Wörter mittendrin zu trennen
                wrapped_text = self._smart_wrap_text(text, chars_per_line, font, draw)
                
                if hasattr(draw, 'multiline_textbbox'):
                    text_bbox = draw.multiline_textbbox((0,0), wrapped_text, font=font)
                    text_h = text_bbox[3] - text_bbox[1]
                    text_w = text_bbox[2] - text_bbox[0]
                else:
                    text_w, text_h = draw.textsize(wrapped_text, font=font)
                
                # Prüfe ob es passt
                if text_h <= available_h and text_w <= available_w:
                    # Gefunden! Nimm die größte passende Schrift
                    return (font, wrapped_text, text_w, text_h)
                    
            except Exception:
                continue
        
        # Fallback: kleinste Schrift
        font = self._load_font(min_fontsize)
        bbox = font.getbbox("M")
        char_w = bbox[2] - bbox[0] if bbox else min_fontsize * 0.6
        chars_per_line = max(1, int(available_w / char_w))
        wrapped_text = self._smart_wrap_text(text, chars_per_line, font, draw)
        
        if hasattr(draw, 'multiline_textbbox'):
            text_bbox = draw.multiline_textbbox((0,0), wrapped_text, font=font)
            text_h = text_bbox[3] - text_bbox[1]
            text_w = text_bbox[2] - text_bbox[0]
        else:
            text_w, text_h = draw.textsize(wrapped_text, font=font)
            
        return (font, wrapped_text, text_w, text_h)

    def _draw_text_in_box(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, w: int, h: int, text_color: str = "black"):
        """\n         Fits text inside a box by iteratively reducing font size and wrapping.
        """
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
                
                # Intelligenter Textumbruch ohne Wörter mittendrin zu trennen
                wrapped_text = self._smart_wrap_text(text, chars_per_line, font, draw)
                
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
             best_wrapped_text = self._smart_wrap_text(text, chars_per_line, best_font, draw)

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

