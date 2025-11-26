from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple

class ImageProcessor:
    def __init__(self):
        pass

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

    def overlay_text(self, image: Image.Image, text_regions: List[Tuple[List[List[int]], str, str]]) -> Image.Image:
        """
        Overlays translated text onto the image.
        
        Args:
            image: The original PIL Image.
            text_regions: List of tuples (bbox, original_text, translated_text).
                         bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
        
        Returns:
            Processed PIL Image.
        """
        draw = ImageDraw.Draw(image)
        
        for bbox, original, translated in text_regions:
            # Calculate bounding rectangle
            pts = np.array(bbox)
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            
            # Draw white rectangle (inpainting)
            draw.rectangle([x_min, y_min, x_max, y_max], fill="white", outline="white")
            
            # Calculate box dimensions
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Draw text
            self._draw_text_in_box(draw, translated, x_min, y_min, box_width, box_height)
            
        return image

    def _draw_text_in_box(self, draw: ImageDraw.ImageDraw, text: str, x: int, y: int, w: int, h: int):
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

        # Center text vertically
        if hasattr(draw, 'multiline_textbbox'):
            final_bbox = draw.multiline_textbbox((0,0), best_wrapped_text, font=best_font)
            final_h = final_bbox[3] - final_bbox[1]
        else:
            _, final_h = draw.textsize(best_wrapped_text, font=best_font)
            
        center_y = y + (h - final_h) // 2
        center_y = max(y, center_y) # Don't go above box
        
        # Draw text (black)
        draw.multiline_text((x + padding, center_y), best_wrapped_text, fill="black", font=best_font, align="center")

    def _load_font(self, fontsize: int):
        """Helper to load a font with fallback"""
        font_names = ["Arial.ttf", "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]
        for name in font_names:
            try:
                return ImageFont.truetype(name, fontsize)
            except:
                continue
        return ImageFont.load_default()

