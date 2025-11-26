from PIL import Image, ImageDraw, ImageFont
import math

def create_logo():
    # Create a 512x512 transparent image
    size = 512
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Colors
    bubble_color = (255, 255, 255) # White
    outline_color = (30, 30, 30)   # Dark Grey
    text_color = (236, 72, 153)    # Streamlit Pink/Red

    # Draw Speech Bubble (Rounded Rectangle)
    padding = 40
    rect = [padding, padding, size-padding, size-120]
    
    # Main bubble body
    draw.rounded_rectangle(rect, radius=60, fill=bubble_color, outline=outline_color, width=15)
    
    # Tail of the bubble
    tail = [(120, size-125), (80, size-40), (200, size-125)]
    draw.polygon(tail, fill=bubble_color, outline=outline_color)
    
    # Fix the outline intersection by redrawing the inner triangle
    inner_tail = [(130, size-130), (85, size-50), (190, size-130)]
    draw.polygon(inner_tail, fill=bubble_color)

    # Draw Text "漢 -> DE" (Kanji for 'China/Text' to DE) or just "Manga\nTL"
    # Let's use "文" (Text/Literature) -> DE
    text = "文\n↓\nDE"
    
    try:
        # Try to find a font
        font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc" # Mac specific
        font = ImageFont.truetype(font_path, 160)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 140)
            text = "Manga\nTL"
        except:
            font = ImageFont.load_default()
            text = "Manga\nTL"

    # Draw text centered
    w, h = size, size-120
    draw.multiline_text((w/2, h/2), text, fill=text_color, font=font, anchor="mm", align="center", spacing=10)

    img.save("logo.png")
    print("Logo generated: logo.png")

if __name__ == "__main__":
    create_logo()
