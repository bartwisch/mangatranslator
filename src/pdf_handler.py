import fitz  # PyMuPDF
from PIL import Image
import io
import os
from typing import List, Union

class PDFHandler:
    def __init__(self):
        pass

    def extract_images_from_pdf(self, pdf_path: str, zoom: int = 2, pages: List[int] = None) -> List[Image.Image]:
        """
        Converts each page of the PDF into a PIL Image.
        
        Args:
            pdf_path: Path to the source PDF file.
            zoom: Zoom factor for higher resolution (default 2 for better OCR).
            pages: Optional list of 0-indexed page numbers to extract. If None, extracts all.
            
        Returns:
            List of PIL Image objects.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        images = []

        # Matrix for zooming (higher resolution for better OCR)
        mat = fitz.Matrix(zoom, zoom)

        # Determine which pages to process
        if pages is None:
            page_indices = range(len(doc))
        else:
            # Filter out invalid page numbers
            page_indices = [p for p in pages if 0 <= p < len(doc)]

        for page_num in page_indices:
            page = doc.load_page(page_num)

            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        doc.close()
        return images

    def save_images_as_pdf(self, images: List[Image.Image], output_path: str):
        """
        Saves a list of PIL Images as a single PDF file.
        
        Args:
            images: List of PIL Image objects.
            output_path: Path where the new PDF should be saved.
        """
        if not images:
            print("No images to save.")
            return

        # Convert PIL images to RGB if necessary and save
        pdf_images = []
        for img in images:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            pdf_images.append(img)

        if pdf_images:
            pdf_images[0].save(
                output_path, 
                save_all=True, 
                append_images=pdf_images[1:], 
                resolution=100.0, 
                quality=95, 
                optimize=True
            )
            print(f"PDF saved successfully at {output_path}")
