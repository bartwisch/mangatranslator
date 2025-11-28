import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ocr_handler import OCRHandler

def test_padding_and_confidence():
    print("Testing padding and confidence parameters...")
    
    # Initialize handler (mock engine or use one that doesn't require heavy load if possible, but we use magi by default)
    # We'll use 'magi' but mock the internal detection to avoid loading the model if possible, 
    # or just rely on the fact that we can call detect_text with new params.
    # Since we can't easily mock without a library, we'll just check if the method signature accepts the args
    # and if the logic inside (which we can't fully run without model) seems reachable.
    
    # Actually, let's just instantiate and call it with a dummy image. 
    # If it fails due to model loading, that's fine, we just want to ensure arguments are passed.
    # But model loading takes time.
    
    # Let's verify the method signature by inspection or just trust the code we wrote.
    # Better: Create a dummy OCRHandler that mocks _detect_with_magi
    
    class MockOCRHandler(OCRHandler):
        def __init__(self):
            self.ocr_engine = 'magi'
            
        def preprocess_image(self, image, mode='gentle'):
            return image
            
        def _detect_with_magi(self, image, scale_factor):
            # Return dummy results: bbox, text
            # bbox: [x1, y1, x2, y2]
            return [
                ([[10, 10], [50, 10], [50, 50], [10, 50]], "LowConf", 0.1),
                ([[60, 60], [100, 60], [100, 100], [60, 100]], "HighConf", 0.9)
            ]

    handler = MockOCRHandler()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Test Confidence Filtering
    results = handler.detect_text(img, confidence_threshold=0.5)
    print(f"Results with conf=0.5: {len(results)}")
    assert len(results) == 1
    assert results[0][1] == "HighConf"
    
    # Test Padding
    # We need to check if padding logic is applied in detect_text.
    # Wait, my MockOCRHandler overrides _detect_with_magi which returns the raw results.
    # The filtering/padding logic is in detect_text AFTER calling _detect_with_*.
    # So I should call the REAL detect_text, but I need to mock the _detect_with_magi call inside it.
    
    # Monkey patch
    # original_detect = OCRHandler._detect_with_magi
    # OCRHandler._detect_with_magi = lambda self, img, scale: [
    #     ([[10, 10], [50, 10], [50, 50], [10, 50]], "Text", 0.9)
    # ]
    
    # Let's just use the Mock approach but properly override the internal method only
    class TestHandler(OCRHandler):
        def __init__(self):
            self.ocr_engine = 'magi'
            
        def preprocess_image(self, image, mode='gentle'):
            return image
            
        def _detect_with_magi(self, image, scale_factor):
             return [([[10, 10], [50, 10], [50, 50], [10, 50]], "Text", 0.9)]

    test_handler = TestHandler()
    
    # Test Padding = 10
    # Original box: 10,10 to 50,50. Center ~30,30.
    # Padding logic: x_min - 10, x_max + 10...
    # New box should be 0,0 to 60,60
    
    results_padded = test_handler.detect_text(img, box_padding=10)
    bbox = results_padded[0][0]
    print(f"Padded bbox: {bbox}")
    
    # Check coordinates
    # bbox is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    x_min = bbox[0][0]
    y_min = bbox[0][1]
    x_max = bbox[2][0]
    y_max = bbox[2][1]
    
    assert x_min == 0
    assert y_min == 0
    assert x_max == 60
    assert y_max == 60
    
    print("✓ Padding and Confidence tests passed")

if __name__ == "__main__":
    test_padding_and_confidence()
