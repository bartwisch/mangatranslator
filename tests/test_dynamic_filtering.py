import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ocr_handler import OCRHandler

def test_dynamic_filtering():
    print("Testing dynamic filtering logic...")
    
    # Create handler
    handler = OCRHandler(ocr_engine='magi')
    
    # Mock raw results (bbox, text, confidence)
    raw_results = [
        ([[0, 0], [10, 0], [10, 10], [0, 10]], "LowConf", 0.2),
        ([[20, 20], [40, 20], [40, 40], [20, 40]], "HighConf", 0.9)
    ]
    
    img_shape = (100, 100)
    
    # Test 1: High Confidence Threshold
    filtered = handler.filter_results(raw_results, confidence_threshold=0.5, box_padding=0, image_shape=img_shape)
    assert len(filtered) == 1
    assert filtered[0][1] == "HighConf"
    print("✓ Confidence filtering works")
    
    # Test 2: Padding
    # Expand by 5px
    filtered = handler.filter_results(raw_results, confidence_threshold=0.0, box_padding=5, image_shape=img_shape)
    assert len(filtered) == 2
    
    # Check first box (LowConf)
    # Original: 0,0 to 10,10
    # Padded: 0-5 -> 0 (clamped), 0-5 -> 0, 10+5 -> 15, 10+5 -> 15
    bbox = filtered[0][0]
    # bbox format is list of points
    x_min = bbox[0][0]
    y_min = bbox[0][1]
    x_max = bbox[2][0]
    y_max = bbox[2][1]
    
    assert x_min == 0
    assert y_min == 0
    assert x_max == 15
    assert y_max == 15
    print("✓ Padding works")

if __name__ == "__main__":
    test_dynamic_filtering()
