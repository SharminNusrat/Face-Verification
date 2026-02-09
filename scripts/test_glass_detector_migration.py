import cv2
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessing.face_glass_detection.glass_detector import glass_detector

def test_glass_detector():
    # Create a dummy image (black)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    print("Running GlassDetector on dummy image...")
    try:
        result = glass_detector.detect_glass(img)
        print("Result:", result)
        
        if "glass_detected" in result and "score_no_glasses" in result:
            print("Test PASSED: Output structure is correct.")
        else:
            print("Test FAILED: Output structure is incorrect.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_glass_detector()
