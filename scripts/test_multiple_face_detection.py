import cv2
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessing.multiple_face_detection.insightface_detector import insightface_detector

def test_multiple_face_detection():
    print("Testing InsightFace Multiple Face Detection...")
    
    # 1. Test with a dummy black image (should detect 0 faces)
    img_black = np.zeros((640, 640, 3), dtype=np.uint8)
    
    try:
        result = insightface_detector.detect_faces(img_black)
        print("Result (Black Image):", result)
        
        if result['face_count'] == 0:
            print("PASS: Correctly detected 0 faces on black image.")
        else:
            print("FAIL: Detected faces on black image.")
            
        if "bboxes" in result and "message" in result:
             print("PASS: Output structure is correct.")
        else:
             print("FAIL: Output structure incorrect.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multiple_face_detection()
