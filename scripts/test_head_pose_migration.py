import cv2
import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow importing app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessing.head_pose_detection.HeadPoseDetector import head_pose_detector

def test_head_pose():
    # Create a dummy image (black)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Draw a simple face-like structure (circle) to see if it even runs without crashing
    # Note: InsightFace needs a real face to return a result, so this dummy image will likely result in "No Face".
    # But the goal here is to check for import errors and basic execution.
    cv2.circle(img, (320, 320), 100, (255, 255, 255), -1) 
    
    print("Running HeadPoseDetector on dummy image...")
    try:
        result = head_pose_detector.get_direction(img)
        print("Result:", result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_head_pose()
