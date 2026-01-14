import sys
import cv2
import os
# Add project root to path so we can import app
sys.path.append(os.getcwd())

from app.preprocessing import preprocess_manager

def test_preprocessing(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not decode image {image_path}")
        return

    print(f"Processing image: {image_path}")
    
    # Run preprocessing
    try:
        results = preprocess_manager.preprocess(img)
        
        # Display results
        face_data = results.get("checks", {}).get("face_detection", {})
        print("\n--- Detection Results ---")
        print(f"Status: {results.get('status')}")
        print(f"Message: {results.get('message')}")
        print(f"Face Count: {face_data.get('face_count')}")
        print(f"BBoxes: {face_data.get('bboxes')}")
        print("-------------------------")
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_preprocessing.py <path_to_image>")
    else:
        test_preprocessing(sys.argv[1])
