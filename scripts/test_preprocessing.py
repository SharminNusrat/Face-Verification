import sys
import cv2
import os
# Add project root to path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

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
        glass_data = results.get("checks", {}).get("glass_detection", {})
        
        print("\n--- Detection Results ---")
        print(f"Status: {results.get('status')}")
        print(f"Message: {results.get('message')}")
        
        if face_data:
            print(f"Face Count: {face_data.get('face_count')}")
            
        if glass_data:
            print(f"Glass Detected: {glass_data.get('glass_detected')}")

        pose_data = results.get("checks", {}).get("head_pose", {})
        if pose_data:
             print(f"Head Direction: {pose_data.get('direction')}")
             angles = pose_data.get('angles')
             if angles:
                 print(f"Angles (Pitch, Yaw, Roll): ({angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f})")
            
        print("-------------------------")
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    # Hardcoded path to image in data/input
    # Since we are in scripts/ directory, we go up one level then into data/input
    image_name = "rakin_portrait.jpg"
    image_path = os.path.join(project_root, "data", "input", image_name)
    test_preprocessing(image_path)
