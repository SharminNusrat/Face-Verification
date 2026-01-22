
from ultralytics import YOLO, settings
import numpy as np
import os
import shutil
import tempfile

# Clean up
if os.path.exists("runs"):
    shutil.rmtree("runs")

# Try to update settings
print("Current runs_dir:", settings['runs_dir'])
# We want to use a temp dir
temp_runs = os.path.join(tempfile.gettempdir(), "yolo_test_runs")
settings.update({'runs_dir': temp_runs})
print("Updated runs_dir:", settings['runs_dir'])

# Run inference with save=False (user's suggestion) BUT with updated global settings
img = np.zeros((640, 640, 3), dtype=np.uint8)
try:
    # Use a dummy model or existing one
    # weights/yolov12n-face.pt might not exist, checking...
    weights_path = "app/preprocessing/multiple_face_detection/weights/yolov12n-face.pt"
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
    else:
        model = YOLO("yolov8n.pt") 
        
    print("Running inference...")
    model(img, verbose=False, save=False)

    if os.path.exists("runs"):
        print("FAIL: 'runs' folder still created in local dir!")
    elif os.path.exists(temp_runs):
        print("SUCCESS: 'runs' folder created in temp dir!")
    else:
        print("SUCCESS: No folder created (unexpected but good)!")

except Exception as e:
    print(f"Error: {e}")
