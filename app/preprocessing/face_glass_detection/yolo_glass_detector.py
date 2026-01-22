# deprecated. this doesn't work well.

from ultralytics import YOLO
import cv2
import numpy as np
import os

class YOLOGlassDetector:
    def __init__(self, model_name: str = "weights/yolo-glass.pt"):
        # Resolve path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_name)
        
        self.model = YOLO(model_path)

    def detect_glasses(self, image_input):
        results = self.model(image_input, verbose=False, save=False, save_txt=False, save_conf=False)
        
        result = results[0]
        boxes = result.boxes
        
        glass_count = len(boxes)
        bboxes = []
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy().tolist()
            bboxes.append(coords)

        message = ""
        glass_detected = glass_count > 0
        
        if glass_detected:
            message = "Glass detected."
        else:
            message = "No glass detected."

        return {
            "glass_detected": glass_detected,
            "glass_count": glass_count,
            "message": message,
            "bboxes": bboxes
        }

yolo_glass_detector = YOLOGlassDetector()
