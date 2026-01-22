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
        """
        Detects glasses in the image.
        Args:
            image_input: path to image or numpy array (cv2 image)
        Returns:
            dict: {
                "glass_detected": bool,
                "glass_count": int,
                "message": str,
                "bboxes": list of [x1, y1, x2, y2]
            }
        """
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
