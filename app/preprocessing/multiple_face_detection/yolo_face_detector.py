from ultralytics import YOLO
import cv2
import numpy as np

import os

class YOLOFaceDetector:
    def __init__(self, model_name: str = "weights/yolov12n-face.pt"):
        # Resolve path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_name)
        
        self.model = YOLO(model_path)

    def detect_faces(self, image_input):
        """
        Detects faces in the image.
        Args:
            image_input: path to image or numpy array (cv2 image)
        Returns:
            dict: {
                "face_count": int,
                "message": str,
                "bboxes": list of [x1, y1, x2, y2]
            }
        """
        # YOLO can accept file path, PIL, cv2, etc.
        results = self.model(image_input, verbose=False, save=False, save_txt=False, save_conf=False)
        
        result = results[0]
        boxes = result.boxes
        
        face_count = len(boxes)
        bboxes = []
        for box in boxes:
            # box.xyxy provides [x1, y1, x2, y2]
            coords = box.xyxy[0].cpu().numpy().tolist()
            bboxes.append(coords)

        message = ""
        if face_count > 1:
            message = "Multiple faces detected."
        elif face_count == 1:
            message = "Single face detected."
        else:
            message = "No face detected."

        return {
            "face_count": face_count,
            "message": message,
            "bboxes": bboxes
        }

# Singleton instance can be created here or in manager
yolo_face_detector = YOLOFaceDetector()
