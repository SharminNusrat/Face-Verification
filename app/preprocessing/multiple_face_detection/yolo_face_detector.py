from ultralytics import YOLO
import cv2
import numpy as np

import os

class YOLOFaceDetector:
    def __init__(self, model_name: str = "weights/yolov12n-face.pt"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_name)
        
        self.model = YOLO(model_path)

    def detect_faces(self, image_input):
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

yolo_face_detector = YOLOFaceDetector()
