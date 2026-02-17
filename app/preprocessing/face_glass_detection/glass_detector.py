import torch
from ultralytics import YOLO
import numpy as np
from app.core.config import settings
from app.core.face_app import FaceAppProvider
from app.core.logger import logger
import os
import cv2

class GlassDetector:
    def __init__(
        self,
        threshold: float = 0.5, 
        input_size: int = 224, 
    ):
        model_path = "./app/preprocessing/face_glass_detection/model/__yolo26x-cls-best.pt"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.input_size = input_size

        self.app = FaceAppProvider.get_app()
        self.model = YOLO(model_path).to(self.device)

    def detect_glass(self, image_numpy: np.ndarray) -> dict:
        """
        Runs inference to specifically detect glasses.
        """
        image_numpy = cv2.copyMakeBorder(image_numpy, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 0, 0])
        face = self.app.get(image_numpy)[0]

        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_numpy.shape[1], x2 + padding)
        y2 = min(image_numpy.shape[0], y2 + padding)
        logger.info("before face cropping")
        face_crop = image_numpy[y1:y2, x1:x2]

        results = self.model.predict(
            source=face_crop,
            conf=0.5
        )
        
        result = results[0]
        top_cls_idx = result.probs.top1
        top_cls_name = result.names[top_cls_idx]
        confidence = result.probs.top1conf.item()

        return {
            "glass_detected": top_cls_name == "glasses",
            "confidence": confidence,
            "message": "glasses detected." if top_cls_name == "glasses" else None
        }

glass_detector = GlassDetector()