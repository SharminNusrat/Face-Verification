import torch
from ultralytics import YOLO
import numpy as np
from app.core.config import settings
import os

class GlassDetector:
    def __init__(
        self,
        threshold: float = 0.5, 
        input_size: int = 160, 
    ):
        model_name = settings.GLASS_DETECTION_YOLO_MODEL
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Ensure the model path points to your weights folder
        model_path = os.path.join(current_dir, "model", f"{model_name}.pt")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.input_size = input_size
        
        # Load the model once during initialization
        self.model = YOLO(model_path).to(self.device)

    def detect_glass(self, image_numpy: np.ndarray) -> dict:
        """
        Runs inference to specifically detect glasses.
        """
        # YOLO handles the scaling and normalization internally via predict
        results = self.model.predict(
            source=image_numpy,
            imgsz=self.input_size,
            conf=self.threshold,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        detections = []

        for box in result.boxes:
            # Since nc=1, class_id will always be 0 for 'glasses'
            detections.append({
                "box": box.xyxy[0].cpu().numpy().tolist(), # [xmin, ymin, xmax, ymax]
                "confidence": round(float(box.conf[0]), 4),
                "label": "glasses"
            })

        return {
            "glass_detected": len(detections) > 0,
            "confidence": detections[0]['confidence'],
            "message": "glasses detected."
        }

glass_detector = GlassDetector()