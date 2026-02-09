# currently in use.
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class GlassDetector:
    def __init__(
        self,
        model_name: str = "glasses_detection.pth",
        threshold: float = 0.5, # Adjusted for sigmoid output
        input_size: tuple[int, int] = (160, 160),
    ):
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "model", model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.input_size = input_size

        # Initialize EfficientNet-B0
        self.model = models.efficientnet_b0(weights=None) 
        # Modify classifier for binary classification (matches training script)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Glass detection model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load glass detection model: {e}")
        else:
             print(f"Warning: Glass detection model not found at {model_path}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect_glass(self, image_numpy: np.ndarray) -> dict:
        try:
            # Convert BGR (OpenCV) to RGB
            img_rgb = image_numpy[..., ::-1].copy() 
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(img_tensor)
                score = torch.sigmoid(logits).item()

            # Logic: If score > threshold (e.g. 0.5), it is class 1 (No Glasses?? OR Glasses??)
            # Need to match training labeling. 
            # Usually strict subsets are: Class 0 vs Class 1. 
            # In your old script: "glass_detected = score_no_glasses < threshold". 
            # This implies Class 1 was "No Glasses" and Class 0 was "Glasses".
            # Let's assume standard ImageFolder alphabetical order: 
            # If folders are "0_glasses" and "1_no_glasses" -> then 1=no_glasses.
            # If folders are "glasses" and "no_glasses" -> then 0=glasses, 1=no_glasses.
            
            # To match previous behavior: "score_no_glasses"
            score_no_glasses = score 
            glass_detected = score_no_glasses < self.threshold

            message = "Glass detected." if glass_detected else "No glass detected."
            
            return {
                "glass_detected": bool(glass_detected),
                "score_no_glasses": float(score_no_glasses),
                "message": message,
            }

        except Exception as e:
            print(f"Error during glass detection: {e}")
            return {
                "glass_detected": False,
                "score_no_glasses": 0.0,
                "message": f"Error: {str(e)}",
            }

glass_detector = GlassDetector()