import torch
from torchvision import models
import torch.nn as nn
import os

def create_dummy_model():
    model = models.efficientnet_b0()
    # Modify classifier to match our training script (Binary classification)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    
    save_path = "app/preprocessing/face_glass_detection/model/glasses_detection.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Dummy model saved to {save_path}")

if __name__ == "__main__":
    create_dummy_model()
