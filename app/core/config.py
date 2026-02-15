import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "InsightFace Verification"
    VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env") 
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DEVICE: str = "cpu"
    ENABLE_MULTIPLE_FACE_DETECTION: bool = True
    ENABLE_FACE_GLASS_DETECTION: bool = True
    ENABLE_HEAD_POSE_DETECTION: bool = True
    # InsightFace model settings
    MODEL_NAME: str = "buffalo_l"
    GLASS_DETECTION_YOLO_MODEL: str = 'yolov8s' # yolov8n, yolov26n
    DET_SIZE: tuple = (640, 640)
        
settings = Settings()
