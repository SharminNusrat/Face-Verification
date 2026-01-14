import os

class Settings:
    PROJECT_NAME: str = "InsightFace Verification"
    VERSION: str = "1.0.0"
    # InsightFace model settings
    DET_SIZE: tuple = (640, 640)
    MODEL_NAME: str = "buffalo_l"
    
settings = Settings()
