from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Face Verification API"
    API_V1_STR: str = "/api/v1"
    
    # Model settings
    FACE_MODEL_NAME: str = "ArcFace" # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib
    DISTANCE_METRIC: str = "cosine" # Options: cosine, euclidean, euclidean_l2

    class Config:
        env_file = ".env"

settings = Settings()
