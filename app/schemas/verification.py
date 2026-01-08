from pydantic import BaseModel, Field

class FaceVerificationRequest(BaseModel):
    # We will accept images as UploadFile, so this schema might just be for metadata if needed
    # Or if we accept base64 strings. For now, let's keep it empty or defined for responses.
    pass

class FaceVerificationResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    similarity_metric: str
