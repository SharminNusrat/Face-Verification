from pydantic import BaseModel, Field

class FaceVerificationRequest(BaseModel):
    # We will accept images as UploadFile, so this schema might just be for metadata if needed
    # Or if we accept base64 strings. For now, let's keep it empty or defined for responses.
    pass

class FaceVerificationResponse(BaseModel):
    verified: bool
    distance: float | None = None
    threshold: float | None = None
    model: str | None = None
    similarity_metric: str | None = None
