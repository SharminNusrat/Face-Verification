from pydantic import BaseModel

class VerifyResponse(BaseModel):
    match: bool
    score: float
    error: str | None = None
