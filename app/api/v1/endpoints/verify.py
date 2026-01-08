from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.verification import FaceVerificationResponse
from app.services.face_matcher import FaceMatcher
from app.core.config import settings
import cv2
import numpy as np

router = APIRouter()

@router.post("/verify", response_model=FaceVerificationResponse)
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Verify if two face images belong to the same person.
    """
    try:
        # Read image files into numpy arrays
        contents1 = await file1.read()
        nparr1 = np.frombuffer(contents1, np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)

        contents2 = await file2.read()
        nparr2 = np.frombuffer(contents2, np.uint8)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
             raise HTTPException(status_code=400, detail="Invalid image data")

        # Call the service passing numpy arrays directly
        result = FaceMatcher.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=settings.FACE_MODEL_NAME,
            metric=settings.DISTANCE_METRIC
        )
        
        print(f"Result: {result}")
        
        return FaceVerificationResponse(
            verified=result.get("verified"),
            distance=result.get("distance"),
            threshold=result.get("threshold"),
            model=result.get("model"),
            similarity_metric=result.get("similarity_metric")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
