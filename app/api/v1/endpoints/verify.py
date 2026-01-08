from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.verification import FaceVerificationResponse
from app.services.face_matcher import FaceMatcher
from app.core.config import settings
import shutil
import os
import tempfile

router = APIRouter()

@router.post("/verify", response_model=FaceVerificationResponse)
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Verify if two face images belong to the same person.
    """
    # Create temp files
    # Use .jpg as default suffix if filename is missing extension or handle generically
    suffix1 = os.path.splitext(file1.filename)[1] if file1.filename else ".jpg"
    suffix2 = os.path.splitext(file2.filename)[1] if file2.filename else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix1) as tmp1:
        shutil.copyfileobj(file1.file, tmp1)
        tmp1_path = tmp1.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix2) as tmp2:
        shutil.copyfileobj(file2.file, tmp2)
        tmp2_path = tmp2.name

    try:
        # Call the service
        result = FaceMatcher.verify(
            img1_path=tmp1_path,
            img2_path=tmp2_path,
            model_name=settings.FACE_MODEL_NAME,
            metric=settings.DISTANCE_METRIC
        )
        
        return FaceVerificationResponse(
            verified=result.get("verified"),
            distance=result.get("distance"),
            threshold=result.get("threshold"),
            model=result.get("model"),
            similarity_metric=result.get("similarity_metric")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp files
        if os.path.exists(tmp1_path):
            try:
                os.remove(tmp1_path)
            except:
                pass
        if os.path.exists(tmp2_path):
            try:
                os.remove(tmp2_path)
            except:
                pass
