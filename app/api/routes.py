from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.face_matcher import face_matcher
from app.schemas.verify import VerifyResponse

router = APIRouter()

@router.post("/verify", response_model=VerifyResponse)
async def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        
        result = face_matcher.compare_faces(img1_bytes, img2_bytes)
        return result
    except ValueError as e:
        return {"match": False, "score": 0.0, "error": str(e)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
