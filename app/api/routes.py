from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2 
from app.services.face_matcher import face_matcher
from app.schemas.verify import VerifyResponse

from app.preprocessing import preprocess_manager as weeding_manager

router = APIRouter()

@router.post("/face/verify", response_model=VerifyResponse)
async def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        img1_np = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2_np = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img1_np is None or img2_np is None:
            raise HTTPException(status_code=400, detail="Could not decode one or both images")

        result_1 = weeding_manager.preprocess(img1_np)
        result_2 = weeding_manager.preprocess(img2_np)

        if result_1["status"] != "success" or result_2["status"] != "success":
            return {
                "match": False,
                "score": 0.0,
                "error": "provided images did not follow the guidelines !!!"
            }

        result = face_matcher.compare_faces(img1_bytes, img2_bytes)
        print(result)  # debugging log

        return result

    except ValueError as e:
        print("ValueError: ", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("Exception: ", e)
        raise HTTPException(status_code=500, detail="Internal server error")
