from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2 
from app.services.face_matcher import face_matcher
from app.schemas.verify import VerifyResponse

from app.preprocessing import preprocess_manager as weeding_manager

router = APIRouter()

@router.post("/face/verify", response_model=VerifyResponse)
async def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        print("Received images") # debugging log
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()

        img1_np = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2_np = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img1_np is None or img2_np is None:
            raise HTTPException(status_code=400, detail="Could not decode one or both images")

        quality_check_1 = weeding_manager.preprocess(img1_np)
        quality_check_2 = weeding_manager.preprocess(img2_np)
        # quality_check_1 and 2 structure
            # "status": "0", # 5 == multiple face, 7 glasses and 9 rotated head
            # "checks": {},
            # # results['checks']['face_detection'] ->
            #     # "face_count": face_count, # this means results['checks']['face_detection']['face_count'] = face count in the img
            #     # "message": message,
            #     # "bboxes": bboxes
            # # results['checks']['glass_detection'] ->
            #     # "glass_detected": bool(glass_detected),
            #     # "score_no_glasses": float(score_no_glasses),
            #     # "message": message,
            # # results['checks']['head_pose_detection'] =
            #     # "head_pose": text != "Looking Forward",
            #     # "message": text,
            #     # "angles": (x, y, z)
        
            # "message": ""
        
        # gatekeeping logic harsher type.
        if '5' in quality_check_1['status'] or '5' in  quality_check_2['status']:
            msg = quality_check_1['message'] + " | " + quality_check_2['message']
            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": msg
                }
            )
        elif '79' in quality_check_1['status'] or '79' in quality_check_2['status']:
            msg = quality_check_1['message'] + " | " + quality_check_2['message']
            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": msg
                }
            )


        result = face_matcher.compare_faces(img1_bytes, img2_bytes)
        # gatekeeping but lighter type
        if quality_check_1['status'] == '7' or quality_check_2['status'] == '7':
            result['error'] = quality_check_1['message'] + " | " + quality_check_2['message']

        elif quality_check_1['status'] == '9' or quality_check_2['status'] == '9':
            result['error'] = quality_check_1['message'] + " | " + quality_check_2['message']

        return result

    except ValueError as e:
        print("ValueError: ", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("Exception: ", e)
        raise HTTPException(status_code=500, detail="Internal server error")
