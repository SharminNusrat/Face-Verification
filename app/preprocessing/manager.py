from app.preprocessing.multiple_face_detection.yolo_face_detector import yolo_face_detector
from app.preprocessing.face_glass_detection.glass_detector import glass_detector
from app.preprocessing.head_pose_detection.HeadPoseDetector import head_pose_detector
from app.core.config import settings
import numpy as np

class PreprocessingManager:
    def __init__(self):
        self.enable_multiple_face_detection = settings.ENABLE_MULTIPLE_FACE_DETECTION
        self.enable_face_glass_detection = settings.ENABLE_FACE_GLASS_DETECTION
        self.enable_head_pose_detection = settings.ENABLE_HEAD_POSE_DETECTION

    def preprocess(self, image: np.ndarray):
        results = {
            "status": "0", # 5 == multiple face, 7 glasses and 9 rotated head
            "checks": {},
            # results['checks']['face_detection'] ->
                # "face_count": face_count, # this means results['checks']['face_detection']['face_count'] = face count in the img
                # "message": message,
                # "bboxes": bboxes
            # results['checks']['glass_detection'] ->
                # "glass_detected": bool(glass_detected),
                # "score_no_glasses": float(score_no_glasses),
                # "message": message,
            # results['checks']['head_pose_detection'] =
                # "head_pose": text != "Looking Forward",
                # "message": text,
                # "angles": (x, y, z)
        
            "message": ""
        }

        if self.enable_multiple_face_detection:
            face_result = yolo_face_detector.detect_faces(image)
            results["checks"]["face_detection"] = face_result
            
            if face_result["face_count"] > 1:
                results["status"] = "5"
                results["message"] = face_result["message"]
            elif face_result["face_count"] == 0:
                results["status"] = "5"
                results["message"] = "No face detected"
        
        if self.enable_face_glass_detection:
            glass_result = glass_detector.detect_glass(image)
            results["checks"]["glass_detection"] = glass_result
            
            if glass_result["glass_detected"]:
                if results['status'] == '0':
                    results["status"] = "7"
                else:
                    results['status'] += '7'

                current_msg = results.get("message", "")
                if current_msg:
                    results["message"] = f"{current_msg} | {glass_result['message']}"
                else:
                    results["message"] = glass_result["message"]

        if self.enable_head_pose_detection: 
            head_pose_result = head_pose_detector.get_direction(image)
            results["checks"]["head_pose_detection"] = head_pose_result
            
            if head_pose_result["head_pose"]:
                if results['status'] == '0':
                    results['status'] = '9'
                else:
                    results["status"] += '9'


                current_msg = results.get("message", "")
                if current_msg:
                    results["message"] = f"{current_msg} | {head_pose_result['message']}"
                else:
                    results["message"] = head_pose_result["message"]

        return results

preprocess_manager = PreprocessingManager()
