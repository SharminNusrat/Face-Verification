from app.preprocessing.multiple_face_detection.yolo_face_detector import yolo_face_detector
from app.preprocessing.face_glass_detection.glass_detector import glass_detector
from app.preprocessing.head_pose_detection.HeadPoseDetector import head_pose_detector
from app.core.config import settings
import numpy as np

class PreprocessingManager:
    def __init__(self):
        # Flags to enable/disable specific tasks
        self.enable_multiple_face_detection = settings.ENABLE_MULTIPLE_FACE_DETECTION
        self.enable_face_glass_detection = settings.ENABLE_FACE_GLASS_DETECTION
        self.enable_head_pose_detection = settings.ENABLE_HEAD_POSE_DETECTION
        # self.enable_task3 = False # Placeholder

    def preprocess(self, image: np.ndarray):
        """
        Main preprocessing method.
        Args:
            image: cv2 numpy array
        Returns:
            dict: Aggregated results of enabled preprocessing steps
        """
        results = {
            "status": "success",
            "checks": {}, # if we want we can remove this. kept for future proofing
            "message": ""
        }

        # Task 1: Multiple Face Detection
        if self.enable_multiple_face_detection:
            face_result = yolo_face_detector.detect_faces(image)
            results["checks"]["face_detection"] = face_result
            
            if face_result["face_count"] > 1:
                results["status"] = "severe" #  if we want we can return from here as well
                results["message"] = face_result["message"]
            elif face_result["face_count"] == 0:
                results["status"] = "error"
                results["message"] = "No face detected"
        
        # Task 2: Face Glass Detection
        if self.enable_face_glass_detection:
            glass_result = glass_detector.detect_glass(image)
            results["checks"]["glass_detection"] = glass_result
            
            if glass_result["glass_detected"]:
                 results["status"] = "warning"
                 current_msg = results.get("message", "")
                 if current_msg:
                     results["message"] = f"{current_msg} | {glass_result['message']}"
                 else:
                     results["message"] = glass_result["message"]

        # Task 3: Head Pose Detection
        if self.enable_head_pose_detection: # we can set threshold for yaw, pitch, roll
            # yaw -> rotation in the spinal axis
            # roll -> rotation in the nose axis
            # pitch -> rotation in the ear axis # draw a line between two ears
            head_pose_result = head_pose_detector.get_direction(image)
            results["checks"]["head_pose_detection"] = head_pose_result
            
            if head_pose_result["head_pose"]:
                 results["status"] = "warning"
                 current_msg = results.get("message", "")
                 if current_msg:
                     results["message"] = f"{current_msg} | {head_pose_result['message']}"
                 else:
                     results["message"] = head_pose_result["message"]
                 
        # Task 3: Placeholder
        # if self.enable_task3:
        #     pass

        return results

preprocess_manager = PreprocessingManager()
