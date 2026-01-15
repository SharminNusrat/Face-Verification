from app.preprocessing.multiple_face_detection.yolo_face_detector import yolo_face_detector
from app.preprocessing.face_glass_detection.yolo_glass_detector import yolo_glass_detector
from app.preprocessing.head_pose_detection.head_pose_detector import head_pose_detector
import numpy as np

class PreprocessingManager:
    def __init__(self):
        # Flags to enable/disable specific tasks
        self.enable_multiple_face_detection = True
        self.enable_face_glass_detection = True
        self.enable_head_pose_detection = True

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
            "checks": {}
        }

        # Task 1: Multiple Face Detection
        if self.enable_multiple_face_detection:
            face_result = yolo_face_detector.detect_faces(image)
            results["checks"]["face_detection"] = face_result
            
            if face_result["face_count"] > 1:
                results["status"] = "warning"
                results["message"] = face_result["message"]
            elif face_result["face_count"] == 0:
                results["status"] = "error"
                results["message"] = "No face detected"
        
        # Task 2: Face Glass Detection
        if self.enable_face_glass_detection:
            glass_result = yolo_glass_detector.detect_glasses(image)
            results["checks"]["glass_detection"] = glass_result
            
            if glass_result["glass_detected"]:
                 results["status"] = "warning"
                 current_msg = results.get("message", "")
                 if current_msg:
                     results["message"] = f"{current_msg} | {glass_result['message']}"
                 else:
                     results["message"] = glass_result["message"]
                 
        # Task 3: Head Pose Detection
        if self.enable_head_pose_detection:
            pose_result = head_pose_detector.get_direction(image)
            results["checks"]["head_pose"] = pose_result
            
            # Logic: If not looking forward, warn?
            if pose_result["direction"] != "Looking Forward":
                 # We don't overwrite "error" status, only success or warning
                 if results["status"] != "error":
                      results["status"] = "warning"
                 
                 current_msg = results.get("message", "")
                 msg_part = f"Head Pose: {pose_result['direction']}"
                 if current_msg:
                     results["message"] = f"{current_msg} | {msg_part}"
                 else:
                     results["message"] = msg_part

        return results

preprocess_manager = PreprocessingManager()
