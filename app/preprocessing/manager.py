from app.preprocessing.multiple_face_detection.yolo_detector import yolo_detector
import numpy as np

class PreprocessingManager:
    def __init__(self):
        # Flags to enable/disable specific tasks
        self.enable_multiple_face_detection = True
        # self.enable_task2 = False # Placeholder
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
            "checks": {}
        }

        # Task 1: Multiple Face Detection
        if self.enable_multiple_face_detection:
            face_result = yolo_detector.detect_faces(image)
            results["checks"]["face_detection"] = face_result
            
            if face_result["face_count"] > 1:
                results["status"] = "warning"
                results["message"] = face_result["message"]
            elif face_result["face_count"] == 0:
                results["status"] = "error"
                results["message"] = "No face detected"
        
        # Task 2: Placeholder
        # if self.enable_task2:
        #     pass

        # Task 3: Placeholder
        # if self.enable_task3:
        #     pass

        return results

preprocess_manager = PreprocessingManager()
