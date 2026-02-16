# currently in use.
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from app.core.face_app import FaceAppProvider

class HeadPoseDetector:
    def __init__(self):
        """
        Initializes the InsightFace FaceAnalysis app using the FaceAppProvider singleton.
        """
        self.app = FaceAppProvider.get_app()

    def get_direction(self, image_bgr, horizontal_threshold=20, vertical_threshold=20):
        """
        Takes a BGR numpy image and returns the direction string using InsightFace.
        Returns: (direction_string, (pitch, yaw, roll))
        """
        faces = self.app.get(image_bgr)

        if not faces:
            return {
                "head_pose": False,
                "message": "No Face",
                "angles": (0, 0, 0)
            }

        # Assume the largest face is the target
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]

        if face.pose is None:
             return {
                "head_pose": False,
                "message": "Pose Not Detected",
                "angles": (0, 0, 0)
            }

        # InsightFace returns pose as [pitch, yaw, roll] in degrees
        pitch, yaw, roll = face.pose

        # InsightFace co-ordinate system might differ slightly from MediaPipe's
        # Adjust logic based on standard InsightFace outputs:
        # Pitch: +ve (Up), -ve (Down)
        # Yaw: +ve (Right), -ve (Left)
        # Roll: +ve (Right tilt), -ve (Left tilt)
        
        # Note: Thresholds might need tuning as InsightFace degrees can be different scale
        
        if yaw > horizontal_threshold:
            text = "Looking Right"
        elif yaw < -horizontal_threshold:
            text = "Looking Left"
        elif pitch > vertical_threshold:
            text = "Looking Up"
        elif pitch < -vertical_threshold:
            text = "Looking Down"
        else:
            text = "Looking Forward"
            
        return {
            "head_pose": text != "Looking Forward",
            "message": text,
            "angles": (float(pitch), float(yaw), float(roll))
        }

head_pose_detector = HeadPoseDetector()