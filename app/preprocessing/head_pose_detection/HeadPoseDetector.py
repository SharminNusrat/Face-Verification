# currently in use.
import numpy as np
from insightface.app import FaceAnalysis
from app.core.face_app import FaceAppProvider
from app.core.logger import logger

class HeadPoseDetector:
    def __init__(self):
        """
        Initializes the InsightFace FaceAnalysis app using the FaceAppProvider singleton.
        """
        logger.info("HeadPoseDetector has been initialized.")
        self.app = FaceAppProvider.get_app()

    def get_direction(self, image_bgr, horizontal_threshold=20, vertical_threshold=20):
        """
        Takes a BGR numpy image and returns the direction string using InsightFace.
        Returns: (direction_string, (pitch, yaw, roll))
        """

        faces = self.app.get(image_bgr)
        logger.info("faces have been extracted.") # debugging log
        if not faces:
            return {
                "head_pose": False,
                "message": "No Face",
                "angles": (0, 0, 0)
            }

        # in the image there will only be one face. so No need to sort and get the first
        face = faces[0]
        logger.info("face have been extracted.")
        if face.pose is None:
             return {
                "head_pose": False,
                "message": "Pose Not Detected",
                "angles": (0, 0, 0)
            }

        # InsightFace returns pose as [pitch, yaw, roll] in degrees
        pitch, yaw, roll = face.pose
        
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
        
        logger.info(f"the person is {text}.")
        return {
            "head_pose": text != "Looking Forward",
            "message": text,
            "angles": (float(pitch), float(yaw), float(roll))
        }

head_pose_detector = HeadPoseDetector()