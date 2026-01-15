import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Yaw → left–right head turn (looking sideways)
# Pitch → up–down (looking up/down)
# Roll → head tilt (ear toward shoulder)

class HeadPoseDetector:
    def __init__(self, model_name: str = "models/face_landmarker.task"):
        # Resolve path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face landmarker model not found at: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE)

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_direction(self, image_bgr, horizontal_threshold=10, vertical_threshold=30):
        """
        Takes a BGR numpy image and returns the direction string and angles.
        Returns: 
            dict: {
                "direction": str,
                "angles": tuple(pitch, yaw, roll) | None,
                "message": str
            }
        """
        if image_bgr is None:
             return {"direction": "Error", "angles": None, "message": "Image is None"}

        # 1. Convert BGR to RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # 3. Detect
        detection_result = self.landmarker.detect(mp_image)

        # 4. If no face, return values indicating that
        if not detection_result.face_landmarks:
             return {
                 "direction": "Unknown", 
                 "angles": None, 
                 "message": "No face detected by landmarker"
             }

        # 5. Extract Landmarks & Calculate PnP
        face_landmarks = detection_result.face_landmarks[0]
        img_h, img_w, _ = image_bgr.shape
        
        face_3d = []
        face_2d = []
        
        # Indices: Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
        landmark_indices = [1, 199, 33, 263, 61, 291]

        for idx in landmark_indices:
            lm = face_landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
            
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                [0, focal_length, img_h / 2],
                                [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        # 6. Calculate Angles
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        x = angles[0] * 360 # Pitch
        y = angles[1] * 360 # Yaw
        z = angles[2] * 360 # Roll

        # 7. Determine Direction
        if y < -horizontal_threshold:
            text = "Looking Left"
        elif y > horizontal_threshold:
            text = "Looking Right"
        elif x < -vertical_threshold:
            text = "Looking Down"
        elif x > vertical_threshold:
            text = "Looking Up"
        else:
            text = "Looking Forward"
            
        return {
            "direction": text,
            "angles": (x, y, z),
            "message": text
        }

head_pose_detector = HeadPoseDetector()
