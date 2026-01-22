# currently in use.
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HeadPoseDetector:
    def __init__(self, model_path='./app/preprocessing/head_pose_detection/assets/face_landmarker.task'):
        """
        Initializes the MediaPipe FaceLandmarker.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE) # Changed to IMAGE mode for single usage
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_direction(self, image_bgr, horizontal_threshold=10, vertical_threshold=20):
        """
        Takes a BGR numpy image and returns the direction string.
        Returns: (direction_string, (pitch, yaw, roll))
        """
        # 1. Convert BGR to RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # 3. Detect
        detection_result = self.landmarker.detect(mp_image)

        # 4. If no face, return "No Face" with proper dict format
        if not detection_result.face_landmarks:
            return {
                "head_pose": False,  # Assuming no face detected is handled by face detector, or we could set True if we want to flag it here too. 
                                     # But if we treat 'head_pose' as 'is_pose_invalid', No Face is technically invalid but maybe redundant. 
                                     # Let's default to False to avoid overwriting Error with Warning in manager.
                "message": "No Face",
                "angles": (0, 0, 0)
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
        # Note: If you flip the image BEFORE passing it here, signs might be reversed.
        # This logic assumes the standard "Mirror" view (user sees themselves like a mirror)
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
            "head_pose": text != "Looking Forward",
            "message": text,
            "angles": (x, y, z)
        }

head_pose_detector = HeadPoseDetector()