import cv2
import numpy as np
from insightface.app import FaceAnalysis
from app.core.config import settings
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocessing import manager as preprocess_manager

class FaceMatcher:
    def __init__(self):
        self.app = FaceAnalysis(name=settings.MODEL_NAME)
        self.app.prepare(ctx_id=0, det_size=settings.DET_SIZE)

    def get_embedding(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        
        faces = self.app.get(img)
        if not faces:
            return None 
        
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].embedding

    def compare_faces(self, img1_bytes: bytes, img2_bytes: bytes):
        emb1 = self.get_embedding(img1_bytes)
        emb2 = self.get_embedding(img2_bytes)

        if emb1 is None or emb2 is None:
            return {
                "match": False,
                "score": 0.0,
                "threshold": threshold,
                "error": "Face not detected in one or both images"
            }

        score = cosine_similarity([emb1], [emb2])[0][0]
        threshold = 0.5 
        matched = bool(score > threshold)

        return {
            "match": matched,
            "score": float(score),
            "threshold": threshold,
        }

face_matcher = FaceMatcher()
