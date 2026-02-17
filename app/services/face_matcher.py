import cv2
import numpy as np
from insightface.app import FaceAnalysis
from app.core.config import settings
from sklearn.metrics.pairwise import cosine_similarity
from app.core.logger import logger
from app.core.face_app import FaceAppProvider
from app.preprocessing import manager as preprocess_manager

class FaceMatcher:
    def __init__(self):
        logger.info("face matcher initialized.")
        self.app = FaceAppProvider.get_app()

    def get_embedding(self, image: np.ndarray):

        if image is None:
            logger.error("bro you gave an empty image.")
            raise ValueError("You provided an empty image.")
        
        faces = self.app.get(image)
        if not faces:
            logger.error("No faces have been detected.")
            return None 
        
        return faces[0].embedding

    def compare_faces(self, img1_np: np.ndarray, img2_np: np.ndarray):
        threshold = settings.SIMILARITY_THRESHOLD

        emb1 = self.get_embedding(img1_np)
        emb2 = self.get_embedding(img2_np)

        if emb1 is None or emb2 is None:
            if emb1 is None:
                logger.error("image 1 didn't produce any embeddings.")
            if emb2 is None:
                logger.error("image 2 didn't produce any embeddings.")
            if emb1 is None and emb2 is None:
                logger.error("both images didn't produce any embeddings.")
            
            return {
                "match": False,
                "score": 0.0,
                "threshold": threshold,
                "error": "Face not detected in one or both images"
            }

        score = cosine_similarity([emb1], [emb2])[0][0]
        logger.info(f"{score} similarity between the faces.")
        matched = bool(score > threshold)

        return {
            "match": matched,
            "score": float(score),
            "threshold": threshold,
        }

face_matcher = FaceMatcher()
