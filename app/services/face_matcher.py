import cv2
import numpy as np
from insightface.app import FaceAnalysis
from app.core.config import settings
from sklearn.metrics.pairwise import cosine_similarity

class FaceMatcher:
    def __init__(self):
        # providers options: ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # We default to CPU to be safe, but user can change if they have GPU setup.
        # Note: onnxruntime-gpu is needed for CUDA.
        self.app = FaceAnalysis(name=settings.MODEL_NAME)
        self.app.prepare(ctx_id=0, det_size=settings.DET_SIZE)

    def get_embedding(self, image_bytes: bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        
        faces = self.app.get(img)
        if not faces:
            return None # No face detected
        
        # Assume the largest face is the target? Or strictly the first one.
        # Let's sort by area just in case
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].embedding

    def compare_faces(self, img1_bytes: bytes, img2_bytes: bytes):
        emb1 = self.get_embedding(img1_bytes)
        emb2 = self.get_embedding(img2_bytes)

        if emb1 is None or emb2 is None:
            return {
                "match": False,
                "score": 0.0,
                "error": "Face not detected in one or both images"
            }

        # Calculate cosine similarity
        # embeddings are usually normalized in InsightFace, but let's use cosine_similarity for safety
        score = cosine_similarity([emb1], [emb2])[0][0]
        
        # Threshold for match (this can be tuned)
        # Common threshold for arcface/insightface is around 0.3 - 0.5 depending on loss, 
        # but cosine similarity range is [-1, 1].
        # InsightFace usually recommends matching if distance < 1.1 (Euclidean) or Sim > threshold.
        # Let's use a conservative threshold of 0.5 for now, user can tune.
        threshold = 0.5 
        matched = bool(score > threshold)

        return {
            "match": matched,
            "score": float(score),
            "threshold": threshold
        }

face_matcher = FaceMatcher()
