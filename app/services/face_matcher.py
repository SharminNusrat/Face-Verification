from deepface import DeepFace
# from app.core.config import settings

class FaceMatcher:
    @staticmethod
    def verify(img1_path: str, img2_path: str, model_name: str = "ArcFace", metric: str = "cosine"):
        """
        Verifies if two images belong to the same person.
        """
        try:
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                distance_metric=metric,
                enforce_detection=False # Set to False to avoid errors if face is not clear, can be parameterized
            )
            print(f"Result: {result}")
            return result
        except Exception as e:
            # Handle deepface specific exceptions or re-raise
            # For now, let's print and re-raise or return a structured error
            print(f"Error in FaceMatcher: {e}")
            raise e
