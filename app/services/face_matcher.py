from deepface import DeepFace
from app.core.config import settings

class FaceMatcher:
    @staticmethod
    def verify(img1_path, img2_path, model_name: str = "ArcFace", metric: str = "cosine"):
        """
        Verifies if two images belong to the same person.
        Accepts image paths (str) or numpy arrays (BGR).
        """
        try:
            # Check for multiple faces in both images
            FaceMatcher._ensure_single_face(img1_path, "Image 1")
            FaceMatcher._ensure_single_face(img2_path, "Image 2")

            # DeepFace.verify accepts numpy arrays for img1_path and img2_path arguments
            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name=model_name,
                distance_metric=metric,
                enforce_detection=False 
            )
            # Apply custom threshold
            distance = result.get('distance')
            threshold = settings.FACE_MATCH_THRESHOLD
            
            # Override verification decision based on custom threshold
            result['verified'] = distance <= threshold
            result['threshold'] = threshold
            
            print(f"Result: {result}")
            return result
        except Exception as e:
            # Handle deepface specific exceptions or re-raise
            # For now, let's print and re-raise or return a structured error
            print(f"Error in FaceMatcher: {e}")
            raise e

    @staticmethod
    def _ensure_single_face(img, img_label):
        """
        Ensures that exactly one face is detected in the image.
        Uses retinaface backend for high accuracy.
        """
        try:
            # "Detector backend" refers to the specific algorithm or model used to locate faces within the image.
            # We use 'retinaface' as it is widely considered one of the most accurate face detectors available,
            # especially for handling different poses, lighting, and occlusions, though it may be slower than 'opencv'.
            
            faces = DeepFace.extract_faces(img_path=img, detector_backend='retinaface', enforce_detection=False)
            
            if len(faces) > 1:
                raise ValueError(f"Multiple faces detected in {img_label}. Please upload an image with a single face.")
            if len(faces) == 0:
                 raise ValueError(f"No face detected in {img_label}. Please upload an image with a visible face.")
                 
        except ValueError as e:
            raise e
        except Exception as e:
            # extract_faces might raise error if no face found and enforce_detection=True (but we set False)
            print(f"Error validating faces in {img_label}: {e}")
            raise ValueError(f"Could not process faces in {img_label}")

