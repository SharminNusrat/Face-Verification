from app.core.face_app import FaceAppProvider

class InsightFaceDetector:
    def __init__(self):
        self.app = FaceAppProvider.get_app()

    def detect_faces(self, image_input):
        """
        Detects faces in the input image using InsightFace.
        Returns a dictionary with face count, message, and bounding boxes.
        """
        faces = self.app.get(image_input)
        
        face_count = len(faces)
        bboxes = []
        for face in faces:
            # face.bbox is usually [x1, y1, x2, y2]
            # Convert to float/int list
            box = face.bbox.astype(int).tolist()
            bboxes.append(box)

        message = ""
        if face_count > 1:
            message = "Multiple faces detected."
        elif face_count == 1:
            message = "Single face detected."
        else:
            message = "No face detected."

        return {
            "face_count": face_count,
            "message": message,
            "bboxes": bboxes
        }

insightface_detector = InsightFaceDetector()
