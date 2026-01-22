# currently in use. 

import numpy as np
import tensorflow as tf
import keras


class GlassDetector:
    def __init__(
        self,
        model_path: str = "./app/preprocessing/face_glass_detection/model/glasses_detection.keras",
        threshold: float = 0.9, # koto percent sure je glass nai.  
        input_size: tuple[int, int] = (160, 160),
    ):
        self.model = keras.models.load_model(model_path)
        self.threshold = threshold
        self.input_size = input_size

    def _prepare_image(self, image_numpy: np.ndarray) -> tf.Tensor:
        if not isinstance(image_numpy, np.ndarray):
            raise ValueError("image_numpy must be a NumPy array")

        img = image_numpy

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {img.shape}")

        img = img.astype("float32")
        img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, self.input_size)
        img = tf.expand_dims(img, axis=0)
        return img

    def detect_glass(self, image_numpy: np.ndarray) -> dict:
        try:
            img_tensor = self._prepare_image(image_numpy)
            logits = self.model(img_tensor, training=False)
            logits = tf.reshape(logits, [-1])[0]  # scalar
            score_no_glasses = tf.nn.sigmoid(logits).numpy().item()
            glass_detected = score_no_glasses < self.threshold
            message = "Glass detected." if glass_detected else "No glass detected."
            return {
                "glass_detected": bool(glass_detected),
                "score_no_glasses": float(score_no_glasses),
                "message": message,
            }

        except Exception as e:
            print(f"Error during detection: {e}")
            return {
                "glass_detected": False,
                "score_no_glasses": None,
                "message": f"Error: {str(e)}",
            }

glass_detector = GlassDetector()