# glass_detector.py

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
        """
        Wrapper around the trained Keras model for glasses detection.

        The saved model already includes:
        - tf.keras.applications.mobilenet_v2.preprocess_input
        - MobileNetV2 base
        - GlobalAveragePooling2D
        - Dense(1) (logit)

        So we should only:
        - convert to float32
        - resize to input_size
        - add batch dimension
        """
        self.model = keras.models.load_model(model_path)
        self.threshold = threshold
        self.input_size = input_size

    def _prepare_image(self, image_numpy: np.ndarray) -> tf.Tensor:
        """
        Prepare a single image (NumPy array) for the model.

        Accepts:
            - H x W (grayscale)
            - H x W x 1
            - H x W x 3
        """
        if not isinstance(image_numpy, np.ndarray):
            raise ValueError("image_numpy must be a NumPy array")

        img = image_numpy

        # Handle grayscale: (H, W) -> (H, W, 3)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected image with 3 channels, got shape {img.shape}")

        # Convert to float32, keep in [0, 255] range
        img = img.astype("float32")

        # To tensor and resize to (160, 160)
        img = tf.convert_to_tensor(img)
        img = tf.image.resize(img, self.input_size)

        # Add batch dimension: (H, W, 3) -> (1, H, W, 3)
        img = tf.expand_dims(img, axis=0)
        return img

    def detect_glass(self, image_numpy: np.ndarray) -> dict:
        """
        Detects glasses in the image using the Keras classification model.

        Args:
            image_numpy: NumPy image array (H x W x C or H x W).

        Returns:
            dict: {
                "glass_detected": bool,
                "score_no_glasses": float | None,  # P(no_glasses)
                "message": str
            }
        """
        try:
            # --- 1. PREPARE INPUT ---
            img_tensor = self._prepare_image(image_numpy)

            # --- 2. PREDICTION ---
            # Model outputs logits (shape (1, 1) or (1,))
            logits = self.model(img_tensor, training=False)
            logits = tf.reshape(logits, [-1])[0]  # scalar

            # Sigmoid -> probability of label "1"
            # With BinaryCrossentropy(from_logits=True):
            #   y_true in {0,1}, y_pred is logit for class "1"
            score_no_glasses = tf.nn.sigmoid(logits).numpy().item()
            print("score no glass: ", score_no_glasses) # debugging log

            # --- 3. DECISION LOGIC ---
            # If your dataset had:
            #   class 0 -> "glasses"
            #   class 1 -> "no_glasses"
            # then score_no_glasses = P(no_glasses)
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