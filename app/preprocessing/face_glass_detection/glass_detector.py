import numpy as np
import tensorflow as tf
import keras


model = keras.models.load_model("./app/preprocessing/face_glass_detection/model/glasses_detection.keras")

class GlassDetector:
    def __init__(self):
        self.threshold = 0.5 
    
    def detect_glass(self, image_numpy: np.ndarray):
        """
        Detects glasses in the image using Keras Classification model.
        Args:
            image_numpy: NumPy array image in HxWxC or HxW format (uint8 or float).
        Returns:
            dict: {
                "glass_detected": bool,
                "message": str
            }
        """
        try:
            # --- 1. PREPROCESSING ---
            if not isinstance(image_numpy, np.ndarray):
                raise ValueError("image_numpy must be a NumPy array")

            img_array = image_numpy

            # Handle grayscale images: (H, W) or (H, W, 1) -> (H, W, 3)
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.ndim == 3 and img_array.shape[2] == 1:
                img_array = np.repeat(img_array, 3, axis=2)

            # Convert to float32
            img_array = img_array.astype("float32")

            # Resize to 160x160 (model input size)
            img_tensor = tf.convert_to_tensor(img_array)
            img_tensor = tf.image.resize(img_tensor, (160, 160))

            # MobileNetV2-style preprocessing: scale to [-1, 1]
            img_tensor = (img_tensor / 127.5) - 1.0

            # Add batch dimension: (160, 160, 3) -> (1, 160, 160, 3)
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            # --- 2. PREDICTION ---
            prediction = model.predict(img_tensor, verbose=0)

            # Apply Sigmoid to get probability (0.0 to 1.0)
            score = tf.nn.sigmoid(prediction)[0][0].numpy()

            # --- 3. LOGIC ---
            # If score is LOW (< threshold), it is 'glasses'
            glass_detected = score < self.threshold

            # --- 4. FORMAT OUTPUT ---
            if glass_detected:
                message = "Glass detected."
            else:
                message = "No glass detected."

            return {
                "glass_detected": glass_detected,
                "message": message
            }

        except Exception as e:
            print(f"Error during detection: {e}")
            return {
                "glass_detected": False,
                "message": f"Error: {str(e)}"
            }

glass_detector = GlassDetector()