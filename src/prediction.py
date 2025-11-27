import numpy as np
import cv2
from tensorflow import keras
from datetime import datetime


class TrafficSignPredictor:
    def __init__(self, model_path, class_names):
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = (64, 64)

    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_path, top_k=3):
        """Make prediction"""
        start_time = datetime.now()

        # Preprocess
        img = self.preprocess_image(image_path)

        # Predict
        predictions = self.model.predict(img, verbose=0)

        # Get top-k
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]

        latency = (datetime.now() - start_time).total_seconds() * 1000

        return {
            'predicted_class': int(top_indices[0]),
            'predicted_label': self.class_names.get(int(top_indices[0]), 'Unknown'),
            'confidence': float(predictions[0][top_indices[0]]),
            'top_k': [
                {
                    'class': int(idx),
                    'label': self.class_names.get(int(idx), 'Unknown'),
                    'probability': float(predictions[0][idx])
                }
                for idx in top_indices
            ],
            'latency_ms': latency
        }