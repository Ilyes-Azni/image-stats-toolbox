import tensorflow as tf

from tensorflow.keras.applications import MobileNetV3Large

from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

import numpy as np

from PIL import Image

class FeatureExtractor:
    def __init__(self, input_shape=(224, 224)):
        self.input_shape = input_shape
        self.model = MobileNetV3Large(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    
    def extract_features(self, image_path):
        """Extract normalized features from a single image"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_shape)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Extract features
        features = self.model.predict(img_array)
        
        # Normalize features
        normalized_features = features / np.linalg.norm(features)
        return normalized_features.flatten()
    
    def batch_extract(self, image_paths):
        """Extract features from multiple images"""
        features = []
        for path in image_paths:
            features.append(self.extract_features(path))
        return np.array(features)
