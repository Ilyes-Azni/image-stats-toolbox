# Implementation of Mahalanobis distance on image for outlier detection
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.outliers.outlier import Outlier
from src.utils.image_loader import ImageLoader
from skimage import io
from scipy.linalg import inv
from src.utils.visualisation import DatasetVisualizer

class mahalanobis(Outlier):
    def __init__(self, image_loader: ImageLoader, class_name: str = None):
        self.image_loader = image_loader
        self.class_name = class_name
        
        if class_name:
            self.image_paths = image_loader.get_images_by_class(class_name)
        else:
            self.image_paths = []
            for ext_paths in image_loader.dataset_index.values():
                self.image_paths.extend(ext_paths)
                
        features = self._extract_dataset_features()
        super().__init__(features)
        self.mean = np.mean(features, axis=0)
        self.covariance = np.cov(features, rowvar=False)
        # Computing inverse covariance matrix for Mahalanobis distance
        self.inv_covariance = inv(self.covariance)

    def detect(self, threshold: float = 3.0) -> List[int]:
        """
        Detects outliers using Mahalanobis distance
        
        Args:
            threshold: Distance threshold for outlier detection
        """
        distances = []
        for x in self.features:
            # Mahalanobis distance formula: sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))
            diff = x - self.mean
            dist = np.sqrt(diff.dot(self.inv_covariance).dot(diff))
            distances.append(dist)
            
        distances = np.array(distances)
        self.outlier_scores = {i: d for i, d in enumerate(distances)}
        self.outlier_indices = [i for i, d in enumerate(distances) if d > threshold]
        return self.outlier_indices
    
    def _extract_dataset_features(self) -> np.ndarray:
        """Extract features from all images in the dataset"""
        features = []
        for path in self.image_paths:
            img = io.imread(path)
            features.append(self._extract_features(img))
        return np.array(features)
    
    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract features from a single image"""
        features = []
        # Color statistics
        features.extend(np.mean(img, axis=(0,1)))
        features.extend(np.std(img, axis=(0,1)))
        # Shape features
        features.append(img.shape[0])  # height
        features.append(img.shape[1])  # width
        features.append(img.shape[0] * img.shape[1])  # area
        
        return np.array(features)

    def get_outlier_paths(self) -> List[str]:
        """Returns the file paths of detected outlier images"""
        return [self.image_paths[i] for i in self.outlier_indices]
    
    def visualize_outliers(self, num_samples: int = 5):
        """Displays a grid of detected outlier images for visual inspection"""
        # Create a temporary ImageLoader instance just for visualization
        temp_loader = ImageLoader(self.image_loader.root_path)
        temp_loader.dataset_index = {"outliers": self.get_outlier_paths()}
        temp_loader.class_mapping = {"outliers": self.get_outlier_paths()}
        
        # Use our existing visualization tool
        DatasetVisualizer.viz(temp_loader, images_per_class=num_samples)

