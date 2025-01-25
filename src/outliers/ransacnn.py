# outliers/ransacnn.py

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from .outlier import Outlier

class RANSACNN(Outlier):
    def __init__(self, features: np.ndarray):
        super().__init__(features)
        self.n_samples = features.shape[0]
        
    def detect(self, 
               sample_ratio: float = 0.05,
               threshold_iter: int = 500,
               **kwargs) -> List[int]:
        """
        Detect outliers using RANSAC-NN algorithm
        
        Args:
            sample_ratio: Ratio of samples to use in each iteration (default: 0.05)
            threshold_iter: Number of threshold iterations for TS stage (default: 500)
            
        Returns:
            List of outlier indices
        """
        # Normalize features as per paper
        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
        
        # Stage 1: Inlier Score Prediction (ISP)
        m = max(1, int(self.n_samples * sample_ratio))
        s = max(1, int(np.ceil(self.n_samples / m)))
        inlier_scores = self._isp(m, s)
        
        # Stage 2: Threshold Sampling (TS)
        outlier_scores = self._ts(inlier_scores, m, threshold_iter)
        
        # Store scores and determine outliers
        self.outlier_scores = {i: float(outlier_scores[i]) for i in range(self.n_samples)}
        self.outlier_indices = sorted(
            self.outlier_scores.keys(), 
            key=lambda x: self.outlier_scores[x], 
            reverse=True
        )[:int(self.n_samples * sample_ratio)]  # Default to top sample_ratio% as outliers
        
        return self.outlier_indices

    def _isp(self, m: int, s: int) -> np.ndarray:
        """Inlier Score Prediction stage"""
        eta = np.ones(self.n_samples)
        
        for _ in range(s):
            # Random sample without replacement
            sample_idx = np.random.choice(self.n_samples, size=m, replace=False)
            sample_features = self.features[sample_idx]
            
            # Compute cosine similarity with all features
            similarities = cosine_similarity(self.features, sample_features)
            max_similarities = np.max(similarities, axis=1)
            
            # Update eta with element-wise minimum
            eta = np.minimum(eta, max_similarities)
            
        return eta

    def _ts(self, eta: np.ndarray, m: int, t: int) -> np.ndarray:
        """Threshold Sampling stage"""
        sigma = np.zeros(self.n_samples)
        
        for k in range(1, t+1):
            tau = (k-1)/t
            eligible_mask = eta > tau
            eligible_features = self.features[eligible_mask]
            
            if len(eligible_features) == 0:
                continue
                
            # Sample from eligible features
            sample_size = min(m, len(eligible_features))
            sample_idx = np.random.choice(len(eligible_features), size=sample_size, replace=False)
            sample_features = eligible_features[sample_idx]
            
            # Compute similarities
            similarities = cosine_similarity(self.features, sample_features)
            max_similarities = np.max(similarities, axis=1)
            
            # Update outlier scores
            sigma = ((k-1)*sigma + (max_similarities < tau).astype(float)) / k
            
        return sigma