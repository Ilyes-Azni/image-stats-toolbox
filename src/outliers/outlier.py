# Initialization of outlier class

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from pathlib import Path

class Outlier(ABC):
    def __init__(self, features: np.ndarray):
        """
        Base class for outlier detection methods
        
        Args:
            features (np.ndarray): Array of features to analyze for outliers
        """
        self.features = features
        self.outlier_indices: List[int] = []
        self.outlier_scores: Dict[int, float] = {}
        
    @abstractmethod
    def detect(self, **kwargs) -> List[int]:
        """
        Abstract method to detect outliers
        
        Returns:
            List[int]: Indices of detected outliers
        """
        pass
    
    def get_outlier_indices(self) -> List[int]:
        """
        Returns the indices of detected outliers
        """
        return self.outlier_indices
    
    def get_outlier_scores(self) -> Dict[int, float]:
        """
        Returns the outlier scores for each sample
        """
        return self.outlier_scores
    
    def get_inlier_indices(self) -> List[int]:
        """
        Returns indices of samples that are not outliers
        """
        return [i for i in range(len(self.features)) if i not in self.outlier_indices]
