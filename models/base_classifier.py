# models/base_classifier.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.base import BaseEstimator

class BaseClassifier(ABC, BaseEstimator):
    """Abstract base class for all classifiers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """Train the classifier"""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
        
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""
        pass
        
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance scores"""
        pass
        
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get classifier parameters"""
        return self.config
        
    def set_params(self, **params) -> 'BaseClassifier':
        """Set classifier parameters"""
        self.config.update(params)
        return self