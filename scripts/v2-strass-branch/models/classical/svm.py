# models/classical/svm.py
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any

from models.base_classifier import BaseClassifier

class SVMClassifier(BaseClassifier):
    """Support Vector Machine classifier"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """Train SVM classifier"""
        # Create pipeline with scaling
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                C=self.config.get('C', 1.0),
                kernel=self.config.get('kernel', 'rbf'),
                gamma=self.config.get('gamma', 'scale'),
                probability=True,
                random_state=self.config.get('random_state', 42)
            ))
        ])
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance (SVM doesn't have built-in importance)"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # For SVM, return absolute values of support vectors' weights
        svm_model = self.model.named_steps['svm']
        if hasattr(svm_model, 'coef_'):
            return np.abs(svm_model.coef_[0])
        else:
            # For non-linear kernels, return zeros
            return np.zeros(self.model.named_steps['scaler'].n_features_in_)