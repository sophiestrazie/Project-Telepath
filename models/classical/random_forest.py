# models/classical/random_forest.py
"""
Random Forest classifier implementation for multimodal fMRI stimulus prediction.

This module implements a Random Forest classifier that inherits from BaseClassifier
and follows SOLID principles for flexible, extensible machine learning workflows.
"""

from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any, Optional
import logging

from models.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest classifier with preprocessing pipeline.
    
    This implementation provides a wrapper around sklearn's RandomForestClassifier
    with automatic feature scaling and comprehensive feature importance analysis.
    
    Attributes:
        scaler (StandardScaler): Feature scaling transformer
        model (Pipeline): Complete ML pipeline with preprocessing
        feature_names (Optional[np.ndarray]): Names of input features
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Random Forest classifier with configuration.
        
        Args:
            config: Configuration dictionary containing hyperparameters
                   Expected keys:
                   - n_estimators: Number of trees (default: 100)
                   - max_depth: Maximum tree depth (default: None)
                   - min_samples_split: Min samples to split (default: 2)
                   - min_samples_leaf: Min samples per leaf (default: 1)
                   - max_features: Features per split (default: 'sqrt')
                   - bootstrap: Bootstrap sampling (default: True)
                   - random_state: Random seed (default: 42)
                   - n_jobs: Parallel jobs (default: -1)
                   - class_weight: Class balancing (default: None)
        """
        super().__init__(config)
        self.scaler = StandardScaler()
        self.feature_names: Optional[np.ndarray] = None
        
        # Validate configuration parameters
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_numeric_params = {
            'n_estimators': (int, 1, float('inf')),
            'min_samples_split': (int, 2, float('inf')),
            'min_samples_leaf': (int, 1, float('inf'))
        }
        
        for param, (param_type, min_val, max_val) in required_numeric_params.items():
            if param in self.config:
                value = self.config[param]
                if not isinstance(value, param_type) or not (min_val <= value <= max_val):
                    raise ValueError(f"{param} must be {param_type.__name__} in range [{min_val}, {max_val}]")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Train Random Forest classifier with preprocessing pipeline.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            self: Fitted classifier instance
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If fitting fails
        """
        if X.size == 0 or y.size == 0:
            raise ValueError("Input data cannot be empty")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        try:
            # Store feature information
            self.feature_names = np.arange(X.shape[1])
            
            # Create preprocessing and model pipeline
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', SKRandomForest(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', None),
                    min_samples_split=self.config.get('min_samples_split', 2),
                    min_samples_leaf=self.config.get('min_samples_leaf', 1),
                    max_features=self.config.get('max_features', 'sqrt'),
                    bootstrap=self.config.get('bootstrap', True),
                    random_state=self.config.get('random_state', 42),
                    n_jobs=self.config.get('n_jobs', -1),
                    class_weight=self.config.get('class_weight', None),
                    verbose=0
                ))
            ])
            
            # Fit the complete pipeline
            self.model.fit(X, y)
            self.is_fitted = True
            
            logger.info(f"Random Forest fitted with {X.shape[0]} samples, {X.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Failed to fit Random Forest: {str(e)}")
            raise RuntimeError(f"Model fitting failed: {str(e)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
            
        Raises:
            RuntimeError: If model not fitted
            ValueError: If input shape is invalid
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        try:
            predictions = self.model.predict(X)
            logger.debug(f"Generated predictions for {X.shape[0]} samples")
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities for all classes.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
            
        Raises:
            RuntimeError: If model not fitted
            ValueError: If input shape is invalid
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        try:
            probabilities = self.model.predict_proba(X)
            logger.debug(f"Generated probabilities for {X.shape[0]} samples")
            return probabilities
        except Exception as e:
            logger.error(f"Probability prediction failed: {str(e)}")
            raise RuntimeError(f"Probability prediction failed: {str(e)}")
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Return feature importance scores from the Random Forest.
        
        Feature importance is calculated as the mean decrease in impurity
        across all trees in the forest.
        
        Returns:
            Feature importance scores of shape (n_features,)
            
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        try:
            rf_model = self.model.named_steps['rf']
            importance_scores = rf_model.feature_importances_
            
            logger.debug(f"Computed feature importance for {len(importance_scores)} features")
            return importance_scores
        except Exception as e:
            logger.error(f"Feature importance computation failed: {str(e)}")
            raise RuntimeError(f"Feature importance computation failed: {str(e)}")
    
    def get_tree_feature_importance(self, normalize: bool = True) -> np.ndarray:
        """
        Get detailed feature importance statistics across all trees.
        
        Args:
            normalize: Whether to normalize importance scores to sum to 1
            
        Returns:
            Array of shape (n_estimators, n_features) with per-tree importance
            
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting tree-level importance")
        
        rf_model = self.model.named_steps['rf']
        tree_importances = np.array([
            tree.feature_importances_ for tree in rf_model.estimators_
        ])
        
        if normalize:
            # Normalize each tree's importance to sum to 1
            tree_sums = tree_importances.sum(axis=1, keepdims=True)
            tree_sums[tree_sums == 0] = 1  # Prevent division by zero
            tree_importances = tree_importances / tree_sums
        
        return tree_importances
    
    def get_oob_score(self) -> float:
        """
        Get out-of-bag score if bootstrap=True was used.
        
        Returns:
            Out-of-bag accuracy score
            
        Raises:
            RuntimeError: If model not fitted or OOB not available
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting OOB score")
        
        rf_model = self.model.named_steps['rf']
        
        if not rf_model.bootstrap:
            raise RuntimeError("OOB score not available when bootstrap=False")
        
        if not hasattr(rf_model, 'oob_score_'):
            # Need to refit with oob_score=True
            logger.warning("OOB score was not computed during fitting")
            return np.nan
        
        return rf_model.oob_score_
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Get model complexity metrics.
        
        Returns:
            Dictionary containing complexity metrics:
            - total_nodes: Total nodes across all trees
            - mean_depth: Average tree depth
            - max_depth: Maximum tree depth
            - total_leaves: Total leaf nodes
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting complexity metrics")
        
        rf_model = self.model.named_steps['rf']
        
        depths = [tree.tree_.max_depth for tree in rf_model.estimators_]
        node_counts = [tree.tree_.node_count for tree in rf_model.estimators_]
        leaf_counts = [tree.tree_.n_leaves for tree in rf_model.estimators_]
        
        return {
            'n_estimators': rf_model.n_estimators,
            'total_nodes': sum(node_counts),
            'mean_nodes_per_tree': np.mean(node_counts),
            'mean_depth': np.mean(depths),
            'max_depth': max(depths),
            'total_leaves': sum(leaf_counts),
            'mean_leaves_per_tree': np.mean(leaf_counts)
        }