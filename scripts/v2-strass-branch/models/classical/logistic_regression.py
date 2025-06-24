# models/classical/logistic_regression.py
"""
Logistic Regression classifier implementation for multimodal fMRI stimulus prediction.

This module implements a Logistic Regression classifier that inherits from BaseClassifier
and follows SOLID principles for flexible, extensible machine learning workflows.
"""

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning

from models.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression classifier with preprocessing pipeline.
    
    This implementation provides a wrapper around sklearn's LogisticRegression
    with automatic feature scaling and comprehensive coefficient analysis.
    Supports both binary and multiclass classification.
    
    Attributes:
        scaler (StandardScaler): Feature scaling transformer
        model (Pipeline): Complete ML pipeline with preprocessing
        feature_names (Optional[np.ndarray]): Names of input features
        classes_ (Optional[np.ndarray]): Unique class labels
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Logistic Regression classifier with configuration.
        
        Args:
            config: Configuration dictionary containing hyperparameters
                   Expected keys:
                   - C: Regularization strength (default: 1.0)
                   - penalty: Regularization type (default: 'l2')
                   - solver: Optimization algorithm (default: 'lbfgs')
                   - max_iter: Maximum iterations (default: 1000)
                   - multi_class: Multiclass strategy (default: 'auto')
                   - class_weight: Class balancing (default: None)
                   - random_state: Random seed (default: 42)
                   - tol: Convergence tolerance (default: 1e-4)
                   - fit_intercept: Whether to fit intercept (default: True)
        """
        super().__init__(config)
        self.scaler = StandardScaler()
        self.feature_names: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        
        # Validate configuration parameters
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate C parameter
        c_value = self.config.get('C', 1.0)
        if not isinstance(c_value, (int, float)) or c_value <= 0:
            raise ValueError("C must be a positive number")
        
        # Validate penalty
        penalty = self.config.get('penalty', 'l2')
        valid_penalties = ['l1', 'l2', 'elasticnet', 'none']
        if penalty not in valid_penalties:
            raise ValueError(f"penalty must be one of {valid_penalties}")
        
        # Validate solver
        solver = self.config.get('solver', 'lbfgs')
        valid_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
        if solver not in valid_solvers:
            raise ValueError(f"solver must be one of {valid_solvers}")
        
        # Validate solver-penalty compatibility
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            raise ValueError("L1 penalty requires 'liblinear' or 'saga' solver")
        
        if penalty == 'elasticnet' and solver != 'saga':
            raise ValueError("elasticnet penalty requires 'saga' solver")
        
        # Validate max_iter
        max_iter = self.config.get('max_iter', 1000)
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionClassifier':
        """
        Train Logistic Regression classifier with preprocessing pipeline.
        
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
            # Store feature and class information
            self.feature_names = np.arange(X.shape[1])
            self.classes_ = np.unique(y)
            
            # Determine multi_class strategy if not specified
            multi_class = self.config.get('multi_class', 'auto')
            if multi_class == 'auto':
                multi_class = 'ovr' if len(self.classes_) == 2 else 'multinomial'
            
            # Handle l1_ratio for elasticnet penalty
            lr_params = {
                'C': self.config.get('C', 1.0),
                'penalty': self.config.get('penalty', 'l2'),
                'solver': self.config.get('solver', 'lbfgs'),
                'max_iter': self.config.get('max_iter', 1000),
                'multi_class': multi_class,
                'class_weight': self.config.get('class_weight', None),
                'random_state': self.config.get('random_state', 42),
                'tol': self.config.get('tol', 1e-4),
                'fit_intercept': self.config.get('fit_intercept', True),
                'n_jobs': self.config.get('n_jobs', None)
            }
            
            # Add l1_ratio for elasticnet penalty
            if lr_params['penalty'] == 'elasticnet':
                lr_params['l1_ratio'] = self.config.get('l1_ratio', 0.5)
            
            # Create preprocessing and model pipeline
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', SKLogisticRegression(**lr_params))
            ])
            
            # Fit the complete pipeline
            self.model.fit(X, y)
            self.is_fitted = True
            
            logger.info(f"Logistic Regression fitted with {X.shape[0]} samples, "
                       f"{X.shape[1]} features, {len(self.classes_)} classes")
            
        except Exception as e:
            logger.error(f"Failed to fit Logistic Regression: {str(e)}")
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
        Return feature importance scores based on coefficient magnitudes.
        
        For multiclass problems, returns the mean absolute coefficient
        values across all classes.
        
        Returns:
            Feature importance scores of shape (n_features,)
            
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        try:
            lr_model = self.model.named_steps['lr']
            coefficients = lr_model.coef_
            
            # Handle binary vs multiclass cases
            if coefficients.ndim == 1:
                # Binary classification
                importance_scores = np.abs(coefficients)
            else:
                # Multiclass classification - take mean absolute value across classes
                importance_scores = np.mean(np.abs(coefficients), axis=0)
            
            logger.debug(f"Computed feature importance for {len(importance_scores)} features")
            return importance_scores
        except Exception as e:
            logger.error(f"Feature importance computation failed: {str(e)}")
            raise RuntimeError(f"Feature importance computation failed: {str(e)}")
    
    def get_coefficients(self) -> np.ndarray:
        """
        Return raw model coefficients.
        
        Returns:
            Model coefficients of shape (n_classes, n_features) for multiclass
            or (n_features,) for binary classification
            
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting coefficients")
        
        lr_model = self.model.named_steps['lr']
        return lr_model.coef_
    
    def get_intercept(self) -> Union[float, np.ndarray]:
        """
        Return model intercept(s).
        
        Returns:
            Intercept value(s) - scalar for binary, array for multiclass
            
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting intercept")
        
        lr_model = self.model.named_steps['lr']
        intercept = lr_model.intercept_
        
        # Return scalar for binary classification
        if len(intercept) == 1:
            return intercept[0]
        return intercept
    
    def get_class_probabilities_log_odds(self, X: np.ndarray) -> np.ndarray:
        """
        Return log-odds (decision function values) for predictions.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Log-odds values of shape (n_samples,) for binary classification
            or (n_samples, n_classes) for multiclass
            
        Raises:
            RuntimeError: If model not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing log-odds")
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        try:
            log_odds = self.model.decision_function(X)
            logger.debug(f"Computed log-odds for {X.shape[0]} samples")
            return log_odds
        except Exception as e:
            logger.error(f"Log-odds computation failed: {str(e)}")
            raise RuntimeError(f"Log-odds computation failed: {str(e)}")
    
    def get_regularization_path(self, X: np.ndarray, y: np.ndarray, 
                              C_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute regularization path for different C values.
        
        Args:
            X: Training features
            y: Training labels  
            C_values: Array of regularization strengths to test
            
        Returns:
            Dictionary with 'C_values' and 'coefficients' arrays
            
        Raises:
            RuntimeError: If model not fitted or computation fails
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing regularization path")
        
        try:
            from sklearn.linear_model import LogisticRegressionCV
            
            # Create a cross-validated logistic regression
            lr_cv = LogisticRegressionCV(
                Cs=C_values,
                cv=5,
                penalty=self.config.get('penalty', 'l2'),
                solver=self.config.get('solver', 'lbfgs'),
                max_iter=self.config.get('max_iter', 1000),
                random_state=self.config.get('random_state', 42)
            )
            
            # Scale the features
            X_scaled = self.scaler.transform(X)
            lr_cv.fit(X_scaled, y)
            
            return {
                'C_values': C_values,
                'coefficients': lr_cv.coefs_paths_[1],  # Path for positive class
                'scores': lr_cv.scores_[1]  # CV scores for positive class
            }
            
        except Exception as e:
            logger.error(f"Regularization path computation failed: {str(e)}")
            raise RuntimeError(f"Regularization path computation failed: {str(e)}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive model statistics.
        
        Returns:
            Dictionary containing model statistics:
            - n_features: Number of input features
            - n_classes: Number of classes
            - n_iter: Number of iterations to convergence
            - regularization_strength: C parameter value
            - penalty_type: Type of regularization
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting statistics")
        
        lr_model = self.model.named_steps['lr']
        
        stats = {
            'n_features': len(self.feature_names),
            'n_classes': len(self.classes_),
            'classes': self.classes_.tolist(),
            'regularization_strength': lr_model.C,
            'penalty_type': lr_model.penalty,
            'solver': lr_model.solver,
            'max_iter': lr_model.max_iter,
            'fit_intercept': lr_model.fit_intercept
        }
        
        # Add convergence information if available
        if hasattr(lr_model, 'n_iter_'):
            stats['n_iter'] = lr_model.n_iter_.tolist() if hasattr(lr_model.n_iter_, 'tolist') else lr_model.n_iter_
        
        return stats