# models/ensemble/voting.py
"""
Voting ensemble classifier for multimodal fMRI stimulus prediction.

This module implements voting ensemble methods that combine predictions from
multiple base classifiers using either hard voting (majority) or soft voting
(probability averaging) following SOLID principles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib

# Ensure project imports work
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.base_classifier import BaseClassifier
from models import ClassifierFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VotingEnsemble(BaseClassifier):
    """
    Voting ensemble classifier that combines multiple base classifiers.
    
    Supports both hard voting (majority vote) and soft voting (probability averaging).
    Follows the Composite pattern to treat ensemble as a single classifier.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize voting ensemble.
        
        Args:
            config: Configuration dictionary containing:
                - base_classifiers: List of (name, classifier_type, classifier_config) tuples
                - voting_type: 'hard' or 'soft'
                - weights: Optional list of weights for each classifier
                - use_probabilities: Whether to use predict_proba for soft voting
        """
        super().__init__(config)
        
        # Ensemble configuration
        self.base_classifiers_config = config.get('base_classifiers', [])
        self.voting_type = config.get('voting_type', 'soft').lower()
        self.weights = config.get('weights', None)
        self.use_probabilities = config.get('use_probabilities', True)
        
        # Validation
        if self.voting_type not in ['hard', 'soft']:
            raise ValueError("voting_type must be 'hard' or 'soft'")
        
        if self.voting_type == 'soft' and not self.use_probabilities:
            warnings.warn("Soft voting requires probability estimates. Setting use_probabilities=True")
            self.use_probabilities = True
        
        # Initialize base classifiers
        self.base_classifiers = []
        self.classifier_names = []
        self._initialize_base_classifiers()
        
        # Ensemble state
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.n_classes_ = None
        
        # Performance tracking
        self.base_classifier_scores_ = {}
        self.ensemble_score_ = None
        
    def _initialize_base_classifiers(self) -> None:
        """Initialize base classifiers from configuration."""
        for clf_config in self.base_classifiers_config:
            if isinstance(clf_config, dict):
                name = clf_config.get('name', f"classifier_{len(self.base_classifiers)}")
                clf_type = clf_config['type']
                clf_params = clf_config.get('config', {})
            elif isinstance(clf_config, (list, tuple)) and len(clf_config) >= 2:
                if len(clf_config) == 3:
                    name, clf_type, clf_params = clf_config
                else:
                    name, clf_type = clf_config
                    clf_params = {}
            else:
                raise ValueError(f"Invalid base classifier configuration: {clf_config}")
            
            # Create classifier instance
            try:
                classifier = ClassifierFactory.create_classifier(clf_type, clf_params)
                self.base_classifiers.append(classifier)
                self.classifier_names.append(name)
                logger.info(f"Initialized base classifier: {name} ({clf_type})")
            except Exception as e:
                logger.error(f"Failed to create classifier {name} ({clf_type}): {e}")
                raise
        
        # Validate weights
        if self.weights is not None:
            if len(self.weights) != len(self.base_classifiers):
                raise ValueError("Number of weights must match number of base classifiers")
            
            # Normalize weights
            self.weights = np.array(self.weights)
            self.weights = self.weights / np.sum(self.weights)
        else:
            # Equal weights
            self.weights = np.ones(len(self.base_classifiers)) / len(self.base_classifiers)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """
        Fit all base classifiers.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        # Validate input
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Store class information
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        
        logger.info(f"Fitting voting ensemble with {len(self.base_classifiers)} base classifiers")
        
        # Fit each base classifier
        fitted_classifiers = []
        fitted_names = []
        for i, (name, classifier) in enumerate(zip(self.classifier_names, self.base_classifiers)):
            logger.info(f"Fitting classifier {i+1}/{len(self.base_classifiers)}: {name}")
            
            try:
                # Fit classifier
                classifier.fit(X, y)
                fitted_classifiers.append(classifier)
                fitted_names.append(name)
                
                # Evaluate classifier performance
                if hasattr(classifier, 'score'):
                    score = classifier.score(X, y)
                    self.base_classifier_scores_[name] = score
                    logger.info(f"  Training accuracy: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to fit classifier {name}: {e}")
                logger.warning(f"Skipping classifier {name} due to fitting error")
                continue
        
        # Update base classifiers list to only include successfully fitted ones
        if len(fitted_classifiers) != len(self.base_classifiers):
            logger.warning(f"Only {len(fitted_classifiers)}/{len(self.base_classifiers)} classifiers fitted successfully")
            self.base_classifiers = fitted_classifiers
            self.classifier_names = fitted_names
            # Adjust weights accordingly
            self.weights = self.weights[:len(fitted_classifiers)]
            if len(self.weights) > 0:
                self.weights = self.weights / np.sum(self.weights)
        
        if len(self.base_classifiers) == 0:
            raise RuntimeError("No base classifiers were successfully fitted")
        
        self.is_fitted = True
        logger.info("Voting ensemble fitting completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before making predictions")
        
        X = check_array(X)
        
        if self.voting_type == 'hard':
            return self._predict_hard_voting(X)
        else:
            return self._predict_soft_voting(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the ensemble.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before making predictions")
        
        if self.voting_type == 'hard':
            warnings.warn("predict_proba with hard voting may not be meaningful")
        
        X = check_array(X)
        return self._predict_proba_ensemble(X)
    
    def _predict_hard_voting(self, X: np.ndarray) -> np.ndarray:
        """Perform hard voting (majority vote)."""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.base_classifiers)))
        
        # Collect predictions from all classifiers
        for i, classifier in enumerate(self.base_classifiers):
            try:
                pred = classifier.predict(X)
                # Encode predictions to ensure consistency
                pred_encoded = self.label_encoder.transform(pred)
                predictions[:, i] = pred_encoded
            except Exception as e:
                logger.warning(f"Classifier {i} failed to predict: {e}")
                # Use majority class as fallback
                predictions[:, i] = np.full(n_samples, 0)  # Encoded majority class
        
        # Apply weights if specified
        if self.weights is not None and len(self.weights) == len(self.base_classifiers):
            # Weighted voting
            final_predictions = np.zeros(n_samples)
            for i in range(n_samples):
                votes = {}
                for j, pred in enumerate(predictions[i]):
                    pred = int(pred)
                    votes[pred] = votes.get(pred, 0) + self.weights[j]
                final_predictions[i] = max(votes, key=votes.get)
        else:
            # Simple majority voting
            final_predictions = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), 
                axis=1, 
                arr=predictions
            )
        
        # Decode back to original labels
        return self.label_encoder.inverse_transform(final_predictions.astype(int))
    
    def _predict_soft_voting(self, X: np.ndarray) -> np.ndarray:
        """Perform soft voting (probability averaging)."""
        probabilities = self._predict_proba_ensemble(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
    
    def _predict_proba_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble probability predictions."""
        n_samples = X.shape[0]
        ensemble_probas = np.zeros((n_samples, self.n_classes_))
        
        total_weight = 0
        for i, classifier in enumerate(self.base_classifiers):
            try:
                if hasattr(classifier, 'predict_proba'):
                    probas = classifier.predict_proba(X)
                    
                    # Ensure probabilities match our class ordering
                    if hasattr(classifier, 'classes_'):
                        clf_classes = classifier.classes_
                        # Reorder probabilities to match our class order
                        probas_reordered = np.zeros((n_samples, self.n_classes_))
                        for j, cls in enumerate(clf_classes):
                            if cls in self.classes_:
                                cls_idx = np.where(self.classes_ == cls)[0][0]
                                probas_reordered[:, cls_idx] = probas[:, j]
                        probas = probas_reordered
                    
                    # Apply weight
                    weight = self.weights[i] if self.weights is not None else 1.0
                    ensemble_probas += weight * probas
                    total_weight += weight
                    
                else:
                    # Fallback: use hard predictions and convert to probabilities
                    predictions = classifier.predict(X)
                    pred_encoded = self.label_encoder.transform(predictions)
                    
                    probas = np.zeros((n_samples, self.n_classes_))
                    for j, pred in enumerate(pred_encoded):
                        probas[j, pred] = 1.0
                    
                    weight = self.weights[i] if self.weights is not None else 1.0
                    ensemble_probas += weight * probas
                    total_weight += weight
                    
            except Exception as e:
                logger.warning(f"Classifier {i} failed to predict probabilities: {e}")
                continue
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_probas /= total_weight
        
        return ensemble_probas
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get ensemble feature importance by averaging individual importances.
        
        Returns:
            Average feature importance across base classifiers
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before getting feature importance")
        
        importances = []
        weights_used = []
        
        for i, classifier in enumerate(self.base_classifiers):
            try:
                if hasattr(classifier, 'get_feature_importance'):
                    importance = classifier.get_feature_importance()
                    importances.append(importance)
                    weights_used.append(self.weights[i] if self.weights is not None else 1.0)
                elif hasattr(classifier, 'feature_importances_'):
                    importance = classifier.feature_importances_
                    importances.append(importance)
                    weights_used.append(self.weights[i] if self.weights is not None else 1.0)
                elif hasattr(classifier, 'coef_'):
                    # For linear models, use absolute coefficients
                    coef = classifier.coef_
                    if coef.ndim > 1:
                        importance = np.mean(np.abs(coef), axis=0)
                    else:
                        importance = np.abs(coef)
                    importances.append(importance)
                    weights_used.append(self.weights[i] if self.weights is not None else 1.0)
            except Exception as e:
                logger.warning(f"Could not get feature importance from classifier {i}: {e}")
                continue
        
        if not importances:
            logger.warning("No classifiers provided feature importance")
            return np.zeros(1)  # Fallback
        
        # Convert to arrays and ensure same length
        importances = [np.array(imp) for imp in importances]
        min_length = min(len(imp) for imp in importances)
        importances = [imp[:min_length] for imp in importances]
        
        # Weighted average
        ensemble_importance = np.zeros(min_length)
        total_weight = 0
        
        for importance, weight in zip(importances, weights_used):
            ensemble_importance += weight * importance
            total_weight += weight
        
        if total_weight > 0:
            ensemble_importance /= total_weight
        
        return ensemble_importance
    
    def get_base_classifier_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base classifiers.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping classifier names to their predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before making predictions")
        
        X = check_array(X)
        predictions = {}
        
        for name, classifier in zip(self.classifier_names, self.base_classifiers):
            try:
                pred = classifier.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Classifier {name} failed to predict: {e}")
                predictions[name] = np.array([])
        
        return predictions
    
    def get_base_classifier_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get probability predictions from individual base classifiers.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary mapping classifier names to their probability predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before making predictions")
        
        X = check_array(X)
        probabilities = {}
        
        for name, classifier in zip(self.classifier_names, self.base_classifiers):
            try:
                if hasattr(classifier, 'predict_proba'):
                    probas = classifier.predict_proba(X)
                    probabilities[name] = probas
                else:
                    logger.info(f"Classifier {name} does not support probability prediction")
                    probabilities[name] = None
            except Exception as e:
                logger.warning(f"Classifier {name} failed to predict probabilities: {e}")
                probabilities[name] = None
        
        return probabilities
    
    def evaluate_base_classifiers(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Evaluate individual base classifiers using cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with evaluation results for each classifier
        """
        results = {}
        
        for name, classifier in zip(self.classifier_names, self.base_classifiers):
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                
                results[name] = {
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'cv_scores': cv_scores.tolist()
                }
                
                logger.info(f"{name}: CV Accuracy = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate classifier {name}: {e}")
                results[name] = {
                    'cv_mean': 0.0,
                    'cv_std': 0.0,
                    'cv_scores': [],
                    'error': str(e)
                }
        
        return results
    
    def get_diversity_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate diversity metrics for the ensemble.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary containing diversity metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before calculating diversity")
        
        # Get predictions from all classifiers
        all_predictions = []
        for classifier in self.base_classifiers:
            try:
                pred = classifier.predict(X)
                pred_encoded = self.label_encoder.transform(pred)
                all_predictions.append(pred_encoded)
            except Exception as e:
                logger.warning(f"Skipping classifier in diversity calculation: {e}")
                continue
        
        if len(all_predictions) < 2:
            return {'error': 'Need at least 2 classifiers for diversity calculation'}
        
        all_predictions = np.array(all_predictions).T  # Shape: (n_samples, n_classifiers)
        y_encoded = self.label_encoder.transform(y)
        
        # Calculate Q-statistic (pairwise diversity)
        n_classifiers = all_predictions.shape[1]
        q_stats = []
        
        for i in range(n_classifiers):
            for j in range(i + 1, n_classifiers):
                pred_i = all_predictions[:, i]
                pred_j = all_predictions[:, j]
                
                # Create contingency table
                n11 = np.sum((pred_i == y_encoded) & (pred_j == y_encoded))  # Both correct
                n10 = np.sum((pred_i == y_encoded) & (pred_j != y_encoded))  # i correct, j wrong
                n01 = np.sum((pred_i != y_encoded) & (pred_j == y_encoded))  # i wrong, j correct
                n00 = np.sum((pred_i != y_encoded) & (pred_j != y_encoded))  # Both wrong
                
                # Q-statistic
                denominator = (n11 * n00 + n01 * n10)
                if denominator != 0:
                    q_stat = (n11 * n00 - n01 * n10) / denominator
                    q_stats.append(q_stat)
        
        # Disagreement measure
        disagreement = 0
        for i in range(n_classifiers):
            for j in range(i + 1, n_classifiers):
                disagreement += np.mean(all_predictions[:, i] != all_predictions[:, j])
        
        disagreement /= (n_classifiers * (n_classifiers - 1) / 2)
        
        return {
            'mean_q_statistic': np.mean(q_stats) if q_stats else 0.0,
            'std_q_statistic': np.std(q_stats) if q_stats else 0.0,
            'disagreement': disagreement,
            'n_classifiers': n_classifiers
        }
    
    def save_ensemble(self, filepath: str) -> None:
        """
        Save the ensemble to disk.
        
        Args:
            filepath: Path to save the ensemble
        """
        ensemble_data = {
            'config': self.config,
            'base_classifiers': self.base_classifiers,
            'classifier_names': self.classifier_names,
            'weights': self.weights,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str) -> 'VotingEnsemble':
        """
        Load ensemble from disk.
        
        Args:
            filepath: Path to load the ensemble from
            
        Returns:
            Loaded VotingEnsemble instance
        """
        ensemble_data = joblib.load(filepath)
        
        # Create new instance
        ensemble = cls(ensemble_data['config'])
        
        # Restore state
        ensemble.base_classifiers = ensemble_data['base_classifiers']
        ensemble.classifier_names = ensemble_data['classifier_names']
        ensemble.weights = ensemble_data['weights']
        ensemble.label_encoder = ensemble_data['label_encoder']
        ensemble.classes_ = ensemble_data['classes_']
        ensemble.n_classes_ = ensemble_data['n_classes_']
        ensemble.is_fitted = ensemble_data['is_fitted']
        
        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble


# Convenience function for creating voting ensembles
def create_voting_ensemble(base_classifiers: List[Tuple[str, str, Dict[str, Any]]],
                          voting_type: str = 'soft',
                          weights: Optional[List[float]] = None) -> VotingEnsemble:
    """
    Convenience function to create a voting ensemble.
    
    Args:
        base_classifiers: List of (name, classifier_type, config) tuples
        voting_type: 'hard' or 'soft'
        weights: Optional weights for each classifier
        
    Returns:
        VotingEnsemble instance
    """
    config = {
        'base_classifiers': base_classifiers,
        'voting_type': voting_type,
        'weights': weights
    }
    
    return VotingEnsemble(config)


def get_algonauts_optimized_voting_config() -> Dict[str, Any]:
    """
    Create optimized voting configuration designed to outperform Algonauts baseline.
    
    Returns:
        Optimized voting configuration for fMRI stimulus prediction
    """
    return {
        'base_classifiers': [
            ('svm_rbf_high', 'svm', {
                'C': 10.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'cache_size': 1000
            }),
            ('svm_linear', 'svm', {
                'C': 1.0,
                'kernel': 'linear',
                'probability': True,
                'cache_size': 1000
            }),
            ('rf_balanced', 'random_forest', {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'bootstrap': True
            }),
            ('lr_balanced', 'logistic_regression', {
                'C': 10.0,
                'penalty': 'l2',
                'max_iter': 2000,
                'class_weight': 'balanced'
            })
        ],
        'voting_type': 'soft',
        'weights': [0.3, 0.25, 0.25, 0.2],  # Slightly favor SVM-RBF
        'use_probabilities': True
    }


def create_neuroimaging_voting_config() -> Dict[str, Any]:
    """
    Create voting configuration specifically optimized for neuroimaging data.
    
    Returns:
        Neuroimaging-optimized voting configuration
    """
    return {
        'base_classifiers': [
            ('svm_high_c', 'svm', {
                'C': 100.0,
                'kernel': 'linear',
                'probability': True,
                'cache_size': 1000
            }),
            ('rf_deep', 'random_forest', {
                'n_estimators': 500,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True
            }),
            ('lr_elastic', 'logistic_regression', {
                'C': 1.0,
                'penalty': 'elasticnet',
                'l1_ratio': 0.5,
                'solver': 'saga',
                'max_iter': 5000
            })
        ],
        'voting_type': 'soft',
        'weights': None,  # Equal weights
        'use_probabilities': True
    }


def create_diverse_voting_config() -> Dict[str, Any]:
    """
    Create diverse voting configuration with multiple algorithm types.
    
    Returns:
        Diverse voting configuration
    """
    return {
        'base_classifiers': [
            ('svm_rbf', 'svm', {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True
            }),
            ('svm_poly', 'svm', {
                'C': 1.0,
                'kernel': 'poly',
                'degree': 3,
                'probability': True
            }),
            ('rf_default', 'random_forest', {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }),
            ('rf_deep', 'random_forest', {
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 5,
                'random_state': 42
            }),
            ('lr_l1', 'logistic_regression', {
                'C': 1.0,
                'penalty': 'l1',
                'solver': 'liblinear',
                'random_state': 42
            }),
            ('lr_l2', 'logistic_regression', {
                'C': 10.0,
                'penalty': 'l2',
                'max_iter': 1000,
                'random_state': 42
            })
        ],
        'voting_type': 'soft',
        'weights': None,  # Equal weights for diversity
        'use_probabilities': True
    }


# Example usage and testing
if __name__ == "__main__":
    # Example configuration for testing
    test_config = {
        'base_classifiers': [
            ('svm', 'svm', {'C': 1.0, 'kernel': 'rbf'}),
            ('rf', 'random_forest', {'n_estimators': 100}),
            ('lr', 'logistic_regression', {'C': 1.0})
        ],
        'voting_type': 'soft',
        'weights': [0.4, 0.4, 0.2]
    }
    
    # Create ensemble
    ensemble = VotingEnsemble(test_config)
    print(f"Created voting ensemble with {len(ensemble.base_classifiers)} base classifiers")
    print(f"Voting type: {ensemble.voting_type}")
    print(f"Weights: {ensemble.weights}")