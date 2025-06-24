# models/ensemble/stacking.py
"""
Stacking ensemble classifier for multimodal fMRI stimulus prediction.

This module implements stacking (stacked generalization) ensemble methods where
base classifier predictions are used as features for a meta-learner. Supports
both cross-validation based stacking and holdout-based stacking following
SOLID principles.

Mathematical Foundation:
$$\hat{y} = f_{meta}(h_1(x), h_2(x), ..., h_k(x))$$

Where:
- $h_i(x)$ are base classifier predictions
- $f_{meta}$ is the meta-learner
- $\hat{y}$ is the final prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, cross_val_score
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


class StackingEnsemble(BaseClassifier):
    """
    Stacking ensemble classifier that uses base classifier predictions as features.
    
    The ensemble works in two levels:
    1. Base level: Multiple diverse classifiers make predictions
    2. Meta level: A meta-learner combines base predictions to make final prediction
    
    Supports both probability-based and class-based stacking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stacking ensemble.
        
        Args:
            config: Configuration dictionary containing:
                - base_classifiers: List of (name, classifier_type, classifier_config) tuples
                - meta_learner: Dictionary with meta-learner configuration
                - cv_folds: Number of folds for cross-validation (default: 5)
                - use_probabilities: Whether to use predict_proba for base predictions
                - use_original_features: Whether to include original features in meta-learner
                - passthrough: Whether to pass original features to meta-learner
                - method: 'cv' for cross-validation or 'holdout' for holdout validation
                - holdout_size: Fraction of data for holdout (if method='holdout')
        """
        super().__init__(config)
        
        # Ensemble configuration
        self.base_classifiers_config = config.get('base_classifiers', [])
        self.meta_learner_config = config.get('meta_learner', {'type': 'logistic_regression', 'config': {}})
        self.cv_folds = config.get('cv_folds', 5)
        self.use_probabilities = config.get('use_probabilities', True)
        self.use_original_features = config.get('use_original_features', False)
        self.passthrough = config.get('passthrough', False)  # Alias for use_original_features
        self.method = config.get('method', 'cv').lower()
        self.holdout_size = config.get('holdout_size', 0.2)
        
        # Resolve passthrough vs use_original_features
        if self.passthrough:
            self.use_original_features = True
        
        # Validation
        if self.method not in ['cv', 'holdout']:
            raise ValueError("method must be 'cv' or 'holdout'")
        
        if not 0 < self.holdout_size < 1:
            raise ValueError("holdout_size must be between 0 and 1")
        
        # Initialize base classifiers
        self.base_classifiers = []
        self.classifier_names = []
        self._initialize_base_classifiers()
        
        # Initialize meta-learner
        self.meta_learner = None
        self._initialize_meta_learner()
        
        # Ensemble state
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.n_classes_ = None
        
        # Training data for meta-features
        self.meta_features_train_ = None
        self.original_features_shape_ = None
        
        # Performance tracking
        self.base_classifier_scores_ = {}
        self.meta_learner_score_ = None
        self.ensemble_score_ = None
        
    def _initialize_base_classifiers(self) -> None:
        """Initialize base classifiers from configuration."""
        for clf_config in self.base_classifiers_config:
            if isinstance(clf_config, dict):
                name = clf_config.get('name', f"base_clf_{len(self.base_classifiers)}")
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
                logger.error(f"Failed to create base classifier {name} ({clf_type}): {e}")
                raise
        
        if len(self.base_classifiers) == 0:
            raise ValueError("At least one base classifier must be specified")
    
    def _initialize_meta_learner(self) -> None:
        """Initialize meta-learner from configuration."""
        meta_type = self.meta_learner_config.get('type', 'logistic_regression')
        meta_params = self.meta_learner_config.get('config', {})
        
        try:
            self.meta_learner = ClassifierFactory.create_classifier(meta_type, meta_params)
            logger.info(f"Initialized meta-learner: {meta_type}")
        except Exception as e:
            logger.error(f"Failed to create meta-learner {meta_type}: {e}")
            raise
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        # Validate input
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        
        # Store original feature shape
        self.original_features_shape_ = X.shape[1]
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Store class information
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        
        logger.info(f"Fitting stacking ensemble with {len(self.base_classifiers)} base classifiers")
        logger.info(f"Using {self.method} method for meta-feature generation")
        
        if self.method == 'cv':
            meta_features = self._fit_cv_stacking(X, y)
        else:
            meta_features = self._fit_holdout_stacking(X, y)
        
        # Store meta-features for later use
        self.meta_features_train_ = meta_features
        
        # Fit meta-learner
        logger.info("Fitting meta-learner")
        self.meta_learner.fit(meta_features, y)
        
        # Evaluate meta-learner
        if hasattr(self.meta_learner, 'score'):
            self.meta_learner_score_ = self.meta_learner.score(meta_features, y)
            logger.info(f"Meta-learner training accuracy: {self.meta_learner_score_:.4f}")
        
        self.is_fitted = True
        logger.info("Stacking ensemble fitting completed")
        
        return self
    
    def _fit_cv_stacking(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit base classifiers using cross-validation stacking.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Meta-features array
        """
        logger.info(f"Using {self.cv_folds}-fold cross-validation for meta-feature generation")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Generate meta-features using cross-validation
        meta_features_list = []
        
        for i, (name, classifier) in enumerate(zip(self.classifier_names, self.base_classifiers)):
            logger.info(f"Generating meta-features for {name} ({i+1}/{len(self.base_classifiers)})")
            
            try:
                if self.use_probabilities and hasattr(classifier, 'predict_proba'):
                    # Use cross-validation to get out-of-fold probability predictions
                    cv_probas = cross_val_predict(
                        classifier, X, y, cv=cv, method='predict_proba', n_jobs=1
                    )
                    meta_features_list.append(cv_probas)
                    
                    logger.info(f"  Generated probability features: shape {cv_probas.shape}")
                    
                else:
                    # Use cross-validation to get out-of-fold class predictions
                    cv_preds = cross_val_predict(classifier, X, y, cv=cv, n_jobs=1)
                    
                    # Convert to one-hot encoding
                    cv_preds_encoded = self.label_encoder.transform(cv_preds)
                    cv_preds_onehot = np.zeros((len(cv_preds), self.n_classes_))
                    cv_preds_onehot[np.arange(len(cv_preds)), cv_preds_encoded] = 1
                    
                    meta_features_list.append(cv_preds_onehot)
                    logger.info(f"  Generated class features: shape {cv_preds_onehot.shape}")
                
                # Evaluate base classifier
                cv_scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
                self.base_classifier_scores_[name] = {
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores)
                }
                logger.info(f"  CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to generate meta-features for {name}: {e}")
                # Create dummy features as fallback
                dummy_features = np.zeros((X.shape[0], self.n_classes_))
                meta_features_list.append(dummy_features)
                continue
        
        # Now fit all base classifiers on full training data
        logger.info("Fitting base classifiers on full training data")
        for name, classifier in zip(self.classifier_names, self.base_classifiers):
            try:
                classifier.fit(X, y)
                logger.info(f"  Fitted {name}")
            except Exception as e:
                logger.error(f"Failed to fit {name}: {e}")
        
        # Combine meta-features
        meta_features = np.hstack(meta_features_list)
        
        # Add original features if requested
        if self.use_original_features:
            meta_features = np.hstack([meta_features, X])
            logger.info(f"Added original features to meta-features")
        
        logger.info(f"Final meta-features shape: {meta_features.shape}")
        return meta_features
    
    def _fit_holdout_stacking(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit base classifiers using holdout stacking.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Meta-features array
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Using holdout validation ({self.holdout_size:.1%}) for meta-feature generation")
        
        # Split data into base and meta sets
        X_base, X_meta, y_base, y_meta = train_test_split(
            X, y, test_size=self.holdout_size, 
            stratify=y, random_state=42
        )
        
        logger.info(f"Base set size: {X_base.shape[0]}, Meta set size: {X_meta.shape[0]}")
        
        # Fit base classifiers on base set and predict on meta set
        meta_features_list = []
        
        for i, (name, classifier) in enumerate(zip(self.classifier_names, self.base_classifiers)):
            logger.info(f"Training {name} on base set ({i+1}/{len(self.base_classifiers)})")
            
            try:
                # Fit on base set
                classifier.fit(X_base, y_base)
                
                if self.use_probabilities and hasattr(classifier, 'predict_proba'):
                    # Predict probabilities on meta set
                    meta_probas = classifier.predict_proba(X_meta)
                    meta_features_list.append(meta_probas)
                    logger.info(f"  Generated probability features: shape {meta_probas.shape}")
                    
                else:
                    # Predict classes on meta set
                    meta_preds = classifier.predict(X_meta)
                    
                    # Convert to one-hot encoding
                    meta_preds_encoded = self.label_encoder.transform(meta_preds)
                    meta_preds_onehot = np.zeros((len(meta_preds), self.n_classes_))
                    meta_preds_onehot[np.arange(len(meta_preds)), meta_preds_encoded] = 1
                    
                    meta_features_list.append(meta_preds_onehot)
                    logger.info(f"  Generated class features: shape {meta_preds_onehot.shape}")
                
                # Evaluate on meta set
                meta_score = accuracy_score(y_meta, classifier.predict(X_meta))
                self.base_classifier_scores_[name] = {'holdout_accuracy': meta_score}
                logger.info(f"  Holdout accuracy: {meta_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                # Create dummy features as fallback
                dummy_features = np.zeros((X_meta.shape[0], self.n_classes_))
                meta_features_list.append(dummy_features)
                continue
        
        # Refit all base classifiers on full training data
        logger.info("Refitting base classifiers on full training data")
        for name, classifier in zip(self.classifier_names, self.base_classifiers):
            try:
                classifier.fit(X, y)
                logger.info(f"  Refitted {name}")
            except Exception as e:
                logger.error(f"Failed to refit {name}: {e}")
        
        # Combine meta-features
        meta_features = np.hstack(meta_features_list)
        
        # Add original features if requested (from meta set)
        if self.use_original_features:
            meta_features = np.hstack([meta_features, X_meta])
            logger.info(f"Added original features to meta-features")
        
        # Store meta set targets for meta-learner training
        self._y_meta = y_meta
        
        logger.info(f"Final meta-features shape: {meta_features.shape}")
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before making predictions")
        
        X = check_array(X)
        
        # Generate meta-features from base classifiers
        meta_features = self._generate_meta_features(X)
        
        # Make prediction using meta-learner
        predictions = self.meta_learner.predict(meta_features)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the stacking ensemble.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before making predictions")
        
        X = check_array(X)
        
        # Generate meta-features from base classifiers
        meta_features = self._generate_meta_features(X)
        
        # Make probability prediction using meta-learner
        if hasattr(self.meta_learner, 'predict_proba'):
            probabilities = self.meta_learner.predict_proba(meta_features)
        else:
            # Fallback to hard predictions
            predictions = self.meta_learner.predict(meta_features)
            pred_encoded = self.label_encoder.transform(predictions)
            probabilities = np.zeros((len(predictions), self.n_classes_))
            probabilities[np.arange(len(predictions)), pred_encoded] = 1.0
        
        return probabilities
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate meta-features from base classifier predictions.
        
        Args:
            X: Input features
            
        Returns:
            Meta-features array
        """
        meta_features_list = []
        
        for name, classifier in zip(self.classifier_names, self.base_classifiers):
            try:
                if self.use_probabilities and hasattr(classifier, 'predict_proba'):
                    # Get probability predictions
                    probas = classifier.predict_proba(X)
                    meta_features_list.append(probas)
                else:
                    # Get class predictions and convert to one-hot
                    preds = classifier.predict(X)
                    pred_encoded = self.label_encoder.transform(preds)
                    pred_onehot = np.zeros((len(preds), self.n_classes_))
                    pred_onehot[np.arange(len(preds)), pred_encoded] = 1
                    meta_features_list.append(pred_onehot)
                    
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
                # Create dummy features as fallback
                dummy_features = np.zeros((X.shape[0], self.n_classes_))
                meta_features_list.append(dummy_features)
        
        # Combine meta-features
        meta_features = np.hstack(meta_features_list)
        
        # Add original features if requested
        if self.use_original_features:
            meta_features = np.hstack([meta_features, X])
        
        return meta_features
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the meta-learner.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before getting feature importance")
        
        try:
            if hasattr(self.meta_learner, 'get_feature_importance'):
                return self.meta_learner.get_feature_importance()
            elif hasattr(self.meta_learner, 'feature_importances_'):
                return self.meta_learner.feature_importances_
            elif hasattr(self.meta_learner, 'coef_'):
                coef = self.meta_learner.coef_
                if coef.ndim > 1:
                    return np.mean(np.abs(coef), axis=0)
                else:
                    return np.abs(coef)
            else:
                logger.warning("Meta-learner does not provide feature importance")
                return np.zeros(1)
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return np.zeros(1)
    
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
    
    def get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Get meta-features that would be fed to the meta-learner.
        
        Args:
            X: Input features
            
        Returns:
            Meta-features array
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before generating meta-features")
        
        return self._generate_meta_features(X)
    
    def analyze_meta_learner_importance(self) -> Dict[str, Any]:
        """
        Analyze the importance of different components in the meta-learner.
        
        Returns:
            Dictionary containing importance analysis
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before analysis")
        
        feature_importance = self.get_feature_importance()
        
        # Calculate importance per base classifier
        n_base_classifiers = len(self.base_classifiers)
        features_per_classifier = self.n_classes_ if self.use_probabilities else self.n_classes_
        
        classifier_importance = {}
        start_idx = 0
        
        for i, name in enumerate(self.classifier_names):
            end_idx = start_idx + features_per_classifier
            if end_idx <= len(feature_importance):
                importance = np.sum(feature_importance[start_idx:end_idx])
                classifier_importance[name] = importance
                start_idx = end_idx
        
        # Original features importance (if used)
        original_features_importance = 0.0
        if self.use_original_features and start_idx < len(feature_importance):
            original_features_importance = np.sum(feature_importance[start_idx:])
        
        return {
            'total_features': len(feature_importance),
            'base_classifier_importance': classifier_importance,
            'original_features_importance': original_features_importance,
            'feature_importance': feature_importance.tolist(),
            'most_important_classifier': max(classifier_importance.items(), key=lambda x: x[1])[0] if classifier_importance else None
        }
    
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
            'meta_learner': self.meta_learner,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_,
            'n_classes_': self.n_classes_,
            'original_features_shape_': self.original_features_shape_,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Stacking ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str) -> 'StackingEnsemble':
        """
        Load ensemble from disk.
        
        Args:
            filepath: Path to load the ensemble from
            
        Returns:
            Loaded StackingEnsemble instance
        """
        ensemble_data = joblib.load(filepath)
        
        # Create new instance
        ensemble = cls(ensemble_data['config'])
        
        # Restore state
        ensemble.base_classifiers = ensemble_data['base_classifiers']
        ensemble.classifier_names = ensemble_data['classifier_names']
        ensemble.meta_learner = ensemble_data['meta_learner']
        ensemble.label_encoder = ensemble_data['label_encoder']
        ensemble.classes_ = ensemble_data['classes_']
        ensemble.n_classes_ = ensemble_data['n_classes_']
        ensemble.original_features_shape_ = ensemble_data['original_features_shape_']
        ensemble.is_fitted = ensemble_data['is_fitted']
        
        logger.info(f"Stacking ensemble loaded from {filepath}")
        return ensemble


# Convenience function for creating stacking ensembles
def create_stacking_ensemble(base_classifiers: List[Tuple[str, str, Dict[str, Any]]],
                           meta_learner: Dict[str, Any] = None,
                           cv_folds: int = 5,
                           use_probabilities: bool = True,
                           use_original_features: bool = False) -> StackingEnsemble:
    """
    Convenience function to create a stacking ensemble.
    
    Args:
        base_classifiers: List of (name, classifier_type, config) tuples
        meta_learner: Meta-learner configuration
        cv_folds: Number of CV folds for meta-feature generation
        use_probabilities: Whether to use probability predictions
        use_original_features: Whether to include original features
        
    Returns:
        StackingEnsemble instance
    """
    if meta_learner is None:
        meta_learner = {'type': 'logistic_regression', 'config': {'C': 1.0}}
    
    config = {
        'base_classifiers': base_classifiers,
        'meta_learner': meta_learner,
        'cv_folds': cv_folds,
        'use_probabilities': use_probabilities,
        'use_original_features': use_original_features
    }
    
    return StackingEnsemble(config)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration for testing
    test_config = {
        'base_classifiers': [
            ('svm', 'svm', {'C': 1.0, 'kernel': 'rbf'}),
            ('rf', 'random_forest', {'n_estimators': 100}),
            ('lr', 'logistic_regression', {'C': 1.0})
        ],
        'meta_learner': {
            'type': 'logistic_regression',
            'config': {'C': 10.0, 'max_iter': 1000}
        },
        'cv_folds': 5,
        'use_probabilities': True,
        'use_original_features': False,
        'method': 'cv'
    }
    
    # Create ensemble
    ensemble = StackingEnsemble(test_config)
    print(f"Created stacking ensemble with {len(ensemble.base_classifiers)} base classifiers")
    print(f"Meta-learner: {ensemble.meta_learner_config['type']}")
    print(f"CV folds: {ensemble.cv_folds}")
    print(f"Use probabilities: {ensemble.use_probabilities}")
    print(f"Use original features: {ensemble.use_original_features}")


def get_algonauts_optimized_stacking_config() -> Dict[str, Any]:
    """
    Create optimized stacking configuration designed to outperform Algonauts baseline.
    
    Returns:
        Optimized stacking configuration for fMRI stimulus prediction
    """
    return {
        'base_classifiers': [
            # Diverse set of strong base learners
            ('svm_rbf', 'svm', {
                'C': 10.0, 
                'kernel': 'rbf', 
                'gamma': 'scale',
                'probability': True
            }),
            ('svm_linear', 'svm', {
                'C': 1.0, 
                'kernel': 'linear', 
                'probability': True
            }),
            ('rf_deep', 'random_forest', {
                'n_estimators': 200, 
                'max_depth': 20,
                'min_samples_split': 2,
                'max_features': 'sqrt',
                'bootstrap': True
            }),
            ('rf_shallow', 'random_forest', {
                'n_estimators': 100, 
                'max_depth': 10,
                'min_samples_split': 5,
                'max_features': 'log2'
            }),
            ('lr_l2', 'logistic_regression', {
                'C': 10.0, 
                'penalty': 'l2',
                'max_iter': 2000,
                'class_weight': 'balanced'
            }),
            ('lr_l1', 'logistic_regression', {
                'C': 1.0, 
                'penalty': 'l1',
                'solver': 'liblinear',
                'max_iter': 2000
            })
        ],
        'meta_learner': {
            'type': 'logistic_regression',
            'config': {
                'C': 10.0,
                'penalty': 'l2', 
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        },
        'cv_folds': 5,
        'use_probabilities': True,
        'use_original_features': False,  # Focus on meta-features for better generalization
        'method': 'cv'
    }


def create_neuroimaging_stacking_config() -> Dict[str, Any]:
    """
    Create stacking configuration specifically optimized for neuroimaging data.
    
    Returns:
        Neuroimaging-optimized stacking configuration
    """
    return {
        'base_classifiers': [
            ('svm_high_c', 'svm', {
                'C': 100.0,  # High regularization for high-dimensional data
                'kernel': 'linear',
                'probability': True,
                'cache_size': 1000
            }),
            ('rf_balanced', 'random_forest', {
                'n_estimators': 500,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
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
        'meta_learner': {
            'type': 'random_forest',  # Non-linear meta-learner
            'config': {
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 5,
                'random_state': 42
            }
        },
        'cv_folds': 10,  # More folds for robust meta-features
        'use_probabilities': True,
        'use_original_features': True,  # Include original features for neuroimaging
        'method': 'cv'
    }
