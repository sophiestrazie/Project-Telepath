# utils/metrics.py
"""
Evaluation metrics and performance assessment utilities for fMRI stimulus prediction.

This module provides comprehensive evaluation tools for multimodal fMRI classification
including accuracy metrics, confusion matrices, ROC analysis, and feature importance
statistical assessments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

class ModelEvaluator:
    """
    Comprehensive model evaluation class for fMRI stimulus classification.
    
    Provides methods for computing various classification metrics, generating
    performance visualizations, and conducting statistical analyses.
    """
    
    def __init__(self, class_labels: Optional[List[str]] = None):
        """
        Initialize the ModelEvaluator.
        
        Args:
            class_labels: Optional list of class labels for better visualization
        """
        self.class_labels = class_labels
        self.results_cache = {}
        
    def evaluate_classifier(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None,
                           classifier_name: str = "Unknown") -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single classifier.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            classifier_name: Name of the classifier for identification
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {
            'classifier_name': classifier_name,
            'basic_metrics': self._compute_basic_metrics(y_true, y_pred),
            'confusion_matrix': self._compute_confusion_matrix_metrics(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
        }
        
        # Add probabilistic metrics if probabilities are available
        if y_pred_proba is not None:
            results['roc_metrics'] = self._compute_roc_metrics(y_true, y_pred_proba)
            results['precision_recall_metrics'] = self._compute_precision_recall_metrics(
                y_true, y_pred_proba
            )
        
        # Cache results for comparison
        self.results_cache[classifier_name] = results
        
        return results
    
    def _compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
    
    def _compute_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute confusion matrix and related metrics."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Compute per-class metrics
        per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        
        return {
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_normalized': cm_normalized.tolist(),
            'per_class_accuracy': per_class_accuracy.tolist(),
            'support': np.sum(cm, axis=1).tolist()
        }
    
    def _compute_roc_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Compute ROC curves and AUC scores."""
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            return {
                'binary_roc': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': roc_auc
                },
                'auc_score': roc_auc
            }
        else:
            # Multi-class classification
            roc_curves = {}
            auc_scores = {}
            
            for i, class_label in enumerate(unique_classes):
                # One-vs-rest approach
                y_true_binary = (y_true == class_label).astype(int)
                y_score = y_pred_proba[:, i]
                
                fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                class_name = str(class_label)
                roc_curves[class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': roc_auc
                }
                auc_scores[class_name] = roc_auc
            
            # Compute macro and micro average AUC
            try:
                macro_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                micro_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='micro')
            except ValueError:
                macro_auc = np.mean(list(auc_scores.values()))
                micro_auc = macro_auc  # Fallback
            
            return {
                'multiclass_roc': roc_curves,
                'auc_scores': auc_scores,
                'macro_auc': macro_auc,
                'micro_auc': micro_auc
            }
    
    def _compute_precision_recall_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Compute precision-recall curves and average precision scores."""
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
            
            return {
                'binary_pr': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': thresholds.tolist(),
                    'average_precision': avg_precision
                }
            }
        else:
            # Multi-class classification
            pr_curves = {}
            ap_scores = {}
            
            for i, class_label in enumerate(unique_classes):
                y_true_binary = (y_true == class_label).astype(int)
                y_score = y_pred_proba[:, i]
                
                precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
                avg_precision = average_precision_score(y_true_binary, y_score)
                
                class_name = str(class_label)
                pr_curves[class_name] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': thresholds.tolist(),
                    'average_precision': avg_precision
                }
                ap_scores[class_name] = avg_precision
            
            return {
                'multiclass_pr': pr_curves,
                'ap_scores': ap_scores,
                'macro_ap': np.mean(list(ap_scores.values()))
            }
    
    def compare_classifiers(self, results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple classifiers based on their evaluation results.
        
        Args:
            results_dict: Dictionary mapping classifier names to their evaluation results
            
        Returns:
            DataFrame containing comparison metrics
        """
        comparison_data = []
        
        for clf_name, results in results_dict.items():
            basic_metrics = results.get('basic_metrics', {})
            
            row = {
                'Classifier': clf_name,
                'Accuracy': basic_metrics.get('accuracy', 0.0),
                'Precision (Macro)': basic_metrics.get('precision_macro', 0.0),
                'Recall (Macro)': basic_metrics.get('recall_macro', 0.0),
                'F1 (Macro)': basic_metrics.get('f1_macro', 0.0),
                'F1 (Weighted)': basic_metrics.get('f1_weighted', 0.0),
            }
            
            # Add AUC if available
            if 'roc_metrics' in results:
                if 'auc_score' in results['roc_metrics']:
                    row['AUC'] = results['roc_metrics']['auc_score']
                elif 'macro_auc' in results['roc_metrics']:
                    row['AUC'] = results['roc_metrics']['macro_auc']
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Accuracy', ascending=False)


def compute_classification_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Standalone function to compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities (optional)
        
    Returns:
        Dictionary containing all computed metrics
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_classifier(y_true, y_pred, y_pred_proba)


def compute_confusion_matrix(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           normalize: Optional[str] = None) -> np.ndarray:
    """
    Compute confusion matrix with optional normalization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        
    Returns:
        Confusion matrix as numpy array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0)
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
    
    return cm


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_labels: Optional[List[str]] = None,
                         normalize: Optional[str] = None,
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: Optional class labels for better visualization
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        title: Plot title
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    cm = compute_confusion_matrix(y_true, y_pred, normalize)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', ax=ax, 
                xticklabels=class_labels, yticklabels=class_labels)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    return fig


def compute_roc_curves(y_true: np.ndarray, 
                      y_pred_proba: np.ndarray) -> Dict[str, Any]:
    """
    Compute ROC curves for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Prediction probabilities
        
    Returns:
        Dictionary containing ROC curve data and AUC scores
    """
    evaluator = ModelEvaluator()
    return evaluator._compute_roc_metrics(y_true, y_pred_proba)


def compute_feature_importance_stats(feature_importance: np.ndarray,
                                   feature_names: Optional[List[str]] = None,
                                   top_k: int = 20) -> Dict[str, Any]:
    """
    Compute statistical analysis of feature importance scores.
    
    Args:
        feature_importance: Array of feature importance scores
        feature_names: Optional feature names
        top_k: Number of top features to return
        
    Returns:
        Dictionary containing feature importance statistics
    """
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
    
    # Basic statistics
    stats_dict = {
        'mean_importance': np.mean(feature_importance),
        'std_importance': np.std(feature_importance),
        'min_importance': np.min(feature_importance),
        'max_importance': np.max(feature_importance),
        'median_importance': np.median(feature_importance),
        'q25_importance': np.percentile(feature_importance, 25),
        'q75_importance': np.percentile(feature_importance, 75),
    }
    
    # Top features
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    top_features = {
        'indices': top_indices.tolist(),
        'names': [feature_names[i] for i in top_indices],
        'scores': feature_importance[top_indices].tolist()
    }
    
    # Statistical tests
    # Test for normality
    try:
        normality_stat, normality_p = stats.normaltest(feature_importance)
        stats_dict['normality_test'] = {
            'statistic': float(normality_stat),
            'p_value': float(normality_p),
            'is_normal': normality_p > 0.05
        }
    except Exception as e:
        warnings.warn(f"Could not perform normality test: {e}")
        stats_dict['normality_test'] = None
    
    return {
        'statistics': stats_dict,
        'top_features': top_features,
        'feature_count': len(feature_importance),
        'non_zero_features': np.sum(feature_importance != 0)
    }


def bootstrap_metric(y_true: np.ndarray, 
                    y_pred: np.ndarray,
                    metric_func: callable,
                    n_bootstrap: int = 1000,
                    confidence_level: float = 0.95,
                    random_state: int = 42) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for a given metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Metric function (e.g., accuracy_score)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing metric statistics and confidence intervals
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'mean': np.mean(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'median': np.median(bootstrap_scores),
        'confidence_interval': {
            'lower': np.percentile(bootstrap_scores, lower_percentile),
            'upper': np.percentile(bootstrap_scores, upper_percentile),
            'level': confidence_level
        },
        'bootstrap_scores': bootstrap_scores.tolist()
    }


def cross_validation_summary(cv_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for cross-validation scores.
    
    Args:
        cv_scores: Array of cross-validation scores
        
    Returns:
        Dictionary containing summary statistics
    """
    return {
        'mean': np.mean(cv_scores),
        'std': np.std(cv_scores),
        'min': np.min(cv_scores),
        'max': np.max(cv_scores),
        'median': np.median(cv_scores),
        'q25': np.percentile(cv_scores, 25),
        'q75': np.percentile(cv_scores, 75),
        'cv': np.std(cv_scores) / np.mean(cv_scores) if np.mean(cv_scores) != 0 else 0,  # Coefficient of variation
        'scores': cv_scores.tolist()
    }