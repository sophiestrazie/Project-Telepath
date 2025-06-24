# data/preprocessors.py
"""
Data preprocessing module for multimodal fMRI stimulus prediction.

This module provides comprehensive preprocessing capabilities including
scaling, dimensionality reduction, feature selection, and data quality
control following SOLID principles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, VarianceThreshold,
    mutual_info_classif, f_classif, chi2
)
from sklearn.manifold import TSNE
from sklearn.utils.validation import check_X_y, check_array
import joblib

# Try to import advanced preprocessing tools
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.impute import SimpleImputer, KNNImputer
    IMPUTER_AVAILABLE = True
except ImportError:
    IMPUTER_AVAILABLE = False

# Ensure project imports work
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessing operations.
    
    Follows the Strategy pattern to enable different preprocessing approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BasePreprocessor':
        """
        Fit the preprocessor.
        
        Args:
            X: Input features
            y: Target labels (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        pass
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform the input data.
        
        Args:
            X: Input features
            y: Target labels (optional)
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)


class StandardizeData(BasePreprocessor):
    """
    Data standardization preprocessor with multiple scaling options.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize standardization preprocessor.
        
        Args:
            config: Configuration dictionary containing:
                - method: 'standard', 'minmax', or 'robust'
                - with_mean: Whether to center data (default: True)
                - with_std: Whether to scale to unit variance (default: True)
                - feature_range: Range for MinMaxScaler (default: (0, 1))
        """
        super().__init__(config)
        
        self.method = config.get('method', 'standard').lower()
        self.with_mean = config.get('with_mean', True)
        self.with_std = config.get('with_std', True)
        self.feature_range = config.get('feature_range', (0, 1))
        
        # Initialize scaler based on method
        if self.method == 'standard':
            self.scaler = StandardScaler(
                with_mean=self.with_mean,
                with_std=self.with_std
            )
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
        elif self.method == 'robust':
            self.scaler = RobustScaler(
                with_centering=self.with_mean,
                with_scaling=self.with_std
            )
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        logger.info(f"Initialized {self.method} scaler")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StandardizeData':
        """
        Fit the scaler to the data.
        
        Args:
            X: Input features
            y: Target labels (unused)
            
        Returns:
            Self for method chaining
        """
        X = check_array(X)
        
        logger.info(f"Fitting {self.method} scaler on data shape: {X.shape}")
        self.scaler.fit(X)
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using fitted scaler.
        
        Args:
            X: Input features
            
        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        X = check_array(X)
        return self.scaler.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation.
        
        Args:
            X: Scaled features
            
        Returns:
            Original scale features
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(X)


class DimensionalityReducer(BasePreprocessor):
    """
    Dimensionality reduction preprocessor supporting multiple algorithms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dimensionality reduction preprocessor.
        
        Args:
            config: Configuration dictionary containing:
                - method: 'pca', 'ica', 'svd', 'tsne', 'umap'
                - n_components: Number of components to keep
                - random_state: Random seed
                - Additional method-specific parameters
        """
        super().__init__(config)
        
        self.method = config.get('method', 'pca').lower()
        self.n_components = config.get('n_components', 100)
        self.random_state = config.get('random_state', 42)
        
        # Initialize reducer based on method
        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'ica':
            self.reducer = FastICA(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=config.get('max_iter', 200),
                tol=config.get('tol', 1e-4)
            )
        elif self.method == 'svd':
            self.reducer = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'tsne':
            self.reducer = TSNE(
                n_components=min(self.n_components, 3),  # t-SNE limited to 3D
                random_state=self.random_state,
                perplexity=config.get('perplexity', 30),
                learning_rate=config.get('learning_rate', 200)
            )
        elif self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            self.reducer = UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=config.get('n_neighbors', 15),
                min_dist=config.get('min_dist', 0.1)
            )
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.method}")
        
        logger.info(f"Initialized {self.method} reducer with {self.n_components} components")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DimensionalityReducer':
        """
        Fit the dimensionality reducer.
        
        Args:
            X: Input features
            y: Target labels (unused for most methods)
            
        Returns:
            Self for method chaining
        """
        X = check_array(X)
        
        logger.info(f"Fitting {self.method} on data shape: {X.shape}")
        
        # t-SNE doesn't have separate fit/transform
        if self.method == 'tsne':
            self.transformed_data_ = self.reducer.fit_transform(X)
        else:
            self.reducer.fit(X)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using fitted reducer.
        
        Args:
            X: Input features
            
        Returns:
            Reduced dimensionality features
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer must be fitted before transform")
        
        X = check_array(X)
        
        # t-SNE requires special handling
        if self.method == 'tsne':
            warnings.warn("t-SNE transform uses stored fit_transform result")
            return self.transformed_data_
        
        return self.reducer.transform(X)
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio for methods that support it.
        
        Returns:
            Explained variance ratio or None if not available
        """
        if hasattr(self.reducer, 'explained_variance_ratio_'):
            return self.reducer.explained_variance_ratio_
        return None
    
    def get_components(self) -> Optional[np.ndarray]:
        """
        Get components for methods that support it.
        
        Returns:
            Components matrix or None if not available
        """
        if hasattr(self.reducer, 'components_'):
            return self.reducer.components_
        return None


class FeatureSelector(BasePreprocessor):
    """
    Feature selection preprocessor with multiple selection strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature selection preprocessor.
        
        Args:
            config: Configuration dictionary containing:
                - method: 'variance', 'kbest', 'percentile'
                - score_func: Scoring function for kbest/percentile
                - k: Number of features for kbest (default: 1000)
                - percentile: Percentile for percentile selection (default: 10)
                - threshold: Variance threshold (default: 0.0)
        """
        super().__init__(config)
        
        self.method = config.get('method', 'variance').lower()
        self.k = config.get('k', 1000)
        self.percentile = config.get('percentile', 10)
        self.threshold = config.get('threshold', 0.0)
        
        # Get scoring function
        score_func_name = config.get('score_func', 'f_classif')
        if score_func_name == 'f_classif':
            self.score_func = f_classif
        elif score_func_name == 'mutual_info':
            self.score_func = mutual_info_classif
        elif score_func_name == 'chi2':
            self.score_func = chi2
        else:
            raise ValueError(f"Unknown score function: {score_func_name}")
        
        # Initialize selector based on method
        if self.method == 'variance':
            self.selector = VarianceThreshold(threshold=self.threshold)
        elif self.method == 'kbest':
            self.selector = SelectKBest(score_func=self.score_func, k=self.k)
        elif self.method == 'percentile':
            self.selector = SelectPercentile(
                score_func=self.score_func,
                percentile=self.percentile
            )
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        
        logger.info(f"Initialized {self.method} feature selector")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X: Input features
            y: Target labels (required for supervised methods)
            
        Returns:
            Self for method chaining
        """
        X = check_array(X)
        
        if self.method in ['kbest', 'percentile'] and y is None:
            raise ValueError(f"{self.method} requires target labels")
        
        logger.info(f"Fitting {self.method} feature selector on data shape: {X.shape}")
        
        if y is not None:
            X, y = check_X_y(X, y)
            self.selector.fit(X, y)
        else:
            self.selector.fit(X)
        
        self.is_fitted = True
        
        # Log number of selected features
        if hasattr(self.selector, 'get_support'):
            n_selected = np.sum(self.selector.get_support())
            logger.info(f"Selected {n_selected} features out of {X.shape[1]}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using fitted selector.
        
        Args:
            X: Input features
            
        Returns:
            Selected features
        """
        if not self.is_fitted:
            raise RuntimeError("Selector must be fitted before transform")
        
        X = check_array(X)
        return self.selector.transform(X)
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get selected feature indices or mask.
        
        Args:
            indices: Whether to return indices (True) or mask (False)
            
        Returns:
            Selected feature indices or boolean mask
        """
        if not self.is_fitted:
            raise RuntimeError("Selector must be fitted before get_support")
        
        return self.selector.get_support(indices=indices)
    
    def get_scores(self) -> Optional[np.ndarray]:
        """
        Get feature scores for methods that support it.
        
        Returns:
            Feature scores or None if not available
        """
        if hasattr(self.selector, 'scores_'):
            return self.selector.scores_
        return None


class DataQualityController(BasePreprocessor):
    """
    Data quality control and cleaning preprocessor.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data quality controller.
        
        Args:
            config: Configuration dictionary containing:
                - handle_missing: How to handle missing values ('drop', 'impute')
                - imputation_strategy: Strategy for imputation ('mean', 'median', 'knn')
                - outlier_detection: Whether to detect outliers
                - outlier_method: Method for outlier detection ('iqr', 'zscore')
                - outlier_threshold: Threshold for outlier detection
        """
        super().__init__(config)
        
        self.handle_missing = config.get('handle_missing', 'impute')
        self.imputation_strategy = config.get('imputation_strategy', 'mean')
        self.outlier_detection = config.get('outlier_detection', False)
        self.outlier_method = config.get('outlier_method', 'iqr')
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        
        # Initialize imputer if needed
        if self.handle_missing == 'impute':
            if not IMPUTER_AVAILABLE:
                warnings.warn("Advanced imputers not available. Using simple mean imputation.")
                from sklearn.preprocessing import Imputer
                self.imputer = Imputer(strategy=self.imputation_strategy)
            else:
                if self.imputation_strategy in ['mean', 'median', 'most_frequent']:
                    self.imputer = SimpleImputer(strategy=self.imputation_strategy)
                elif self.imputation_strategy == 'knn':
                    self.imputer = KNNImputer(n_neighbors=5)
                else:
                    raise ValueError(f"Unknown imputation strategy: {self.imputation_strategy}")
        
        # Statistics tracking
        self.missing_stats_ = {}
        self.outlier_stats_ = {}
        
        logger.info(f"Initialized data quality controller")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataQualityController':
        """
        Fit the data quality controller.
        
        Args:
            X: Input features
            y: Target labels (unused)
            
        Returns:
            Self for method chaining
        """
        X = check_array(X, force_all_finite=False)
        
        logger.info(f"Analyzing data quality for shape: {X.shape}")
        
        # Analyze missing values
        self._analyze_missing_values(X)
        
        # Fit imputer if needed
        if self.handle_missing == 'impute' and np.isnan(X).any():
            self.imputer.fit(X)
        
        # Analyze outliers
        if self.outlier_detection:
            self._analyze_outliers(X)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using quality control procedures.
        
        Args:
            X: Input features
            
        Returns:
            Cleaned features
        """
        if not self.is_fitted:
            raise RuntimeError("Controller must be fitted before transform")
        
        X = check_array(X, force_all_finite=False)
        
        # Handle missing values
        if self.handle_missing == 'impute' and np.isnan(X).any():
            logger.info("Imputing missing values")
            X = self.imputer.transform(X)
        elif self.handle_missing == 'drop' and np.isnan(X).any():
            logger.warning("Dropping samples with missing values")
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
        
        # Handle outliers
        if self.outlier_detection:
            X = self._handle_outliers(X)
        
        return X
    
    def _analyze_missing_values(self, X: np.ndarray) -> None:
        """Analyze missing value patterns."""
        missing_mask = np.isnan(X)
        
        self.missing_stats_ = {
            'total_missing': np.sum(missing_mask),
            'missing_percentage': np.sum(missing_mask) / X.size * 100,
            'samples_with_missing': np.sum(missing_mask.any(axis=1)),
            'features_with_missing': np.sum(missing_mask.any(axis=0)),
            'missing_per_feature': np.sum(missing_mask, axis=0),
            'missing_per_sample': np.sum(missing_mask, axis=1)
        }
        
        logger.info(f"Missing values: {self.missing_stats_['missing_percentage']:.2f}%")
    
    def _analyze_outliers(self, X: np.ndarray) -> None:
        """Analyze outlier patterns."""
        if self.outlier_method == 'zscore':
            z_scores = np.abs((X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0))
            outlier_mask = z_scores > self.outlier_threshold
        elif self.outlier_method == 'iqr':
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (X < lower_bound) | (X > upper_bound)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.outlier_method}")
        
        self.outlier_stats_ = {
            'total_outliers': np.sum(outlier_mask),
            'outlier_percentage': np.sum(outlier_mask) / X.size * 100,
            'samples_with_outliers': np.sum(outlier_mask.any(axis=1)),
            'features_with_outliers': np.sum(outlier_mask.any(axis=0))
        }
        
        logger.info(f"Outliers detected: {self.outlier_stats_['outlier_percentage']:.2f}%")
    
    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """Handle outliers in the data."""
        if self.outlier_method == 'zscore':
            z_scores = np.abs((X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0))
            outlier_mask = z_scores > self.outlier_threshold
        elif self.outlier_method == 'iqr':
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (X < lower_bound) | (X > upper_bound)
        
        # Clip outliers to bounds
        X_clean = X.copy()
        if self.outlier_method == 'iqr':
            X_clean = np.clip(X_clean, lower_bound, upper_bound)
        elif self.outlier_method == 'zscore':
            # Clip to mean ± threshold * std
            mean_vals = np.nanmean(X, axis=0)
            std_vals = np.nanstd(X, axis=0)
            lower_bound = mean_vals - self.outlier_threshold * std_vals
            upper_bound = mean_vals + self.outlier_threshold * std_vals
            X_clean = np.clip(X_clean, lower_bound, upper_bound)
        
        return X_clean
    
    def get_quality_report(self) -> Dict[str, Any]:
        """
        Get comprehensive data quality report.
        
        Returns:
            Dictionary containing quality statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Controller must be fitted before getting report")
        
        return {
            'missing_stats': self.missing_stats_,
            'outlier_stats': self.outlier_stats_,
            'config': self.config
        }


class TemporalPreprocessor(BasePreprocessor):
    """
    Temporal preprocessing for fMRI time series data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize temporal preprocessor.
        
        Args:
            config: Configuration dictionary containing:
                - window_size: Size of sliding window
                - hop_size: Step size for sliding window
                - aggregation: Aggregation method ('mean', 'max', 'concat')
                - detrend: Whether to detrend signals
                - highpass_filter: High-pass filter cutoff frequency
                - lowpass_filter: Low-pass filter cutoff frequency
        """
        super().__init__(config)
        
        self.window_size = config.get('window_size', 10)
        self.hop_size = config.get('hop_size', 1)
        self.aggregation = config.get('aggregation', 'mean')
        self.detrend = config.get('detrend', False)
        self.highpass_filter = config.get('highpass_filter', None)
        self.lowpass_filter = config.get('lowpass_filter', None)
        
        logger.info(f"Initialized temporal preprocessor with window_size={self.window_size}")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TemporalPreprocessor':
        """
        Fit the temporal preprocessor.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (unused)
            
        Returns:
            Self for method chaining
        """
        X = check_array(X)
        
        logger.info(f"Fitting temporal preprocessor on data shape: {X.shape}")
        
        # Store data characteristics
        self.n_features_ = X.shape[1]
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the temporal data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        X = check_array(X)
        
        # Apply detrending if requested
        if self.detrend:
            X = self._detrend_signals(X)
        
        # Apply filtering if requested
        if self.highpass_filter or self.lowpass_filter:
            X = self._apply_filters(X)
        
        # Create sliding windows
        X_windowed = self._create_sliding_windows(X)
        
        return X_windowed
    
    def _detrend_signals(self, X: np.ndarray) -> np.ndarray:
        """Remove linear trends from signals."""
        from scipy import signal
        
        X_detrended = np.zeros_like(X)
        for i in range(X.shape[1]):
            X_detrended[:, i] = signal.detrend(X[:, i])
        
        return X_detrended
    
    def _apply_filters(self, X: np.ndarray) -> np.ndarray:
        """Apply temporal filters to signals."""
        try:
            from scipy import signal
        except ImportError:
            logger.warning("SciPy not available. Skipping filtering.")
            return X
        
        X_filtered = X.copy()
        
        # Apply high-pass filter
        if self.highpass_filter:
            sos = signal.butter(4, self.highpass_filter, btype='high', output='sos')
            for i in range(X.shape[1]):
                X_filtered[:, i] = signal.sosfilt(sos, X_filtered[:, i])
        
        # Apply low-pass filter
        if self.lowpass_filter:
            sos = signal.butter(4, self.lowpass_filter, btype='low', output='sos')
            for i in range(X.shape[1]):
                X_filtered[:, i] = signal.sosfilt(sos, X_filtered[:, i])
        
        return X_filtered
    
    def _create_sliding_windows(self, X: np.ndarray) -> np.ndarray:
        """Create sliding windows from temporal data."""
        n_samples, n_features = X.shape
        
        # Calculate number of windows
        n_windows = (n_samples - self.window_size) // self.hop_size + 1
        
        if n_windows <= 0:
            raise ValueError(f"Window size {self.window_size} too large for data with {n_samples} samples")
        
        # Create windows
        if self.aggregation == 'concat':
            # Concatenate all time points in window
            output_features = n_features * self.window_size
            X_windowed = np.zeros((n_windows, output_features))
            
            for i in range(n_windows):
                start_idx = i * self.hop_size
                end_idx = start_idx + self.window_size
                window_data = X[start_idx:end_idx, :]
                X_windowed[i, :] = window_data.flatten()
        
        else:
            # Aggregate window contents
            X_windowed = np.zeros((n_windows, n_features))
            
            for i in range(n_windows):
                start_idx = i * self.hop_size
                end_idx = start_idx + self.window_size
                window_data = X[start_idx:end_idx, :]
                
                if self.aggregation == 'mean':
                    X_windowed[i, :] = np.mean(window_data, axis=0)
                elif self.aggregation == 'max':
                    X_windowed[i, :] = np.max(window_data, axis=0)
                elif self.aggregation == 'min':
                    X_windowed[i, :] = np.min(window_data, axis=0)
                elif self.aggregation == 'std':
                    X_windowed[i, :] = np.std(window_data, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        logger.info(f"Created {n_windows} windows with shape {X_windowed.shape}")
        return X_windowed


class FMRIPreprocessor:
    """
    Main preprocessing pipeline for fMRI data.
    
    Orchestrates multiple preprocessing steps in the correct order.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fMRI preprocessor pipeline.
        
        Args:
            config: Configuration dictionary containing preprocessing steps
        """
        self.config = config
        self.pipeline_steps = []
        self.is_fitted = False
        
        # Build preprocessing pipeline based on config
        self._build_pipeline()
        
        logger.info(f"Initialized fMRI preprocessing pipeline with {len(self.pipeline_steps)} steps")
    
    def _build_pipeline(self) -> None:
        """Build preprocessing pipeline from configuration."""
        
        # Data quality control (first step)
        if 'quality_control' in self.config:
            quality_controller = DataQualityController(self.config['quality_control'])
            self.pipeline_steps.append(('quality_control', quality_controller))
        
        # Temporal preprocessing
        if 'temporal' in self.config:
            temporal_processor = TemporalPreprocessor(self.config['temporal'])
            self.pipeline_steps.append(('temporal', temporal_processor))
        
        # Standardization
        if 'standardization' in self.config:
            standardizer = StandardizeData(self.config['standardization'])
            self.pipeline_steps.append(('standardization', standardizer))
        
        # Feature selection
        if 'feature_selection' in self.config:
            feature_selector = FeatureSelector(self.config['feature_selection'])
            self.pipeline_steps.append(('feature_selection', feature_selector))
        
        # Dimensionality reduction (last step)
        if 'dimensionality_reduction' in self.config:
            dim_reducer = DimensionalityReducer(self.config['dimensionality_reduction'])
            self.pipeline_steps.append(('dimensionality_reduction', dim_reducer))
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'FMRIPreprocessor':
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Self for method chaining
        """
        X_current = X.copy()
        
        logger.info(f"Fitting preprocessing pipeline on data shape: {X.shape}")
        
        for step_name, processor in self.pipeline_steps:
            logger.info(f"Fitting {step_name}")
            processor.fit(X_current, y)
            X_current = processor.transform(X_current)
            logger.info(f"  Output shape: {X_current.shape}")
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        X_current = X.copy()
        
        for step_name, processor in self.pipeline_steps:
            X_current = processor.transform(X_current)
        
        return X_current
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing pipeline.
        
        Returns:
            Dictionary containing pipeline information
        """
        info = {
            'n_steps': len(self.pipeline_steps),
            'steps': [step_name for step_name, _ in self.pipeline_steps],
            'is_fitted': self.is_fitted
        }
        
        # Add step-specific information if fitted
        if self.is_fitted:
            for step_name, processor in self.pipeline_steps:
                if hasattr(processor, 'get_quality_report'):
                    info[f'{step_name}_report'] = processor.get_quality_report()
                elif hasattr(processor, 'get_explained_variance_ratio'):
                    variance_ratio = processor.get_explained_variance_ratio()
                    if variance_ratio is not None:
                        info[f'{step_name}_variance_explained'] = np.sum(variance_ratio)
        
        return info
    
    def save_pipeline(self, filepath: str) -> None:
        """
        Save the fitted pipeline.
        
        Args:
            filepath: Path to save the pipeline
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'config': self.config,
            'pipeline_steps': self.pipeline_steps,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str) -> 'FMRIPreprocessor':
        """
        Load a fitted pipeline.
        
        Args:
            filepath: Path to load the pipeline from
            
        Returns:
            Loaded FMRIPreprocessor instance
        """
        pipeline_data = joblib.load(filepath)
        
        instance = cls(pipeline_data['config'])
        instance.pipeline_steps = pipeline_data['pipeline_steps']
        instance.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"Pipeline loaded from {filepath}")
        return instance


# Convenience functions for creating common preprocessing configurations
def get_default_fmri_config() -> Dict[str, Any]:
    """
    Get default preprocessing configuration for fMRI data.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'quality_control': {
            'handle_missing': 'impute',
            'imputation_strategy': 'mean',
            'outlier_detection': True,
            'outlier_method': 'iqr'
        },
        'standardization': {
            'method': 'standard',
            'with_mean': True,
            'with_std': True
        },
        'feature_selection': {
            'method': 'variance',
            'threshold': 0.01
        },
        'dimensionality_reduction': {
            'method': 'pca',
            'n_components': 1000
        }
    }


def get_algonauts_optimized_config() -> Dict[str, Any]:
    """
    Get preprocessing configuration optimized for Algonauts competition.
    
    Returns:
        Optimized configuration dictionary
    """
    return {
        'quality_control': {
            'handle_missing': 'impute',
            'imputation_strategy': 'knn',
            'outlier_detection': True,
            'outlier_method': 'iqr'
        },
        'standardization': {
            'method': 'robust',
            'with_mean': True,
            'with_std': True
        },
        'feature_selection': {
            'method': 'kbest',
            'score_func': 'f_classif',
            'k': 2000
        },
        'dimensionality_reduction': {
            'method': 'pca',
            'n_components': 500
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    X_sample = np.random.randn(100, 1000)
    y_sample = np.random.randint(0, 5, 100)
    
    # Test preprocessing pipeline
    config = get_default_fmri_config()
    preprocessor = FMRIPreprocessor(config)
    
    X_transformed = preprocessor.fit_transform(X_sample, y_sample)
    
    print(f"Original shape: {X_sample.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Pipeline info: {preprocessor.get_pipeline_info()}")
    
    print("Preprocessing module test completed")