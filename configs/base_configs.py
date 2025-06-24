# config/base_config.py
"""
Base configuration module for multimodal fMRI stimulus prediction.

This module provides core configuration settings that are shared across
different components of the system, following SOLID principles.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseConfig:
    """
    Base configuration class containing shared settings.
    
    This class follows the Single Responsibility Principle by managing
    only configuration-related functionality.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize base configuration.
        
        Args:
            config_dict: Optional dictionary to override default settings
        """
        self.config = self._get_default_config()
        if config_dict:
            self.config.update(config_dict)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            # Data paths
            'data_path': 'data/raw',
            'output_path': 'results',
            'cache_path': 'cache',
            
            # General settings
            'random_state': 42,
            'n_jobs': -1,
            'verbose': True,
            
            # Cross-validation settings
            'cv_folds': 5,
            'test_size': 0.2,
            'validation_size': 0.2,
            
            # Preprocessing settings
            'standardize_features': True,
            'handle_missing_values': True,
            'missing_value_strategy': 'mean',
            
            # fMRI specific settings
            'tr': 1.49,  # Repetition time in seconds
            'hrf_delay': 3,  # HRF delay in TRs
            'excluded_samples_start': 5,
            'excluded_samples_end': 5,
            
            # Feature extraction settings
            'pca_components': 1000,
            'feature_selection': True,
            'feature_selection_k': 'all',
            
            # Model evaluation
            'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1'],
            'primary_metric': 'accuracy',
            
            # Logging
            'log_level': 'INFO',
            'log_file': 'fmri_analysis.log',
            
            # Performance
            'use_gpu': True,
            'gpu_device': 'cuda:0',
            'batch_size': 32,
            'num_workers': 4,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            config_dict: Dictionary of new configuration values
        """
        self.config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate paths
        data_path = Path(self.config['data_path'])
        if not data_path.exists():
            logger.warning(f"Data path does not exist: {data_path}")
        
        # Validate numeric parameters
        if self.config['cv_folds'] < 2:
            raise ValueError("cv_folds must be >= 2")
        
        if not 0 < self.config['test_size'] < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if self.config['tr'] <= 0:
            raise ValueError("tr must be positive")
        
        if self.config['hrf_delay'] < 0:
            raise ValueError("hrf_delay must be non-negative")
        
        return True
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.config['output_path'],
            self.config['cache_path'],
            'logs',
            'models',
            'figures'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")


class EnvironmentConfig:
    """
    Environment-specific configuration management.
    
    Handles different configurations for development, testing, and production.
    """
    
    def __init__(self, environment: str = 'development'):
        """
        Initialize environment configuration.
        
        Args:
            environment: Environment name ('development', 'testing', 'production')
        """
        self.environment = environment
        self.config = self._get_environment_config()
    
    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        configs = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'use_cache': True,
                'strict_validation': False,
                'n_jobs': 2,  # Limit parallelization for debugging
            },
            'testing': {
                'debug': False,
                'log_level': 'INFO',
                'use_cache': False,
                'strict_validation': True,
                'random_state': 42,  # Fixed for reproducibility
            },
            'production': {
                'debug': False,
                'log_level': 'WARNING',
                'use_cache': True,
                'strict_validation': True,
                'optimize_performance': True,
            }
        }
        
        return configs.get(self.environment, configs['development'])
    
    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config.copy()


def get_config(environment: Optional[str] = None, 
               custom_config: Optional[Dict[str, Any]] = None) -> BaseConfig:
    """
    Factory function to create configuration instance.
    
    Args:
        environment: Environment name
        custom_config: Custom configuration overrides
        
    Returns:
        BaseConfig instance
    """
    # Start with base configuration
    base_config = BaseConfig()
    
    # Apply environment-specific settings
    if environment:
        env_config = EnvironmentConfig(environment)
        base_config.update(env_config.get_config())
    
    # Apply custom overrides
    if custom_config:
        base_config.update(custom_config)
    
    # Validate and setup
    base_config.validate()
    base_config.setup_directories()
    
    return base_config