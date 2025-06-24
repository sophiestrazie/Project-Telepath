# models/neural/mlp.py
"""
Multi-Layer Perceptron (MLP) classifier for multimodal fMRI stimulus prediction.

This module implements a flexible MLP architecture with configurable layers,
dropout, batch normalization, and advanced training features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
from pathlib import Path

from models.base_classifier import BaseClassifier



class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron network with configurable architecture.
    
    This class implements the core MLP architecture with support for:
    - Variable number of hidden layers
    - Batch normalization
    - Dropout regularization
    - Residual connections (optional)
    - Multiple activation functions
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 n_classes: int,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 use_batch_norm: bool = True,
                 use_residual: bool = False):
        """
        Initialize MLP network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            n_classes: Number of output classes
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'gelu', 'swish')
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super(MLPNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.use_residual = use_residual
        
        # Activation function selection
        self.activation_fn = self._get_activation_function(activation)
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization for ReLU-like activations
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        residual = None
        
        for i, layer in enumerate(self.layers):
            # Store residual connection input
            if self.use_residual and i > 0 and x.shape[1] == layer.out_features:
                residual = x
            
            # Linear transformation
            x = layer(x)
            
            # Batch normalization
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation_fn(x)
            
            # Residual connection
            if self.use_residual and residual is not None and x.shape == residual.shape:
                x = x + residual
                residual = None
            
            # Dropout
            x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

class MLPClassifier(BaseClassifier):
    """
    Multi-Layer Perceptron classifier for fMRI stimulus prediction.
    
    This classifier provides a flexible MLP architecture with advanced training
    features including learning rate scheduling, early stopping, and gradient clipping.
    
    Args:
        config: Configuration dictionary containing model parameters
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model: Optional[MLPNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # Training history
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Configuration validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate and set default configuration parameters."""
        # Set default values first
        defaults = {
            'hidden_dims': [512, 256, 128],  # Reasonable default for fMRI data
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'batch_size': 32,
            'activation': 'relu',
            'use_batch_norm': True,
            'use_residual': False,
            'early_stopping_patience': 10,
            'grad_clip_norm': 1.0,
            'scheduler_type': 'plateau',
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'validation_split': 0.2,
            'random_state': 42
        }
        
        # Update defaults with any user-provided config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
        
        # Validate the types of critical parameters
        if not isinstance(self.config['hidden_dims'], list):
            raise ValueError("hidden_dims must be a list of integers")
        if len(self.config['hidden_dims']) == 0:
            raise ValueError("hidden_dims cannot be empty")