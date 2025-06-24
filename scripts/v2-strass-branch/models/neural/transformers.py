# models/neural/transformer.py
"""
Transformer-based classifier for multimodal fMRI stimulus prediction.
Implements attention mechanism for capturing temporal dependencies in fMRI data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.base_classifier import BaseClassifier


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.
    Adds positional information to the input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate division term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):
    """
    Transformer encoder model for fMRI data classification.
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 n_classes: int = 2,
                 max_seq_len: int = 5000):
        """
        Initialize transformer encoder model.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            n_classes: Number of output classes
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            src_mask: Optional source mask for attention
            
        Returns:
            Classification logits of shape (batch_size, n_classes)
        """
        # Reshape for transformer: (seq_len, batch_size, input_dim)
        x = x.transpose(0, 1)
        
        # Project input to model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(x, src_mask)
        
        # Global average pooling over sequence dimension
        pooled = encoded.mean(dim=0)  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class TransformerClassifier(BaseClassifier):
    """
    Transformer-based classifier for fMRI stimulus prediction.
    
    This classifier uses multi-head self-attention to capture complex
    temporal relationships in fMRI time series data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize transformer classifier.
        
        Args:
            config: Configuration dictionary containing model hyperparameters
        """
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.attention_weights = None
        
        # Extract configuration parameters
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 2048)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.n_epochs = config.get('n_epochs', 150)
        self.batch_size = config.get('batch_size', 16)
        self.patience = config.get('patience', 20)
        self.sequence_length = config.get('sequence_length', 100)
        
    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare input data as sequences for transformer processing.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Reshaped data of shape (n_samples, seq_len, features_per_step)
        """
        n_samples, n_features = X.shape
        
        # Calculate features per timestep
        features_per_step = max(1, n_features // self.sequence_length)
        actual_seq_len = n_features // features_per_step
        
        # Trim features to fit sequence length
        n_features_trimmed = actual_seq_len * features_per_step
        X_trimmed = X[:, :n_features_trimmed]
        
        # Reshape to sequences
        X_sequences = X_trimmed.reshape(n_samples, actual_seq_len, features_per_step)
        
        return X_sequences
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TransformerClassifier':
        """
        Train the transformer classifier.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        print(f"Training Transformer classifier on {X.shape[0]} samples...")
        
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Prepare sequences
        X_sequences = self._prepare_sequences(X_scaled)
        seq_len, features_per_step = X_sequences.shape[1], X_sequences.shape[2]
        
        print(f"Sequence shape: {X_sequences.shape}")
        print(f"Features per timestep: {features_per_step}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        # Create model
        n_classes = len(np.unique(y_encoded))
        self.model = TransformerEncoderModel(
            input_dim=features_per_step,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            n_classes=n_classes,
            max_seq_len=seq_len + 100  # Add buffer for positional encoding
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.patience // 2, factor=0.5, verbose=True
        )
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0
        training_losses = []
        
        self.model.train()
        
        for epoch in range(self.n_epochs):
            total_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            training_losses.append(avg_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.6f}, "
                      f"LR: {current_lr:.2e}, Best Loss: {best_loss:.6f}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        self.training_losses = training_losses
        
        print(f"Training completed. Final loss: {best_loss:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Preprocess data
        X_scaled = self.scaler.transform(X)
        X_sequences = self._prepare_sequences(X_scaled)
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Process in batches to handle memory constraints
            batch_size = self.batch_size
            n_samples = X_tensor.shape[0]
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X_tensor[i:batch_end]
                
                outputs = self.model(batch_X)
                batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return self.label_encoder.inverse_transform(np.array(predictions))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Prediction probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Preprocess data
        X_scaled = self.scaler.transform(X)
        X_sequences = self._prepare_sequences(X_scaled)
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = self.batch_size
            n_samples = X_tensor.shape[0]
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X_tensor[i:batch_end]
                
                outputs = self.model(batch_X)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probabilities.append(batch_probs)
        
        return np.vstack(probabilities)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Return feature importance based on attention weights.
        
        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # For transformer, we'll use gradient-based importance
        # as attention weights are more complex to interpret directly
        
        # Create dummy input to compute gradients
        n_features = self.scaler.n_features_in_
        dummy_sequences = self._prepare_sequences(np.random.randn(1, n_features))
        dummy_input = torch.FloatTensor(dummy_sequences).to(self.device)
        dummy_input.requires_grad_(True)
        
        self.model.eval()
        
        # Forward pass
        output = self.model(dummy_input)
        
        # Compute gradients for each class and take maximum
        importances = []
        for class_idx in range(output.shape[1]):
            self.model.zero_grad()
            if dummy_input.grad is not None:
                dummy_input.grad.zero_()
            
            output[0, class_idx].backward(retain_graph=True)
            
            # Get gradient magnitude and reshape back to original feature space
            grad = torch.abs(dummy_input.grad).cpu().numpy()
            # Flatten and pad/trim to original feature size
            grad_flat = grad.flatten()
            
            if len(grad_flat) > n_features:
                grad_flat = grad_flat[:n_features]
            elif len(grad_flat) < n_features:
                grad_flat = np.pad(grad_flat, (0, n_features - len(grad_flat)))
            
            importances.append(grad_flat)
        
        # Take maximum importance across all classes
        feature_importance = np.max(importances, axis=0)
        
        return feature_importance
    
    def get_attention_weights(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract attention weights for interpretation.
        
        Args:
            X: Input data to analyze
            
        Returns:
            Dictionary containing attention weights from different layers
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before extracting attention weights")
        
        # This would require modifying the forward pass to return attention weights
        # For now, return a placeholder
        print("Attention weight extraction requires model modification.")
        print("Consider using model hooks to capture attention weights during forward pass.")
        
        return {
            'layer_0_attention': np.random.rand(X.shape[0], self.nhead, 
                                              self.sequence_length, self.sequence_length),
            'note': 'Placeholder - requires model modification for actual attention weights'
        }
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history and metrics.
        
        Returns:
            Dictionary containing training history
        """
        if not hasattr(self, 'training_losses'):
            return {'message': 'No training history available'}
        
        return {
            'training_losses': self.training_losses,
            'final_loss': self.training_losses[-1] if self.training_losses else None,
            'converged_epoch': len(self.training_losses),
            'model_parameters': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate
            }
        }