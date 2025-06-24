# models/neural/cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math

from models.base_classifier import BaseClassifier


class CNNClassifier(BaseClassifier):
    """Convolutional Neural Network classifier for fMRI data using PyTorch"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _reshape_for_cnn(self, X: np.ndarray) -> np.ndarray:
        """Reshape 2D fMRI data to pseudo-3D for CNN processing"""
        n_samples, n_features = X.shape
        
        # Create pseudo-spatial dimensions
        # Assume voxels can be arranged in a cubic-like structure
        spatial_dim = int(math.ceil(math.sqrt(n_features)))
        
        # Pad if necessary to make it square
        if spatial_dim ** 2 != n_features:
            pad_size = spatial_dim ** 2 - n_features
            X_padded = np.pad(X, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        else:
            X_padded = X
        
        # Reshape to (batch_size, 1, spatial_dim, spatial_dim)
        X_reshaped = X_padded.reshape(n_samples, 1, spatial_dim, spatial_dim)
        
        return X_reshaped, (spatial_dim, spatial_dim)
    
    def _create_model(self, input_shape: Tuple[int, int], n_classes: int) -> nn.Module:
        """Create CNN model"""
        n_filters = self.config.get('n_filters', [64, 128, 256])
        kernel_size = self.config.get('kernel_size', 3)
        dropout_rate = self.config.get('dropout_rate', 0.5)
        
        class CNNModel(nn.Module):
            def __init__(self, input_shape, n_filters, kernel_size, dropout_rate, n_classes):
                super(CNNModel, self).__init__()
                
                # Convolutional layers
                self.conv_layers = nn.ModuleList()
                in_channels = 1
                
                for out_channels in n_filters:
                    self.conv_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Dropout2d(dropout_rate * 0.5)  # Lower dropout for conv layers
                    ))
                    in_channels = out_channels
                
                # Calculate size after convolutions
                h, w = input_shape
                for _ in n_filters:
                    h = h // 2  # MaxPool2d reduces by factor of 2
                    w = w // 2
                
                # Fully connected layers
                fc_input_size = n_filters[-1] * max(1, h) * max(1, w)
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(fc_input_size, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, n_classes)
                )
                
            def forward(self, x):
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
                x = self.classifier(x)
                return x
        
        return CNNModel(input_shape, n_filters, kernel_size, dropout_rate, n_classes)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CNNClassifier':
        """Train CNN classifier"""
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Reshape for CNN
        X_reshaped, self.spatial_shape = self._reshape_for_cnn(X_scaled)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        # Create model
        n_classes = len(np.unique(y_encoded))
        self.model = self._create_model(self.spatial_shape, n_classes).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = min(self.config.get('batch_size', 32), len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # Training loop
        n_epochs = self.config.get('n_epochs', 100)
        self.model.train()
        
        best_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = self.config.get('early_stopping_patience', 20)
        
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_reshaped, _ = self._reshape_for_cnn(X_scaled)
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        
        self.model.eval()
        predictions = []
        
        # Process in batches to handle memory constraints
        batch_size = self.config.get('inference_batch_size', 64)
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = self.model(batch)
                batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return self.label_encoder.inverse_transform(np.array(predictions))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_reshaped, _ = self._reshape_for_cnn(X_scaled)
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        
        self.model.eval()
        probabilities = []
        
        # Process in batches to handle memory constraints
        batch_size = self.config.get('inference_batch_size', 64)
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = self.model(batch)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probabilities.append(batch_probs)
        
        return np.vstack(probabilities)
    
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance using gradient-based method"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # Create dummy input with spatial shape
        dummy_input = torch.randn(1, 1, *self.spatial_shape).to(self.device)
        dummy_input.requires_grad_(True)
        
        self.model.eval()
        output = self.model(dummy_input)
        
        # Use max output for gradient computation
        max_output = output.max()
        max_output.backward()
        
        # Get gradients and flatten to match original feature space
        gradients = dummy_input.grad.abs().squeeze().cpu().numpy()
        
        # Flatten and trim to original feature size
        gradients_flat = gradients.flatten()
        original_size = self.scaler.n_features_in_
        
        return gradients_flat[:original_size]