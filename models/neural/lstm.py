# models/neural/lstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.base_classifier import BaseClassifier


class LSTMClassifier(BaseClassifier):
    """Long Short-Term Memory classifier for fMRI time series data using PyTorch"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_sequences(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences from fMRI data for LSTM processing"""
        n_samples, n_features = X.shape
        
        # If we have fewer time points than sequence length, pad or adjust
        if n_samples < sequence_length:
            # Pad with zeros or repeat the sequence
            pad_length = sequence_length - n_samples
            X_padded = np.pad(X, ((0, pad_length), (0, 0)), mode='edge')
            return X_padded.reshape(1, sequence_length, n_features)
        
        # Create sliding window sequences
        sequences = []
        for i in range(n_samples - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        
        return np.array(sequences)
    
    def _create_model(self, input_size: int, n_classes: int) -> nn.Module:
        """Create LSTM model"""
        hidden_size = self.config.get('hidden_size', 128)
        num_layers = self.config.get('num_layers', 2)
        dropout_rate = self.config.get('dropout_rate', 0.3)
        bidirectional = self.config.get('bidirectional', True)
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, n_classes, 
                        dropout_rate, bidirectional):
                super(LSTMModel, self).__init__()
                
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                
                # LSTM layer
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout_rate if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )
                
                # Attention mechanism (optional)
                self.use_attention = True
                if self.use_attention:
                    lstm_output_size = hidden_size * (2 if bidirectional else 1)
                    self.attention = nn.Sequential(
                        nn.Linear(lstm_output_size, lstm_output_size // 2),
                        nn.Tanh(),
                        nn.Linear(lstm_output_size // 2, 1),
                        nn.Softmax(dim=1)
                    )
                
                # Classifier layers
                classifier_input_size = hidden_size * (2 if bidirectional else 1)
                
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(classifier_input_size, classifier_input_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(classifier_input_size // 2, classifier_input_size // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(classifier_input_size // 4, n_classes)
                )
                
            def forward(self, x):
                # LSTM forward pass
                lstm_out, (hidden, cell) = self.lstm(x)
                
                if self.use_attention:
                    # Apply attention mechanism
                    attention_weights = self.attention(lstm_out)
                    # Weighted sum of LSTM outputs
                    attended_output = torch.sum(lstm_out * attention_weights, dim=1)
                else:
                    # Use last output or average pooling
                    if self.bidirectional:
                        # Concatenate forward and backward final states
                        attended_output = torch.cat([hidden[-2], hidden[-1]], dim=1)
                    else:
                        attended_output = hidden[-1]
                
                # Classification
                output = self.classifier(attended_output)
                return output
        
        return LSTMModel(input_size, hidden_size, num_layers, n_classes, 
                        dropout_rate, bidirectional)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMClassifier':
        """Train LSTM classifier"""
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create sequences
        sequence_length = self.config.get('sequence_length', min(50, X_scaled.shape[0] // 2))
        X_sequences = self._create_sequences(X_scaled, sequence_length)
        
        # For each sequence, we need a corresponding label
        # Use the label from the last time point of each sequence
        if X_sequences.shape[0] > 1:
            y_sequences = y_encoded[sequence_length-1:]
        else:
            # If only one sequence, use the most common label
            y_sequences = np.array([np.bincount(y_encoded).argmax()])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        y_tensor = torch.LongTensor(y_sequences).to(self.device)
        
        # Create model
        input_size = X_sequences.shape[2]  # Number of features
        n_classes = len(np.unique(y_encoded))
        self.model = self._create_model(input_size, n_classes).to(self.device)
        self.sequence_length = sequence_length
        
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
        batch_size = min(self.config.get('batch_size', 16), len(dataset))
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
        early_stopping_patience = self.config.get('early_stopping_patience', 15)
        
        for epoch in range(n_epochs):
            total_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping for RNNs
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
            
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled, self.sequence_length)
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        
        self.model.eval()
        predictions = []
        
        # Process in batches
        batch_size = self.config.get('inference_batch_size', 32)
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = self.model(batch)
                batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        predictions = np.array(predictions)
        
        # If we created multiple sequences from the input, 
        # we need to map back to original sample predictions
        if len(predictions) == 1:
            # Single sequence case - repeat prediction for all original samples
            return np.full(X.shape[0], 
                          self.label_encoder.inverse_transform([predictions[0]])[0])
        else:
            # Multiple sequences - extend predictions to cover all samples
            extended_predictions = np.zeros(X.shape[0], dtype=predictions.dtype)
            extended_predictions[:self.sequence_length-1] = predictions[0]  # First samples
            extended_predictions[self.sequence_length-1:] = predictions  # Remaining samples
            
            return self.label_encoder.inverse_transform(extended_predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled, self.sequence_length)
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        
        self.model.eval()
        probabilities = []
        
        # Process in batches
        batch_size = self.config.get('inference_batch_size', 32)
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = self.model(batch)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probabilities.append(batch_probs)
        
        probabilities = np.vstack(probabilities)
        
        # Handle mapping back to original samples
        if len(probabilities) == 1:
            # Single sequence case
            return np.tile(probabilities, (X.shape[0], 1))
        else:
            # Multiple sequences case
            extended_probs = np.zeros((X.shape[0], probabilities.shape[1]))
            extended_probs[:self.sequence_length-1] = probabilities[0]  # First samples
            extended_probs[self.sequence_length-1:] = probabilities  # Remaining samples
            
            return extended_probs
    
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance using attention weights and gradients"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # Create dummy input sequence
        dummy_input = torch.randn(1, self.sequence_length, 
                                 self.scaler.n_features_in_).to(self.device)
        dummy_input.requires_grad_(True)
        
        self.model.eval()
        output = self.model(dummy_input)
        
        # Use max output for gradient computation
        max_output = output.max()
        max_output.backward()
        
        # Get gradients across time and features
        gradients = dummy_input.grad.abs().cpu().numpy()
        
        # Average across time dimension to get feature importance
        feature_importance = np.mean(gradients, axis=(0, 1))
        
        return feature_importance