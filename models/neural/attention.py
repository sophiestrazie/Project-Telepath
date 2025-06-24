# models/neural/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.base_classifier import BaseClassifier

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for fMRI data"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, seq_len, seq_len)
        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear transformation
        output = self.w_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_output, attn_weights = self.attention(x, mask)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(ff_output + attn_output)
        
        return output, attn_weights

class SelfAttentionClassifier(BaseClassifier):
    """Self-attention based classifier for fMRI data"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention_weights = None
        
    def _create_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Create self-attention model"""
        d_model = self.config.get('d_model', 256)
        n_heads = self.config.get('n_heads', 8)
        n_layers = self.config.get('n_layers', 6)
        d_ff = self.config.get('d_ff', 1024)
        dropout = self.config.get('dropout_rate', 0.1)
        max_seq_len = self.config.get('max_seq_len', 1000)
        
        return SelfAttentionNetwork(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            n_classes=n_classes,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

class SelfAttentionNetwork(nn.Module):
    """Complete self-attention network for fMRI classification"""
    
    def __init__(self, input_dim: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, n_classes: int, dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        """
        if len(x.shape) == 2:
            # Reshape from (batch_size, input_dim) to (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Store attention weights
        all_attention_weights = []
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            if return_attention:
                all_attention_weights.append(attn_weights)
        
        # Global pooling over sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)
        
        if return_attention:
            return logits, all_attention_weights
        return logits

class MLPClassifier(BaseClassifier):
    """Multi-Layer Perceptron classifier using PyTorch"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Create MLP model"""
        hidden_dims = self.config.get('hidden_dims', [512, 256, 128])
        dropout_rate = self.config.get('dropout_rate', 0.3)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, n_classes))
        
        return nn.Sequential(*layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifier':
        """Train MLP classifier"""
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        # Create model
        input_dim = X_scaled.shape[1]
        n_classes = len(np.unique(y_encoded))
        self.model = self._create_model(input_dim, n_classes).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        # Training loop
        n_epochs = self.config.get('n_epochs', 100)
        self.model.train()
        
        for epoch in range(n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
        return probabilities
    
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance (gradient-based)"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
            
        # Use gradient-based feature importance
        dummy_input = torch.randn(1, self.scaler.n_features_in_).to(self.device)
        dummy_input.requires_grad_(True)
        
        self.model.eval()
        output = self.model(dummy_input)
        output.backward(torch.ones_like(output))
        
        importance = torch.abs(dummy_input.grad).mean(dim=0).cpu().numpy()
        return importance