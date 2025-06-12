# Flexible Multi-Modal Classifier Architecture for fMRI Prediction

A comprehensive, extensible framework for experimenting with different pretrained classifiers on multimodal fMRI and stimulus data. This architecture provides a unified interface for Vision Transformers, CNNs, and multimodal fusion models.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for larger models

### Installation

```powershell
# Create and activate virtual environment
python -m venv brain-env
.\brain-env\Scripts\Activate.ps1

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Basic Usage

```python
from multimodal_stimulus_fmri_predict.core.classifier_factory import ClassifierFactory
from multimodal_stimulus_fmri_predict.utils.experiment_runner import ExperimentRunner

# Create a Vision Transformer classifier
config = {
    'pretrained': True,
    'image_size': 224,
    'num_classes': 2,
    'learning_rate': 1e-4
}
classifier = ClassifierFactory.create_classifier('vit', config)

# Train and evaluate
history = classifier.train(train_loader, val_loader, epochs=10)
test_loss, test_acc = classifier.evaluate(test_loader)
```

## 🏗️ Architecture Overview

### Core Components

1. **Base Classifier** (`base_classifier.py`)
   - Abstract base class defining the common interface
   - Handles training, evaluation, and prediction workflows
   - Device management and logging

2. **Classifier Factory** (`classifier_factory.py`)
   - Central registry for all classifier types
   - Instantiates classifiers based on configuration
   - Extensible design for adding new models

3. **Experiment Runner** (`experiment_runner.py`)
   - Orchestrates multiple experiments
   - Automated result collection and analysis
   - Saves detailed logs and metrics

### Supported Classifiers

| Classifier | Description | Key Features |
|------------|-------------|--------------|
| **Vision Transformer (ViT)** | Transformer-based image classifier | - Pretrained on ImageNet<br>- Configurable patch sizes<br>- Attention mechanisms |
| **ResNet** | Residual CNN architecture | - Multiple variants (18, 34, 50, 101)<br>- Skip connections<br>- Proven performance |
| **EfficientNet** | Efficient CNN with compound scaling | - Optimized for efficiency<br>- State-of-the-art accuracy<br>- Multiple model sizes |
| **MultiModal Fusion** | Combines image and fMRI data | - Late fusion architecture<br>- Configurable encoders<br>- Cross-modal learning |

## 📊 Experiment Configuration

### Single Classifier Example

```python
# Vision Transformer configuration
vit_config = {
    'classifier_type': 'vit',
    'classifier_config': {
        'pretrained': True,
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 2,
        'learning_rate': 1e-4
    },
    'training_config': {'epochs': 20}
}
```

### Batch Experiments

```python
# Run multiple configurations automatically
from multimodal_stimulus_fmri_predict.configs.experiment_configs import (
    get_vit_configs, get_resnet_configs, get_multimodal_configs
)

all_configs = get_vit_configs() + get_resnet_configs() + get_multimodal_configs()
runner = ExperimentRunner(results_dir="experiments")
results = runner.run_multiple_experiments(all_configs, train_loader, val_loader, test_loader)
```

## 🔧 Customization & Extension

### Adding New Classifiers

1. **Create Classifier Class**:
```python
from multimodal_stimulus_fmri_predict.core.base_classifier import BaseClassifier

class MyCustomClassifier(BaseClassifier):
    def build_model(self):
        # Implement your model architecture
        return my_model
    
    def preprocess_data(self, data):
        # Implement data preprocessing
        return processed_data
```

2. **Register with Factory**:
```python
ClassifierFactory.register_classifier('custom', MyCustomClassifier)
```

### Configuration Templates

The framework includes predefined configuration templates in `configs/experiment_configs.py`:

- `get_vit_configs()`: Vision Transformer variants
- `get_resnet_configs()`: ResNet architectures  
- `get_multimodal_configs()`: Multimodal fusion models

## 📈 Performance & Benchmarking

### Expected Runtimes

| Model Type | Training (10 epochs) | Inference (1000 samples) |
|------------|---------------------|---------------------------|
| ResNet-18 | ~15 minutes | ~30 seconds |
| ResNet-50 | ~25 minutes | ~45 seconds |
| EfficientNet-B0 | ~20 minutes | ~35 seconds |
| ViT-Base | ~30 minutes | ~60 seconds |
| MultiModal Fusion | ~40 minutes | ~75 seconds |

*Times estimated on RTX 3080 with batch size 32*

### Memory Requirements

| Model | GPU Memory | System RAM |
|-------|------------|------------|
| ResNet-18 | 4GB | 8GB |
| ResNet-50 | 6GB | 12GB |
| EfficientNet-B0 | 5GB | 10GB |
| ViT-Base | 8GB | 16GB |
| MultiModal | 10GB | 20GB |

## 🗂️ Project Structure

```
multimodal_stimulus_fmri_predict/
├── core/
│   ├── base_classifier.py      # Abstract base class
│   └── classifier_factory.py   # Model instantiation
├── classifiers/
│   ├── vision_transformer.py   # ViT implementation
│   ├── resnet.py              # ResNet variants
│   ├── efficientnet.py       # EfficientNet models
│   └── multimodal_classifier.py # Fusion models
├── utils/
│   └── experiment_runner.py   # Experiment orchestration
├── configs/
│   └── experiment_configs.py  # Configuration templates
└── examples/
    └── usage_example.py       # Complete examples
```

## 📋 Data Format Requirements

### Image Data
- **Format**: PyTorch tensors with shape `(batch_size, channels, height, width)`
- **Channels**: 1 (grayscale) or 3 (RGB) - automatically converted
- **Size**: Resized to model requirements (typically 224×224)
- **Normalization**: Applied automatically using ImageNet statistics

### fMRI Data
- **Format**: PyTorch tensors with shape `(batch_size, features)`
- **Features**: Configurable dimension (e.g., 1000 voxels)
- **Preprocessing**: Z-score normalization applied automatically

### Multimodal Data
- **Format**: Tuple of `(image_tensor, fmri_tensor)`
- **Labels**: Integer class indices starting from 0

## 🎯 Best Practices

### Model Selection
1. **Start Simple**: Begin with ResNet-18 for rapid prototyping
2. **Scale Gradually**: Move to larger models (ResNet-50, ViT) as needed
3. **Consider Efficiency**: EfficientNet offers good accuracy/efficiency trade-offs
4. **Multimodal Last**: Add fMRI fusion after establishing image baselines

### Training Tips
1. **Learning Rates**: Start with 1e-4, reduce to 5e-5 for fine-tuning
2. **Batch Sizes**: Use largest batch size that fits in GPU memory
3. **Early Stopping**: Monitor validation accuracy to prevent overfitting
4. **Data Augmentation**: Implement in your DataLoader for better generalization

### Experiment Design
1. **Systematic Comparison**: Use consistent splits and metrics across models
2. **Statistical Significance**: Run multiple seeds and report confidence intervals
3. **Ablation Studies**: Test individual components (pretrained vs. scratch)
4. **Cross-Validation**: Implement k-fold CV for robust evaluation

## 🔍 Troubleshooting

### Common Issues

**OutOfMemoryError**
```python
# Reduce batch size
train_loader = DataLoader(dataset, batch_size=16)  # Instead of 32

# Enable gradient checkpointing (for supported models)
model.gradient_checkpointing_enable()
```

**Slow Training**
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

**Model Not Converging**
```python
# Adjust learning rate
config['learning_rate'] = 5e-5  # Reduce LR

# Add learning rate scheduling
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

## 📊 Results Analysis

The framework automatically generates:

- **Training Curves**: Loss and accuracy over epochs
- **Performance Metrics**: Accuracy, F1-score, AUC-ROC
- **Timing Information**: Training duration and inference speed
- **Model Comparisons**: Side-by-side performance analysis

Results are saved in JSON format and summarized in CSV files for easy analysis.

## 🤝 Contributing

### Adding New Models

1. Inherit from `BaseClassifier`
2. Implement `build_model()` and `preprocess_data()`
3. Add configuration template
4. Update documentation

### Code Style
- Follow PEP 8 conventions
- Use type hints where possible
- Add docstrings for public methods
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for ViT implementation
- PyTorch team for deep learning framework
- torchvision for pretrained CNN models
- Research community for multimodal learning advances

## 📞 Support

For questions and issues:
1. Check existing GitHub issues
2. Review troubleshooting section
3. Create detailed issue with code samples
4. Include system specifications and error messages

---

**Estimated Development Time:**
- For neurodivergent/burned out developers: 8-12 hours to implement basic version, 16-24 hours for full feature set
- For average regulated developers: 6-8 hours for basic version, 12-16 hours for full feature set

**Mathematical Dependencies:**

For optimal results, ensure the following packages are available:

```latex
% Required LaTeX packages for mathematical notation
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}

% Loss function formulation
L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)

% Multimodal fusion equation
h_{fusion} = \text{MLP}([h_{image}; h_{fmri}])

% Attention mechanism (for ViT)
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```