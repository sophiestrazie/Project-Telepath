# Multimodal fMRI Stimulus Prediction Repository

## 🧠 Overview

This repository provides a modular, extensible framework for experimenting with different machine learning classifiers on multimodal fMRI data. The architecture follows SOLID principles and uses factory patterns for easy classifier swapping, enabling researchers to quickly compare various approaches for stimulus prediction tasks.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [Mathematical Foundations](#mathematical-foundations)

## ✨ Features

- **Modular Architecture**: Easy to extend with new classifiers
- **Multiple Classifier Types**: Classical ML, Neural Networks, and Ensemble methods
- **Flexible Data Pipeline**: Support for various fMRI data formats
- **Comprehensive Evaluation**: Cross-validation, metrics, and visualization
- **Hyperparameter Optimization**: Built-in grid search capabilities
- **Experiment Management**: YAML-based configuration system
- **Extensible Design**: Factory pattern for seamless classifier addition

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

```powershell
# Clone the repository
git clone https://github.com/your-username/multimodal-stimulus-fmri-predict.git
cd multimodal-stimulus-fmri-predict

# Create virtual environment
python -m venv brain-env
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\brain-env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Dependencies

```text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.9.0
torchvision>=0.10.0
nibabel>=3.2.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
pyyaml>=5.4.0
joblib>=1.0.0
tqdm>=4.62.0
```

## 🏃‍♂️ Quick Start

### Basic Usage

```powershell
# Run baseline experiment with classical ML methods
python main.py --config config/experiment_configs.yaml --experiment baseline --subjects 01 02 03 04 05 --output results/baseline --plot

# Run neural network experiment
python main.py --config config/experiment_configs.yaml --experiment neural_networks --subjects 01 02 03 04 05 --output results/neural --plot
```

### Python API

```python
from experiments.experiment_runner import ExperimentRunner
from data.loaders import FMRIDataLoader

# Initialize experiment runner
runner = ExperimentRunner('config/experiment_configs.yaml')

# Run experiment
results = runner.run_experiment('baseline', ['01', '02', '03'])

# Save results
runner.save_experiment_results(results, 'results/my_experiment.json')
```

## 🏗️ Architecture

### Project Structure

```
multimodal_stimulus_fmri_predict/
├── config/                    # Configuration files
│   ├── base_config.py
│   ├── model_configs.py
│   └── experiment_configs.yaml
├── data/                      # Data handling
│   ├── loaders.py            # fMRI data loading
│   ├── preprocessors.py      # Data preprocessing
│   └── transforms.py         # Data transformations
├── models/                    # Classifier implementations
│   ├── base_classifier.py    # Abstract base class
│   ├── classical/            # Traditional ML methods
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   └── logistic_regression.py
│   ├── neural/               # Neural network methods
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   └── ensemble/             # Ensemble methods
│       ├── voting.py
│       └── stacking.py
├── utils/                     # Utility functions
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py     # Plotting functions
│   └── io_utils.py          # I/O operations
├── experiments/              # Experiment management
│   ├── experiment_runner.py
│   └── hyperparameter_search.py
└── main.py                   # Main execution script
```

### Core Design Patterns

#### 1. Abstract Base Classifier

All classifiers inherit from `BaseClassifier`, ensuring consistent interface:

```python
class BaseClassifier(ABC, BaseEstimator):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        pass
```

#### 2. Factory Pattern

The `ClassifierFactory` enables dynamic classifier creation:

```python
classifier = ClassifierFactory.create_classifier('svm', config)
```

#### 3. Configuration-Driven Experiments

YAML configuration files define experiment parameters:

```yaml
experiments:
  baseline:
    classifiers:
      - type: "svm"
        config:
          C: 1.0
          kernel: "rbf"
```

## 💻 Usage

### Running Experiments

#### Command Line Interface

```powershell
# Basic experiment
python main.py --config config/experiment_configs.yaml --experiment baseline --subjects 01 02 03 --output results/baseline

# With visualization
python main.py --config config/experiment_configs.yaml --experiment neural_networks --subjects 01 02 03 04 05 --output results/neural --plot

# Custom configuration
python main.py --config my_custom_config.yaml --experiment advanced --subjects 01 02 03 04 05 06 07 08 09 10 --output results/advanced --plot
```

#### Python API

```python
from experiments.experiment_runner import ExperimentRunner
from models import ClassifierFactory

# Initialize runner
runner = ExperimentRunner('config/experiment_configs.yaml')

# Run single experiment
results = runner.run_experiment('baseline', ['01', '02', '03'])

# Create custom classifier
config = {'C': 10.0, 'kernel': 'linear'}
classifier = ClassifierFactory.create_classifier('svm', config)

# Load data and train
from data.loaders import FMRIDataLoader
loader = FMRIDataLoader({'data_path': 'data/raw'})
X, _ = loader.load_fmri_data('01')
y = loader.load_stimulus_labels('01')

classifier.fit(X, y)
predictions = classifier.predict(X)
```

### Data Format Requirements

#### fMRI Data
- **Format**: NIfTI files (`.nii.gz`)
- **Structure**: 4D arrays (x, y, z, time)
- **Naming**: `sub-{subject_id}_task-stimuli_bold.nii.gz`

#### Labels
- **Format**: TSV files (`.tsv`)
- **Columns**: Must include `stimulus_type`
- **Naming**: `sub-{subject_id}_task-stimuli_events.tsv`

#### Directory Structure
```
data/
├── sub-01/
│   ├── func/
│   │   └── sub-01_task-stimuli_bold.nii.gz
│   └── sub-01_task-stimuli_events.tsv
├── sub-02/
│   └── ...
```

## ⚙️ Configuration

### Experiment Configuration

```yaml
# config/experiment_configs.yaml
experiments:
  baseline:
    classifiers:
      - type: "svm"
        config:
          C: 1.0
          kernel: "rbf"
          gamma: "scale"
      - type: "random_forest"
        config:
          n_estimators: 100
          max_depth: 10
          random_state: 42

data:
  data_path: "data/raw"
  test_size: 0.2
  random_state: 42

preprocessing:
  standardize: true
  dimensionality_reduction:
    method: "pca"
    n_components: 1000

cross_validation:
  n_folds: 5
  shuffle: true
  random_state: 42
```

### Classifier-Specific Configurations

#### SVM Configuration
```yaml
svm_config:
  C: 1.0                    # Regularization parameter
  kernel: "rbf"             # Kernel type
  gamma: "scale"            # Kernel coefficient
  probability: true         # Enable probability estimates
```

#### Neural Network Configuration
```yaml
mlp_config:
  hidden_dims: [512, 256, 128]  # Hidden layer dimensions
  dropout_rate: 0.3             # Dropout probability
  learning_rate: 0.001          # Learning rate
  n_epochs: 100                 # Training epochs
  batch_size: 32                # Batch size
```

## 📚 API Reference

### Core Classes

#### BaseClassifier
Abstract base class for all classifiers.

**Methods:**
- `fit(X, y)`: Train the classifier
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get prediction probabilities
- `get_feature_importance()`: Get feature importance scores

#### ClassifierFactory
Factory class for creating classifier instances.

**Methods:**
- `create_classifier(classifier_type, config)`: Create classifier instance
- `get_available_classifiers()`: List available classifier types

#### FMRIDataLoader
Handles loading and preprocessing of fMRI data.

**Methods:**
- `load_fmri_data(subject_id)`: Load fMRI data for subject
- `load_stimulus_labels(subject_id)`: Load stimulus labels
- `create_train_test_split(X, y)`: Create train/test splits

#### ExperimentRunner
Main class for running experiments.

**Methods:**
- `run_experiment(experiment_name, subject_ids)`: Run complete experiment
- `save_experiment_results(results, output_path)`: Save results
- `run_hyperparameter_search(classifier_type, param_grid, X, y)`: Optimize parameters

### Available Classifiers

#### Classical Methods
- **SVM**: Support Vector Machine with RBF/Linear kernels
- **Random Forest**: Ensemble of decision trees
- **Logistic Regression**: Linear probabilistic classifier

#### Neural Networks
- **MLP**: Multi-Layer Perceptron with dropout and batch normalization
- **CNN**: Convolutional Neural Network for spatial patterns
- **LSTM**: Long Short-Term Memory for temporal dependencies
- **Transformer**: Attention-based architecture

#### Ensemble Methods
- **Voting Classifier**: Majority/weighted voting
- **Stacking Classifier**: Meta-learner approach

## 📊 Examples

### Example 1: Basic Comparison

```python
from experiments.experiment_runner import ExperimentRunner

# Initialize runner
runner = ExperimentRunner('config/experiment_configs.yaml')

# Run baseline experiment
results = runner.run_experiment('baseline', ['01', '02', '03', '04', '05'])

# Print results
print(f"Best classifier: {results['summary']['best_classifier']}")
print(f"Best accuracy: {results['summary']['best_accuracy']:.4f}")
```

**Expected Runtime:**
- Neurodivergent (burned out): 45-60 minutes
- Average person: 20-30 minutes

### Example 2: Hyperparameter Optimization

```python
from experiments.experiment_runner import ExperimentRunner
from data.loaders import FMRIDataLoader

# Load data
loader = FMRIDataLoader({'data_path': 'data/raw'})
X, _ = loader.load_fmri_data('01')
y = loader.load_stimulus_labels('01')

# Define parameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

# Run hyperparameter search
runner = ExperimentRunner('config/experiment_configs.yaml')
results = runner.run_hyperparameter_search('svm', param_grid, X, y)

print(f"Best parameters: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.4f}")
```

**Expected Runtime:**
- Neurodivergent (burned out): 2-3 hours
- Average person: 1-1.5 hours

### Example 3: Custom Classifier Implementation

```python
from models.base_classifier import BaseClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.model = GaussianNB()
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        # Naive Bayes doesn't have feature importance
        return np.zeros(self.model.n_features_in_)

# Register with factory
from models import ClassifierFactory
ClassifierFactory._classifiers['naive_bayes'] = NaiveBayesClassifier
```

## 🤝 Contributing

### Adding New Classifiers

1. **Create classifier class** inheriting from `BaseClassifier`
2. **Implement required methods**: `fit`, `predict`, `predict_proba`, `get_feature_importance`
3. **Register with factory** in `models/__init__.py`
4. **Add configuration** in `experiment_configs.yaml`
5. **Write tests** in `tests/test_classifiers.py`

### Development Setup

```powershell
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 .
black .
```

### Pull Request Guidelines

- Follow PEP 8 style guidelines
- Include comprehensive tests
- Update documentation
- Add example usage
- Estimate completion times for tasks

## 🔬 Mathematical Foundations

### Support Vector Machine

The SVM optimization problem:

```latex
$$\min_{w,b,\xi} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$
```
Subject to:

```latex
$$y_i(w^T\phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$
```
Where:

```latex
- $w$: weight vector
- $b$: bias term
- $\xi_i$: slack variables
- $C$: regularization parameter
- $\phi(x_i)$: feature mapping function
```

### Neural Network Loss

Multi-class cross-entropy loss:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{i,k}\log(\hat{y}_{i,k})$$

Softmax activation:
$$\hat{y}_{i,k} = \frac{e^{z_{i,k}}}{\sum_{j=1}^{K}e^{z_{i,j}}}$$

### Cross-Validation

K-fold cross-validation score:
$$CV_{score} = \frac{1}{K}\sum_{k=1}^{K} \text{Accuracy}_k$$

Standard error:
$$SE = \sqrt{\frac{\sum_{k=1}^{K}(\text{Accuracy}_k - CV_{score})^2}{K-1}}$$

## 📈 Performance Expectations

### Estimated Completion Times

#### Full Implementation
**Neurodivergent Person (Burned Out):**
- Setup and Environment: 2-3 hours
- Base Architecture: 8-12 hours
- Classifier Implementation: 12-16 hours
- Experiment Framework: 6-8 hours
- Testing and Debugging: 4-6 hours
- Documentation: 2-3 hours
- **Total: 34-48 hours**

**Average Person (Regulated):**
- Setup and Environment: 1-1.5 hours
- Base Architecture: 4-6 hours
- Classifier Implementation: 6-8 hours
- Experiment Framework: 3-4 hours
- Testing and Debugging: 2-3 hours
- Documentation: 1-1.5 hours
- **Total: 17-24 hours**

#### Single Experiment Run
**Neurodivergent (Burned Out):** 45-60 minutes
**Average Person:** 20-30 minutes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

## 🙏 Acknowledgments

- neuroimaging community for data standards
- scikit-learn for machine learning utilities
- PyTorch for neural network implementations
- nibabel for neuroimaging data handling