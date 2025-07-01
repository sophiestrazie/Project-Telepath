

# 🧠 Project Cere

#### Multimodal AI for Social Good

*AI4Good Lab @Mila • Montreal 2025 Cohort*

## 🔍 Project Overview
 
Project Cere develops **multimodal machine learning models** that integrate visual, textual, and audio data to address pressing social challenges. This repository contains our codebase, experiments, and documentation for creating interpretable AI systems with real-world impact.

## ✨ Features

- **Modular Architecture**: Easy to extend with new classifiers
- **Multiple Classifier Types**: Classical ML, Neural Networks, and Ensemble methods
- **Flexible Data Pipeline**: Support for various fMRI data formats
- **Comprehensive Evaluation**: Cross-validation, metrics, and visualization
- **Hyperparameter Optimization**: Built-in grid search capabilities
- **Experiment Management**: YAML-based configuration system
- **Extensible Design**: Factory pattern for seamless classifier addition

## 📋 Table of Contents

- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
- [Features](#-features)
- [Usage](#-usage)
- [Project Roadmap](#-project-roadmap)
- [Team](#-team)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)


-------------------------


## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Git

### Setup
1. **Clone the repository**:

   ```bash
   git clone https://github.com/marialagakos/AI4Good-MTL-Group-2.git
   cd AI4Good-MTL-Group-2
   ```

2. **Create and activate virtual environment**:

   ```bash
   python -m venv cere-env
   # Linux/MacOS
   source cere-env/bin/activate
   # Windows (PowerShell)
   .\cere-env\Scripts\Activate.ps1
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -e .  # Editable install for development
   pip install -r requirements.txt  # Optional: Full dependency install
   ```

Here’s a concise **team onboarding guide** and your **personal command cheat sheet** for working with this repository:


### **Creating A Fork**

1. **Fork the Repository**  
   - Each member creates a personal fork:  
     - Go to [github.com/marialagakos/AI4Good-MTL-Group-2](https://github.com/marialagakos/AI4Good-MTL-Group-2) → Click **"Fork"** (top-right).

2. **Clone Your Fork**  
   ```bash
   git clone https://github.com/YOUR-USERNAME/AI4Good-MTL-Group-2.git
   cd AI4Good-MTL-Group-2
   ```

3. **Set Up Remotes**  
   ```bash
   git remote add upstream https://github.com/marialagakos/AI4Good-MTL-Group-2.git
   git remote -v  # Verify: origin=your fork, upstream=original
   ```

4. **Sync with Upstream**  
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main  # Keep your fork updated
   ```


## 📂 Repository Structure

```
Project-Cere/
├── main.py                 # Main execution script
├── README.md               # Project documentation
├── data/                   # Raw and processed datasets
│   ├── feature_extraction.py    # Feature extraction utilities
│   ├── DATA_INSTRUCTIONS.md
│   ├── .ipnyb_checkpoints/
│   ├── src/                 
│   │   ├── telepath/
│   │   ├── telepath.egg-info/
│   │   └── temp_audio_chunks/
│   ├── pca_data/
│   ├── loaders.py            # loading (fmri, audio, text, visual)
│   ├── preprocessors.py      # preprocessing
│   ├── transforms.py         # transformations
│   ├── fmri/                    # fMRI data
│   ├── audio/                   # Audio samples
│   ├── transcripts/             # Text corpora
│   └── visual/                  # Image/video data
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
├── .gitignore              # File control
├── docs/                   # Technical documentation
├── tests/                  # Unit and integration tests
└── LICENSE.md
```


-------------------------


## 🚀 Usage

### Running the Pipeline

```bash
python src/main.py --modality all --config configs/default.yaml
```

### Key Arguments

- `--modality`: Choose `audio`, `text`, `visual`, or `all`
- `--config`: Path to YAML configuration file

### Jupyter Notebooks

```bash
jupyter lab notebooks/
```


### **Maintaining Forks**  

#### **1. Start a New Feature**  
```bash
git checkout -b feature/your-feature-name  # e.g., feature/login-form
```

#### **2. Commit & Push to Your Fork**  
```bash
git add .
git commit -m "Description of changes"
git push -u origin feature/your-feature-name
```

#### **3. Sync with Upstream**  
```bash
git checkout main
git fetch upstream
git merge upstream/main  # Or use `git rebase upstream/main`
git push origin main
```

#### **4. Update Your Feature Branch**  
```bash
git checkout feature/your-feature-name
git rebase main  # Apply your changes on top of latest updates
git push --force  # Only if you've rebased
```

#### **5. Create a Pull Request (PR)**  
1. Go to your fork on GitHub.  
2. Click **"Compare & Pull Request"** for your branch.  
3. Target `marialagakos/AI4Good-MTL-Group-2:main` as the base.

---

### **Key Rules for the Team**  
- **Never push directly to `upstream`** (only PRs).  
- **Always branch from `main`** (no direct commits to `main`).  
- **Rebase instead of merge** to keep history clean (use `git rebase main`).  


### **Troubleshooting**  

- **Permission denied?**  
  ```bash
  git remote set-url origin https://github.com/YOUR-USERNAME/AI4Good-MTL-Group-2.git
  ```  
- **Broken branch?**  
  ```bash
  git checkout main
  git branch -D broken-branch
  ```  

Print this or save it as a text file! Need a visual workflow diagram? Let me know.


## 🗺️ Project Roadmap

| Phase          | Key Deliverables                          |
|----------------|------------------------------------------|
| Data Analysis  | EDA reports, preprocessing pipelines     |
| Modeling       | Multimodal fusion architectures          |
| Evaluation     | Cross-modal attention visualizations     |
| Deployment     | Flask API for model serving              |

## 👥 Team

- [Maria Lagakos](https://github.com/marialagakos) - Feature Extraction
- [Sophie Strassmann](https://github.com/sophiestrazie) - Creative Director, Pipeline Architecture, and Classification Team
- [Yujie Chen](https://github.com/huricaneee) - Classification Team
- [Keyu Liang](https://github.com/Keyu17) - Feature Extraction and Data Migration
- [Maria Gallamoso](https://github.com/mariagarcia) - Feature Extraction Team
- [Catherina Medeiros](https://github.com/cathmedeiros) Director of Imaging, Feature Extraction Team


## 📜 License

This project is licensed under the **MIT License** - see [LICENSE.md](LICENSE.md) for details.

## 🙏 Acknowledgments

We gratefully acknowledge:

- **Jennifer Addison** and **Yosra Kazemi** for their expertise and leadership
- The AI4Good Lab Montreal and Mila team for their support
- Our TA **Hugo Berard** and **Laetitia Constantin**

Consulting Scholars and Mentors:

- Rose Landry - Mila
- Adel Halawa - McGill University
- Dr. Lune Bellec - Université de Montréal
- Dr. Mayada Elsabbagh - Transforming Autism Care Consortium
- The Algonauts Project
- Compute Canada for their computational resources
- The Digital Research Alliance of Canada
