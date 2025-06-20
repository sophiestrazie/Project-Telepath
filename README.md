# AI4Good Lab - Project Cere

*Montreal 2025*

## 🧠 Project Overview

This repository contains the work of **Project Cere** for the AI4Good Montreal program 2025 cohort. Our project focuses on developing and evaluating multimodal machine learning models that integrate visual, textual, and audio data to address real-world social challenges.


## Table of Contents

What to expect in this repository:

- [Installation and Usage](#installation-and-usage)
- [Repository Structure](#repository-structure)
- [Team Members](#team-members)
- [License](#license)
- [Acknowledgments](#acknowledgments)


## 📂 Repository Structure


## 🚀 Installation & Usage

Using Python 3.10

Create virtual environment and install dependencies:

```bash
python -m venv brain-env
```
```bash
python -m pip install --upgrade pip
```

Activate the environement

```bash
# bash
source .brain-env/bin/activate
```
OR

```PowerShell
# PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\brain-env\Scripts\Activate.ps1
```


Then install the package:

```bash
# bash
pip install -e . # run each time multimodal_stimulus_fmri_predict is updated

```

Install the dependencies (optional):

```bash
#bash
pip install -r requirements.txt
```

### How to run the code


### Backup: Pulling Repo

```PowerShell
git status

git pull
```

## Project Roadmap

### Related Work

### Data

### Methodology

### Performance 

### Conclusion

## 👥 Team Members

[Team Member 1 Name] (GitHub Profile)

[Team Member 2 Name] (GitHub Profile)

[Team Member 3 Name] (GitHub Profile)

[Team Member 4 Name] (GitHub Profile)

## 📝 License
This project is licensed under the [License Name] - see the LICENSE.md file for details.

## 🙏 Acknowledgments
AI4Good Montreal organizers and mentors

[Any other organizations or individuals you want to acknowledge]



-------------------------




# 🧠 Project Cere - Multimodal ML for Social Good

*AI4Good Lab Montreal • 2025 Cohort*

## 🌟 Project Overview

Project Cere develops **multimodal machine learning models** that integrate visual, textual, and audio data to address pressing social challenges. This repository contains our codebase, experiments, and documentation for creating interpretable AI systems with real-world impact.

## 📋 Table of Contents

- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
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

## 📂 Repository Structure

```
Project-Cere/
├── data/                   # Raw and processed datasets
│   ├── audio/              # Audio samples
│   ├── text/               # Text corpora
│   └── visual/             # Image/video data
├── models/                 # Pretrained models and checkpoints
├── notebooks/              # Exploratory analysis and prototyping
├── src/                    # Core source code
│   ├── preprocessing/      # Data pipelines
│   ├── modeling/           # Model architectures
│   └── evaluation/         # Metrics and analysis
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

## 🗺️ Project Roadmap

| Phase          | Key Deliverables                          |
|----------------|------------------------------------------|
| Data Analysis  | EDA reports, preprocessing pipelines     |
| Modeling       | Multimodal fusion architectures          |
| Evaluation     | Cross-modal attention visualizations     |
| Deployment     | Flask API for model serving              |

## 👥 Team

- [Jane Doe](https://github.com/janedoe) - Data Pipelines
- [John Smith](https://github.com/johnsmith) - Model Architecture
- [Alex Chen](https://github.com/alexchen) - Evaluation Metrics
- [Maria Garcia](https://github.com/mariagarcia) - Deployment

## 📜 License
This project is licensed under the **MIT License** - see [LICENSE.md](LICENSE.md) for details.

## 🙏 Acknowledgments

We gratefully acknowledge:

- The AI4Good Lab Montreal organizers
- Our project mentors and TAs.... 
- Compute Canada for providing advanced computing resources


