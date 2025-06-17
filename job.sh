#!/bin/bash
#SBATCH --job-name=cpu_pyproject_test     # Your job name
#SBATCH --time=00:30:00                   # 30 minutes
#SBATCH --cpus-per-task=2                 # CPU cores (adjust if needed)
#SBATCH --mem=4G                          # RAM
#SBATCH --output=logs/output_%j.txt       # Save output logs to a file

# Load Python module
module load python/3.10

# Create temporary virtual environment in SLURM's job-local directory
VENV_DIR=$(mktemp -d $SLURM_TMPDIR/venv.XXXXXX)
python -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Upgrade pip and install necessary build tools
pip install --upgrade pip
pip install build

# (Optional) Add any runtime dependencies here:
# pip install pandas torch wandb etc.

# Move to your project directory in $SCRATCH
cd $SCRATCH/AI4Good-MTL-Group-2

# Install your package from pyproject.toml
pip install -e .

# Run your Python script
python /scripts/run_feature_extraction_pipeline.py

# Archive any output data and move it to your Project folder
tar -cf $project/project-repo/data/results/result-archive.tar data/output/results.pt results.csv
