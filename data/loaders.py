# data/loaders.py
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import nibabel as nib
from pathlib import Path

class MultimodalDataLoader:
    """
    Loader class for handling fMRI data and associated stimulus labels.
    
    This class is designed to:
    - Load 4D fMRI brain imaging data (x, y, z, time) and convert it to 2D (time, voxels).
    - Load corresponding stimulus labels for supervised learning.
    - Split the data into training and testing sets.

    Parameters:
        config (Dict[str, Any]): Dictionary containing configuration such as the root data path.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Convert string path to Path object for easier manipulation
        self.data_path = Path(config['data_path'])
        
    def load_fmri_data(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and reshape fMRI data for a single subject.

        fMRI data is stored in 4D: (x, y, z, time).
        To use it with ML models, we reshape it to 2D: (time, voxels),
        where each row is a timepoint and each column is a voxel (brain region).

        Args:
            subject_id (str): ID of the subject to load.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - 2D fMRI data (timepoints x voxels)
                - Affine matrix (used for spatial metadata, not directly used in modeling)
        """
        # Construct the expected path to the subject’s fMRI data
        fmri_file = Path(r"C:\Users\PC\Desktop\AI4Good - Brain Project\AI4Good-MTL-Group-2\scripts\v2-strass-branch\data\sub-01_ses-001_task-s01e02a_space-T1w_boldref.nii.gz")
        #fmri_file = f"C:\Users\PC\Desktop\AI4Good - Brain Project\AI4Good-MTL-Group-2\scripts\v2-strass-branch\data\sub-01_ses-001_task-s01e02a_space-T1w_boldref.nii.gz"
        #self.data_path / f"sub-{subject_id}_ses-001_task-s01e02a_space-T1w_boldref.nii.gz"
        #f"sub-{subject_id}" / "func" / f"sub-{subject_id}_task-stimuli_bold.nii.gz"
        
        # Construct the expected path to the subject’s fMRI data
        if not fmri_file.exists():
            raise FileNotFoundError(f"fMRI data not found: {fmri_file}")
            
        # Load the .nii.gz file (NIfTI format) using nibabel
        img = nib.load(str(fmri_file))
        data = img.get_fdata()  # Get numerical voxel data as a NumPy array
        
        # Reshape from 4D (x, y, z, time) to 2D (time, voxels)
        # The last dimension is time. We reshape to (time, features)
        n_timepoints = data.shape[-1]
        data_2d = data.reshape(-1, n_timepoints).T # Transpose to (time, voxels)
        
        return data_2d, img.affine
    
    def load_stimulus_labels(self, subject_id: str) -> np.ndarray:
        """
        Load stimulus labels for a subject from a corresponding .tsv file.
        
        Each row in the file corresponds to a stimulus presented at a specific timepoint.

        Args:
            subject_id (str): ID of the subject to load.

        Returns:
            np.ndarray: Array of stimulus labels (e.g., category names or IDs)
        """
        # Path to the stimulus file
        label_file = self.data_path / f"sub-{subject_id}" / f"sub-{subject_id}_task-stimuli_events.tsv"
        
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        # Read the tab-separated file using pandas
        events = pd.read_csv(label_file, sep='\t')
        
        # Return the 'stimulus_type' column as a NumPy array
        return events['stimulus_type'].values
    
    def create_train_test_split(self, 
                              X: np.ndarray, 
                              y: np.ndarray, 
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the fMRI data and labels into training and testing sets.

        Stratified splitting ensures that each class is proportionally represented in both sets.

        Args:
            X (np.ndarray): Feature matrix (timepoints x voxels)
            y (np.ndarray): Labels corresponding to each timepoint
            test_size (float): Fraction of data to reserve for testing (default is 0.2)
            random_state (int): Seed for reproducibility

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)
    