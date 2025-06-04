# src/your_package_name/audio/extract.py

# src/your_package_name/audio/extract.py

import os
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import librosa
import h5py

from .config import AudioConfig

class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config

    def process(self, episode_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio chunks of a movie file.

        Parameters
        ----------
        episode_path : str
            Path to the movie file.

        Returns
        -------
        np.ndarray
            Extracted MFCC features.
        """
        clip = VideoFileClip(episode_path)
        start_times = np.arange(0, clip.duration, self.config.chunk_duration)[:-1]

        temp_dir = os.path.join(self.config.save_dir_temp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        features = []

        with tqdm(total=len(start_times), desc="Extracting audio features") as pbar:
            for start in start_times:
                subclip = clip.subclip(start, start + self.config.chunk_duration)
                wav_path = os.path.join(temp_dir, "chunk.wav")
                subclip.audio.write_audiofile(wav_path, verbose=False, logger=None)

                y, sr = librosa.load(wav_path, sr=self.config.sampling_rate, mono=True)
                mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
                features.append(mfcc)
                pbar.update(1)

        return np.array(features, dtype=np.float32)

    def save_features(self, features: np.ndarray, episode_id: str):
        """
        Save extracted features to disk.

        Parameters
        ----------
        features : np.ndarray
            Extracted MFCC features.
        episode_id : str
            Identifier of the episode (e.g., 's01e01a').
        """
        if self.config.save_format == "npy":
            out_path = os.path.join(self.config.save_dir_features, f"{episode_id}_audio_features.npy")
            np.save(out_path, features)
        elif self.config.save_format == "h5":
            out_path = os.path.join(self.config.save_dir_features, f"{episode_id}_audio_features.h5")
            with h5py.File(out_path, "w") as f:
                group = f.create_group(episode_id)
                group.create_dataset("audio", data=features, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported save format: {self.config.save_format}")

        print(f"Features saved to: {out_path}")
