# src/your_package_name/audio/extract.py

import os
import numpy as np
import librosa
import h5py
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from pathlib import Path



def extract_audio_features(
    episode_path, tr, sr,
    save_dir_temp, save_dir_features,
    device='cpu',
    save_format='h5',   # "h5" or "npy"
    episode_id='s01e01a'
):
    """
    Extracts and saves audio features from a video file using MFCC.

    Parameters
    ----------
    episode_path : str
        Path to the .mkv video file.
    tr : float
        Chunk duration (in seconds).
    sr : int
        Audio sampling rate.
    save_dir_temp : str
        Path to save temp audio chunks.
    save_dir_features : str
        Path to save final feature file.
    device : str
        Placeholder ('cpu' or 'gpu') — not used.
    save_format : str
        Format to save: "h5" or "npy".
    episode_id : str
        Episode identifier, used in file naming.

    Returns
    -------
    np.ndarray
        Extracted features (shape: [n_chunks, n_mfcc])
    """

    clip = VideoFileClip(episode_path)
    start_times = np.arange(0, clip.duration, tr)[:-1]

    # Prepare temp folder
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Feature list
    audio_features = []

    with tqdm(total=len(start_times), desc="Extracting audio features") as pbar:
        for start in start_times:
            chunk_path = os.path.join(temp_dir, 'audio_chunk.wav')
            clip_chunk = clip.subclip(start, start + tr)
            clip_chunk.audio.write_audiofile(chunk_path, verbose=False, logger=None)

            y, _ = librosa.load(chunk_path, sr=sr, mono=True)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
            audio_features.append(mfcc)
            pbar.update(1)

    # Convert to array
    audio_features = np.array(audio_features, dtype='float32')

    # Save
    os.makedirs(save_dir_features, exist_ok=True)

    if save_format == 'npy':
        out_file = os.path.join(save_dir_features, f'{episode_id}_audio_features.npy')
        np.save(out_file, audio_features)
        print(f"Audio features saved to {out_file}")

    elif save_format == 'h5':
        out_file = os.path.join(save_dir_features, f'{episode_id}_audio_features.h5')
        mode = 'a' if Path(out_file).exists() else 'w'
        with h5py.File(out_file, mode) as f:
            if episode_id in f:
                del f[episode_id]
            group = f.create_group(episode_id)
            group.create_dataset('audio', data=audio_features, dtype=np.float32)
        print(f"Audio features saved to {out_file}")

    return audio_features
