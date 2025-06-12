import os
import numpy as np
import torch
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from torchvision.models.feature_extraction import create_feature_extractor
from .config import VideoConfig
from .transform_utils import define_frames_transform
import h5py
from pathlib import Path



class VideoProcessor:
    def __init__(self, config: VideoConfig):
        self.config = config
        self.device = config.device
        self.transform = define_frames_transform()
        self.model, self.model_layer = self.get_vision_model()

    def get_vision_model(self):

        """
        Load a pre-trained slow_r50 video model and set up the feature extractor.

        Parameters
        ----------
        device : torch.device
            The device on which the model will run (i.e., 'cpu' or 'cuda').

        Returns
        -------
        feature_extractor : torch.nn.Module
            The feature extractor model.
        model_layer : str
            The layer from which visual features will be extracted.

        """

        # Load the model
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50',
            pretrained=True)

        # Select 'blocks.5.pool' as the feature extractor layer
        model_layer = 'blocks.5.pool'
        feature_extractor = create_feature_extractor(model,
            return_nodes=[model_layer])
        feature_extractor.to(self.device)
        feature_extractor.eval()

        return feature_extractor, model_layer

    def process(self, episode_path: str) -> np.ndarray:
        """
    Extract visual features from a movie using a pre-trained video model.

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the visual features are extracted.
    tr : float
        Duration of each chunk, in seconds (aligned with the fMRI repetition
        time, or TR).
    feature_extractor : torch.nn.Module
        Pre-trained feature extractor model.
    model_layer : str
        The model layer from which the visual features are extracted.
    transform : torchvision.transforms.Compose
        Transformation pipeline for processing video frames.
    device : torch.device
        Device for computation ('cpu' or 'cuda').
    save_dir_temp : str
        Directory where the chunked movie clips are temporarily stored for
        feature extraction.
    save_dir_features : str
        Directory where the extracted visual features are saved.

    Returns
    -------
    visual_features : float
        Array containing the extracted visual features.

    """
        
        clip = VideoFileClip(episode_path)
        start_times = [x for x in np.arange(0, clip.duration, self.config.chunk_duration)][:-1]
        # Create the directory where the movie chunks are temporarily saved
        temp_dir = os.path.join(self.config.save_dir_temp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Empty features list
        visual_features = []

        # Loop over chunks
        with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
            for start in start_times:

                # Divide the movie in chunks of length TR, and save the resulting
                # clips as '.mp4' files
                clip_chunk = clip.subclip(start, start+self.config.chunk_duration)
                chunk_path = os.path.join(temp_dir, 'visual_chunk.mp4')
                clip_chunk.write_videofile(chunk_path, verbose=False, audio=False,
                    logger=None)
                # Load the frames from the chunked movie clip
                video_clip = VideoFileClip(chunk_path)
                chunk_frames = [frame for frame in video_clip.iter_frames()]

                # Format the frames to shape:
                # (batch_size, channels, num_frames, height, width)
                frames_array = np.transpose(np.array(chunk_frames), (3, 0, 1, 2))
                # Convert the video frames to tensor
                inputs = torch.from_numpy(frames_array).float()
                # Preprocess the video frames
                inputs = self.transform(inputs).unsqueeze(0).to(self.device)

                # Extract the visual features
                with torch.no_grad():
                    preds = self.model(inputs)
                visual_features.append(np.reshape(preds[self.model_layer].cpu().numpy(), -1))

                # Update the progress bar
                pbar.update(1)

        # Convert the visual features to float32
        visual_features = np.array(visual_features, dtype='float32')

        return visual_features
    
    def save_features(self, visual_features: np.ndarray, episode_id: str):

        #Save the visual features
        if self.config.save_format == "npy":
            out_path = os.path.join(self.config.save_dir_features, f"{episode_id}_video_features.npy")
            np.save(out_path, visual_features)
        elif self.config.save_format == "h5":  
            out_file_visual = os.path.join(
            self.config.save_dir_features, f"{episode_id}_video_features.h5")
            with h5py.File(out_file_visual, 'a' if Path(out_file_visual).exists() else 'w') as f:
                group = f.create_group(episode_id)
                group.create_dataset('visual', data=visual_features, dtype=np.float32)
        else: 
            raise ValueError(f"Unsupported save format: {self.config.save_format}")
        
        print(f"Visual features saved to {out_file_visual}")
        


