
import os
import string
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import librosa
import cv2
from PIL import Image

from tqdm import tqdm
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

from moviepy.editor import VideoFileClip
from IPython.display import Video, display, clear_output
import ipywidgets as widgets

from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.models.feature_extraction import create_feature_extractor

from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale

from transformers import (
    BertTokenizer, BertModel,
    AutoImageProcessor, AutoModelForVideoClassification
)


# Visual feature extraction
# 1) slow_r50
def define_frames_transform():
    """Defines the preprocessing pipeline for the video frames. Note that this
    transform is specific to the slow_r50 model."""
    transform = Compose(
        [
            UniformTemporalSubsample(8),
            Lambda(lambda x: x/255.0),
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(size=256),
            CenterCrop(256)
        ]
    
  )
    print("finished defining frames transform")
    return transform
def get_vision_model(device):
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
    feature_extractor.to(device)
    feature_extractor.eval()

    print("get_vision_model: Model loaded and feature extractor created.")
    return feature_extractor, model_layer
def extract_visual_features_slowr50(episode_path, tr, feature_extractor, model_layer,
    transform, device, save_dir_temp, save_dir_features, video_id):
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

    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Empty features list
    visual_features = []

    # Loop over chunks
    with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
        for start in start_times:

            # Divide the movie in chunks of length TR, and save the resulting
            # clips as '.mp4' files
            clip_chunk = clip.subclip(start, start+tr)
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
            inputs = transform(inputs).unsqueeze(0).to(device)

            # Extract the visual features
            with torch.no_grad():
                preds = feature_extractor(inputs)
            visual_features.append(np.reshape(preds[model_layer].cpu().numpy(), -1))

            # Update the progress bar
            pbar.update(1)

    # Convert the visual features to float32
    visual_features = np.array(visual_features, dtype='float32')

    # # Save the visual features
    # out_file_visual = os.path.join(
    #     save_dir_features, f'friends_s02e01a_features_visual_slow_r50.h5')
    # with h5py.File(out_file_visual, 'a' if Path(out_file_visual).exists() else 'w') as f:
    #     group = f.create_group("s02e01a")
    #     group.create_dataset('visual', data=visual_features, dtype=np.float32)
    # print(f"Visual features saved to {out_file_visual}")

    # Output
    #return visual_features

    # Save features to HDF5 file


    return visual_features


# 2) TimeSformer 
def get_timesformer_model(device):
    """
    Load Hugging Face Timesformer model and its processor.
    """
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model.to(device)
    model.eval()
    return model, processor
def extract_visual_features_timesformer(
    episode_path, tr, model, processor,
    device, save_dir_temp, save_dir_features,
    num_frames=8 # Add parameter for the number of frames to extract
):
    """
    Extract video chunk-level features using Timesformer from Hugging Face.
    """

    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]

    temp_dir = os.path.join(save_dir_temp, 'temp_timesformer')
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(save_dir_features, exist_ok=True)

    visual_features = []

    # Use OpenCV to open the video file for more precise frame extraction
    cap = cv2.VideoCapture(episode_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {episode_path}")
        return [] # Return empty list or raise an error

    fps = cap.get(cv2.CAP_PROP_FPS)

    with tqdm(total=len(start_times), desc="Extracting visual features (Timesformer)") as pbar:
        for start in start_times:
            # Calculate the start frame index based on the start time and FPS
            start_frame_index = int(start * fps)
            # Set the video position to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    # Handle cases where we might not get enough frames at the end
                    break
                # Convert the frame from BGR to RGB (OpenCV reads in BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

            # If we didn't get enough frames, skip this chunk or pad if necessary
            if len(frames) < num_frames:
                 # You might choose to pad with a black frame or skip
                 print(f"Warning: Not enough frames for chunk starting at {start}s. Skipping.")
                 continue # Skip the current chunk

            # Prepare input batch: processor will handle resizing, normalization, stacking frames
            # Ensure frames is a list of PIL images
            inputs = processor(frames, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Extract last hidden states or logits
            # outputs.logits shape: (batch_size=1, num_classes=400)
            # For features, use logits (or if you want deeper features, you'd need to modify model)
            feats = outputs.logits.cpu().numpy().reshape(-1)  # flatten logits vector
            visual_features.append(feats)

            pbar.update(1)

    cap.release() # Release the video capture object

    visual_features = np.array(visual_features, dtype=np.float32)

    return visual_features

# 3) VIT
def get_vit_model(device):
    """
    Load Hugging Face ViT model and its feature extractor.
    """
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    model.to(device)
    model.eval()

    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    return model, feature_extractor
def extract_visual_features_vit_hf(episode_path, tr, model, feature_extractor,
                                   device, save_dir_temp, save_dir_features):
    """
    Extract frame-level visual features using Hugging Face ViT, and average over TR segments.
    """
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]

    temp_dir = os.path.join(save_dir_temp, 'temp_vit_hf')
    os.makedirs(temp_dir, exist_ok=True)

    # Ensure the directory for saving features exists
    os.makedirs(save_dir_features, exist_ok=True)

    visual_features = []

    with tqdm(total=len(start_times), desc="Extracting visual features (HF ViT)") as pbar:
        for start in start_times:
            clip_chunk = clip.subclip(start, start + tr)
            chunk_path = os.path.join(temp_dir, 'visual_chunk.mp4')
            clip_chunk.write_videofile(chunk_path, verbose=False, audio=False, logger=None)

            video_clip = VideoFileClip(chunk_path)
            chunk_frames = [Image.fromarray(frame) for frame in video_clip.iter_frames()]

            frame_feats = []
            for frame in chunk_frames:
                inputs = feature_extractor(images=frame, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # [1, num_tokens, hidden_size]
                cls_token = last_hidden_state[:, 0, :]  # CLS token
                frame_feats.append(cls_token.squeeze(0).cpu().numpy())

            # Mean pool across frames for this TR
            if frame_feats:
                avg_feat = np.mean(np.stack(frame_feats), axis=0)
                visual_features.append(avg_feat)

            pbar.update(1)

    visual_features = np.array(visual_features, dtype='float32')

    return visual_features


# Audio feature extraction
# 1) MFCCs 
def extract_audio_features_MFCCs(episode_path, tr, sr, device, save_dir_temp,
    save_dir_features, audio_id):
    """
    Extract audio features from a movie using Mel-frequency cepstral
    coefficients (MFCCs).

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the audio features are extracted.
    tr : float
        Duration of each chunk, in seconds (aligned with the fMRI repetition
        time, or TR).
    sr : int
        Audio sampling rate.
    device : str
        Device to perform computations ('cpu' or 'gpu').
    save_dir_temp : str
        Directory where the chunked movie clips are temporarily stored for
        feature extraction.
    save_dir_features : str
        Directory where the extracted audio features are saved.

    Returns
    -------
    audio_features : float
        Array containing the extracted audio features.

    """

    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Empty features list
    audio_features = []

    ### Loop over chunks ###
    with tqdm(total=len(start_times), desc="Extracting audio features") as pbar:
        for start in start_times:

            # Divide the movie in chunks of length TR, and save the resulting
            # audio clips as '.wav' files
            clip_chunk = clip.subclip(start, start+tr)
            chunk_audio_path = os.path.join(temp_dir, 'audio_s01e01b.wav')
            clip_chunk.audio.write_audiofile(chunk_audio_path, verbose=False,
                logger=None)
            # Load the audio samples from the chunked movie clip
            y, sr = librosa.load(chunk_audio_path, sr=sr, mono=True)

            # Extract the audio features (MFCC)
            mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
            audio_features.append(mfcc_features)
            # Update the progress bar
            pbar.update(1)

    ### Convert the visual features to float32 ###
    audio_features = np.array(audio_features, dtype='float32')

    # # Save the audio features
    # out_file_audio = os.path.join(
    #     save_dir_features, f'friends_s02e01a_features_audio.h5')
    # with h5py.File(out_file_audio, 'a' if Path(out_file_audio).exists() else 'w') as f:
    #     group = f.create_group("s02e01a")
    #     group.create_dataset('audio', data=audio_features, dtype=np.float32)
    # print(f"Audio features saved to {out_file_audio}")


    ### Output ###
    return audio_features




# Text feature extraction
# 1) BERT
def get_language_model_BERT(device):
    """
    Load a pre-trained bert-base-uncased language model and its corresponding
    tokenizer.

    Parameters
    ----------
    device : torch.device
        Device on which the model will run (e.g., 'cpu' or 'cuda').

    Returns
    -------
    model : object
        Pre-trained language model.
    tokenizer : object
        Tokenizer corresponding to the language model.

    """

    ### Load the model ###
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval().to(device)

    ### Load the tokenizer ###
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
        do_lower_case=True)

    ### Output ###
    return model, tokenizer
def extract_language_features_BERT(episode_path, model, tokenizer, num_used_tokens,
    kept_tokens_last_hidden_state, device, save_dir_features, video_id):
    """
    Extract language features from a movie using a pre-trained language model.

    Parameters
    ----------
    episode_path : str
        Path to the movie transcripts for which the language features are
        extracted.
    model : object
        Pre-trained language model.
    tokenizer : object
        Tokenizer corresponding to the language model.
    num_used_tokens : int
        Total number of tokens that are fed to the language model for each
        chunk, including the tokens from the chunk of interest plus N tokens
        from previous chunks (the maximum allowed by the model is 510).
    kept_tokens_last_hidden_state : int
        Number of features retained for the last_hidden_state, where each
        feature corresponds to a token, starting from the most recent token.
    device : str
        Device to perform computations ('cpu' or 'gpu').
    save_dir_features : str
        Directory where the extracted language features are saved.

    Returns
    -------
    pooler_output : list
        List containing the pooler_output features for each chunk.
    last_hidden_state : list
        List containing the last_hidden_state features for each chunk

    """

    ### Load the transcript ###
    df = pd.read_csv(episode_path, sep='\t')
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    ### Initialize the tokens and features lists ###
    tokens, np_tokens, pooler_output, last_hidden_state = [], [], [], []

    ### Loop over text chunks ###
    for i in tqdm(range(df.shape[0]), desc="Extracting language features"):

        ### Tokenize raw text ###
        if not df.iloc[i]["is_na"]: # Only tokenize if words were spoken during a chunk (i.e., if the chunk is not empty)
            # Tokenize raw text with puntuation (for pooler_output features)
            tr_text = df.iloc[i]["text_per_tr"]
            tokens.extend(tokenizer.tokenize(tr_text))
            # Tokenize without punctuation (for last_hidden_state features)
            tr_np_tokens = tokenizer.tokenize(
                tr_text.translate(str.maketrans('', '', string.punctuation)))
            np_tokens.extend(tr_np_tokens)

        ### Extract the pooler_output features ###
        if len(tokens) > 0: # Only extract features if there are tokens available
            # Select the number of tokens used from the current and past chunks,
            # and convert them into IDs
            used_tokens = tokenizer.convert_tokens_to_ids(
                tokens[-(num_used_tokens):])
            # IDs 101 and 102 are special tokens that indicate the beginning and
            # end of an input sequence, respectively.
            input_ids = [101] + used_tokens + [102]
            tensor_tokens = torch.tensor(input_ids).unsqueeze(0).to(device)
            # Extract and store the pooler_output features
            with torch.no_grad():
                outputs = model(tensor_tokens)
                pooler_output.append(outputs['pooler_output'][0].cpu().numpy())
        else: # Store NaN values if no tokes are available
            pooler_output.append(np.full(768, np.nan, dtype='float32'))

        ### Extract the last_hidden_state features ###
        if len(np_tokens) > 0: # Only extract features if there are tokens available
            np_feat = np.full((kept_tokens_last_hidden_state, 768), np.nan, dtype='float32')
            # Select the number of tokens used from the current and past chunks,
            # and convert them into IDs
            used_tokens = tokenizer.convert_tokens_to_ids(
                np_tokens[-(num_used_tokens):])
            # IDs 101 and 102 are special tokens that indicate the beginning and
            # end of an input sequence, respectively.
            np_input_ids = [101] + used_tokens + [102]
            np_tensor_tokens = torch.tensor(np_input_ids).unsqueeze(0).to(device)
            # Extract and store the last_hidden_state features
            with torch.no_grad():
                np_outputs = model(np_tensor_tokens)
                np_outputs = np_outputs['last_hidden_state'][0][1:-1].cpu().numpy()
            tk_idx = min(kept_tokens_last_hidden_state, len(np_tokens))
            np_feat[-tk_idx:, :] = np_outputs[-tk_idx:]
            last_hidden_state.append(np_feat)
        else: # Store NaN values if no tokens are available
            last_hidden_state.append(np.full(
                (kept_tokens_last_hidden_state, 768), np.nan, dtype='float32'))

    ### Convert the language features to float32 ###
    pooler_output = np.array(pooler_output, dtype='float32')
    last_hidden_state = np.array(last_hidden_state, dtype='float32')



    ### Output ###
    return pooler_output, last_hidden_state

# 2) Roberta
def get_language_model_Roberta(device):

    ### Load the model ###
    model = RobertaModel.from_pretrained('roberta-large')
    model.eval().to(device)

    ### Load the tokenizer ###
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    hidden_size = model.config.hidden_size

    ### Output ###
    return model, tokenizer, hidden_size
def extract_language_features_Roberta(
    episode_path,
    model,
    tokenizer,
    hidden_size,
    num_used_tokens,
    kept_tokens_last_hidden_state,
    device,
    save_dir_features
):
    # Load transcript
    df = pd.read_csv(episode_path, sep='\t')
    df.insert(loc=0, column="is_na", value=df["text_per_tr"].isna())

    tokens, np_tokens = [], []
    pooler_output, last_hidden_state = [], []

    episode_id = Path(episode_path).stem
    model_tag = "roberta-large"
    os.makedirs(save_dir_features, exist_ok=True)
    out_file = os.path.join(save_dir_features, f"{episode_id}_features_language.h5")

    for i in tqdm(range(df.shape[0]), desc=f"Extracting features from {episode_id}"):

        if not df.iloc[i]["is_na"]:
            text = df.iloc[i]["text_per_tr"]
            tokens = tokenizer.tokenize(text)
            clean_text = text.translate(str.maketrans('', '', string.punctuation))
            np_tokens = tokenizer.tokenize(clean_text)
        else:
            tokens, np_tokens = [], []

        # Sentence embedding using first token (RoBERTa [CLS]-equivalent)
        if tokens:
            token_ids = tokenizer.convert_tokens_to_ids(tokens[-num_used_tokens:])
            input_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # First token
                pooler_output.append(cls_embedding.squeeze().cpu().numpy())
        else:
            pooler_output.append(np.full(hidden_size, np.nan, dtype='float32'))

        # Token-level embedding from last_hidden_state
        if np_tokens:
            token_ids = tokenizer.convert_tokens_to_ids(np_tokens[-num_used_tokens:])
            input_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]
            input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

            with torch.no_grad():
                token_outputs = model(input_tensor)
                token_embeddings = token_outputs.last_hidden_state[0][1:-1].cpu().numpy()  # skip special tokens

            np_feat = np.full((kept_tokens_last_hidden_state, hidden_size), np.nan, dtype='float32')
            tk_idx = min(kept_tokens_last_hidden_state, token_embeddings.shape[0])
            np_feat[-tk_idx:, :] = token_embeddings[-tk_idx:]
            last_hidden_state.append(np_feat)
        else:
            last_hidden_state.append(np.full((kept_tokens_last_hidden_state, hidden_size), np.nan, dtype='float32'))

    # Convert to arrays
    pooler_output = np.array(pooler_output, dtype='float32')
    last_hidden_state = np.array(last_hidden_state, dtype='float32')

    # Save to .h5
    with h5py.File(out_file, 'w') as f:
        episode_group = f.create_group(episode_id)
        language_group = episode_group.create_group("language")
        language_group.create_dataset("pooler_output", data=np.array(pooler_output, dtype='float32'))
        language_group.create_dataset("last_hidden_state", data=np.array(last_hidden_state, dtype='float32'))

    print(f" Saved features to: {out_file}")

    return pooler_output, last_hidden_state

# 3) miniLM
def get_minilm_model(device):
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    hidden_size = model.get_sentence_embedding_dimension()
    return model, hidden_size
def extract_minilm_language_features(episode_path, model, hidden_size, device, save_dir_features):
    # Load transcript
    df = pd.read_csv(episode_path, sep='\t')
    texts = df["text_per_tr"].fillna("").tolist()

    # Extract sentence embeddings
    print(f"Extracting sentence embeddings for {len(texts)} samples...")
    embeddings = model.encode(texts, convert_to_numpy=True, device=device, show_progress_bar=True)


    return embeddings


# Replace  NaN values in the stimulus features with zeros, and z-score the features
def preprocess_features(features):
    """
    Rplaces NaN values in the stimulus features with zeros, and z-score the
    features.

    Parameters
    ----------
    features : float
        Stimulus features.

    Returns
    -------
    prepr_features : float
        Preprocessed stimulus features.

    """

    ### Convert NaN values to zeros ###
    features = np.nan_to_num(features)

    ### Z-score the features ###
    scaler = StandardScaler()
    prepr_features = scaler.fit_transform(features)

    ### Output ###
    return prepr_features
# Update or create a .npy dictionary file
def update_npy_dict(new_key, new_features, save_path):
    """
    Update or create a .npy dictionary file.
    
    If the file exists, load it, update the dictionary with the new key and array,
    and save it back. If it doesn't exist, create a new dictionary and save it.
    
    Parameters:
        new_key (str): The key to insert or update (e.g., "s01e01a")
        new_features (np.ndarray): The features to store
        save_path (str): Path to the .npy file
    """
    if os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True).item()
        data[new_key] = new_features
        print(f"Updated existing file: added/overwritten key '{new_key}'")
    else:
        data = {new_key: new_features}
        print(f"Created new file with key '{new_key}'")
    
    np.save(save_path, data)



# Extract different modalities of features, store them in .npy files, and return the features
def extract_features_video(episode_path, video_id, video_extractor, movie):
    
    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Saving directories
    save_dir_temp = "./visual_features"
    #save_dir_features = root_data_dir +  "/stimulus_features/raw/visual/"
    save_dir_features = "./Visual"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = define_frames_transform()
    feature_extractor, model_layer = get_vision_model(device)

    # Execute visual feature extraction
    if video_extractor == "slow_r50":
        print("Extracting visual features using slow_r50 model...")
        visual_features = extract_visual_features_slowr50(episode_path, tr, feature_extractor,
            model_layer, transform, device, save_dir_temp, save_dir_features, video_id)
    elif video_extractor == "TimeSformer":
        print("Extracting visual features using Timesformer model...")
        model, processor = get_timesformer_model(device)
        visual_features = extract_visual_features_timesformer(
            episode_path, tr, model, processor, device, save_dir_temp,
            save_dir_features, num_frames=8)
    elif video_extractor == "ViT":
        print("Extracting visual features using ViT model...")
        model, feature_extractor = get_vit_model(device)
        visual_features = extract_visual_features_vit_hf(
            episode_path, tr, model, feature_extractor, device, save_dir_temp,
            save_dir_features)
    
    features = preprocess_features(visual_features)


    out_file_visual = os.path.join(save_dir_features, f'{movie}_{video_id}_features_visual_{video_extractor}.h5')
    os.makedirs(save_dir_features, exist_ok=True)

    with h5py.File(out_file_visual, 'a') as f:
        if 'visual' in f:
            del f['visual']  # Overwrite existing dataset if it already exists
        f.create_dataset('visual', data=features, dtype=np.float32)

    print(f"Visual features saved directly to dataset 'visual' in {out_file_visual}")
    


    # # add visual features to npy_visual.npy
    # update_npy_dict(video_id, features, "./Visual/visual_npy.npy")
    
    return features

def extract_features_audio(episode_path, video_id, audio_extractor, movie):


    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Audio sampling rate
    sr = 22050

    # Saving directories
    save_dir_temp = "./audio_features"
    #save_dir_features = root_data_dir +  "/stimulus_features/raw/audio/"
    save_dir_features = "./Audio"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if audio_extractor == "MFCCs":
        # Execute audio feature extraction
        audio_f = extract_audio_features_MFCCs(episode_path, tr, sr, device, save_dir_temp, save_dir_features, video_id)
    
    features = preprocess_features(audio_f)
    # add audio features to npy_audio.npy
    # update_npy_dict(video_id, features, "./Audio/audio_npy.npy")

    out_file_audio = os.path.join(save_dir_features, f'{movie}_{video_id}_features_audio_{audio_extractor}.h5')
    os.makedirs(save_dir_features, exist_ok=True)

    with h5py.File(out_file_audio, 'a') as f:
        if 'audio' in f:
            del f['audio']  # Overwrite existing dataset if it already exists
        f.create_dataset('audio', data=features, dtype=np.float32)

    print(f"Visual features saved directly to dataset 'visual' in {out_file_audio}")
    


    return features

    """
    Extract audio features from a movie using the specified audio extractor.

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the audio features are extracted.
    video_id : str
        Identifier for the video (e.g., 's01e01a').
    audio_extractor : str
        The audio extractor to use (e.g., 'mfcc').

    Returns
    -------
    audio_features : float
        Array containing the extracted audio features.

    """

    # Duration of each movie chunk, aligned with the fMRI TR of 1.49 seconds
    tr = 1.49

    # Audio sampling rate
    sr = 16000

    # Saving directories
    save_dir_temp = "./audio_features"
    save_dir_features = "./Audio"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Execute audio feature extraction
    if audio_extractor == "mfcc":
        print("Extracting audio features using MFCC...")
        audio_features = extract_audio_features(episode_path, tr, sr, device,
            save_dir_temp, save_dir_features)

    return audio_features

def extract_features_text(episode_path, video_id, text_extractor, movie):
    # Saving directory
    #save_dir_features = root_data_dir +  "/stimulus_features/raw/language/"
    save_dir_features = "./Text" # Other parameters
    num_used_tokens = 510
    kept_tokens_last_hidden_state = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Execute language feature extraction
    if text_extractor == "BERT":
        print("Extracting language features using BERT model...")
        model, tokenizer = get_language_model_BERT(device)
        pooler, last_state = extract_language_features_BERT(episode_path,
            model, tokenizer, num_used_tokens, kept_tokens_last_hidden_state, device,
            save_dir_features, video_id)
        last_hidden_flat = last_state.reshape(pooler.shape[0], -1)

        language_feature = np.append(pooler, last_hidden_flat, axis=1)
        
    elif text_extractor == "Roberta":
        print("Extracting language features using RoBERTa model...")
        model, tokenizer, hidden_size = get_language_model_Roberta(device)
        pooler, last_state = extract_language_features_Roberta(
            episode_path, model, tokenizer, hidden_size, num_used_tokens,
            kept_tokens_last_hidden_state, device, save_dir_features)
        
        language_feature = np.append(pooler, last_state, axis=1)
    
    elif text_extractor == "miniLM":
        print("Extracting language features using MiniLM model...")
        model, hidden_size = get_minilm_model(device)
        language_feature = extract_minilm_language_features(
            episode_path, model, hidden_size, device, save_dir_features)
        
    features = preprocess_features(language_feature)
    # add text features to npy_text.npy
    # update_npy_dict(video_id, features, "./Text/text_npy.npy")

    out_file_text = os.path.join(save_dir_features, f'{movie}_{video_id}_features_text_{text_extractor}.h5')
    os.makedirs(save_dir_features, exist_ok=True)

    with h5py.File(out_file_text, 'a') as f:
        if 'text' in f:
            del f['text']  # Overwrite existing dataset if it already exists
        f.create_dataset('text', data=features, dtype=np.float32)

    print(f"Visual features saved directly to dataset 'visual' in {out_file_text}")
    


    return features

# # merge features from 3 npy files, which each contain features from one modality
# def load_stimulus_features(modality):



#     """
#     Load the stimulus features.

#     Parameters
#     ----------
#     root_data_dir : str
#         Root data directory.
#     modality : str
#         Used feature modality.

#     Returns
#     -------
#     features : dict
#         Dictionary containing the stimulus features.

#     """

#     features = {}

#     ### Load the visual features ###
#     if modality == 'visual' or modality == 'all':
     
#         features['visual'] = np.load("./Visual/visual_npy.npy", allow_pickle=True).item()

#     ### Load the audio features ###
#     if modality == 'audio' or modality == 'all':

#         features['audio'] = np.load("./Audio/audio_npy.npy", allow_pickle=True).item()

#     ### Load the language features ###
#     if modality == 'language' or modality == 'all':
#         features['language'] = np.load("./Text/text_npy.npy", allow_pickle=True).item()

#     ### Output ###
#     return features
# def load_fmri(subject, video_id):
#     """
#     Load the fMRI responses for the selected subject.

#     Parameters
#     ----------
#     root_data_dir : str
#         Root data directory.
#     subject : int
#         Subject used to train and validate the encoding model.

#     Returns
#     -------
#     fmri : dict
#         Dictionary containing the  fMRI responses.

#     """

#     fmri = {}

#     ### Load the fMRI responses for Friends ###
#     # Data directory
#     fmri_file = f'sub-0{subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5'
#     #fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors','fmri', f'sub-0{subject}', 'func', fmri_file)
#     fmri_dir = os.path.join("./func", fmri_file)


#     # Load the the fMRI responses
#     fmri_friends = h5py.File(fmri_dir, 'r')
#     for key, val in fmri_friends.items():
#         if video_id in key:
#             fmri[str(key[13:])] = val[:].astype(np.float32)
#     del fmri_friends


#     # ### Load the fMRI responses for Movie10 ###
#     # # Data directory
#     # fmri_file = f'sub-0{subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5'
#     # fmri_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
#     #     'fmri', f'sub-0{subject}', 'func', fmri_file)
#     # # Load the the fMRI responses
#     # fmri_movie10 = h5py.File(fmri_dir, 'r')
#     # for key, val in fmri_movie10.items():
#     #     fmri[key[13:]] = val[:].astype(np.float32)
#     # del fmri_movie10
#     # # Average the fMRI responses across the two repeats for 'figures'
#     # keys_all = fmri.keys()
#     # figures_splits = 12
#     # for s in range(figures_splits):
#     #     movie = 'figures' + format(s+1, '02')
#     #     keys_movie = [rep for rep in keys_all if movie in rep]
#     #     fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
#     #     del fmri[keys_movie[0]]
#     #     del fmri[keys_movie[1]]
#     # # Average the fMRI responses across the two repeats for 'life'
#     # keys_all = fmri.keys()
#     # life_splits = 5
#     # for s in range(life_splits):
#     #     movie = 'life' + format(s+1, '02')
#     #     keys_movie = [rep for rep in keys_all if movie in rep]
#     #     fmri[movie] = ((fmri[keys_movie[0]] + fmri[keys_movie[1]]) / 2).astype(np.float32)
#     #     del fmri[keys_movie[0]]
#     #     del fmri[keys_movie[1]]

#     ### Output ###
#     return fmri
# def align_features_and_fmri_samples(features, fmri, excluded_samples_start,
#     excluded_samples_end, hrf_delay, stimulus_window, movies):
#     """
#     Align the stimulus feature with the fMRI response samples for the selected
#     movies, later used to train and validate the encoding models.

#     Parameters
#     ----------
#     features : dict
#         Dictionary containing the stimulus features.
#     fmri : dict
#         Dictionary containing the fMRI responses.
#     excluded_trs_start : int
#         Integer indicating the first N fMRI TRs that will be excluded and not
#         used for model training. The reason for excluding these TRs is that due
#         to the latency of the hemodynamic response the fMRI responses of first
#         few fMRI TRs do not yet contain stimulus-related information.
#     excluded_trs_end : int
#         Integer indicating the last N fMRI TRs that will be excluded and not
#         used for model training. The reason for excluding these TRs is that
#         stimulus feature samples (i.e., the stimulus chunks) can be shorter than
#         the fMRI samples (i.e., the fMRI TRs), since in some cases the fMRI run
#         ran longer than the actual movie. However, keep in mind that the fMRI
#         timeseries onset is ALWAYS SYNCHRONIZED with movie onset (i.e., the
#         first fMRI TR is always synchronized with the first stimulus chunk).
#     hrf_delay : int
#         fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
#         that reflects changes in blood oxygenation levels in response to
#         activity in the brain. Blood flow increases to a given brain region in
#         response to its activity. This vascular response, which follows the
#         hemodynamic response function (HRF), takes time. Typically, the HRF
#         peaks around 5–6 seconds after a neural event: this delay reflects the
#         time needed for blood oxygenation changes to propagate and for the fMRI
#         signal to capture them. Therefore, this parameter introduces a delay
#         between stimulus chunks and fMRI samples for a better correspondence
#         between input stimuli and the brain response. For example, with a
#         hrf_delay of 3, if the stimulus chunk of interest is 17, the
#         corresponding fMRI sample will be 20.
#     stimulus_window : int
#         Integer indicating how many stimulus features' chunks are used to model
#         each fMRI TR, starting from the chunk corresponding to the TR of
#         interest, and going back in time. For example, with a stimulus_window of
#         5, if the fMRI TR of interest is 20, it will be modeled with stimulus
#         chunks [16, 17, 18, 19, 20]. Note that this only applies to visual and
#         audio features, since the language features were already extracted using
#         transcript words spanning several movie chunks (thus, each fMRI TR will
#         only be modeled using the corresponding language feature chunk). Also
#         note that a larger stimulus window will increase compute time, since it
#         increases the amount of stimulus features used to train and test the
#         fMRI encoding models.
#     movies: list
#         List of strings indicating the movies for which the fMRI responses and
#         stimulus features are aligned, out of the first six seasons of Friends
#         ["friends-s01", "friends-s02", "friends-s03", "friends-s04",
#         "friends-s05", "friends-s06"], and the four movies from Movie10
#         ["movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"].

#     Returns
#     -------
#     aligned_features : float
#         Aligned stimulus features for the selected movies.
#     aligned_fmri : float
#         Aligned fMRI responses for the selected movies.

#     """

#     ### Empty data variables ###
#     aligned_features = []
#     aligned_fmri = np.empty((0,1000), dtype=np.float32)

#     ### Loop across movies ###
#     for movie in movies:

#         ### Get the IDs of all movies splits for the selected movie ###
#         if movie[:7] == 'friends':
#             id = movie[8:]
#         elif movie[:7] == 'movie10':
#             id = movie[8:]
#         movie_splits = [key for key in fmri if id in key[:len(id)]]

#         ### Loop over movie splits ###
#         for split in movie_splits:

#             ### Extract the fMRI ###
#             fmri_split = fmri[split]
#             # Exclude the first and last fMRI samples
#             fmri_split = fmri_split[excluded_samples_start:-excluded_samples_end]
#             aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

#             ### Loop over fMRI samples ###
#             for s in range(len(fmri_split)):
#                 # Empty variable containing the stimulus features of all
#                 # modalities for each fMRI sample
#                 f_all = np.empty(0)

#                 ### Loop across modalities ###
#                 for mod in features.keys():

#                     ### Visual and audio features ###
#                     # If visual or audio modality, model each fMRI sample using
#                     # the N stimulus feature samples up to the fMRI sample of
#                     # interest minus the hrf_delay (where N is defined by the
#                     # 'stimulus_window' variable)
#                     if mod == 'visual' or mod == 'audio':
#                         # In case there are not N stimulus feature samples up to
#                         # the fMRI sample of interest minus the hrf_delay (where
#                         # N is defined by the 'stimulus_window' variable), model
#                         # the fMRI sample using the first N stimulus feature
#                         # samples
#                         if s < (stimulus_window + hrf_delay):
#                             idx_start = excluded_samples_start
#                             idx_end = idx_start + stimulus_window
#                         else:
#                             idx_start = s + excluded_samples_start - hrf_delay \
#                                 - stimulus_window + 1
#                             idx_end = idx_start + stimulus_window
#                         # In case there are less visual/audio feature samples
#                         # than fMRI samples minus the hrf_delay, use the last N
#                         # visual/audio feature samples available (where N is
#                         # defined by the 'stimulus_window' variable)
#                         if idx_end > (len(features[mod][split])):
#                             idx_end = len(features[mod][split])
#                             idx_start = idx_end - stimulus_window
#                         f = features[mod][split][idx_start:idx_end]
#                         f_all = np.append(f_all, f.flatten())

#                     ### Language features ###
#                     # Since language features already consist of embeddings
#                     # spanning several samples, only model each fMRI sample
#                     # using the corresponding stimulus feature sample minus the
#                     # hrf_delay
#                     elif mod == 'language':
#                         # In case there are no language features for the fMRI
#                         # sample of interest minus the hrf_delay, model the fMRI
#                         # sample using the first language feature sample
#                         if s < hrf_delay:
#                             idx = excluded_samples_start
#                         else:
#                             idx = s + excluded_samples_start - hrf_delay
#                         # In case there are fewer language feature samples than
#                         # fMRI samples minus the hrf_delay, use the last
#                         # language feature sample available
#                         if idx >= (len(features[mod][split]) - hrf_delay):
#                             f = features[mod][split][-1,:]
#                         else:
#                             f = features[mod][split][idx]
#                         f_all = np.append(f_all, f.flatten())

#                  ### Append the stimulus features of all modalities for this sample ###
#                 aligned_features.append(f_all)

#     ### Convert the aligned features to a numpy array ###
#     aligned_features = np.asarray(aligned_features, dtype=np.float32)

#     ### Output ###
#     return aligned_features, aligned_fmri
# def merge_features(subject, video_id, phase):

#     modality = "all"  #@param ["visual", "audio", "language", "all"]
#     #modality = "all"

#     excluded_samples_start = 0  #@param {type:"slider", min:0, max:20, step:1}

#     excluded_samples_end = 1  #@param {type:"slider", min:0, max:20, step:1}

#     hrf_delay = 0  #@param {type:"slider", min:0, max:10, step:1}
#     #hrf_delay = 3
#     stimulus_window = 5  #@param {type:"slider", min:1, max:20, step:1}

#     movies_train = ["friends-s01"] # @param {allow-input: true}

#     #movies_train = ["friends-s01", "friends-s02", "friends-s03", "friends-s04", "friends-s05", "movie10-bourne", "movie10-figures", "movie10-life", "movie10-wolf"] # @param {allow-input: true}

#     movies_val = ["friends-s02"]
#     #movies_val = ["friends-s06"] # @param {allow-input: true}
    
#     # Load the stimulus features
#     features = load_stimulus_features(modality)
#     fmri = load_fmri(subject, video_id)

#     aligned_features, aligned_fmri = align_features_and_fmri_samples(features, fmri, excluded_samples_start,
#     excluded_samples_end, hrf_delay, stimulus_window,
#     movies_train)

#     if phase == "train":

#         update_npy_dict(video_id, aligned_features, "./features/features_train.npy")
#         update_npy_dict(video_id, aligned_fmri, "./features/fmri_train.npy")
    
#     elif phase == "test":
        
#         update_npy_dict(video_id, aligned_features, "./features/features_test.npy")
#         update_npy_dict(video_id, aligned_fmri, "./features/fmri_test.npy")


#     return aligned_features, aligned_fmri







if __name__ == "__main__":


    # training
    path_s01e01a_video = "./s01e01a/friends_s01e01a.mkv"
    path_s01e01a_text = "./s01e01a/friends_s01e01a.tsv"

    feature_1 = extract_features_video(path_s01e01a_video, "s01e01a", "slow_r50","friends")
    print(feature_1)
    feature_2 = extract_features_audio(path_s01e01a_video, "s01e01a", "MFCCs", "friends")
    print(feature_2)
    feature_3 = extract_features_text(path_s01e01a_text, "s01e01a", "BERT", "friends")
    print(feature_3)

    # features_train, fmri_train = merge_features(1, "s01e01a", "train")    #subject = 1, video_id = "s01e01a"


    # #testing 
    # path_s02e01a_video = "./s02e01a/friends_s02e01a.mkv"
    # path_s02e01a_text = "./s02e01a/friends_s02e01a.tsv"
    # feature_1 = extract_features_video(path_s02e01a_video, "s02e01a", "slow_r50")
    # feature_2 = extract_features_audio(path_s02e01a_video, "s02e01a", "MFCCs")
    # feature_3 = extract_features_text(path_s02e01a_text, "s02e01a", "BERT")
    # features_val, fmri_val = merge_features(1, "s02e01a", "test")  # subject = 1, video_id = "s02e01a"
  



