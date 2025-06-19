import os
import torch
import ffmpeg
import pandas as pd
import h5py
from tqdm import tqdm
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.data import (
    load_and_transform_text,
    load_and_transform_vision_data,
    load_and_transform_audio_data,
)

# Base directories
BASE_DIR = os.path.join("scratch", "algonauts_2025.competitors", "friends.stimuli", "s1")
RESULTS_DIR = os.path.join("scratch", "ImageBind", "results")
CHECKPOINT_PATH = ".checkpoints/imagebind_huge.pth"

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# File paths
TSV_PATH = os.path.join(BASE_DIR, "friends_s01e01a.tsv")
MKV_PATH = os.path.join(BASE_DIR, "friends_s01e01a.mkv")
OUTPUT_PT_PATH = os.path.join(RESULTS_DIR, "friends_s01e01a.pt")

# Config
TR_DURATION = 1.49
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load ImageBind model
model = imagebind_model.imagebind_huge(pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.to(DEVICE)
model.eval()


def extract_middle_frame(video_path, out_path, start):
    ffmpeg.input(video_path, ss=start + TR_DURATION / 2).filter('scale', 224, 224) \
        .output(out_path, vframes=1).overwrite_output().run(quiet=True)


def extract_audio_segment(video_path, out_path, start):
    ffmpeg.input(video_path, ss=start, t=TR_DURATION).output(out_path, ac=1, ar='16000') \
        .overwrite_output().run(quiet=True)


def get_transcript(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    return df['text_per_tr'].fillna('').tolist()


def get_imagebind_embeddings(text, frame_path, audio_path):
    inputs = {
        ModalityType.TEXT: load_and_transform_text([text], device=DEVICE),
        ModalityType.VISION: load_and_transform_vision_data([frame_path], device=DEVICE),
        ModalityType.AUDIO: load_and_transform_audio_data([audio_path], device=DEVICE)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return (
        embeddings[ModalityType.TEXT],
        embeddings[ModalityType.VISION],
        embeddings[ModalityType.AUDIO]
    )


def process_file(tsv_file, mkv_file, out_pt_file):
    transcript = get_transcript(tsv_file)
    text_embeddings, vision_embeddings, audio_embeddings = [], [], []

    for i, text in tqdm(enumerate(transcript), total=len(transcript), desc=os.path.basename(tsv_file)):
        start_time = i * TR_DURATION
        frame_file = f"tmp_frame_{i}.jpg"
        audio_file = f"tmp_audio_{i}.wav"

        extract_middle_frame(mkv_file, frame_file, start_time)
        extract_audio_segment(mkv_file, audio_file, start_time)

        try:
            t, v, a = get_imagebind_embeddings(text, frame_file, audio_file)
            text_embeddings.append(t.cpu())
            vision_embeddings.append(v.cpu())
            audio_embeddings.append(a.cpu())
        except Exception as e:
            print(f"Skipping TR {i} due to error: {e}")
        finally:
            for f in [frame_file, audio_file]:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass

    embeddings_dict = {
        "text": torch.cat(text_embeddings, dim=0),
        "vision": torch.cat(vision_embeddings, dim=0),
        "audio": torch.cat(audio_embeddings, dim=0),
    }

    torch.save(embeddings_dict, out_pt_file)
    basename = os.path.splitext(os.path.basename(tsv_file))[0]

    for modality in ["text", "vision", "audio"]:
        data = embeddings_dict[modality]
        out_path = os.path.join(RESULTS_DIR, f"{basename}_{modality}_ImageBind.h5")

        if os.path.exists(out_path):
            print(f"[SKIP] {out_path} already exists.")
            continue

        with h5py.File(out_path, "w") as f:
            f.create_dataset("embedding", data=data.cpu().numpy())
            f.attrs["modality"] = modality
            f.attrs["episode"] = basename
            f.attrs["shape"] = data.shape
            print(f"[WRITE] Saved {out_path}")


# Run processing
process_file(TSV_PATH, MKV_PATH, OUTPUT_PT_PATH)
