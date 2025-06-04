# import argparse

# def main(args):
#     ...

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run feature extraction pipeline")
#     parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
#     parser.add_argument("--output", type=str, required=True, help="Output directory for extracted features")
    
#     args = parser.parse_args()
    
#     main(args)

# scripts/run_feature_extraction_pipeline.py

from your_package_name.audio.extract import extract_audio_features
import os

def main():
    # Set input and output paths
    episode_path = "data/movie/friends/friends_s01e01a.mkv"
    save_dir_temp = "temp_audio_chunks"
    save_dir_features = "extracted_features/audio"

    # Parameters
    tr = 1.49
    sr = 22050
    device = "cpu"
    save_format = "h5"
    episode_id = "s01e01a"

    # Ensure directories exist
    os.makedirs(save_dir_temp, exist_ok=True)
    os.makedirs(save_dir_features, exist_ok=True)

    # Run extraction
    features = extract_audio_features(
        episode_path=episode_path,
        tr=tr,
        sr=sr,
        save_dir_temp=save_dir_temp,
        save_dir_features=save_dir_features,
        device=device,
        save_format=save_format,
        episode_id=episode_id
    )

    print("Audio feature extraction completed.")
    print("Extracted feature shape:", features.shape)


if __name__ == "__main__":
    main()
