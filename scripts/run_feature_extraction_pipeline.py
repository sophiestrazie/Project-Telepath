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

from multimodal_stimulus_fmri_prediction.audio import AudioConfig, AudioProcessor
from multimodal_stimulus_fmri_prediction.video import VideoConfig, VideoProcessor
from multimodal_stimulus_fmri_prediction.language import LanguageConfig, LanguageProcessor
import os

def main():
    #episode_path = "data/movie/friends/friends_s01e01a.mkv"
    episode_id = "s01e01a"

    # config = AudioConfig(
    #     save_dir_temp="temp_audio_chunks",
    #     save_dir_features="extracted_features/audio",
    # )

    # os.makedirs(config.save_dir_temp, exist_ok=True)
    # os.makedirs(config.save_dir_features, exist_ok=True)

    # processor = AudioProcessor(config)
    # features = processor.process(episode_path)
    # processor.save_features(features, episode_id)

    # print("Audio feature extraction completed.")
    # print("Extracted feature shape:", features.shape)

    # video_config = VideoConfig()

    # os.makedirs(video_config.save_dir_temp, exist_ok=True)
    # os.makedirs(video_config.save_dir_features, exist_ok=True)

    # processor = VideoProcessor(video_config)
    # features = processor.process(episode_path)
    # processor.save_features(features, episode_id)

    # print("Video feature extraction completed.")
    # print("Extracted feature shape:", features.shape)

    # Other parameters
    episode_path = "data/transcripts/friends_s01e01a.tsv"

    num_used_tokens = 510
    kept_tokens_last_hidden_state = 10

    language_config = LanguageConfig(num_used_tokens=num_used_tokens, kept_tokens_last_hidden_state=kept_tokens_last_hidden_state)

    os.makedirs(language_config.save_dir_features, exist_ok=True)

    processor = LanguageProcessor(language_config)
    pooler_output, last_hidden_state = processor.process(episode_path)
    processor.save_features(pooler_output, last_hidden_state, episode_id)

    print("Language feature extraction completed.")
    print("Extracted feature shape:", pooler_output.shape)

if __name__ == "__main__":
    main()
