from dataclasses import dataclass


@dataclass
class VideoConfig:
    chunk_duration: float = 1.49
    device: str = "cpu"
    save_dir_temp: str = "temp_video_chunks"
    save_dir_features: str = "extracted_features/video"
    save_format = "h5"
