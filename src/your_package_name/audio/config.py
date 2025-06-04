from dataclasses import dataclass

@dataclass
class AudioConfig:
    chunk_duration: float = 1.49
    sampling_rate: int = 22050
    device: str = "cpu"
    save_dir_temp: str = "./temp_audio_chunks"
    save_dir_features: str = "./features/audio"
    save_format: str = "h5"
