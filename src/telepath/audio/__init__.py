# from dataclasses import dataclass

# @dataclass
# class AudioConfig:
#     sample_rate: int = 16000
#     num_channels: int = 1
#     duration: float = 5.0

# config = AudioConfig(sample_rate=100)

# def get_audio_config(config):
#     return config.sample_rate

# class AudioProcessor():
#     def __init__(self, config):
#         self.config = config

#     def process(self, audio_data):
#         # Dummy processing logic
#         return audio_data * self.config.sample_rate / 16000

from .config import AudioConfig
from .extract import AudioProcessor

__all__ = ["AudioConfig", "AudioProcessor"]


