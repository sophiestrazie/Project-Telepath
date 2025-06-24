from dataclasses import dataclass


@dataclass
class LanguageConfig:
    num_used_tokens:int
    kept_tokens_last_hidden_state : int
    chunk_duration: float = 1.49
    device: str = "cpu"
    save_dir_features: str = "extracted_features/language"
    save_format = "h5"