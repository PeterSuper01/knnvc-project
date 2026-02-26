from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
import torch


class Settings(BaseSettings):
    hf_token: str
    source_wav_path: str
    matching_wav_path: str
    output_wav_path: str
    encoder_sample_rate: int = 16000
    vocoder_sample_rate: int = 16000
    k: int = 4

    model_config = SettingsConfigDict(env_file=".env")

    @computed_field
    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
