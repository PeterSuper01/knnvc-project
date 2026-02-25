from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str
    source_wav_path: str
    matching_wav_path: str
    output_wav_path: str
    encoder_sample_rate: int = 16000
    vocoder_sample_rate: int = 16000
    k: int = 4
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
