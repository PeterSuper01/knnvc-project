import os

from dotenv import load_dotenv
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import torch
from torch import Tensor

from audioprocess.readwav import WavReader
from config import settings


class WavEncoder:
    def __init__(self, settings):
        self.device = settings.device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-large", token=settings.hf_token
        )
        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large", token=settings.hf_token
        ).to(self.device)

        self.model.eval()

    def __call__(self, audio: Tensor) -> Tensor:
        """
        audio: [T]
        """
        inputs = self.processor(
            audio, return_tensors="pt", sampling_rate=settings.encoder_sample_rate
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # return 6th layer's hidden state
        return outputs.hidden_states[6]


if __name__ == "__main__":
    encoder = WavEncoder()
    audio = WavReader("data/miko.wav").process_wav()
    print(encoder(audio).shape)
