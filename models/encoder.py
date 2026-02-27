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
        self.sr = settings.encoder_sample_rate
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-large", token=settings.hf_token
        )
        self.model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large", token=settings.hf_token
        ).to(self.device)

        self.model.eval()

    def __call__(self, audio: Tensor) -> Tensor:
        """
        audio: [L]
        return: [B, T, 1024]
        T is the number of frames in the audio, which is L / 320
        """
        window_sec = 20.0
        overlap_sec = 2.0

        window_size = int(window_sec * self.sr)
        overlap_size = int(overlap_sec * self.sr)
        hop_size = window_size - overlap_size

        if audio.shape[0] < window_size:
            return self._forward(audio)

        all_hidden_states = []
        for start in range(0, len(audio), hop_size):
            end = min(start + window_size, len(audio))
            audio_chunk = audio[start:end]

            if len(audio_chunk) < self.sr:
                break

            hidden = self._forward(audio_chunk)  # [1, L_chunk, 1024]

            cut = int(overlap_size / 320 / 2)
            if start == 0:
                curr_hidden = hidden[:, :-cut, :]
            elif end == len(audio):
                curr_hidden = hidden[:, cut:, :]
            else:
                curr_hidden = hidden[:, cut:-cut, :]

            all_hidden_states.append(curr_hidden)

        return torch.cat(all_hidden_states, dim=1)

    def _forward(self, audio_chunk: Tensor) -> Tensor:
        inputs = self.processor(audio_chunk, return_tensors="pt", sampling_rate=self.sr)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[6]


if __name__ == "__main__":
    encoder = WavEncoder(settings)
    audio = WavReader("data/miko.wav").process_wav(settings)
    print(encoder(audio).shape)
