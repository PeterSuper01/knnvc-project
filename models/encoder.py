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
        self.window_sec = settings.encoder_window_sec
        self.overlap_sec = settings.encoder_overlap_sec
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
        window_size = int(self.window_sec * self.sr)
        overlap_size = int(self.overlap_sec * self.sr)
        hop_size = window_size - overlap_size

        if audio.shape[0] < window_size:
            return self._forward(audio)

        all_hidden_states = []
        break_flag = False

        for start in range(0, len(audio), hop_size):
            if start + 2 * window_size - 1 * overlap_size > len(audio):
                end = len(audio)
                break_flag = True
            else:
                end = start + window_size

            audio_chunk = audio[start:end]

            hidden = self._forward(audio_chunk)  # [1, L_chunk, 1024]

            cut = int(overlap_size / 320 / 2)
            if start == 0:
                curr_hidden = hidden[:, :-cut, :]
            elif end == len(audio):
                curr_hidden = hidden[:, cut:, :]
            else:
                curr_hidden = hidden[:, cut:-cut, :]

            all_hidden_states.append(curr_hidden)
            if break_flag:
                break

        return torch.cat(all_hidden_states, dim=1)

    def _forward(self, audio_chunk: Tensor) -> Tensor:
        inputs = self.processor(audio_chunk, return_tensors="pt", sampling_rate=self.sr)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # print(inputs["input_values"].shape)
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[6]


if __name__ == "__main__":
    encoder = WavEncoder(settings)
    audio = WavReader("data/hinako_all.wav").process_wav(settings)
    print(audio.shape)
    print(encoder(audio).shape)
