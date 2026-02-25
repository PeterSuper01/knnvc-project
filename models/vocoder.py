import torch
from torch import Tensor

from models.encoder import WavEncoder
from audioprocess.readwav import WavReader


class HifiGANVocoder:
    def __init__(self):
        self.model = torch.hub.load(
            "bshall/knn-vc", "hifigan_wavlm", trust_repo=True, device="cpu"
        )[0]
        self.model.eval()

    def __call__(self, x: Tensor) -> Tensor:
        """
        x: [Batch, T, 1024]
        """
        with torch.no_grad():
            wav = self.model(x)
            return wav.squeeze()


if __name__ == "__main__":
    vocoder = HifiGANVocoder()
    test_feature = torch.randn(1, 100, 1024)

    audio = vocoder(test_feature)
    print(audio.shape)
