import torchaudio

from config import settings
from models.encoder import WavEncoder
from models.vocoder import HifiGANVocoder
from audioprocess.readwav import WavReader
from models.knn import knn_search


def main():
    encoder = WavEncoder(settings)
    vocoder = HifiGANVocoder(settings)

    source_audio = (
        WavReader(settings.source_wav_path).process_wav(settings).to(settings.device)
    )
    matching_audio = (
        WavReader(settings.matching_wav_path).process_wav(settings).to(settings.device)
    )

    encoded_source_audio = encoder(source_audio)[0]
    encoded_matching_audio = encoder(matching_audio)[0]

    pred_source_audio = knn_search(
        encoded_source_audio, encoded_matching_audio, k=4, device=settings.device
    )
    decoded_audio = vocoder(pred_source_audio.unsqueeze(0))

    torchaudio.save(
        settings.output_wav_path, decoded_audio, settings.vocoder_sample_rate
    )


if __name__ == "__main__":
    main()
