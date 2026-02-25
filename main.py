import torchaudio

from config import settings
from models.encoder import WavEncoder
from models.vocoder import HifiGANVocoder
from audioprocess.readwav import WavReader
from models.knn import knn_search


def main():
    encoder = WavEncoder()
    vocoder = HifiGANVocoder()

    source_audio = WavReader(settings.source_wav_path).process_wav()
    matching_audio = WavReader(settings.matching_wav_path).process_wav()

    encoded_source_audio = encoder(source_audio)[0]
    encoded_matching_audio = encoder(matching_audio)[0]

    pred_source_audio = knn_search(encoded_source_audio, encoded_matching_audio, k=4)
    decoded_audio = vocoder(pred_source_audio.unsqueeze(0))

    print(decoded_audio.shape)
    torchaudio.save(
        settings.output_wav_path,
        decoded_audio,
        settings.vocoder_sample_rate,
        format="wav",
    )


if __name__ == "__main__":
    main()
