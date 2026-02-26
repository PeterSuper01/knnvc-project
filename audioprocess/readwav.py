import torchaudio


class WavReader:
    def __init__(self, file_path: str):
        self.waveform, self.sample_rate = torchaudio.load(file_path)

    def read_wav(self):
        return self.waveform, self.sample_rate

    def mono_wav(self):
        if self.waveform.shape[0] > 1:
            self.waveform = self.waveform.mean(dim=0).unsqueeze(0)
        return self.waveform

    def resample_wav(self, target_sample_rate: int = 16000):
        self._resampler = torchaudio.transforms.Resample(
            self.sample_rate, target_sample_rate
        )
        return self._resampler(self.waveform)

    def process_wav(self, settings):
        self.mono_wav()
        processed_wav = self.resample_wav(settings.encoder_sample_rate).to(
            settings.device
        )
        return processed_wav.squeeze(0)


if __name__ == "__main__":
    wav_reader = WavReader("data/miko.wav")
    processed_wav = wav_reader.process_wav()
    print(processed_wav.shape)
