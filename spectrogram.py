import torch
import torchaudio
from config import TARGET_SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH

class SpectrogramGenerator:
    def __init__(self):
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=TARGET_SAMPLE_RATE
        )

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

    def generate(self, waveform, sample_rate):
        # Convert stereo → mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = self.resampler(waveform)

        # Mel spectrogram
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)  # log compression

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        return mel
