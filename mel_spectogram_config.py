import torchaudio

from config import TARGET_SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS


def get_transformation():
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    return mel_transform