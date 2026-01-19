import torchaudio

from config import TARGET_SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_MFCC


def get_mel_transformation():
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    return mel_transform

def get_mfcc_transformation():
    mel_transform = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS
        }
    )
    return mel_transform
