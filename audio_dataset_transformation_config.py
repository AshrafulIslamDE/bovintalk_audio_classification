import torchaudio

from config import TARGET_SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, RNN_TARGET_SAMPLE_RATE, RNN_N_MFCC, RNN_N_FFT, \
    RNN_HOP_LENGTH, RNN_N_MELS


def get_spectrogram_transformation():
    mel_transform = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2
    )
    return mel_transform

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

def get_rnn_mfcc_transformation():
    mel_transform = torchaudio.transforms.MFCC(
        sample_rate=RNN_TARGET_SAMPLE_RATE,
        n_mfcc=RNN_N_MFCC,
        melkwargs={
            "n_fft": RNN_N_FFT,
            "hop_length": RNN_HOP_LENGTH,
            "n_mels": RNN_N_MELS,
            "center": False
        }
    )
    return mel_transform