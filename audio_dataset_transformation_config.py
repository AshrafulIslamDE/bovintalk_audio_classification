import torchaudio

import config



def get_spectrogram_transformation():
    mel_transform = torchaudio.transforms.Spectrogram(
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        power=2
    )
    return mel_transform

def get_mel_transformation():
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.TARGET_SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    )
    return mel_transform

def get_mfcc_transformation():
    mel_transform = torchaudio.transforms.MFCC(
        sample_rate=config.TARGET_SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        melkwargs={
            "n_fft": config.N_FFT,
            "hop_length": config.HOP_LENGTH,
            "n_mels": config.N_MELS
        }
    )
    return mel_transform

def get_rnn_mfcc_transformation():
    mel_transform = torchaudio.transforms.MFCC(
        sample_rate=config.RNN_TARGET_SAMPLE_RATE,
        n_mfcc=config.RNN_N_MFCC,
        melkwargs={
            "n_fft": config.RNN_N_FFT,
            "hop_length": config.RNN_HOP_LENGTH,
            "n_mels": config.RNN_N_MELS,
            "center": False
        }
    )
    return mel_transform