import torch
import torchaudio
import random

class AudioAugmentor:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def add_noise(self, signal, noise_level=0.005):
        noise = torch.randn_like(signal) * noise_level
        return signal + noise

    def shift_pitch(self, signal):
        # Shifts pitch by -2 to +2 semitones
        n_steps = random.randint(-2, 2)
        return torchaudio.functional.pitch_shift(signal, self.sr, n_steps)

    def time_stretch(self, signal):
        # Changes speed between 0.8x and 1.2x
        rate = random.uniform(0.8, 1.2)
        return torchaudio.functional.resample(signal, self.sr, int(self.sr * rate))