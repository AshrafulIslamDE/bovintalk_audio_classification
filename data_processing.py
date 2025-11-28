import os
import torch
import torchaudio
from torch.utils.data import Dataset

from spectrogram import SpectrogramGenerator
from config import AUDIO_DIRS


class AudioDataset(Dataset):
    def __init__(self, file_list, labels):
        self.files = file_list
        self.labels = labels
        self.spect = SpectrogramGenerator()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(filepath)
        mel = self.spect.generate(waveform, sr)
        return mel, label

def load_all_files(base_dir="."):
    files = []
    labels = []
    for label_name, folder in AUDIO_DIRS.items():
        folder_path = os.path.join(base_dir, folder)
        label_idx = 0 if label_name == "HFC" else 1
        for file in os.listdir(folder_path):
            if file.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma")):
                files.append(os.path.join(folder_path, file))
                labels.append(label_idx)
    return files, labels
