import torch
from torch.utils.data import Dataset

from utils import get_device


class VGGMelDataset(Dataset):
    def __init__(self, audio_dataset):
        self.audio_dataset = audio_dataset

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, idx):
        mel, label = self.audio_dataset[idx]  # mel shape: (1, mel_bins, time)

        # Resize to (224,224)
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0), size=(224, 224), mode="bilinear"
        ).squeeze(0)

        # Convert 1 → 3 channels
        mel = mel.repeat(3, 1, 1)

        return mel, label
