import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import torch
from dataset.audio_dataset import AudioDataset


class AudioDatasetForSVM(AudioDataset):
    def __getitem__(self, idx):
        # Get the standard transformed signal [1, Freq, Time]
        signal, label = super().__getitem__(idx)

        # 1. Remove channel dim -> [Freq, Time]
        signal = signal.squeeze(0)

        # 2. Statistical Pooling (Common for SVMs)
        # We take the mean across the time dimension to get a fixed-size vector
        # Result shape: [Freq]
        mean_features = torch.mean(signal, dim=1)
        std_features = torch.std(signal, dim=1)

        # Combine mean and std for a more robust feature vector
        combined_features = torch.cat((mean_features, std_features), dim=0)

        # SVM needs numpy arrays, not torch tensors
        return combined_features.numpy(), label