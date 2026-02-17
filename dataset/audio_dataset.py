import torch
import torchaudio
from torch.utils.data import Dataset

from audio_dataset_transformation_config import get_rnn_mfcc_transformation, get_mel_transformation
from config import AUDIO_DIRS, TARGET_SAMPLE_RATE, NUM_SAMPLES
from split_dataset import split_dataset
from utils import get_device


class AudioDataset(Dataset):
    def __init__(self, file_list, labels,transformation):
        self.files = file_list
        self.labels = labels
        self.transformation = transformation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]
        signal, sr = torchaudio.load(filepath)

        signal = self._resample_dataset(signal, sr)
        signal = self._mix_down_dataset(signal)
        signal = self._cut_dataset(signal)
        signal = self._right_pad_dataset(signal)
        signal = self.transformation(signal)
        signal=signal.to(get_device())
        return signal, label



    def _resample_dataset(self,signal, sample_rate):
        if sample_rate != TARGET_SAMPLE_RATE:
          resampler=torchaudio.transforms.Resample(sample_rate,TARGET_SAMPLE_RATE)
          signal = resampler(signal)
        return signal

    def _mix_down_dataset(self, signal):
        if signal.shape[0] > 1:
           signal=torch.mean(signal,dim=0,keepdim=True)
        return signal

    def _cut_dataset(self, signal):
        if signal.shape[1] > NUM_SAMPLES:
            signal = signal[:, :NUM_SAMPLES]
        return signal

    def _right_pad_dataset(self, signal):
        sample_length = signal.shape[1]
        if sample_length < NUM_SAMPLES:
            missing_samples = NUM_SAMPLES - sample_length
            last_dim_padding = (0, missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal



if __name__ == '__main__':
    (train_files, train_labels), _, _ = split_dataset()
    train_data_set = AudioDataset(train_files, train_labels, get_mel_transformation())
    for i in range(11):
        item, label = train_data_set[i]
        print(item.shape)




