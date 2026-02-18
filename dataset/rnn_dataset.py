
from dataset.audio_dataset import AudioDataset
from audio_dataset_transformation_config import get_mfcc_transformation
from config import RNN_SAMPLES_PER_FRAME, RNN_WINDOWS_PER_FRAME, RNN_N_FFT, RNN_HOP_LENGTH
from split_dataset import split_dataset
import torchaudio
import torch
from utils import get_device


class LSTMAudioDataset(AudioDataset):
    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]

        info = torchaudio.info(filepath)

        # Modern torchaudio (2.0+) style
        num_frames = info.num_frames
        sample_rate = info.sample_rate
        duration_sec = num_frames / sample_rate
        #print(f"Duration: {duration_sec:.3f} seconds")

        # Load raw signal
        signal, sr = torchaudio.load(filepath)

        # 1. Preprocess specifically for LSTM
        signal = self._resample_dataset(signal, sr)
        signal = self._mix_down_dataset(signal)

        # --- DYNAMIC CALCULATION STARTS HERE ---
        # Get actual duration from the resampled signal
        actual_samples = signal.shape[1]

        # Determine how many  frames fit
        num_frames = max(1, round(actual_samples / RNN_SAMPLES_PER_FRAME))

        # Calculate EXACT samples needed for the MFCC grid (Total Windows = Frames * 8)
        total_windows_needed = num_frames * RNN_WINDOWS_PER_FRAME

        # Formula: (W - 1) * Hop + N_FFT
        # For 1s (10 frames): (80-1)*275 + 495 = 22220 samples
        dynamic_target = (total_windows_needed - 1) * RNN_HOP_LENGTH + RNN_N_FFT

        # Apply the dynamic length adjustment
        signal = self._adjust_length(signal, dynamic_target)

        # 2. Transform -> Should result in (1, N_MFCC, total_windows_needed)
        signal = self.transformation(signal)

        # 3. Shape for LSTM: (Total_Frames,Windows, N_MFCC)
        signal = signal.squeeze(0).transpose(0, 1) # [Windows, N_MFCC]

        signal = signal.view(num_frames, RNN_WINDOWS_PER_FRAME, -1)

        signal = signal.to(get_device())
        return signal, label

    def _adjust_length(self, signal, target):
        length = signal.shape[1]
        if length > target:
            return signal[:, :target]
        return torch.nn.functional.pad(signal, (0, target - length))

class AudioDatasetOfSpectogramAndMFCCForRNN(AudioDataset):
    def __getitem__(self, idx):
       signal, label = super().__getitem__(idx)
        #Transpose for RNN
       signal=signal.squeeze(0).transpose(0, 1)
       return signal, label



if __name__ == '__main__':
    (train_files, train_labels), _,_ = split_dataset()
    train_data_set=AudioDatasetOfSpectogramAndMFCCForRNN(train_files, train_labels, get_mfcc_transformation())
    for i in range(11):
        item,label=train_data_set[i]
        print(item.shape)
