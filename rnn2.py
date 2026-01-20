from audio_dataset import AudioDataset
from audio_dataset import AudioDataset
from audio_dataset_transformation_config import get_rnn_mfcc_transformation
from config import RNN_SAMPLES_PER_FRAME, RNN_WINDOWS_PER_FRAME, RNN_N_FFT, RNN_HOP_LENGTH, RNN_TARGET_SAMPLE_RATE, \
    RNN_FRAME_DURATION, RNN_OVERLAP_DURATION, RNN_N_MFCC, RNN_N_MELS
from split_dataset import split_dataset
import torchaudio
import torch
from utils import get_device

class LSTMAudioDataset(AudioDataset):
    def __getitem__(self, idx):
        filepath = self.files[idx]
        label = self.labels[idx]

        signal, sr = torchaudio.load(filepath)
        signal = self._resample_dataset(signal, sr)
        signal = self._mix_down_dataset(signal)

        frame_samples = int(RNN_TARGET_SAMPLE_RATE * RNN_FRAME_DURATION)  # 2205
        overlap_samples = int(RNN_TARGET_SAMPLE_RATE * RNN_OVERLAP_DURATION)  # 220
        hop_length = (frame_samples - overlap_samples) // (RNN_WINDOWS_PER_FRAME - 1)
        n_fft = hop_length + overlap_samples

        # number of 0.1s frames
        num_frames = signal.shape[1] // frame_samples
        signal = signal[:, :num_frames * frame_samples]

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=RNN_TARGET_SAMPLE_RATE,
            n_mfcc=RNN_N_MFCC,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "n_mels": RNN_N_MELS,
                "center": False
            }
        )

        frames = []
        for i in range(num_frames):
            frame = signal[:, i*frame_samples:(i+1)*frame_samples]
            mfcc = mfcc_transform(frame)  # (1, 13, 8)
            mfcc = mfcc.squeeze(0).transpose(0, 1)  # (8, 13)
            frames.append(mfcc)

        signal = torch.stack(frames)  # (num_frames, 8, 13)
        return signal.to(get_device()), label

if __name__ == '__main__':
    (train_files, train_labels), _,_ = split_dataset()
    train_data_set=LSTMAudioDataset(train_files,train_labels,get_rnn_mfcc_transformation())
    print(RNN_SAMPLES_PER_FRAME,RNN_HOP_LENGTH,RNN_TARGET_SAMPLE_RATE)
    for i in range(11):
        item,label=train_data_set[i]
        print(item.shape)