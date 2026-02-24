from dataset.audio_dataset import AudioDataset
import torch

class TransferLearningDataset(AudioDataset):
    def __getitem__(self, idx):
        signal,label=super().__getitem__(idx)
        signal= torch.nn.functional.interpolate(
            signal.unsqueeze(0), size=(224, 224), mode="bilinear"
        ).squeeze(0)

        signal=signal.repeat(3,1,1)
        return signal,label
