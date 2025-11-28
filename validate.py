import torch
from torch.utils.data import DataLoader
from data_processing import AudioDataset
from model_architecture import AudioCNN
from split_dataset import split_dataset
from config import BATCH_SIZE, MODEL_PATH

_, (val_files, val_labels), _ = split_dataset()
val_dataset = AudioDataset(val_files, val_labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = AudioCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

correct = 0
with torch.no_grad():
    for mel, label in val_loader:
        mel = mel.unsqueeze(1)
        pred = model(mel)
        correct += (pred.argmax(1) == label).sum().item()

val_acc = correct / len(val_dataset)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
