import torch
from torch.utils.data import DataLoader
from data_processing import AudioDataset
from model_architecture import AudioCNN
from split_dataset import split_dataset
from config import BATCH_SIZE, MODEL_PATH

_, _, (test_files, test_labels) = split_dataset()
test_dataset = AudioDataset(test_files, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = AudioCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

correct = 0
with torch.no_grad():
    for mel, label in test_loader:
        mel = mel.unsqueeze(1)
        pred = model(mel)
        correct += (pred.argmax(1) == label).sum().item()

test_acc = correct / len(test_dataset)
print(f"Test Accuracy: {test_acc*100:.2f}%")
