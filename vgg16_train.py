import torch
from torch.utils.data import DataLoader
from dataset.audio_dataset import AudioDataset
from utils import get_device
from dataset.vgg_mel_dataset import VGGMelDataset  # <-- NEW
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH
from audio_dataset_transformation_config import get_mel_transformation
from split_dataset import split_dataset
from torchvision import models
import torch.nn as nn

# ------------------------
# Load Original Audio Data
# ------------------------
(train_files, train_labels), (val_files, val_labels), _ = split_dataset()

train_audio = AudioDataset(train_files, train_labels, get_mel_transformation())
val_audio   = AudioDataset(val_files, val_labels, get_mel_transformation())

# ------------------------
# Wrap with VGG16 Dataset
# ------------------------
train_dataset = VGGMelDataset(train_audio)
val_dataset   = VGGMelDataset(val_audio)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ------------------------
# Load VGG16 Model
# ------------------------
model = models.vgg16(weights="IMAGENET1K_V1")

# Freeze feature extractor
for p in model.features.parameters():
    p.requires_grad = False

# Replace classifier for 2 classes
model.classifier[6] = nn.Linear(4096, 2)
device=get_device()
model = model.to(device)
print(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss = 0
    correct_train = 0
    for mel, label in train_loader:
        #mel = mel.unsqueeze(1)
        label=label.to(device)
        optimizer.zero_grad()
        pred = model(mel)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_train += (pred.argmax(1) == label).sum().item()
    train_acc = correct_train / len(train_dataset)

    # Validate
    model.eval()
    correct_val = 0
    with torch.no_grad():
        for mel, label in val_loader:
            #mel = mel.unsqueeze(1)
            label=label.to(device)
            pred = model(mel)
            correct_val += (pred.argmax(1) == label).sum().item()
    val_acc = correct_val / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved:", MODEL_PATH)