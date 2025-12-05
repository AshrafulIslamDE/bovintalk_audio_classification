import torch
from torch.utils.data import DataLoader
from audio_dataset import AudioDataset
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH
from mel_spectogram_config import get_transformation
from model_architecture import AudioCNN
from split_dataset import split_dataset

# Load split datasets
(train_files, train_labels), (val_files, val_labels), _ = split_dataset()

train_dataset = AudioDataset(train_files, train_labels,get_transformation())
val_dataset = AudioDataset(val_files, val_labels,get_transformation())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = AudioCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss = 0
    correct_train = 0
    for mel, label in train_loader:
        #mel = mel.unsqueeze(1)
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
            pred = model(mel)
            correct_val += (pred.argmax(1) == label).sum().item()
    val_acc = correct_val / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved:", MODEL_PATH)
