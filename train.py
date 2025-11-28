import torch
from torch.utils.data import DataLoader
from data_processing import AudioDataset
from model_architecture import AudioCNN
from split_dataset import split_dataset
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_PATH
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    # Reshape: (1, Bands, Time) -> (Time, Bands)
    mels = [item[0].squeeze(0).transpose(0, 1) for item in batch]
    labels = [item[1] for item in batch] # Labels are still Python ints here

    # 2. Pad the sequences
    # Output is (Batch, Max_Time, Bands)
    padded_mels = pad_sequence(mels, batch_first=True, padding_value=0.0)

    # 3. CONVERT LABELS TO TENSORS AND STACK (THE FIX)
    # We must convert the list of Python ints into a list of PyTorch Tensors
    # The labels represent classes, so we use dtype=torch.long
    labels_stacked = torch.stack([torch.tensor(label, dtype=torch.long) for label in labels])

    # Output shape: (Batch, Max_Time, Bands), Labels: (Batch)
    return padded_mels, labels_stacked

# Load split datasets
(train_files, train_labels), (val_files, val_labels), _ = split_dataset()

train_dataset = AudioDataset(train_files, train_labels)
val_dataset = AudioDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=custom_collate_fn)
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
        mel = mel.unsqueeze(1)
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
            mel = mel.unsqueeze(1)
            pred = model(mel)
            correct_val += (pred.argmax(1) == label).sum().item()
    val_acc = correct_val / len(val_dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

torch.save(model.state_dict(), MODEL_PATH)
print("Model saved:", MODEL_PATH)
