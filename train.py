import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from audio_dataset import AudioDataset
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH
from audio_dataset_transformation_config import get_mel_transformation, get_mfcc_transformation, \
    get_rnn_mfcc_transformation
from model_architecture import AudioCNN
from rnn_dataset import LSTMAudioDataset
from rnn_model_architecture import RNN_LSTM
from split_dataset import split_dataset
from utils import get_device
import torch.nn as nn

device = get_device()

# for handling variable seq_len of rnn
def collate_fn(batch):
    # Sort batch by sequence length (descending) is often helpful for RNNs
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    signals, labels = zip(*batch)

    # Pad the 'Time' dimension (dim 0 of the signal)
    # signals[0] is (Time, 8, 13)
    padded_signals = pad_sequence(signals, batch_first=True, padding_value=0)

    labels = torch.tensor(labels)
    return padded_signals, labels

def train(transformation,collate_fn=None,model:nn.Module=AudioCNN()):
        # Load split datasets
        (train_files, train_labels), (val_files, val_labels), _ = split_dataset()

        train_dataset = LSTMAudioDataset(train_files, train_labels,transformation)
        val_dataset = LSTMAudioDataset(val_files, val_labels,transformation)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,collate_fn=collate_fn)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            # Train
            model.train()
            running_loss = 0
            correct_train = 0
            for mel, label in train_loader:
                #mel = mel.unsqueeze(1)
                label = label.to(device)
                optimizer.zero_grad()
                pred = model(mel)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct_train += (pred.argmax(1) .eq(label) ).sum().item()
            train_acc = correct_train / len(train_dataset)

            # Validate
            model.eval()
            correct_val = 0
            with torch.no_grad():
                for mel, label in val_loader:
                    #mel = mel.unsqueeze(1)
                    pred = model(mel)
                    correct_val += (pred.argmax(1) .eq(label) ).sum().item()
            val_acc = correct_val / len(val_dataset)

            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved:", MODEL_PATH)

if __name__ == '__main__':
     #train(get_mel_transformation())
     #train(get_mfcc_transformation())
     train(get_rnn_mfcc_transformation(),model=RNN_LSTM(),collate_fn=collate_fn)


