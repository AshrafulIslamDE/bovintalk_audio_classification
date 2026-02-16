import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from audio_dataset import AudioDataset
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH, N_MELS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
from audio_dataset_transformation_config import get_mel_transformation, get_mfcc_transformation, \
    get_rnn_mfcc_transformation
from model_architecture import AudioCNN
from rnn_dataset import LSTMAudioDataset, AudioDatasetSpectogram
from rnn_model_architecture_for_spectrum import RNN_Spectogram, RNN_LSTM_Spectogram, RNN_GRU_Spectogram
from split_dataset import split_dataset, load_all_files
from utils import get_device
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torch.utils.data import random_split

device = get_device()
f1_scores = []

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

def train(train_dataloader:DataLoader,val_dataloader:DataLoader, model:nn.Module=AudioCNN()):

        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            # Train
            model.train()
            running_loss = 0
            correct_train = 0
            for mel, label in train_dataloader:
                #mel = mel.unsqueeze(1)
                label = label.to(device)
                optimizer.zero_grad()
                pred = model(mel)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct_train += (pred.argmax(1) .eq(label) ).sum().item()
            train_acc = correct_train / len(train_dataloader.dataset)

            # Validate
            model.eval()
            correct_val = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for mel, label in val_dataloader:
                    #mel = mel.unsqueeze(1)
                    pred = model(mel)
                    predicted = pred.argmax(1)
                    correct_val += (predicted .eq(label) ).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label.cpu().numpy())

            val_acc = correct_val / len(val_dataloader.dataset)

            # calculate f1 score
            f1 = f1_score(all_labels, all_preds, average='weighted')
            f1_scores.append(f1)

            print(f"Epoch {epoch + 1}/{EPOCHS} | "
                  f"Loss: {running_loss:.4f} | "
                  f"Train Acc: {train_acc * 100:.2f}% | "
                  f"Val Acc: {val_acc * 100:.2f}% | "
                  f"F1 Score: {f1:.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved:", MODEL_PATH)

def draw_f1_score():
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS +1), f1_scores, marker='o', color='b')
    plt.title("F1 Score per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.xticks(range(1, EPOCHS+1 ))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def get_dataloader(transformation,collate_fn=None,dataset_class=AudioDataset):
    files, labels=load_all_files()
    dataset=dataset_class(files,labels,transformation)
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset,_=random_split(dataset,[TRAIN_RATIO, VAL_RATIO,TEST_RATIO],generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return train_loader, val_loader

if __name__ == '__main__':
     #train(get_mel_transformation())
     #train(get_mfcc_transformation())
     train(*get_dataloader(get_mel_transformation(),collate_fn=collate_fn,
                          dataset_class=AudioDatasetSpectogram),model=RNN_GRU_Spectogram(input_size=N_MELS))
     draw_f1_score()

