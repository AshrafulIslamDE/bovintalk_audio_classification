import datetime
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import random_split
import os
import json

# Your custom imports
from audio_dataset import AudioDataset
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, N_MELS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
from audio_dataset_transformation_config import get_mel_transformation
from rnn_dataset import AudioDatasetSpectogram
from rnn_model_architecture_for_spectrum import RNN_Spectogram, LSTM_Spectogram, GRU_Spectogram, \
    BiLSTM_Spectogram
from split_dataset import load_all_files
from utils import get_device

device = get_device()
f1_scores = []

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    signals, labels = zip(*batch)
    padded_signals = pad_sequence(signals, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    padded_signals = padded_signals.to(device)
    labels.to(device)
    return padded_signals, labels


def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: nn.Module, model_name_str: str):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    final_train_acc = 0
    final_val_acc = 0
    final_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        correct_train = 0
        for mel, label in train_dataloader:
            # Reassigning to device properly
            mel = mel.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred = model(mel)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_train += (pred.argmax(1).eq(label)).sum().item()

        train_acc = correct_train / len(train_dataloader.dataset)

        # Validate
        model.eval()
        correct_val = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for mel, label in val_dataloader:
                mel = mel.to(device)
                label = label.to(device)
                pred = model(mel)
                predicted = pred.argmax(1)
                correct_val += (predicted.eq(label)).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_acc = correct_val / len(val_dataloader.dataset)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        f1_scores.append(f1)

        # Update final metrics for saving
        final_train_acc, final_val_acc, final_f1 = train_acc, val_acc, f1

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss:.4f} | "
              f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | "
              f"F1 Score: {f1:.4f} ")

    # --- SAVE MODEL FILE ---
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d")
    model_filename = (f"{EPOCHS}_{final_train_acc * 100:.1f}_{final_val_acc * 100:.1f}_"
                      f"{final_f1:.3f}_{timestamp}_{model_name_str}_{LEARNING_RATE}.pth")
    model_save_path = os.path.join("models", model_filename)  # models/filename.pth
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    # --- WRITE TO CENTRAL LOG FILE ---
    log_entry = (f"Model: {model_name_str} | Epochs: {EPOCHS} | "
                 f"Train Acc: {final_train_acc * 100:.2f}% | Val Acc: {final_val_acc * 100:.2f}% | "
                 f"F1 Score: {final_f1:.4f} | Timestamp: {timestamp}\n")
    log_path = os.path.join("logs", "all_models_summary.txt")
    with open(log_path, "a") as f:
        f.write(log_entry)


def draw_f1_score(model_name):
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plot_data", exist_ok=True)

    # --- 1. Save the actual Image ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), f1_scores, marker='o', color='b', label='F1 Score')
    plt.title(f"F1 Score per Epoch - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend(loc='lower right')
    plt.xticks(range(1, EPOCHS + 1))
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plot_path = os.path.join("plots", f"{model_name}_f1_plot.png")
    plt.savefig(plot_path)
    plt.close()  # Close figure to free up memory
    print(f"Plot image saved to: {plot_path}")

    # --- 2. Save Raw Data for later use in IDE ---
    data_path = os.path.join("plot_data", f"{model_name}_f1_values.json")
    with open(data_path, 'w') as f:json.dump(f1_scores, f)
    print(f"Raw F1 data saved to: {data_path}")


def get_dataloader(transformation, collate_fn=None, dataset_class=AudioDataset):
    files, labels = load_all_files()
    dataset = dataset_class(files, labels, transformation)
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset, _ = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO], generator)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return train_loader, val_loader


if __name__ == '__main__':
    print(f" executing device: {device}")
    model_classes = [
        RNN_Spectogram,
        LSTM_Spectogram,
        BiLSTM_Spectogram,
        GRU_Spectogram
    ]

    for model_class in model_classes:
        curr_model_name = model_class.__name__
        print(f"\n--- Starting Training: {curr_model_name} ---")

        f1_scores = []  # Reset for each model architecture

        train_loader, val_loader = get_dataloader(
            get_mel_transformation(),
            collate_fn=collate_fn,
            dataset_class=AudioDatasetSpectogram
        )

        train(
            train_loader,
            val_loader,
            model=model_class(input_size=N_MELS),
            model_name_str=curr_model_name
        )

        draw_f1_score(curr_model_name)
