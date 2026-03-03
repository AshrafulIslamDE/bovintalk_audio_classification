import datetime

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib

import os
import numpy as np

import config
from training_module.training_config_utils import get_optimizer
from training_module.training_visualization import draw_confusion_matrix, draw_f1_score
from utils import get_device

from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score

# to support backend plotting (https://matplotlib.org/stable/users/explain/figure/backends.html)
matplotlib.use('Agg')
# to access GPU or CPU
device = get_device()
print('Using device:', device)

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
    optimizer = get_optimizer(model)

    final_train_acc = 0
    final_val_acc = 0
    final_f1 = 0
    f1_scores = []
    for epoch in range(config.EPOCHS):
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

        val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

        # ----- F1 score calculation -----
        f1 = f1_score(all_labels, all_preds, average='weighted')
        f1_scores.append(f1)

        # ----- Precision, Recall calculation ----
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Update final metrics for saving
        final_train_acc, final_val_acc, final_f1 = train_acc, val_acc, f1

        print(f"Epoch {epoch + 1}/{config.EPOCHS} | Loss: {running_loss:.4f} | "
              f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} ")

    param_string = (
        f"{model_name_str}_ep{config.EPOCHS}_lr{config.LEARNING_RATE}_"
        f"bs{config.BATCH_SIZE}_{config.OPTIMIZER_TYPE}_mfcc{config.N_MFCC}"
    )
    # ----Generate Confusion Matrix --------
    cm = confusion_matrix(all_labels, all_preds)
    draw_confusion_matrix(cm, param_string)

    draw_f1_score(model_name=param_string,f1_scores=f1_scores)

    # --- SAVE MODEL FILE ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d")
    model_filename = (f"{config.EPOCHS}_{final_train_acc * 100:.1f}_{final_val_acc * 100:.1f}_"
                      f"{final_f1:.3f}_{timestamp}_{model_name_str}_{config.LEARNING_RATE}.pth")

    model_save_path = os.path.join("models", model_filename)  # models/filename.pth
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    # --- WRITE TO CENTRAL LOG FILE ---
    log_entry = (f"Model: {model_name_str} | Epochs: {config.EPOCHS} | "
                 f"Train Acc: {final_train_acc * 100:.2f}% | Val Acc: {final_val_acc * 100:.2f}% | "
                 f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} |"
                 f"Batch Size: {config.BATCH_SIZE} | Learning Rate: {config.LEARNING_RATE} | "
                 f"Optimizer Type: {config.OPTIMIZER_TYPE} | N_MFCC: {config.N_MFCC} | Timestamp: {timestamp}\n")
    log_path = os.path.join("logs", "all_models_summary.txt")
    with open(log_path, "a") as f:
        f.write(log_entry)






