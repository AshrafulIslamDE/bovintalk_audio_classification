import torch
from data_processing import load_all_files
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED

def split_dataset(base_dir="."):
    files, labels = load_all_files(base_dir)
    total_len = len(files)

    train_len = int(TRAIN_RATIO * total_len)
    val_len = int(VAL_RATIO * total_len)
    test_len = total_len - train_len - val_len

    # Shuffle indices
    torch.manual_seed(SEED)
    indices = torch.randperm(total_len).tolist()

    train_files = [files[i] for i in indices[:train_len]]
    train_labels = [labels[i] for i in indices[:train_len]]

    val_files = [files[i] for i in indices[train_len:train_len+val_len]]
    val_labels = [labels[i] for i in indices[train_len:train_len+val_len]]

    test_files = [files[i] for i in indices[train_len+val_len:]]
    test_labels = [labels[i] for i in indices[train_len+val_len:]]

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)
