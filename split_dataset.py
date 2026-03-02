import torch
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED, AUDIO_DIRS

def load_all_files():
    files = []
    labels = []

    for label_name, folder_path in AUDIO_DIRS.items():
        label_idx = 0 if label_name == "HFC" else 1

        if not folder_path.exists():
            print("Missing:", folder_path)
            continue

        for file in folder_path.iterdir():
            if file.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".wma"]:
                files.append(str(file))
                labels.append(label_idx)

    return files, labels


import random


def load_all_files_balanced():
    files = []
    labels = []

    # Store files by category first to calculate the gap
    categorized_files = {"HFC": [], "LFC": []}

    for label_name, folder_path in AUDIO_DIRS.items():
        if not folder_path.exists():
            continue
        for file in folder_path.iterdir():
            if file.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]:
                categorized_files[label_name].append(str(file))

    hfc_count = len(categorized_files["HFC"])  # 952
    lfc_count = len(categorized_files["LFC"])  # 192

    # 1. Add all HFC files as is
    files.extend(categorized_files["HFC"])
    labels.extend([0] * hfc_count)

    # 2. Add all original LFC files
    files.extend(categorized_files["LFC"])
    labels.extend([1] * lfc_count)

    # 3. Calculate how many augmented samples we need for LFC
    gap = hfc_count - lfc_count

    print(f"Balancing dataset... Adding {gap} augmented samples to LFC.")

    # Randomly pick from existing LFC files and duplicate them until the gap is filled
    for _ in range(gap):
        random_file = random.choice(categorized_files["LFC"])
        files.append(random_file)
        labels.append(1)

    return files, labels

def split_dataset():
    files, labels = load_all_files()
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

if __name__ == "__main__":
    load_all_files()

