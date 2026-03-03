import torch
from pandas.io.sas.sas_constants import dataset_length
from torch.utils.data import DataLoader, random_split

import config
from dataset.audio_dataset import AudioDataset
from split_dataset import load_all_files, load_all_files_balanced


def get_dataloader(transformation, collate_fn=None, dataset_class=AudioDataset):
    files, labels = load_all_files()
    dataset = dataset_class(files, labels, transformation)
    generator = torch.Generator().manual_seed(config.SEED)
    dataset_length=len(dataset)
    # print(dataset_length)
    batch_size=config.BATCH_SIZE
    val_ratio=config.VAL_RATIO
    train_ratio=config.TRAIN_RATIO

    test_dataset_length=dataset_length- (int(dataset_length*train_ratio)+int(dataset_length*val_ratio))
    split_dataset_length=[int(dataset_length*train_ratio), int(dataset_length*val_ratio), test_dataset_length]
    train_dataset, val_dataset, _ = random_split(dataset, split_dataset_length, generator)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_dataset_loader, validation_dataset_loader

if __name__ == '__main__':
    print()