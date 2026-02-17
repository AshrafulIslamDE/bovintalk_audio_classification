import torch
from pandas.io.sas.sas_constants import dataset_length
from torch.utils.data import DataLoader, random_split

from config import TRAIN_RATIO, SEED, VAL_RATIO, TEST_RATIO, BATCH_SIZE
from dataset.audio_dataset import AudioDataset
from split_dataset import load_all_files


def get_dataloader(transformation, collate_fn=None, dataset_class=AudioDataset):
    files, labels = load_all_files()
    dataset = dataset_class(files, labels, transformation)
    generator = torch.Generator().manual_seed(SEED)
    dataset_length=len(dataset)
    print(f"Dataset length: {dataset_length}")
    test_dataset_length=dataset_length- (int(dataset_length*TRAIN_RATIO)+int(dataset_length*VAL_RATIO))
    split_dataset_length=[int(dataset_length*TRAIN_RATIO), int(dataset_length*VAL_RATIO), test_dataset_length]
    train_dataset, val_dataset, _ = random_split(dataset, split_dataset_length, generator)
    train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    validation_dataset_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return train_dataset_loader, validation_dataset_loader

if __name__ == '__main__':
    print()