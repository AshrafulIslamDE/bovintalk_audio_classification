from torch.nn.utils.rnn import pad_sequence
import torch.utils.data

def collate_fn(batch):
    # Sort batch by sequence length (descending) is often helpful for RNNs
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    signals, labels = zip(*batch)

    # Pad the 'Time' dimension (dim 0 of the signal)
    # signals[0] is (Time, 8, 13)
    padded_signals = pad_sequence(signals, batch_first=True, padding_value=0)

    labels = torch.tensor(labels)
    return padded_signals, labels


# When creating your DataLoader:
train_loader = torch.utils.data.DataLoader(
    train_data_set,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn  # Use our custom padder
)