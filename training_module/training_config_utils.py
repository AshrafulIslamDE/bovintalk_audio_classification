import argparse
import config
import torch


def update_config_from_args():
    parser = argparse.ArgumentParser(description="Update training configuration.")

    # Define flags with default=None so we can check if they were actually passed
    parser.add_argument("--lr", type=float, default=None, help="Override LEARNING_RATE")
    parser.add_argument("--epochs", type=int, default=None, help="Override EPOCHS")
    parser.add_argument("--batch_size", type=int, default=None, help="Override BATCH_SIZE")
    parser.add_argument("--n_mfcc", type=int, default=None, help="Override MFCC number")
    parser.add_argument("--optimizer", type=str, default=None, choices=['sgd', 'adam'], help="Optimizer choice")

    args = parser.parse_args()

    # Only update the config module if the argument was provided
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.n_mfcc is not None:
        config.N_MFCC = args.n_mfcc

    if args.optimizer is not None:
        # We store the choice in config to access it in the train function
        config.OPTIMIZER_TYPE = args.optimizer
    else:
        # Default to SGD if not specified in config or CLI
        config.OPTIMIZER_TYPE = getattr(config, 'OPTIMIZER_TYPE', 'sgd')
    print(f" learning rate:{config.LEARNING_RATE}, Epochs: {config.EPOCHS},"
          f" Batch Size: {config.BATCH_SIZE}, Optimizer : {config.OPTIMIZER_TYPE} " )
    return config


def get_optimizer(model):
    """Utility to create optimizer based on current config values"""
    if config.OPTIMIZER_TYPE.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    return torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)