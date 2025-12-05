import torch
import torch.nn as nn


def create_convolution_layer(in_channels, out_channels, padding=1, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class AudioCNN(nn.Module):
    def __init__(self, input_shape=(1, 64, 42)):
        super(AudioCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            create_convolution_layer(1, 16),
            create_convolution_layer(16, 32),
            create_convolution_layer(32, 64),
            create_convolution_layer(64, 128),
        )

        self.flatten = nn.Flatten()

        # ---------- DYNAMICALLY COMPUTE LINEAR INPUT SIZE ----------
        with torch.no_grad():
            dummy = torch.rand(1, *input_shape)
            x = self.conv_layers(dummy)
            x = self.flatten(x)
            flattened_size = x.size(1)

        print(f"Auto-calculated Linear Input Features: {flattened_size}")

        self.linear = nn.Linear(flattened_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = self.flatten(x)
        logits = self.linear(x)
        prediction = self.softmax(logits)
        return prediction
