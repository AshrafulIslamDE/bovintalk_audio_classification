import torch
import torch.nn as nn

def create_convolution_layer(in_channels, out_channels, padding=1, kernel_size=3, with_pooling=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
        nn.ReLU()
    ]
    if with_pooling:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class AudioCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.conv_layers = nn.Sequential(
            create_convolution_layer(in_channels, 16),
            create_convolution_layer(16, 32),
            create_convolution_layer(32, 64),
            create_convolution_layer(64, 128),
            create_convolution_layer(128, 256),
        )

        # global_pool will reduce H,W to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((10,1))

        # Dynamically calculate linear input size
        with torch.no_grad():
            dummy_input = torch.rand(1, in_channels, 64, 42)  # arbitrary H,W
            x = self.conv_layers(dummy_input)
            x = self.global_pool(x)
            flattened_size = x.view(1, -1).shape[1]
            print(f"Linear input size: {flattened_size}")

        self.linear = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.linear(x)
        return logits
