import torch
import torch.nn as nn


# Assuming this function is defined correctly in your code
def create_convolution_layer(in_channels, out_channels, padding=1, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
        nn.ReLU(),
        # STANDARD DOWNSAMPLING: Halves the feature map dimensions
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class AudioCNN(nn.Module):
    # Pass the input shape as an argument
    def __init__(self, input_shape=(1, 64, 42)):
        super(AudioCNN, self).__init__()

        self.conv1 = create_convolution_layer(in_channels=1, out_channels=16)
        self.conv2 = create_convolution_layer(in_channels=16, out_channels=32)
        self.conv3 = create_convolution_layer(in_channels=32, out_channels=64)
        self.conv4 = create_convolution_layer(in_channels=64, out_channels=128)
        self.flatten = nn.Flatten()

        # --- DYNAMIC CALCULATION ---
        with torch.no_grad():
            # Create a dummy tensor based on the expected input shape
            dummy_input = torch.rand(1, *input_shape)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            # Use the defined self.flatten layer
            x = self.flatten(x)
            flattened_size = x.size(1)

        print(f"✅ Auto-calculated Linear Input Features: {flattened_size}")

        # --- Define Linear Layer using the calculated size (1024) ---
        self.linear = nn.Linear(in_features=flattened_size, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)  # Flatten must be called here
        logits = self.linear(x)
        prediction = self.softmax(logits)
        return prediction