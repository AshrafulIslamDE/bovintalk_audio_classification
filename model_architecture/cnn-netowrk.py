from torch import torchaudio
from torch import  nn

def create_convolution_layer(in_channels, out_channels, stride=1, padding=1,kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size, stride=stride),
    )
class CNNNetwork(nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
        self.conv1 = create_convolution_layer(in_channels=1, out_channels=16)
        self.conv2 = create_convolution_layer(in_channels=16, out_channels=32)
        self.conv3 = create_convolution_layer(in_channels=32, out_channels=64)
        self.conv4 = create_convolution_layer(in_channels=64, out_channels=128)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128, out_features=2)
