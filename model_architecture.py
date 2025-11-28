import torch.nn as nn
# Note: Remove 'import torch' if it was only added for the failed dynamic sizing

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # --- PERMANENT FIX: Use the calculated size 4608 ---
        self.fc = nn.Sequential(
            nn.Linear(4608, 128),  # <--- CHANGED FROM 4096 to the actual output size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        # ----------------------------------------------------

    def forward(self, inputs):
        # NOTE: Remove the 'if self.fc is None:' block if you used the previous dynamic solution
        x = self.conv(inputs)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)