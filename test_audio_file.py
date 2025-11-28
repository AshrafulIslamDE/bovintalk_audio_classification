import torchaudio
import torch

# Create a dummy wav file
torchaudio.save("test.wav", torch.randn(1, 16000), 16000, backend="soundfile")
print("Save successful")

# Try loading it back
y, sr = torchaudio.load("test.wav", backend="soundfile")
print(f"Load successful: shape {y.shape}")