import torch
import torchaudio
from model_architecture import AudioCNN
from config import MODEL_PATH

def predict(audio_path):
    model = AudioCNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    waveform, sr = torchaudio.load(audio_path)
    spec = SpectrogramGenerator()
    mel = spec.generate(waveform, sr).unsqueeze(0).unsqueeze(0)

    pred = model(mel)
    idx = pred.argmax(1).item()
    return "HFC" if idx == 0 else "LFC"

if __name__ == "__main__":
    file = input("Enter audio file path: ")
    print("Prediction:", predict(file))
