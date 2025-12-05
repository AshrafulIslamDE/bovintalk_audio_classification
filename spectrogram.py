import torchaudio
import matplotlib.pyplot as plt  # Import for plotting
import os  # Import for file path checking

from audio_dataset import AudioDataset
from config import TARGET_SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from split_dataset import load_all_files

mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
if __name__ == '__main__':

        # --- 3. Generate Spectrogram ---
        files, labels = load_all_files()
        audio = AudioDataset(files, labels, mel_transform)
        signal,label = audio[1]

        # generate spectogram
        signal_np = signal.squeeze()  # removes the channel dimension
        signal_np = signal_np.numpy() if hasattr(signal, "numpy") else signal_np

        plt.figure(figsize=(10, 4))
        plt.imshow(signal_np, aspect='auto', origin='lower')
        plt.title(f"Mel-Spectrogram (Label: {label})")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency Bins")
        plt.colorbar(format='%+2.0f dB')
        plt.show()






