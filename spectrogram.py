import torchaudio
import matplotlib.pyplot as plt

from audio_dataset_transformation_config import get_mel_transformation, get_spectrogram_transformation
from dataset.audio_dataset import AudioDataset
from config import TARGET_SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from split_dataset import load_all_files
import numpy as np

if __name__ == '__main__':


        files, labels = load_all_files()
        # Plot signal
        print(labels[0], labels[1024])
        waveform, sample_rate = torchaudio.load(files[1024])

        print("Shape:", waveform.shape)  # (channels, samples)
        print("Sample rate:", sample_rate)

        # ---- Convert to numpy ----
        signal = waveform.squeeze().numpy()  # remove channel dim if mono

        # ---- Create time axis ----
        time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

        # ---- Plot waveform ----
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal)

        plt.title("Raw Audio Signal")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        plt.savefig("waveform_lfc.png", dpi=300)
        plt.show()
        plt.close()








