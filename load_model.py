import streamlit as st
import torch
import torchaudio

from audio_dataset import AudioDataset
from mel_spectogram_config import get_transformation
from model_architecture import AudioCNN

torchaudio.set_audio_backend("ffmpeg")

@st.cache_resource
def load_model():
    model = AudioCNN()
    model.load_state_dict(torch.load("hfc_lfc_cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()
map={0:"HFC",1:"LFC"}
st.title("Audio Classification")
st.write("Upload an audio file and get its predicted class.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file, format="audio/wav")

        # Use your transformation
        transformation = get_transformation()

        # Create a temporary dataset with single audio
        temp_dataset = AudioDataset(
            file_list=["temp_audio.wav"],
            labels=[0],  # dummy label since we don't know it
            transformation=transformation
        )

        # Get the processed signal
        signal, _ = temp_dataset[0]

        # Add batch dimension
        signal = signal.unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(signal)
            predicted_class = torch.argmax(output, dim=1).item()

        st.write(f"Predicted Class: {map[predicted_class]}")