# Global configuration used by all scripts

AUDIO_DIRS = {
    "HFC": "data/HFC_audio",
    "LFC": "data/LFC_audio"
}

# Audio processing
TARGET_SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

# Training
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 0.0001

# Model save path
MODEL_PATH = "hfc_lfc_cnn.pth"

# Split ratio
TRAIN_RATIO = 0.75
VAL_RATIO = 0.05
TEST_RATIO = 0.20

# Random seed
SEED = 42

