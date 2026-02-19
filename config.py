from pathlib import Path

# Global configuration used by all scripts
def get_project_root():
    current = Path(__file__).resolve().parent
    print("enter path file ")
    while current != current.parent:
        if (current / "data").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not find 'data' directory in any parent folder.")


PROJECT_ROOT = get_project_root()

AUDIO_DIRS = {
    "HFC": PROJECT_ROOT / "data" / "HFC_audio",
    "LFC": PROJECT_ROOT / "data" / "LFC_audio"
}

# Audio processing
TARGET_SAMPLE_RATE = 46100
NUM_SAMPLES = 46100
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
N_MFCC = 40

# Audio processing RNN
RNN_TARGET_SAMPLE_RATE = 46100
RNN_NUM_SAMPLES = 46100
RNN_N_MELS = 40
RNN_N_MFCC = 13
RNN_FRAME_DURATION=0.10
RNN_WINDOWS_PER_FRAME=1
RNN_OVERLAP_DURATION=0.01
RNN_SAMPLES_PER_FRAME = int(RNN_TARGET_SAMPLE_RATE * RNN_FRAME_DURATION)
RNN_HOP_LENGTH = RNN_SAMPLES_PER_FRAME // RNN_WINDOWS_PER_FRAME
RNN_OVERLAP_SAMPLES = int(RNN_TARGET_SAMPLE_RATE * RNN_OVERLAP_DURATION)
RNN_N_FFT = RNN_HOP_LENGTH+RNN_OVERLAP_SAMPLES
INPUT_SIZE = RNN_N_MFCC*RNN_WINDOWS_PER_FRAME


# Training
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.0001

# Model save path
MODEL_PATH = "hfc_lfc_cnn.pth"

# Split ratio
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Random seed

SEED = 42

