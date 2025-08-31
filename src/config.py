# src/config.py

import torch

# -- Project Paths --
DATA_DIR = '/kaggle/input/copy-crema-d/cremad/'
# UPDATE THIS LINE to use the new CSV file
CSV_PATH = '/kaggle/working/cremad_paths.csv' 
MODEL_SAVE_PATH = '/kaggle/working/'

# -- Model Hyperparameters --
NUM_CLASSES = 6
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- Audio Preprocessing --
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 384
AUDIO_TIME_STEPS = 384

# -- Image Preprocessing --
IMAGE_SIZE = 224

# -- Emotion Mapping --
EMOTION_MAP = {
    "ANG": 0, "HAP": 1, "SAD": 2,
    "NEU": 3, "FEA": 4, "DIS": 5
}