# src/config.py

import torch

# -- Project Paths --
DATA_DIR = '../data/'
# UPDATE THIS LINE to use the new CSV file
CSV_PATH = '../data/cremad_paths.csv' 
MODEL_SAVE_PATH = '../models/'

# -- Model Hyperparameters --
NUM_CLASSES = 6
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- Audio Preprocessing --
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
AUDIO_TIME_STEPS = 1024

# -- Image Preprocessing --
IMAGE_SIZE = 224

# -- Emotion Mapping --
EMOTION_MAP = {
    "ANG": 0, "HAP": 1, "SAD": 2,
    "NEU": 3, "FEA": 4, "DIS": 5
}