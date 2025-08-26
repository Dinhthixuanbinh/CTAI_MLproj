# src/dataset.py

import torch
import torchaudio
import pandas as pd
import cv2
import numpy as np
import os # <-- Import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio import transforms as T

from src import config

class EmotionDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train
        
        # Image transformations (no changes here)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Audio transformations (no changes here)
        self.audio_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # === MODIFICATION START ===
        video_frame_dir = row['video_frame_dir'] # Get the directory path for frames
        audio_path = row['audio_path']
        emotion_label = config.EMOTION_MAP[row['emotion']]

        # --- Process Video Frame ---
        # List all frame files, sort them, and pick the middle one
        frame_files = sorted(os.listdir(video_frame_dir))
        middle_frame_path = os.path.join(video_frame_dir, frame_files[len(frame_files) // 2])
        
        # Read the image frame using OpenCV
        frame = cv2.imread(middle_frame_path)
        if frame is None:
            # Handle cases where image reading fails
            frame = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image)
        # === MODIFICATION END ===
        
        # --- Process Audio (no changes here) ---
        waveform, sr = torchaudio.load(audio_path)
        if sr != config.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)

        spectrogram = self.audio_transform(waveform)
        spectrogram = self.pad_or_truncate_spectrogram(spectrogram)
        spectrogram = T.AmplitudeToDB()(spectrogram)
        
        return image, spectrogram, torch.tensor(emotion_label, dtype=torch.long)

    def pad_or_truncate_spectrogram(self, spec):
        target_len = config.AUDIO_TIME_STEPS
        current_len = spec.shape[2]
        if current_len > target_len:
            spec = spec[:, :, :target_len]
        elif current_len < target_len:
            padding = target_len - current_len
            spec = torch.nn.functional.pad(spec, (0, padding), 'constant')
        return spec