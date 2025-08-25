# src/dataset.py

import torch
import torchaudio
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio import transforms as T

from src import config

class EmotionDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train
        
        # Define image transformations
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define audio transformations to create Mel spectrogram
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
        video_path = row['video_path']
        audio_path = row['audio_path']
        emotion_label = config.EMOTION_MAP[row['emotion']]

        # --- Process Video Frame ---
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Select the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Handle cases where frame reading fails
            frame = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.image_transform(image)
        
        # --- Process Audio ---
        waveform, sr = torchaudio.load(audio_path)
        if sr != config.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)

        # Create Mel spectrogram
        spectrogram = self.audio_transform(waveform)
        # Pad or truncate spectrogram to a fixed size
        spectrogram = self.pad_or_truncate_spectrogram(spectrogram)
        # Convert to dB scale
        spectrogram = T.AmplitudeToDB()(spectrogram)
        
        return image, spectrogram, torch.tensor(emotion_label, dtype=torch.long)

    def pad_or_truncate_spectrogram(self, spec):
        target_len = config.AUDIO_TIME_STEPS
        current_len = spec.shape[2]
        
        if current_len > target_len:
            # Truncate
            spec = spec[:, :, :target_len]
        elif current_len < target_len:
            # Pad
            padding = target_len - current_len
            spec = torch.nn.functional.pad(spec, (0, padding), 'constant')
            
        return spec