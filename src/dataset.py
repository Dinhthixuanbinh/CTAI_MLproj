import torch
import torchaudio
import pandas as pd
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio import transforms as T

from src import config


class MultiTaskDataset(Dataset):
    """
    A unified dataset for multi-task learning, providing both classification data
    and a corresponding contrastive audio pair for each sample.
    """

    def __init__(
        self,
        df_classification: pd.DataFrame,
        df_contrastive: pd.DataFrame,
        is_train: bool = True,
    ):
        """
        Initializes the dataset with two dataframes.

        Args:
            df_classification (pd.DataFrame): DataFrame for classification data.
            df_contrastive (pd.DataFrame): DataFrame for contrastive data.
            is_train (bool): Flag indicating if the dataset is for training.
        """
        self.df_classification = df_classification
        self.df_contrastive = df_contrastive
        self.is_train = is_train

        # Image transformations
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (config.IMAGE_SIZE, config.IMAGE_SIZE), antialias=True
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Audio transformations
        self.audio_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
        )

    def __len__(self) -> int:
        # The number of samples is the minimum of the two datasets
        return min(len(self.df_classification), len(self.df_contrastive))

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets a single item from the combined dataset.

        Returns:
            A tuple of (images_ce, audio_ce, label_ce, audio_cl1, audio_cl2)
        """
        # --- Classification Data ---
        classification_row = self.df_classification.iloc[idx]

        # Process Video Frames
        video_frame_dir = classification_row["video_frame_dir"]
        frame_files = sorted(os.listdir(video_frame_dir))
        selected_frame = np.random.choice(frame_files)
        middle_frame_path = os.path.join(video_frame_dir, selected_frame)
        frame = cv2.imread(middle_frame_path)
        if frame is None:
            frame = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
        images_ce = self.image_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Process Audio for Classification
        audio_ce_path = classification_row["audio_path"]
        audio_ce = self._load_audio(audio_ce_path)

        label_ce = config.EMOTION_MAP[classification_row["emotion"]]

        # --- Contrastive Data ---
        # The contrastive dataset is much larger, so we will sample randomly to ensure diversity
        contrastive_row = self.df_contrastive.iloc[
            np.random.randint(len(self.df_contrastive))
        ]

        audio_cl1_path = contrastive_row["audio_path_1"]
        audio_cl2_path = contrastive_row["audio_path_2"]

        audio_cl1 = self._load_audio(audio_cl1_path)
        audio_cl2 = self._load_audio(audio_cl2_path)

        return (
            images_ce,
            audio_ce,
            torch.tensor(label_ce, dtype=torch.long),
            audio_cl1,
            audio_cl2,
        )

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Loads and preprocesses an audio clip.
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != config.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)

        spectrogram = self.audio_transform(waveform)
        spectrogram = self.pad_or_truncate_spectrogram(spectrogram)
        spectrogram = T.AmplitudeToDB()(spectrogram)
        return spectrogram

    def pad_or_truncate_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """Pads or truncates a spectrogram to a fixed length."""
        target_len = config.AUDIO_TIME_STEPS
        current_len = spec.shape[2]
        if current_len > target_len:
            spec = spec[:, :, :target_len]
        elif current_len < target_len:
            padding = target_len - current_len
            spec = torch.nn.functional.pad(spec, (0, padding), "constant")
        return spec
