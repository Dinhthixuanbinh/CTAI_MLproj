# src/model.py

import torch
import torch.nn as nn
import timm

from src import config

class MultimodalEmotionRecognizer(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()

        # Visual Model (EfficientNet)
        self.visual_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        visual_feature_size = self.visual_net.num_features

        # Audio Model (Audio Spectrogram Transformer)
        self.audio_net = timm.create_model(
            'ast_p16_128_s10d_t100_in22k',
            pretrained=True,
            num_classes=0,
            n_mels=config.N_MELS,
            time_steps=config.AUDIO_TIME_STEPS,
            in_chans=1
        )
        audio_feature_size = self.audio_net.num_features

        # Fusion and Classifier
        self.fusion = nn.Sequential(
            nn.Linear(visual_feature_size + audio_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, audio_spectrogram):
        if images.dim() == 5: # Handles Phase 2 input (batch, frames, C, H, W)
            # Reshape to treat each frame as a separate batch item
            batch_size, num_frames, C, H, W = images.shape
            images = images.view(batch_size * num_frames, C, H, W)
            
            # Extract features for all frames
            visual_features = self.visual_net(images)
            
            # Aggregate features by taking the mean over the frames
            visual_features = visual_features.view(batch_size, num_frames, -1).mean(dim=1)
        else: # Handles Phase 1 input (batch, C, H, W)
            visual_features = self.visual_net(images)
        
        audio_spectrogram_3ch = audio_spectrogram.repeat(1, 3, 1, 1)
        audio_features = self.audio_net(audio_spectrogram_3ch)

        combined_features = torch.cat((visual_features, audio_features), dim=1)
        output = self.fusion(combined_features)
        
        return output