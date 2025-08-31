
# src/model.py

import torch
import torch.nn as nn
import timm

from . import config

class MultimodalEmotionRecognizer(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()

        # --- Visual Model (EfficientNet) ---
        self.visual_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # --- Audio Model (Audio Spectrogram Transformer) ---
        self.audio_net = timm.create_model(
            'deit_base_distilled_patch16_384.fb_in1k',
            pretrained=True,
            num_classes=0
        )

        # === ROBUST FIX: Dynamically determine feature sizes ===
        # Create dummy input tensors to pass through the models
        dummy_image = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        dummy_spectrogram = torch.randn(2, 1, config.N_MELS, config.AUDIO_TIME_STEPS)
        
        # Pass the dummy tensors to get the output shapes
        with torch.no_grad():
            dummy_visual_features = self.visual_net(dummy_image)
            # AST model needs 3 channels, so we repeat the single channel
            dummy_audio_features = self.audio_net(dummy_spectrogram.repeat(1, 3, 1, 1))

        # Get the feature sizes from the shape of the output tensors
        visual_feature_size = dummy_visual_features.shape[1]
        audio_feature_size = dummy_audio_features.shape[1]
        
        print(f"Dynamically determined feature sizes:")
        print(f"   - Visual features: {visual_feature_size}")
        print(f"   - Audio features:  {audio_feature_size}")
        
        # --- Fusion and Classifier ---
        # Now, this will be created with the CORRECT combined feature size
        self.fusion = nn.Sequential(
            nn.Linear(visual_feature_size + audio_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, audio_spectrogram):
        visual_features = self.visual_net(image)
        
        audio_spectrogram_3ch = audio_spectrogram.repeat(1, 3, 1, 1)
        audio_features = self.audio_net(audio_spectrogram_3ch)

        combined_features = torch.cat((visual_features, audio_features), dim=1)
        output = self.fusion(combined_features)
        
        return output