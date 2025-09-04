import torch
import torch.nn as nn
import timm

from src import config

class MultimodalEmotionRecognizer(nn.Module):
    """
    Unified multi-task learning architecture for SER.

    This model combines a visual encoder and an audio encoder, using a shared
    audio backbone for both supervised emotion classification and self-supervised
    contrastive learning.

    Args:
        num_classes (int): The number of emotion classes for classification.
        audio_feature_size (int): The size of the features output by the audio encoder.
        visual_feature_size (int): The size of the features output by the visual encoder.
    """
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        visual_feature_size: int = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0).num_features,
        audio_feature_size: int = timm.create_model(
            'ast_p16_128_s10d_t100_in22k',
            pretrained=True,
            num_classes=0,
            n_mels=config.N_MELS,
            time_steps=config.AUDIO_TIME_STEPS,
            in_chans=1
        ).num_features
    ):
        super().__init__()

        # Visual Encoder (EfficientNet)
        self.visual_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        # Audio Encoder (Audio Spectrogram Transformer)
        # This is a shared encoder for both tasks
        self.audio_net = timm.create_model(
            'ast_p16_128_s10d_t100_in22k',
            pretrained=True,
            num_classes=0,
            n_mels=config.N_MELS,
            time_steps=config.AUDIO_TIME_STEPS,
            in_chans=1
        )

        # Classification Head for supervised emotion classification
        self.classification_head = nn.Sequential(
            nn.Linear(visual_feature_size + audio_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Contrastive Head for self-supervised learning
        # Projects audio features into an embedding space
        self.contrastive_head = nn.Linear(audio_feature_size, 128) # Project to a smaller, fixed dimension

    def forward(
        self,
        images: torch.Tensor,
        audio_ce: torch.Tensor,
        audio_cl1: torch.Tensor,
        audio_cl2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass for the multi-task learning framework.

        Args:
            images (torch.Tensor): A batch of video frames.
            audio_ce (torch.Tensor): A batch of audio spectrograms for the
                                     supervised classification task.
            audio_cl1 (torch.Tensor): A batch of audio spectrograms for the
                                      first clip of a contrastive pair.
            audio_cl2 (torch.Tensor): A batch of audio spectrograms for the
                                      second clip of a contrastive pair.

        Returns:
            A tuple containing:
            - emotion_logits (torch.Tensor): The output logits for emotion classification.
            - audio_embedding_1 (torch.Tensor): The projected embedding for the first
                                                contrastive audio clip.
            - audio_embedding_2 (torch.Tensor): The projected embedding for the second
                                                contrastive audio clip.
        """
        # --- Supervised Classification Path ---
        # Visual Encoder handles both single and multi-frame inputs
        if images.dim() == 5:
            batch_size, num_frames, C, H, W = images.shape
            images = images.view(batch_size * num_frames, C, H, W)
            visual_features = self.visual_net(images)
            visual_features = visual_features.view(batch_size, num_frames, -1).mean(dim=1)
        else:
            visual_features = self.visual_net(images)

        # The Audio Encoder is shared
        audio_features_ce = self.audio_net(audio_ce.repeat(1, 3, 1, 1))

        # Fusion and Classification
        fused_features = torch.cat((visual_features, audio_features_ce), dim=1)
        emotion_logits = self.classification_head(fused_features)

        # --- Self-Supervised Contrastive Learning Path ---
        # The Audio Encoder is reused for the contrastive audio clips
        audio_features_cl1 = self.audio_net(audio_cl1.repeat(1, 3, 1, 1))
        audio_features_cl2 = self.audio_net(audio_cl2.repeat(1, 3, 1, 1))

        # Project features into the new embedding space
        audio_embedding_1 = self.contrastive_head(audio_features_cl1)
        audio_embedding_2 = self.contrastive_head(audio_features_cl2)

        return emotion_logits, audio_embedding_1, audio_embedding_2
