import cv2
import torch
import torch.nn as nn
import timm
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from src import config


class MultimodalEmotionRecognizer(nn.Module):
    """
    Unified multi-task learning architecture for SER with face-based visual features.
    If no face is detected, the whole image is used instead.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        visual_feature_size: int = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        ).num_features,
        audio_feature_size: int = timm.create_model(
            "ast_p16_128_s10d_t100_in22k",
            pretrained=True,
            num_classes=0,
            n_mels=config.N_MELS,
            time_steps=config.AUDIO_TIME_STEPS,
            in_chans=1,
        ).num_features,
    ):
        super().__init__()

        # Visual Encoder (EfficientNet)
        self.visual_net = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )

        # Audio Encoder (Audio Spectrogram Transformer)
        self.audio_net = timm.create_model(
            "ast_p16_128_s10d_t100_in22k",
            pretrained=True,
            num_classes=0,
            n_mels=config.N_MELS,
            time_steps=config.AUDIO_TIME_STEPS,
            in_chans=1,
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(visual_feature_size + audio_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # Contrastive Head
        self.contrastive_head = nn.Linear(audio_feature_size, 128)

        # Load YOLOv11x face detector
        model_path = hf_hub_download(
            repo_id="AdamCodd/YOLOv11x-face-detection", filename="model.pt"
        )
        self.face_detector = YOLO(model_path)

    def extract_face_or_full(self, images: torch.Tensor) -> torch.Tensor:
        """
        Detects faces using YOLO. If found, crops the largest face; otherwise keeps the full image.
        Args:
            images (torch.Tensor): Tensor of shape (B, C, H, W).
        Returns:
            processed_images (torch.Tensor): Tensor of cropped or original images.
        """
        processed_images = []
        b, c, h, w = images.shape

        for i in range(b):
            img = images[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)

            # Run YOLO detection
            results = self.face_detector.predict(img, verbose=False)

            if len(results[0].boxes) > 0:
                # Pick the largest face (by area)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                largest_idx = areas.argmax()
                x1, y1, x2, y2 = boxes[largest_idx].astype(int)

                # Crop face
                face = img[y1:y2, x1:x2, :]
                if face.size == 0:
                    face = img  # fallback if invalid crop
            else:
                # Fallback: whole image
                face = img

            # Resize to visual encoder input size
            face_resized = cv2.resize(face, (224, 224))  # EfficientNet default input
            face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255.0
            processed_images.append(face_tensor)

        return torch.stack(processed_images).to(images.device)

    def forward(
        self,
        images: torch.Tensor,
        audio_ce: torch.Tensor,
        audio_cl1: torch.Tensor,
        audio_cl2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # --- Supervised Classification Path ---
        if images.dim() == 5:
            batch_size, num_frames, C, H, W = images.shape
            images = images.view(batch_size * num_frames, C, H, W)

            # Extract face or fallback to full image
            images = self.extract_face_or_full(images)

            visual_features = self.visual_net(images)
            visual_features = visual_features.view(batch_size, num_frames, -1).mean(
                dim=1
            )
        else:
            images = self.extract_face_or_full(images)
            visual_features = self.visual_net(images)

        # Shared Audio Encoder
        audio_features_ce = self.audio_net(audio_ce.repeat(1, 3, 1, 1))

        # Fusion and Classification
        fused_features = torch.cat((visual_features, audio_features_ce), dim=1)
        emotion_logits = self.classification_head(fused_features)

        # --- Contrastive Path ---
        audio_features_cl1 = self.audio_net(audio_cl1.repeat(1, 3, 1, 1))
        audio_features_cl2 = self.audio_net(audio_cl2.repeat(1, 3, 1, 1))

        audio_embedding_1 = self.contrastive_head(audio_features_cl1)
        audio_embedding_2 = self.contrastive_head(audio_features_cl2)

        return emotion_logits, audio_embedding_1, audio_embedding_2
