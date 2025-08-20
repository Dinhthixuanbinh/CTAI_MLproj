import torch.nn as nn
from .backbones.ma_net_encoder import MANetEncoder
from .backbones.ast_encoder import ASTEncoder
from .fusion_module import FusionModule

class EmotionClassifier(nn.Module):
    def __init__(self, config):
        super(EmotionClassifier, self).__init__()
        self.visual_encoder = MANetEncoder()
        self.audio_encoder = ASTEncoder()
        self.fusion_module = FusionModule(config)
        self.emotion_predictor = nn.Linear(in_features=512, out_features=config['num_classes'])

    def forward(self, audio_spec, visual_frames):
        # Pass through encoders
        audio_features = self.audio_encoder(audio_spec)
        visual_features = self.visual_encoder(visual_frames)
        
        # Fusion
        fused_features = self.fusion_module(audio_features, visual_features)
        
        # Emotion prediction
        output = self.emotion_predictor(fused_features)
        return output