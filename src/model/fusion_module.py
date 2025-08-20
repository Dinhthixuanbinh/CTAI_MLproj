import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, config):
        super(FusionModule, self).__init__()
        self.fusion_method = config['fusion_method']
        if self.fusion_method == 'concat':
            self.fusion_layer = nn.Linear(512 + 512, 512) # Assuming 512-dim features from encoders
        # Implement other fusion methods here as needed

    def forward(self, audio_features, visual_features):
        if self.fusion_method == 'concat':
            fused = torch.cat((audio_features, visual_features), dim=1)
            output = self.fusion_layer(fused)
            return output
        # Handle other fusion methods here
        else:
            raise NotImplementedError(f"Fusion method {self.fusion_method} not implemented.")