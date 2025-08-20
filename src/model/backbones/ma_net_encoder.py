import torch.nn as nn
from torchvision.models import resnet18

class MANetEncoder(nn.Module):
    def __init__(self):
        super(MANetEncoder, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity() # Remove the last fully connected layer

    def forward(self, x):
        # Assuming x is (B, C, T, H, W)
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)
        features = self.resnet(x)
        features = features.view(B, T, -1).mean(dim=1) # Average features across time dimension
        return features