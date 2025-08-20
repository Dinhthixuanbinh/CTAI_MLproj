import torch.nn as nn
from torchvision.models import resnet18

class ASTEncoder(nn.Module):
    def __init__(self):
        super(ASTEncoder, self).__init__()
        # Using a ResNet with 1 input channel for spectrograms
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        # Add a channel dimension for the spectrogram if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        features = self.resnet(x)
        return features