import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class VisualProcessor:
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def process_visual(self, visual_path):
        image_samples = os.listdir(visual_path)
        # Randomly select a fixed number of frames per video clip
        select_index = np.random.choice(len(image_samples), size=self.config['fps'], replace=False)
        select_index.sort()
        images = torch.zeros((self.config['fps'], 3, 224, 224))
        for i in range(self.config['fps']):
            img_path = os.path.join(visual_path, image_samples[select_index[i]])
            img = Image.open(img_path).convert('RGB')
            images[i] = self.transform(img)
        
        return torch.permute(images, (1, 0, 2, 3))