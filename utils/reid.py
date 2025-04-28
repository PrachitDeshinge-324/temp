import sys
import os
sys.path.append(os.path.abspath('./TransReID-SSL/transreid_pytorch'))

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from PIL import Image

class SimpleReID:
    def __init__(self, device='cpu'):
        # Use a pre-trained ResNet18 as a simple feature extractor
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Identity()  # Remove classification head
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, image, bbox):
        # bbox: [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(512)
        inp = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp).cpu().numpy().flatten()
        # Normalize
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        return feat

class ReIDFactory:
    def __init__(self, reid_type='cnn', model_name='osnet_ibn_x1_0', device='cpu', checkpoint=None):
        self.reid_type = reid_type
        self.model_name = model_name
        self.device = device
        if reid_type == 'cnn':
            self.model, self.transform = self._load_cnn_model(model_name)
        else:
            raise ValueError(f"Unknown reid_type: {reid_type}. Only 'cnn' is supported.")
        self.model = self.model.to(device)
        self.model.eval()

    def _load_cnn_model(self, model_name):
        try:
            import torchreid
        except ImportError:
            raise ImportError("Please install torchreid: pip install torchreid")
        # Download pretrained weights if not present
        model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        # Use only the test transform for inference
        _, test_transform = torchreid.data.transforms.build_transforms(
            256, 128, is_train=False
        )
        return model, test_transform

    def extract(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(512)
        # Convert to PIL Image for torchreid transforms
        crop = Image.fromarray(crop[..., ::-1])  # Convert BGR (OpenCV) to RGB (PIL)
        inp = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp).cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        return feat
