import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import os

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
    def __init__(self, reid_type='cnn', model_name='osnet_x1_0', device='cpu'):
        self.reid_type = reid_type
        self.model_name = model_name
        self.device = device
        if reid_type == 'cnn':
            self.model, self.transform = self._load_cnn_model(model_name)
        elif reid_type == 'transformer':
            self.model, self.transform = self._load_transformer_model(model_name)
        else:
            raise ValueError(f"Unknown reid_type: {reid_type}")
        self.model = self.model.to(device)
        self.model.eval()

    def _load_cnn_model(self, model_name):
        # Example: OSNet from torchreid
        try:
            import torchreid
        except ImportError:
            raise ImportError("Please install torchreid: pip install torchreid")
        model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        transform = torchreid.data.transforms.build_transform(
            resize_h=256, resize_w=128, is_train=False
        )
        return model, transform

    def _load_transformer_model(self, model_name):
        # Example: TransReID (ViT-based)
        try:
            from transreid import build_model, build_transform
        except ImportError:
            raise ImportError("Please install TransReID and its dependencies.")
        model = build_model(model_name, pretrained=True)
        transform = build_transform(is_train=False)
        return model, transform

    def extract(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(512)
        inp = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp).cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        return feat
