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

class TransReIDExtractor:
    def __init__(self, model_path, device='cpu'):
        from modeling import build_transformer
        import yaml
        self.device = device
        # Assume config.yaml is in the same directory as model_path
        config_dir = os.path.dirname(model_path)
        config_file = os.path.join(config_dir, 'config.yaml')
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        self.model = build_transformer(cfg, num_class=1)
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((cfg.get('height', 256), cfg.get('width', 128))),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(768)  # TransReID default embedding size
        inp = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        return feat

class ReIDFactory:
    def __init__(self, reid_type='cnn', model_name='osnet_ibn_x1_0', device='cpu', checkpoint=None):
        self.reid_type = reid_type
        self.model_name = model_name
        self.device = device
        if reid_type == 'cnn':
            self.model, self.transform = self._load_cnn_model(model_name)
            self.extract_fn = self._extract_cnn
        elif reid_type == 'transformer':
            if checkpoint is None:
                raise ValueError('Transformer ReID requires a checkpoint path.')
            self.model = TransReIDExtractor(checkpoint, device)
            self.extract_fn = self._extract_transformer
        else:
            raise ValueError(f"Unknown reid_type: {reid_type}. Only 'cnn' and 'transformer' are supported.")
        if reid_type == 'cnn':
            self.model = self.model.to(device)
            self.model.eval()

    def _load_cnn_model(self, model_name):
        try:
            import torchreid
        except ImportError:
            raise ImportError("Please install torchreid: pip install torchreid")
        model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        _, test_transform = torchreid.data.transforms.build_transforms(
            256, 128, is_train=False
        )
        return model, test_transform

    def _extract_cnn(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros(512)
        crop = Image.fromarray(crop[..., ::-1])
        inp = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(inp).cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        return feat

    def _extract_transformer(self, image, bbox):
        return self.model.extract(image, bbox)

    def extract(self, image, bbox):
        return self.extract_fn(image, bbox)
