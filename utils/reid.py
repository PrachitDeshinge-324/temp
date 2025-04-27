import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np

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
