# EfficientDet Detector Stub
import numpy as np
from efficientdet_pytorch import EfficientDet
import torch

class EfficientDetDetector:
    def __init__(self, model_name='efficientdet-d0', confidence_threshold=0.3, device='cpu'):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = EfficientDet.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.input_size = self.model.input_size
        self.class_map = {0: 'person'}  # COCO class 0 is 'person'

    def detect(self, image: np.ndarray):
        """
        Args:
            image (np.ndarray): Input image (BGR, OpenCV format)
        Returns:
            List of detections: [ [x1, y1, x2, y2, confidence, class_id], ... ]
        """
        import cv2
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            boxes, scores, labels = self.model(img_tensor)
        detections = []
        h, w, _ = image.shape
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if int(label) == 0 and float(score) >= self.confidence_threshold:
                x1, y1, x2, y2 = box.cpu().numpy()
                # Scale boxes back to original image size
                x1 = x1 / self.input_size * w
                x2 = x2 / self.input_size * w
                y1 = y1 / self.input_size * h
                y2 = y2 / self.input_size * h
                detections.append([x1, y1, x2, y2, float(score), int(label)])
        return detections