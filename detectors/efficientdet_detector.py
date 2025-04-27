import torch
import cv2
import numpy as np

class EfficientDetDetector:
    def __init__(self, model_name='efficientdet-d0', confidence_threshold=0.3, device='cpu'):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load EfficientDet from the rwightman/efficientdet repo using torch.hub
        self.model = torch.hub.load('rwightman/efficientdet', model_name, pretrained=True)
        self.model = self.model.to(device)
        self.model.eval()

        # Set input size based on model
        self.input_size = self.model.input_size
        self.class_map = {0: 'person'}  # COCO class 0 is 'person'

    def detect(self, image: np.ndarray):
        """
        Detects objects in the input image using EfficientDet.

        Args:
            image (np.ndarray): Input image (BGR, OpenCV format)

        Returns:
            List of detections: [ [x1, y1, x2, y2, confidence, class_id], ... ]
        """
        # Convert BGR to RGB for the model
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
        
        # Prepare the image for inference
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # Perform inference
            boxes, scores, labels = self.model(img_tensor)

        detections = []
        h, w, _ = image.shape
        
        # Process detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if int(label) == 0 and float(score) >= self.confidence_threshold:
                # Convert normalized coordinates back to the original image size
                x1, y1, x2, y2 = box.cpu().numpy()
                x1 = x1 / self.input_size * w
                x2 = x2 / self.input_size * w
                y1 = y1 / self.input_size * h
                y2 = y2 / self.input_size * h
                
                # Append detection
                detections.append([x1, y1, x2, y2, float(score), int(label)])
        
        return detections
