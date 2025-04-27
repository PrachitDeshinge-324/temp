# YOLOv8 Detector Stub
import numpy as np
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_name='yolov8n', confidence_threshold=0.3, device='cpu'):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = YOLO(model_name + '.pt')
        self.model.to(device)

    def detect(self, image: np.ndarray):
        """
        Args:
            image (np.ndarray): Input image (BGR, OpenCV format)
        Returns:
            List of detections: [ [x1, y1, x2, y2, confidence, class_id], ... ]
        """
        results = self.model(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf >= self.confidence_threshold:  # 0 is 'person' in COCO
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf, cls_id])
        return detections