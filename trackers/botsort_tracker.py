# BoT-SORT Tracker Stub
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.reid import SimpleReID
from scipy.spatial.distance import cdist
from utils.reid import ReIDFactory

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
    return o

class Track:
    def __init__(self, bbox, track_id, conf, embedding):
        self.bbox = bbox
        self.track_id = track_id
        self.conf = conf
        self.embedding = embedding
        self.age = 0
        self.time_since_update = 0

class BoTSORTTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_confidence=0.3, device='cpu', reid_type='cnn', reid_model='osnet_x1_0', checkpoint=None):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_confidence = min_confidence
        self.tracks = []
        self.next_id = 1
        self.reid = ReIDFactory(reid_type=reid_type, model_name=reid_model, device=device, checkpoint=checkpoint)

    def update(self, detections, image: np.ndarray = None):
        """
        Args:
            detections: List of detections [ [x1, y1, x2, y2, confidence, class_id], ... ]
            image: Optional, for trackers that use appearance features
        Returns:
            List of tracks: [ [x1, y1, x2, y2, track_id, confidence], ... ]
        """
        # Filter detections by confidence
        detections = [d for d in detections if d[4] >= self.min_confidence]
        dets = np.array([d[:4] for d in detections]) if detections else np.empty((0,4))
        det_confs = [d[4] for d in detections]
        # Extract embeddings for detections
        det_embs = []
        if image is not None and len(detections) > 0:
            for det in detections:
                det_embs.append(self.reid.extract(image, det[:4]))
        else:
            det_embs = [np.zeros(512) for _ in detections]
        # Prepare track bboxes and embeddings
        track_bboxes = np.array([t.bbox for t in self.tracks]) if self.tracks else np.empty((0,4))
        track_embs = np.array([t.embedding for t in self.tracks]) if self.tracks else np.empty((0,512))
        # Association: combine IoU and appearance
        assigned_tracks = set()
        assigned_dets = set()
        if len(track_bboxes) > 0 and len(dets) > 0:
            iou_matrix = np.zeros((len(track_bboxes), len(dets)), dtype=np.float32)
            for t, tb in enumerate(track_bboxes):
                for d, db in enumerate(dets):
                    iou_matrix[t, d] = iou(tb, db)
            # Appearance distance (cosine)
            emb_matrix = cdist(track_embs, det_embs, metric='cosine') if len(track_embs) > 0 and len(det_embs) > 0 else np.zeros_like(iou_matrix)
            # Combine: prefer appearance, fallback to IoU
            cost_matrix = 0.5 * (1 - iou_matrix) + 0.5 * emb_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for t, d in zip(row_ind, col_ind):
                if iou_matrix[t, d] >= self.iou_threshold or emb_matrix[t, d] < 0.4:
                    self.tracks[t].bbox = dets[d]
                    self.tracks[t].conf = det_confs[d]
                    self.tracks[t].embedding = det_embs[d]
                    self.tracks[t].time_since_update = 0
                    assigned_tracks.add(t)
                    assigned_dets.add(d)
            # Unmatched tracks
            for i, track in enumerate(self.tracks):
                if i not in assigned_tracks:
                    track.time_since_update += 1
            # New tracks
            for d, det in enumerate(detections):
                if d not in assigned_dets:
                    self.tracks.append(Track(det[:4], self.next_id, det[4], det_embs[d]))
                    self.next_id += 1
        else:
            # No tracks or no detections: all detections are new tracks
            for idx, det in enumerate(detections):
                self.tracks.append(Track(det[:4], self.next_id, det[4], det_embs[idx]))
                self.next_id += 1
            for track in self.tracks:
                track.time_since_update += 1
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        # Output: [x1, y1, x2, y2, track_id, conf]
        return [[*t.bbox, t.track_id, t.conf] for t in self.tracks if t.time_since_update == 0]