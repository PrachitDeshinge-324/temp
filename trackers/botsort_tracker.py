# BoT-SORT Tracker Stub
import numpy as np
from scipy.optimize import linear_sum_assignment

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
    def __init__(self, bbox, track_id, conf):
        self.bbox = bbox
        self.track_id = track_id
        self.conf = conf
        self.age = 0
        self.time_since_update = 0

class BoTSORTTracker:
    def __init__(self, iou_threshold=0.3, max_age=30, min_confidence=0.3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_confidence = min_confidence
        self.tracks = []
        self.next_id = 1

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
        # Prepare track bboxes
        track_bboxes = np.array([t.bbox for t in self.tracks]) if self.tracks else np.empty((0,4))
        # Association (IoU-based, placeholder for future ReID)
        if len(track_bboxes) > 0 and len(dets) > 0:
            iou_matrix = np.zeros((len(track_bboxes), len(dets)), dtype=np.float32)
            for t, tb in enumerate(track_bboxes):
                for d, db in enumerate(dets):
                    iou_matrix[t, d] = iou(tb, db)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            assigned_tracks = set()
            assigned_dets = set()
            for t, d in zip(row_ind, col_ind):
                if iou_matrix[t, d] >= self.iou_threshold:
                    self.tracks[t].bbox = dets[d]
                    self.tracks[t].conf = det_confs[d]
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
                    self.tracks.append(Track(det[:4], self.next_id, det[4]))
                    self.next_id += 1
        else:
            # No tracks or no detections: all detections are new tracks
            for det in detections:
                self.tracks.append(Track(det[:4], self.next_id, det[4]))
                self.next_id += 1
            for track in self.tracks:
                track.time_since_update += 1
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        # Output: [x1, y1, x2, y2, track_id, conf]
        return [[*t.bbox, t.track_id, t.conf] for t in self.tracks if t.time_since_update == 0]