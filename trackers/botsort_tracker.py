# BoT-SORT Tracker Stub
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from utils.reid import ReIDFactory
from scipy.spatial.distance import cdist

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
        self.kf = self._init_kalman_filter(bbox)
        self.history = []

    def _init_kalman_filter(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.
        y = y1 + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7)
        for i in range(4):
            kf.F[i, i+3] = 1
        kf.H = np.eye(4, 7)
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = np.array([x, y, s, r]).reshape((4,1))
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.kf.x[:4].reshape(-1))
        return self.kf.x[:4].reshape(-1)

    def update(self, bbox, embedding, conf):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.
        y = y1 + h / 2.
        s = w * h
        r = w / float(h + 1e-6)
        self.kf.update(np.array([x, y, s, r]))
        self.bbox = bbox
        self.embedding = embedding
        self.conf = conf
        self.time_since_update = 0

    def get_state(self):
        x, y, s, r = self.kf.x[:4].reshape(-1)
        # Clamp s and r to be positive and nonzero
        s = max(s, 1e-2)
        r = max(r, 1e-2)
        w = np.sqrt(s * r)
        h = s / (w + 1e-6)
        x1 = x - w / 2.
        y1 = y - h / 2.
        x2 = x + w / 2.
        y2 = y + h / 2.
        return [x1, y1, x2, y2]

class BoTSORTTracker:
    def __init__(self, iou_threshold=0.3, max_age=90, min_confidence=0.3, device='cpu', reid_type='cnn', reid_model='osnet_x1_0', checkpoint=None):
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
        # Predict new locations for all tracks
        for t in self.tracks:
            t.predict()
        track_states = np.array([t.get_state() for t in self.tracks]) if self.tracks else np.empty((0,4))
        track_embs = np.array([t.embedding for t in self.tracks]) if self.tracks else np.empty((0,512))
        # Association: combine IoU, appearance, and motion (center distance)
        assigned_tracks = set()
        assigned_dets = set()
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(detections)))
        if len(track_states) > 0 and len(dets) > 0:
            iou_matrix = np.zeros((len(track_states), len(dets)), dtype=np.float32)
            for t, tb in enumerate(track_states):
                for d, db in enumerate(dets):
                    iou_matrix[t, d] = self._iou(tb, db)
            emb_matrix = cdist(track_embs, det_embs, metric='cosine') if len(track_embs) > 0 and len(det_embs) > 0 else np.zeros_like(iou_matrix)
            motion_matrix = np.zeros((len(track_states), len(dets)), dtype=np.float32)
            for t, tb in enumerate(track_states):
                tx, ty = (tb[0]+tb[2])/2, (tb[1]+tb[3])/2
                for d, db in enumerate(dets):
                    dx, dy = (db[0]+db[2])/2, (db[1]+db[3])/2
                    motion_matrix[t, d] = np.linalg.norm([tx-dx, ty-dy])
            if motion_matrix.max() > 0:
                motion_matrix = motion_matrix / (motion_matrix.max() + 1e-6)
            cost_matrix = 0.4 * (1 - iou_matrix) + 0.4 * emb_matrix + 0.2 * motion_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for t, d in zip(row_ind, col_ind):
                if iou_matrix[t, d] >= self.iou_threshold and emb_matrix[t, d] < 0.5:
                    self.tracks[t].update(dets[d], det_embs[d], det_confs[d])
                    assigned_tracks.add(t)
                    assigned_dets.add(d)
            unmatched_tracks -= assigned_tracks
            unmatched_dets -= assigned_dets
            # SECOND STAGE: Try to match remaining detections to unmatched tracks using only embedding
            if len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
                emb_matrix2 = cdist(track_embs[list(unmatched_tracks)], [det_embs[d] for d in unmatched_dets], metric='cosine')
                row2, col2 = linear_sum_assignment(emb_matrix2)
                for idx, (t_idx, d_idx) in enumerate(zip(row2, col2)):
                    t = list(unmatched_tracks)[t_idx]
                    d = list(unmatched_dets)[d_idx]
                    if emb_matrix2[t_idx, d_idx] < 0.6:  # looser threshold for reid-only match
                        self.tracks[t].update(dets[d], det_embs[d], det_confs[d])
                        assigned_tracks.add(t)
                        assigned_dets.add(d)
                unmatched_tracks -= assigned_tracks
                unmatched_dets -= assigned_dets
            # Unmatched tracks: do not update embedding, just increment time_since_update
            for i, track in enumerate(self.tracks):
                if i not in assigned_tracks:
                    track.time_since_update += 1
            # New tracks for unmatched detections
            for d in unmatched_dets:
                self.tracks.append(Track(dets[d], self.next_id, det_confs[d], det_embs[d]))
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
        return [[*t.get_state(), t.track_id, t.conf] for t in self.tracks if t.time_since_update == 0]

    def _iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
        return o