import cv2
import yaml
import torch
import numpy as np
import sys
import csv
import time
from detectors.yolov8_detector import YOLOv8Detector
from detectors.efficientdet_detector import EfficientDetDetector
from trackers.bytetrack_tracker import ByteTrackTracker
from trackers.botsort_tracker import BoTSORTTracker
from utils.visualization import draw_tracks

# Utility: Load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config('config.yaml')
    video_source = config['video_source']
    detector_cfg = config['detector']
    tracker_cfg = config['tracker']
    vis_cfg = config['visualization']

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize detector
    if 'yolov8' in detector_cfg['model']:
        detector = YOLOv8Detector(model_name=detector_cfg['model'], confidence_threshold=detector_cfg['confidence_threshold'], device=device)
    elif 'efficientdet' in detector_cfg['model']:
        detector = EfficientDetDetector(model_name=detector_cfg['model'], confidence_threshold=detector_cfg['confidence_threshold'], device=device)
    else:
        print(f"Unknown detector model: {detector_cfg['model']}")
        sys.exit(1)

    # Initialize tracker
    reid_cfg = config.get('reid', {})
    reid_type = reid_cfg.get('type', 'cnn')
    reid_model = reid_cfg.get('model', 'osnet_x1_0')
    reid_checkpoint = reid_cfg.get('checkpoint', None)
    if tracker_cfg['algorithm'] == 'bytetrack':
        tracker = ByteTrackTracker(iou_threshold=tracker_cfg['iou_threshold'], max_age=tracker_cfg['max_age'], min_confidence=tracker_cfg['min_confidence'])
    elif tracker_cfg['algorithm'] == 'botsort':
        tracker = BoTSORTTracker(
            iou_threshold=tracker_cfg['iou_threshold'],
            max_age=tracker_cfg['max_age'],
            min_confidence=tracker_cfg['min_confidence'],
            device=device,
            reid_type=reid_type,
            reid_model=reid_model,
            checkpoint=reid_checkpoint
        )
    else:
        print(f"Unknown tracker algorithm: {tracker_cfg['algorithm']}")
        sys.exit(1)

    # Prepare logging if enabled
    log_cfg = config.get('logging', {})
    save_tracks = log_cfg.get('save_tracks', False)
    tracks_file = log_cfg.get('tracks_file', 'tracks.csv')
    csv_writer = None
    csv_file = None
    if save_tracks:
        csv_file = open(tracks_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])

    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_source}")
        sys.exit(1)

    # VideoWriter setup if saving output
    out_writer = None
    if vis_cfg.get('save_output', False):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(vis_cfg.get('output_path', 'output.mp4'), fourcc, fps, (width, height))

    frame_id = 0
    max_frames = 15800
    start_frame = 14600
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id < start_frame:
            continue
        if frame_id > max_frames:
            break
        # Detect persons
        try:
            detections = detector.detect(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            break
        # Track persons
        try:
            tracks = tracker.update(detections, frame)
        except Exception as e:
            print(f"Tracking error: {e}")
            break
        # Draw boxes and track IDs on frame
        frame = draw_tracks(frame, tracks)
        # Log tracks if enabled
        if save_tracks and csv_writer:
            for track in tracks:
                x1, y1, x2, y2, track_id, conf = track
                csv_writer.writerow([frame_id, int(track_id), int(x1), int(y1), int(x2), int(y2)])
        # Write frame to output video if enabled
        if out_writer:
            out_writer.write(frame)
        # FPS calculation
        elapsed = time.time() - start_time
        fps = (frame_id + 1) / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if vis_cfg['display']:
            cv2.imshow('Person Detection & Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print(f"Average FPS: {fps:.2f}")

    if save_tracks and csv_file:
        csv_file.close()

    if out_writer:
        out_writer.release()
        print(f"Output video saved to {vis_cfg.get('output_path', 'output.mp4')}")

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")

if __name__ == '__main__':
    main()
