import cv2

def draw_tracks(frame, tracks):
    """
    Draw tracked bounding boxes and IDs on the frame.
    Args:
        frame: np.ndarray (BGR)
        tracks: List of [x1, y1, x2, y2, track_id, confidence]
    Returns:
        frame: np.ndarray with drawings
    """
    for track in tracks:
        x1, y1, x2, y2, track_id, conf = track
        color = (int(track_id * 37) % 256, int(track_id * 17) % 256, int(track_id * 29) % 256)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f'ID:{int(track_id)} {conf:.2f}'
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
