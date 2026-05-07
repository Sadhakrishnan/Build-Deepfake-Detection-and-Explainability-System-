import cv2
import os

def extract_frames(video_path, output_dir=None, frame_interval=10):
    """
    Extracts frames from a video at a specified interval.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    metadata = []
    
    count = 0
    extracted_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Calculate timestamp
            timestamp = count / fps if fps > 0 else 0
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            
            metadata.append({
                "frame_id": extracted_count,
                "timestamp": timestamp_str,
                "original_frame_idx": count
            })
            extracted_count += 1
            
        count += 1
        
    cap.release()
    return frames, metadata
