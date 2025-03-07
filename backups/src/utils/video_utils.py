import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Install with pip install face_recognition")

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available. Install with pip install mediapipe")

def extract_frames(video_path, max_frames=None, fps=25, resize=None):
    """Extract frames from a video file
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        fps: Target frames per second (None to keep original)
        resize: Tuple (width, height) to resize frames
        
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame step if FPS is specified
    frame_step = 1
    if fps and fps < original_fps:
        frame_step = max(1, int(original_fps / fps))
    
    # Set maximum frames to extract if not specified
    if max_frames is None:
        max_frames = frame_count
    
    # Read frames with appropriate step
    frame_idx = 0
    while len(frames) < max_frames and frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Resize if needed
        if resize is not None:
            frame = cv2.resize(frame, resize)
        
        frames.append(frame)
        frame_idx += frame_step
    
    cap.release()
    return frames

def extract_face_landmarks(frames, feature_type="lip", use_mediapipe=True):
    """Extract face landmarks from frames
    
    Args:
        frames: List of video frames
        feature_type: Type of features to extract ("lip", "face")
        use_mediapipe: Whether to use MediaPipe (True) or face_recognition (False)
        
    Returns:
        Array of landmarks features
    """
    if not frames:
        return np.array([])
    
    # Use MediaPipe for higher precision
    if use_mediapipe and MEDIAPIPE_AVAILABLE:
        return _extract_mediapipe_landmarks(frames, feature_type)
    
    # Fall back to face_recognition
    if FACE_RECOGNITION_AVAILABLE:
        return _extract_face_recognition_landmarks(frames, feature_type)
    
    raise ImportError("Neither MediaPipe nor face_recognition is available")

def _extract_mediapipe_landmarks(frames, feature_type):
    """Extract landmarks using MediaPipe"""
    features = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        
        for frame in tqdm(frames, desc="Extracting landmarks"):
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = face_mesh.process(rgb_frame)
            
            # Extract landmarks if face detected
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract specific features based on type
                if feature_type == "lip":
                    # Lip landmarks indices (MediaPipe)
                    lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
                    # Extract coordinates
                    lip_landmarks = np.array([(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z) 
                                            for idx in lip_indices])
                    features.append(lip_landmarks.flatten())
                
                elif feature_type == "face":
                    # Use all landmarks for full face
                    face_landmarks = np.array([(landmark.x, landmark.y, landmark.z) 
                                             for landmark in landmarks])
                    features.append(face_landmarks.flatten())
            else:
                # If no face detected, use zeros as placeholder
                if feature_type == "lip":
                    feature_dim = 20 * 3  # 20 lip landmarks with x,y,z
                else:
                    feature_dim = 468 * 3  # Full face landmarks
                features.append(np.zeros(feature_dim))
    
    return np.array(features)

def _extract_face_recognition_landmarks(frames, feature_type):
    """Extract landmarks using face_recognition library"""
    features = []
    
    for frame in tqdm(frames, desc="Extracting landmarks"):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        
        if face_landmarks_list:
            landmarks = face_landmarks_list[0]
            
            # Extract specific features based on type
            if feature_type == "lip":
                # Extract lips
                top_lip = landmarks["top_lip"]
                bottom_lip = landmarks["bottom_lip"]
                lip_landmarks = top_lip + bottom_lip
                lip_landmarks = np.array(lip_landmarks)
                features.append(lip_landmarks.flatten())
            
            elif feature_type == "face":
                # Use all landmarks for full face
                all_landmarks = []
                for part in landmarks.values():
                    all_landmarks.extend(part)
                all_landmarks = np.array(all_landmarks)
                features.append(all_landmarks.flatten())
        else:
            # If no face detected, use zeros as placeholder
            if feature_type == "lip":
                feature_dim = 20 * 2  # Approximate lip landmarks
            else:
                feature_dim = 68 * 2  # Standard face_recognition landmarks
            features.append(np.zeros(feature_dim))
    
    return np.array(features)

def extract_batch_landmarks(video_paths, max_frames=None, feature_type="lip", n_workers=4):
    """Extract landmarks from multiple videos in parallel"""
    def process_video(video_path):
        frames = extract_frames(video_path, max_frames=max_frames)
        return extract_face_landmarks(frames, feature_type=feature_type)
    
    features = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_video, path) for path in video_paths]
        for future in tqdm(futures, total=len(video_paths), desc="Processing videos"):
            features.append(future.result())
    
    return features