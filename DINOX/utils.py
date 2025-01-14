import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=1):
    """Extract frames from a video file at specified FPS"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    os.makedirs(output_dir, exist_ok=True)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")
    return saved_count

def process_dataset(data_root="data/VSI-Bench", fps=1):
    """Process all videos in the dataset"""
    # Process arkitscenes
    print("Processing ARKitScenes...")
    arkit_dir = os.path.join(data_root, "arkitscenes")
    for video in tqdm(os.listdir(arkit_dir)):
        if video.endswith(".mp4"):
            scene_id = video.split(".")[0]
            video_path = os.path.join(arkit_dir, video)
            output_dir = os.path.join(arkit_dir, scene_id, "frames")
            extract_frames(video_path, output_dir, fps=fps)

    # Process scannet
    print("Processing ScanNet...")
    scannet_dir = os.path.join(data_root, "scannet")
    for video in tqdm(os.listdir(scannet_dir)):
        if video.endswith(".mp4"):
            scene_id = video.split(".")[0]
            video_path = os.path.join(scannet_dir, video)
            output_dir = os.path.join(scannet_dir, scene_id, "frames")
            extract_frames(video_path, output_dir, fps=fps)

    # Process scannetpp (if available)
    scannetpp_dir = os.path.join(data_root, "scannetpp")
    if os.path.exists(scannetpp_dir):
        print("Processing ScanNet++...")
        for video in tqdm(os.listdir(scannetpp_dir)):
            if video.endswith(".mp4"):
                scene_id = video.split(".")[0]
                video_path = os.path.join(scannetpp_dir, video)
                output_dir = os.path.join(scannetpp_dir, scene_id, "frames")
                extract_frames(video_path, output_dir, fps=fps)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/VSI-Bench", help="Path to dataset root")
    parser.add_argument("--fps", type=int, default=1, help="FPS for frame extraction")
    args = parser.parse_args()
    
    process_dataset(args.data_root, args.fps)