import os
import cv2
from glob2 import glob
from tqdm import tqdm

def extract_frames(path, save_dir, fps):
    """
    Extract frames from a video file at a specified rate.

    Args:
        path (str): Path to the video file.
        save_dir (str): Directory to save the extracted frames.
        fps (int): Number of frames to extract per second.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error opening video file: {path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print("Error: Could not get video FPS")
        cap.release()
        return

    interval = int(video_fps / fps)  # Number of video frames per extracted frame
    if interval == 0:
        interval = 1  # Extract every frame if fps > video_fps

    frame_num = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % interval == 0:
            filename = f"{saved_count:05d}.jpg"  # Format: 00000.jpg, 00001.jpg, etc.
            cv2.imwrite(os.path.join(save_dir, filename), frame)
            saved_count += 1

        frame_num += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {save_dir}")

if __name__ == "__main__":
    video_dir = "datasets/XJTU-MMUS-subset-20260401/videos"
    video_paths = glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    with tqdm(total=len(video_paths), desc="Extracting frames") as pbar:
        for video_path in video_paths:
            save_dir = video_path.replace("/videos/", "/images/").replace(".mp4", "")
            extract_frames(video_path, save_dir, fps=2)
            pbar.update(1)
