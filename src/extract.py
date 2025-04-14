import os
import cv2
import shutil
from pathlib import Path
from utils import input_args

FRAME_RATE = 5  # Extract N frames per second
output_dir = "frames"


def extract_frames(start, end, video_path, suffix):
    """
    Extract frames from a video file and save them as images.
    """
    # Set video path and output folder
    output_video_dir = f'{output_dir}/{video_path}{suffix}'

    dirpath = Path(output_video_dir)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    os.makedirs(output_video_dir, exist_ok=True)

    print(f"Extracting frames from video {video_path}")
    print(f"Outputting frames to {output_video_dir}")

    cap = cv2.VideoCapture('videos/'+video_path+'.mp4')
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video at {video_path}")

    # Get native video FPS
    native_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if end is None:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Compute duration in minutes
        end = frame_count / native_fps

    start_frame = int(start * 60 * native_fps)
    end_frame = int(end * 60 * native_fps)

    print(f"Native FPS: {native_fps}")
    print(f"Start frame: {start_frame}, End frame: {end_frame}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_id > end_frame:
            break

        if frame_id % FRAME_RATE == 0:
            # convert to webp at 90% for reduced storage + good quality
            cv2.imwrite(os.path.join(output_video_dir,
                        f"frame_{frame_id:06d}.webp"), frame, [cv2.IMWRITE_WEBP_QUALITY, 90])

        frame_id += 1

    cap.release()
    print("Frame extraction completed.")


def main():
    args = input_args()
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print("start", args.start)
    extract_frames(args.start, args.end,
                   (args.video).split('.')[0], args.suffix)


if main() == "__main__":
    main()
