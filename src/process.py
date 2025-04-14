import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from norfair import Detection, Tracker
from utils import combined_distance, bboxe_save, bboxes_load, input_args

FIXED_SIZE = (256, 128)
CUBIC = True
CUBIC_SUFFIX = '_cubic' if CUBIC else ''
MIN_FRAMES = 5  # Minimum number of frames to keep a track
MIN_LENGTH = 0.00
CONFIDENCE_THRESHOLD = 0.3

frames_path = 'frames'
detect_output_dir = f'plates{CUBIC_SUFFIX}'
track_output_dir = f'filtered_plates{CUBIC_SUFFIX}'


def detect_plates(video_path, detect_output_folder):
    frames_folder = f"{frames_path}/{video_path}"

    print(f"Detecting license plates from {frames_folder}")
    print(f"Outputting detections to {detect_output_folder}")

    dirpath = Path(detect_output_folder)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.makedirs(detect_output_folder, exist_ok=True)

    model = YOLO("license_plate_detector.pt")
    # Instead of a single list, store bboxes per frame
    bboxes_per_frame = {}

    for img_name in sorted(os.listdir(frames_folder)):
        img_number = int(img_name.split("_")[1].split(".")[0])
        if img_number < 30 * 10 * 60:
            print(img_number)
            img_path = os.path.join(frames_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load {img_path}")
                continue

            results = model(img, conf=CONFIDENCE_THRESHOLD)
            frame_bboxes = []

            for r in results:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    width = x2 - x1
                    height = y2 - y1

                    aspect_ratio = width / height
                    area_ratio = (width * height) / \
                        (img.shape[0] * img.shape[1])

                    if 1.5 < aspect_ratio < 6.5 and area_ratio < 0.50:
                        frame_bboxes.append((x1, y1, x2, y2))

                        plate_crop = img[y1:y2, x1:x2]
                        resized = cv2.resize(
                            plate_crop, FIXED_SIZE, interpolation=cv2.INTER_CUBIC) if CUBIC else cv2.resize(
                            plate_crop, FIXED_SIZE)
                        if resized.size > 0:
                            save_path = os.path.join(
                                detect_output_folder, f"{img_name[:-4]}_plate_{i}.png")
                            cv2.imwrite(save_path, resized)

            bboxes_per_frame[img_name] = frame_bboxes

    bboxe_save(bboxes_per_frame, f'_{video_path}{CUBIC_SUFFIX}')

    print("License plates extracted from frames.")


def track(video_path, input_folder, track_output_dir, suffix):
    track_output_folder = f"{track_output_dir}/{video_path}{suffix}"

    print(f"Tracking license plates from {input_folder}")
    print(f"Outputting trackings to {track_output_folder}")

    dirpath = Path(track_output_folder)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.makedirs(track_output_folder, exist_ok=True)

    bboxes_dict = bboxes_load(f'_{video_path}{CUBIC_SUFFIX}')

    print("Tracking license plates across frames...")
    tracker = Tracker(
        distance_function=combined_distance,
        distance_threshold=0.3,  # bigger means more tolerant to change
        hit_counter_max=15,      # number of frames to keep a track alive, ~ 1.5 seconds
        initialization_delay=1,
    )
    plate_sequences = {}

    frame_paths = sorted(os.listdir(input_folder))

    for img_name in frame_paths:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        r_img_name = img_name.split("_plate_")[0] + ".png"
        # Convert bounding boxes to 2D points
        if r_img_name not in bboxes_dict:
            frame_bboxes = []
        else:
            frame_bboxes = bboxes_dict[r_img_name]

        detections = []
        for (x1, y1, x2, y2) in frame_bboxes:
            # shape (1,4)
            box_array = np.array([[x1, y1, x2, y2]], dtype=float)
            # shape (1,)
            score_array = np.array([1.0], dtype=float)

            detection = Detection(
                points=box_array, scores=score_array, data={"origin": img})
            detections.append(detection)

        tracked_objects = tracker.update(detections)

        # Save frames per track ID
        for obj in tracked_objects:
            track_id = obj.id
            if track_id not in plate_sequences:
                plate_sequences[track_id] = []
            plate_sequences[track_id].append(img_path)

    print(plate_sequences)
    # Keep only tracks with ≥5 frames
    for track_id, frames in plate_sequences.items():
        if len(frames) >= MIN_FRAMES:
            track_folder = os.path.join(
                track_output_folder, f"plate{track_id:03}_")
            os.makedirs(track_folder, exist_ok=True)
            for path in frames:
                img = cv2.imread(path)
                img_name = os.path.basename(path)
                if len(img.shape) == 3:
                    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(track_folder, img_name), img_grey)

    print(
        f"Filtered plates saved in {track_output_folder}. Only sequences with ≥{MIN_FRAMES} frames kept.")


def main():
    args = input_args()
    video_path = args.video.split('.')[0]
    track_only = args.track

    os.makedirs(detect_output_dir, exist_ok=True)
    os.makedirs(track_output_dir, exist_ok=True)

    detect_output_folder = f"{detect_output_dir}/{video_path}"
    output_folder = f"{track_output_dir}/{video_path}"
    if not track_only:
        detect_plates(video_path, detect_output_folder)
    track(video_path, detect_output_folder, output_folder, args.suffix)


if __name__ == "__main__":
    main()
