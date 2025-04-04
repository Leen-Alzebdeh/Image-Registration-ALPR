from ultralytics import YOLO
from norfair import Detection, Tracker, distances, Video
import numpy as np
import os
import cv2
from utils import iou_distance

FIXED_SIZE = (256, 128)


def detect_plates(frames_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO("license_plate_detector.pt")

    # Instead of a single list, store bboxes per frame
    bboxes_per_frame = {}

    for img_name in sorted(os.listdir(frames_path)):
        img_path = os.path.join(frames_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            continue

        results = model(img, conf=0.5)
        frame_bboxes = []

        for r in results:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                width = x2 - x1
                height = y2 - y1

                # Keep bounding boxes that are large enough
                if (height/img.shape[0] > 0.03) and (width/img.shape[1] > 0.03):
                    frame_bboxes.append((x1, y1, x2, y2))

                    plate_crop = img[y1:y2, x1:x2]
                    resized = cv2.resize(plate_crop, FIXED_SIZE)
                    if resized.size > 0:
                        save_path = os.path.join(
                            output_folder, f"{img_name[:-4]}_plate_{i}.png")
                        cv2.imwrite(save_path, resized)

        bboxes_per_frame[img_name] = frame_bboxes

    print("License plates extracted from frames.")
    return bboxes_per_frame


def track(input_folder, output_folder, bboxes_dict):
    print("Tracking license plates across frames...")
    tracker = Tracker(
        distance_function=iou_distance,
        distance_threshold=0.9  # bigger means track boxes that have IoU >= 0.1
    )
    plate_sequences = {}

    os.makedirs(output_folder, exist_ok=True)

    frame_paths = sorted(os.listdir(input_folder))

    for img_name in frame_paths:
        img_path = os.path.join(input_folder, img_name)
        print(img_path)

        r_img_name = img_name.split("_plate_")[0] + ".png"
        # Convert bounding boxes to 2D points
        if r_img_name not in bboxes_dict:
            frame_bboxes = []
        else:
            frame_bboxes = bboxes_dict[r_img_name]

        detections = []
        print(frame_bboxes)
        for (x1, y1, x2, y2) in frame_bboxes:
            # shape (1,4)
            box_array = np.array([[x1, y1, x2, y2]], dtype=float)
            # shape (1,)
            score_array = np.array([1.0], dtype=float)

            detection = Detection(points=box_array, scores=score_array)
            detections.append(detection)

        print(detections)
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
        if len(frames) >= 5:
            track_folder = os.path.join(output_folder, f"plate_{track_id}")
            os.makedirs(track_folder, exist_ok=True)
            for path in frames:
                img = cv2.imread(path)
                img_name = os.path.basename(path)
                cv2.imwrite(os.path.join(track_folder, img_name), img)

    print("Filtered plates saved in", output_folder)

    print(
        f"Filtered plates saved in {output_folder}. Only sequences with ≥5 frames kept.")


def main():
    frames_path = "frames/"
    detect_output_folder = "plates/"  # Folder with cropped license plates
    bboxes = detect_plates(frames_path, detect_output_folder)
    output_folder = "filtered_plates/"
    track(detect_output_folder, output_folder, bboxes)


if __name__ == "__main__":
    main()
