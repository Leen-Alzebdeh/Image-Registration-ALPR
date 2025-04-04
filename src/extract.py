import cv2
import os

video_path = "morning-short.mp4"
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 10  # Extract 10 frames per second
start_time = 0 * 60  # Convert minutes to seconds (10 min)
end_time = 40 * 60    # Convert minutes to seconds (40 min)
start_frame = start_time * frame_rate  # Convert to frame index
end_frame = end_time * frame_rate      # Convert to frame index
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if start_frame <= frame_id <= end_frame:
        if frame_id % frame_rate == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_id:06d}.png"), frame)
    
    frame_id += 1
    if frame_id > end_frame:
        break  # Stop when reaching the 40-minute mark

cap.release()
print("Frame extraction completed.")

