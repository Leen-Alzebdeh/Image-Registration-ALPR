import cv2
from utils import align_image_orb_ransac, average_aligned_frames
import os


def process_sequence(plate_frames):
    """
    plate_frames: list of file paths for consecutive frames of the same license plate
    returns: averaged image
    """
    # 1) Load the first image as reference
    ref = cv2.imread(plate_frames[0])
    aligned_images = [ref]

    # 2) Align each subsequent frame to ref
    for p in plate_frames[1:]:
        moving = cv2.imread(p)
        aligned = align_image_orb_ransac(ref, moving)
        aligned_images.append(aligned)

    # 3) Average them
    avg_img = average_aligned_frames(aligned_images)
    return avg_img


def iterate_plates(input_path, output_path):
    for plate_path in os.listdir(input_path):
        plate_path = 'plate_1_GX9'
        frame_paths = []
        print(plate_path)
        for idx, frame in enumerate(os.listdir(input_path+'/'+plate_path)):
            print(frame)
            if frame.endswith('.png'):
                frame_paths.append(
                    input_path+'/'+plate_path+'/'+frame)
                print(frame_paths)
                if idx >= 4:         # start processing at 5+ frames
                    avg_img = process_sequence(frame_paths[-5:])
                    # Save the aligned image
                    os.makedirs(output_path + '/' +
                                plate_path, exist_ok=True)
                    cv2.imwrite(output_path + '/' +
                                plate_path + '/' + frame, avg_img)
        break


def main():
    input_path = 'filtered_plates'
    output_path = 'aligned_plates'
    os.makedirs(output_path, exist_ok=True)
    iterate_plates(input_path, output_path)


main()
