import os
import cv2
import shutil
from pathlib import Path
from utils import align_image_orb_ransac, average_aligned_frames, input_args, align_image_optical_flow, align_image_diffeo

CUBIC = False
CUBIC_SUFFIX = '_cubic' if CUBIC else ''
MIN_FRAME = 4
input_path = f'filtered_plates{CUBIC_SUFFIX}'
output_path = f'aligned_plates{CUBIC_SUFFIX}'


def process_sequence(plate_frames, ir_method):
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
        if ir_method == 'orb':
            aligned = align_image_orb_ransac(ref, moving)
        elif ir_method == 'flow':
            align_image_optical_flow(ref, moving)
        elif ir_method == 'diffeo':
            aligned = align_image_diffeo(ref, moving)
        aligned_images.append(aligned)

    # 3) Average them
    avg_img = average_aligned_frames(aligned_images)
    return avg_img


def iterate_plates(video_path, ir_method):
    input_folder = f"{input_path}/{video_path}"
    output_folder = f"{output_path}/{video_path}"

    print(f"Aligning plates from {input_folder}")
    print(f"Outputting aligned plates to {output_folder}")

    dirpath = Path(output_folder)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.makedirs(output_folder, exist_ok=True)

    for plate_path in os.listdir(input_path):
        frame_paths = []
        full_input_path = input_folder+'/'+plate_path
        full_output_path = output_path + '/' + plate_path
        print(plate_path)
        for idx, frame in enumerate(os.listdir(full_input_path)):
            print(frame)
            if frame.endswith('.png'):
                frame_paths.append(
                    full_input_path)
                print(frame_paths)
                if idx >= MIN_FRAME:         # start processing at 5+ frames
                    avg_img = process_sequence(frame_paths[-5:])
                    # Save the aligned image
                    os.makedirs(full_output_path, exist_ok=True)
                    cv2.imwrite(f'{full_output_path}/{frame}', avg_img)

    print(
        f"Averaged and aligned plates using {ir_method}, used a window of {MIN_FRAME} frames")


def main():
    args = input_args()
    video_path = (args.video).split('.')[0]
    os.makedirs(output_path, exist_ok=True)
    iterate_plates(video_path, args.ir_method)


if __name__ == "__main__":
    main()
