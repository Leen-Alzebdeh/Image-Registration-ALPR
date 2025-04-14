import cv2
import os
import pytesseract
import numpy as np
from difflib import SequenceMatcher
from utils import log_results, log_plate_results, input_args

# Look at another video stabilization metric
# Look at a diffeomorphic metric for image registration
# Look at diffeomorphic method for image registration?
# Try and make it real time

# Report:
# Intro, lit survery, method, results (what i learned subsection), conclusion. 10 page limit

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# TODO:
# 1. Get more data for training (more plates), 50 total, at least 6 per plate, 2 conditions (morning, evening/night)
# 2. Aquire a test set: two short videos with both conditions (morning, evening/night)
# 3. How to get confidence intervals for the OCR results [DONE]
# 4. Implement intensity-based IR: optical flow [DONE]
# 5. Try a diffeomorphic method for image registration [DONE]
# 6. Write report
# 7. Make presentation

pre_dir = 'filtered_plates'
post_dir = 'aligned_plates'


def evaluate_performance(video_path, trial_name):
    total_chars = 0
    total_plates = 0
    pre_sum = np.array([0, 0])
    post_sum = np.array([0, 0])
    pre_conf_all = []
    post_conf_all = []

    if video_path == "all":
        video_dirs = os.listdir(post_dir)
    else:
        video_dirs = [video_path]

    for video_subdir in video_dirs:
        pre_folder = os.path.join(pre_dir, video_subdir)
        post_folder = os.path.join(post_dir, video_subdir)

        if not os.path.exists(pre_folder) or not os.path.exists(post_folder):
            print(f"[WARNING] Skipping {video_subdir} — folder missing.")
            continue

        print(f"Evaluating filtered plates from {pre_folder}")
        print(f"Evaluating aligned plates from {post_folder}")

        for plate_path in os.listdir(post_folder):
            print(plate_path)
            pre_plate_path = pre_folder+'/'+plate_path
            post_plate_path = post_folder+'/'+plate_path
            pre_plate_dir = os.listdir(pre_plate_path)[
                4:]      # skip first 4 frames in filtered folder
            post_plate_dir = os.listdir(post_plate_path)
            assert len(pre_plate_dir) == len(
                post_plate_dir), f"Mismatch in number of frames for {plate_path}"
            gt = plate_path.split('_')[1]  # ground truth
            plate_order = plate_path.split('_')[0][-3:]
            for pre_frame, post_frame in zip(pre_plate_dir, post_plate_dir):
                assert pre_frame == post_plate_dir, f"Mismatch in frame names for {plate_path}"

                # Run OCR on both frames
                pre_ocr, pre_conf = run_ocr(f'{pre_plate_path}/{pre_frame}')
                post_ocr, post_conf = run_ocr(
                    f'{post_plate_path}/{post_frame}')
                # Calculate character accuracy
                pre_errors = character_accuracy(pre_ocr, gt)
                post_errors = character_accuracy(post_ocr, gt)

                pre_sum += pre_errors
                post_sum += post_errors
                pre_conf_all.extend(pre_conf)
                post_conf_all.extend(post_conf)

                # Log results
                log_plate_results(plate_order, gt, pre_ocr,
                                  pre_errors,  post_ocr, post_errors, len(gt))
                total_chars += len(gt)
                total_plates += 1
    log_results(trial_name, video_path, pre_sum, pre_conf_all, post_sum,
                post_conf_all, total_chars, total_plates)


def character_accuracy(pred, truth):
    matches = SequenceMatcher(None, pred, truth).get_matching_blocks()
    exact_match = 1 if pred == truth else 0
    correct = sum(match.size for match in matches)
    return correct, exact_match


def run_ocr(image_path):
    img = cv2.imread(image_path)
    # Denoise
    denoise = cv2.bilateralFilter(img, 11, 17, 17)
    # threshold
    thresh = cv2.adaptiveThreshold(denoise, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)

    # Try:
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return ""

    # make o and 0 interchangable
    # OCR
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    data = pytesseract.image_to_string(
        thresh, config=config, output_type=pytesseract.Output.DICT)
    confidences = [int(conf) for conf in data['conf'] if conf != '-1']
    text = "".join(data['text']).strip().replace(" ", "").replace("\n", "")

    # Clean OCR result
    return text, confidences


def main():
    args = input_args()
    video_path = (args.video).split('.')[0]
    evaluate_performance(video_path, args.trial)


main()
