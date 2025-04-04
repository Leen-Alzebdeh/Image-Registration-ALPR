import pytesseract
from difflib import SequenceMatcher
import cv2
import os
import numpy as np
from utils import log_results, log_plate_results

# Look at another video stabilization metric
# Look at a diffeomorphic metric for image registration
# Look at diffeomorphic method for image registration?
# Try and make it real time

# Report:
# Intro, lit survery, method, results (what i learned subsection), conclusion. 10 page limit

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


def evaluate_performance(pre_path, post_path):
    total_chars = 0
    total_plates = 0
    pre_sum = np.array([0, 0])
    post_sum = np.array([0, 0])
    for plate_path in os.listdir(pre_path):
        plate_path = 'plate1_GX922V'
        print(plate_path)
        pre_plate_path = os.listdir(pre_path+'/'+plate_path)
        post_plate_path = os.listdir(post_path+'/'+plate_path)
        gt = plate_path.split('_')[1]
        print(gt)
        plate_order = plate_path.split('_')[0][-1]
        for pre_frame_path, post_frame_path in zip(pre_plate_path, post_plate_path):
            full_pre_frame_path = pre_path+'/'+plate_path+'/'+pre_frame_path
            full_post_frame_path = post_path+'/'+plate_path+'/'+post_frame_path
            pre_ocr = run_ocr(full_pre_frame_path)
            post_ocr = run_ocr(full_post_frame_path)
            print(pre_ocr, post_ocr)
            # Calculate character accuracy
            pre_errors = character_accuracy(pre_ocr, gt)
            post_errors = character_accuracy(post_ocr, gt)
            pre_sum += pre_errors
            post_sum += post_errors
            log_plate_results(plate_order, gt, pre_ocr,
                              pre_errors,  post_ocr, post_errors, len(gt))
            total_chars += len(gt)
            total_plates += 1
        break
    print(pre_sum, post_sum)
    print(total_chars, total_plates)
    log_results(pre_sum, post_sum, total_chars, total_plates)


def character_accuracy(pred, truth):
    matches = SequenceMatcher(None, pred, truth).get_matching_blocks()
    exact_match = 1 if pred == truth else 0
    correct = sum(match.size for match in matches)
    return correct, exact_match


def run_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return ""

    # Resize image to improve OCR performance
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise (optional)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Adaptive Thresholding (better for varied lighting)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)

    # OCR
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)

    # Clean OCR result
    return text.strip().replace(" ", "").replace("\n", "")


def main():
    evaluate_performance('filtered_plates', 'aligned_plates')


main()
