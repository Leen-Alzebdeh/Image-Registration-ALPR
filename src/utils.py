import cv2
import pickle
import datetime
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.stats import beta


def input_args():
    parser = argparse.ArgumentParser(
        description='Process some integers.')
    parser.add_argument('--video', type=str,
                        help='input video path. If in eval.py, can pass "all" to evaluate all subdirectories', default='morning-short')
    parser.add_argument('--suffix', type=str,
                        help='folder suffix', default='')
    parser.add_argument('--ir', type=str,
                        help='image registration method', default='orb')
    parser.add_argument('--title', type=str,
                        help='trial title', default='Trial')
    parser.add_argument('--start', type=int,
                        help='start time to extract frames (in mins)', default=0)
    parser.add_argument('--end', type=int,
                        help='end time to extract frames (in mins)', default=None)
    parser.add_argument('--track', action='store_true',
                        help='whether to track only (true), or detect and track (false). Default is false.')
    return parser.parse_args()


def bboxe_save(bboxes, suffix):
    with open('bboxes' + suffix + '.pickle', 'wb') as f:
        pickle.dump(bboxes, f, pickle.HIGHEST_PROTOCOL)


def bboxes_load(suffix):
    with open('bboxes' + suffix + '.pickle', 'rb') as f:
        return pickle.load(f)


def crop_to_fixed_size(img, box, fixed_size=(128, 64)):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    crop = img[y1:y2, x1:x2]

    # Resize to fixed size (distorts aspect ratio)
    resized = cv2.resize(crop, fixed_size)
    return resized


def iou_distance(detection_a, tracked_object_b):
    """
    detection_a.points has shape (1,4), detection_b.points has shape (1,4)
    each is the bounding box [x1, y1, x2, y2].
    """
    boxA = detection_a.points[0]  # shape: (4,)
    boxB = tracked_object_b.estimate[0]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-10)

    # Return 1 - iou so smaller is "closer" (Norfair expects smaller => closer)
    return 1.0 - iou


def combined_distance(det_a, tracked_object_b):
    iou = iou_distance(det_a, tracked_object_b)
    app = appearance_distance(det_a, tracked_object_b)
    return 0.7 * iou + 0.3 * app


def appearance_distance(det_a, tracked_object_b):
    img_a = crop_image_from_detection(det_a)
    img_b = crop_image_from_detection(tracked_object_b)
    hist_a = cv2.calcHist([img_a], [0], None, [16], [0, 256])
    hist_b = cv2.calcHist([img_b], [0], None, [16], [0, 256])
    return 1 - cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)


def crop_image_from_detection(det):
    """
    Crop the region in the original image corresponding to the detection's bounding box.
    Compatible with both Detection and TrackedObject.
    """
    print(det)
    if hasattr(det, "last_detection"):
        det = det.last_detection  # If it's a TrackedObject, get the last detection

    if not hasattr(det, 'data') or 'origin' not in det.data:
        raise ValueError(
            "Detection does not contain the original image ('origin').")

    box = det.points[0]  # shape (4,) -> [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)
    return det.data["origin"][y1:y2, x1:x2]


def align_image_orb_ransac(ref, moving):
    """
    Align 'moving' image to 'ref' image using ORB + RANSAC homography.
    Both 'ref' and 'moving' should be grayscale or BGR images of the same size.
    Returns the warped 'moving' image aligned to 'ref'.
    """
    orb = cv2.ORB_create(nfeatures=2000,
                         scaleFactor=1.2,
                         nlevels=8,
                         edgeThreshold=15,
                         patchSize=31,
                         fastThreshold=5)  # Adjust as needed
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 1) Convert to grayscale if needed
    if len(ref.shape) == 3:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref

    if len(moving.shape) == 3:
        mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    else:
        mov_gray = moving

    # 2) Detect keypoints + descriptors
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
    kp_mov, des_mov = orb.detectAndCompute(mov_gray, None)

    if des_ref is None or des_mov is None:
        print("[WARN] No descriptors found; returning original moving image.")
        return moving

    # 3) Match descriptors
    matches = bf.match(des_ref, des_mov)
    if len(matches) < 4:
        print("[WARN] Not enough matches; returning original moving image.")
        return moving

    # 4) Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # 5) Extract matched keypoints
    src_pts = np.float32(
        [kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_mov[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 6) Compute Homography via RANSAC
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("[WARN] Homography not found; returning original moving image.")
        return moving

    # 7) Warp the 'moving' image to 'ref'
    aligned = cv2.warpPerspective(moving, H, (ref.shape[1], ref.shape[0]))
    return aligned


def align_image_optical_flow(ref, moving):
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(mov_gray, ref_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    flow_map = np.column_stack(
        (np.tile(np.arange(w), h), np.repeat(np.arange(h), w)))
    flow_map = flow_map.reshape(h, w, 2).astype(np.float32)
    remap = flow_map + flow
    aligned = cv2.remap(moving, remap[..., 0], remap[..., 1], cv2.INTER_LINEAR)
    return aligned


def align_image_diffeo(ref, moving):
    ref_sitk = sitk.GetImageFromArray(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
    mov_sitk = sitk.GetImageFromArray(cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY))
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(ref_sitk)
    elastix.SetMovingImage(mov_sitk)
    elastix.SetParameterMap(sitk.GetDefaultParameterMap("bspline"))
    elastix.Execute()
    result = elastix.GetResultImage()
    return sitk.GetArrayFromImage(result)


def average_aligned_frames(aligned_frames):
    """
    aligned_frames: list of images [H,W] or [H,W,3], all same size
    returns: average image
    """
    # Convert list of images to float32 for numeric stability
    stack = np.stack([img.astype(np.float32)
                     for img in aligned_frames], axis=0)
    avg_img = np.mean(stack, axis=0)  # average across frames dimension
    # Convert back to uint8
    avg_img = avg_img.round().astype(np.uint8)
    return avg_img


def bayesian_confidence_interval(successes, total, alpha_prior=1, beta_prior=1, cred=0.95):
    """
    Computes a Bayesian confidence interval (credible interval) for OCR accuracy.
    """
    posterior = beta(successes + alpha_prior, total - successes + beta_prior)
    lower = posterior.ppf((1 - cred) / 2)
    upper = posterior.ppf(1 - (1 - cred) / 2)
    mean = posterior.mean()
    return lower, upper, mean


def binary_success_rate(conf_list, threshold=70):
    successes = sum(1 for c in conf_list if c >= threshold)
    return bayesian_confidence_interval(successes, len(conf_list))


def log_results(trial_name, video_path, pre_errors, pre_conf_list, post_errors, post_conf_list, total_chars, total_plates):
    pre_char_acc = pre_errors[0]/total_chars
    pre_exact_acc = pre_errors[1]/total_plates
    post_char_acc = post_errors[0]/total_chars
    post_exact_acc = post_errors[1]/total_plates
    pre_avg_conf = np.mean(pre_conf_list)
    post_avg_conf = np.mean(post_conf_list)
    pre_bayes = binary_success_rate(pre_conf_list)
    post_bayes = binary_success_rate(post_conf_list)

    date = datetime.datetime.now()

    print(f"Pre-Image Registration Results\n")
    print(f"\tAccuracy: {pre_exact_acc:.2%}, \n")
    print(f"\tExact Match: {pre_exact_acc:.2%}\n")
    print(f"\tAverage OCR confidence: {pre_avg_conf:.2%}\n")
    print(f"\tBayesian CI: {pre_bayes}\n\n")

    print(f"Post-Image Registration Results\n")
    print(f"\tAccuracy: {post_char_acc:.2%}, \n")
    print(f"\tExact Match: {post_exact_acc:.2%}\n")
    print(f"\tAverage OCR confidence: {post_avg_conf:.2f}")
    print(f"\tBayesian CI: {post_bayes}\n\n")

    with open("results.md", "a") as f:
        f.write(
            f"| {trial_name} | {video_path} | {pre_char_acc:.2%} | {pre_exact_acc:.2%} | {pre_avg_conf:.2%} | {pre_bayes} \
                | {post_char_acc:.2%} | {post_exact_acc:.2%} | {post_avg_conf:.2%} | {post_bayes} | {date} |\n")


def log_plate_results(plate_order, plate_number, pre_ocr, pre_errors,  post_ocr, post_errors, total_chars):
    pre_char_acc = pre_errors[0]/total_chars
    pre_exact_acc = pre_errors[1]
    post_char_acc = post_errors[0]/total_chars
    post_exact_acc = post_errors[1]

    with open("plate_results.md", "a") as f:
        f.write(
            f"| {plate_order} | {plate_number} | {pre_ocr} | {pre_char_acc:.2%} | {pre_exact_acc:.2%} | {post_ocr} | {post_char_acc:.2%} | {post_exact_acc:.2%} |\n")
