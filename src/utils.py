import cv2
import numpy as np
import datetime
import cv2


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
    print(detection_a.points)
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


def log_results(pre_errors, post_errors, total_chars, total_plates):
    pre_char_acc = pre_errors[0]/total_chars
    pre_exact_acc = pre_errors[1]/total_plates
    post_char_acc = post_errors[0]/total_chars
    post_exact_acc = post_errors[1]/total_plates
    date = datetime.datetime.now()

    print(f"Pre-Image Registration Results\n")
    print(f"\tAccuracy: {pre_exact_acc:.2%}, \n")
    print(f"\tExact Match: {pre_exact_acc:.2%}\n")
    print(f"Post-Image Registration Results\n")
    print(f"\tAccuracy: {post_char_acc:.2%}, \n")
    print(f"\tExact Match: {post_exact_acc:.2%}\n")

    with open("results.md", "a") as f:
        f.write(
            f"| {pre_char_acc:.2%} | {pre_exact_acc:.2%} | {post_char_acc:.2%} | {post_exact_acc:.2%} | {date} |\n")


def log_plate_results(plate_order, plate_number, pre_ocr, pre_errors,  post_ocr, post_errors, total_chars):
    pre_char_acc = pre_errors[0]/total_chars
    pre_exact_acc = pre_errors[1]
    post_char_acc = post_errors[0]/total_chars
    post_exact_acc = post_errors[1]

    print(pre_errors[0], total_chars)
    print(post_errors[0], total_chars)
    with open("plate_results.md", "a") as f:
        f.write(
            f"| {plate_order} | {plate_number} | {pre_ocr} | {pre_char_acc:.2%} | {pre_exact_acc:.2%} | {post_ocr} | {post_char_acc:.2%} | {post_exact_acc:.2%} |\n")
