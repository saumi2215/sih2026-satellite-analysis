import numpy as np
import cv2
from pathlib import Path
def load_image(path, grayscale=False):
    """Load an image from disk (optical = color, SAR = usually grayscale)."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img
def denoise_sar(sar_img, kernel_size=5):
    """
    SAR images have 'speckle noise' due to radar signal interference.
    Median blur is a simple, fast approximation of speckle filtering.
    For production, replace with a proper Lee / Frost filter.
    """
    return cv2.medianBlur(sar_img, kernel_size)


def normalize(img):
    """Scale pixel values to 0-1 range."""
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def resize_to_match(img, target_shape):
    """Resize img (h, w) to match target_shape (h, w)."""
    return cv2.resize(img, (target_shape[1], target_shape[0]))
def coregister(base_gray, moving_img):
    """
    Aligns 'moving_img' onto 'base_gray' using ORB feature matching + Homography.
    Optical and SAR come from different sensors, so pixels don't line up by default.
    Falls back to the original image if not enough matching features are found.
    """
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(base_gray, None)
    kp2, des2 = orb.detectAndCompute(moving_img, None)

    if des1 is None or des2 is None:
        return resize_to_match(moving_img, base_gray.shape)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:50]

    if len(matches) < 4:
        return resize_to_match(moving_img, base_gray.shape)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return resize_to_match(moving_img, base_gray.shape)

    aligned = cv2.warpPerspective(moving_img, H, (base_gray.shape[1], base_gray.shape[0]))
    return aligned
def fuse_optical_sar(optical_img, sar_img_aligned, optical_weight=0.6, sar_weight=0.4):
    """
    Combines optical + SAR into a single 'fused' image.
    SAR carries texture/moisture info that optical alone misses (works day/night, through clouds).
    """
    gray_opt = cv2.cvtColor(optical_img, cv2.COLOR_BGR2GRAY) if len(optical_img.shape) == 3 else optical_img
    sar_denoised = denoise_sar(sar_img_aligned)

    opt_n = (normalize(gray_opt) * 255).astype(np.uint8)
    sar_n = (normalize(sar_denoised) * 255).astype(np.uint8)

    fused = cv2.addWeighted(opt_n, optical_weight, sar_n, sar_weight, 0)
    return fused
def detect_change(img_t1, img_t2, threshold=30):
    """
    Compares two fused images from different time periods to detect changes
    (e.g. flooding, deforestation, construction, disaster damage).
    """
    if img_t1.shape != img_t2.shape:
        img_t2 = resize_to_match(img_t2, img_t1.shape)

    diff = cv2.absdiff(img_t1, img_t2)
    _, change_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    change_percent = (np.count_nonzero(change_mask) / change_mask.size) * 100
    return change_mask, round(change_percent, 2)
def analyze_optical_sar(optical_path, sar_path, optical_t2_path=None, sar_t2_path=None,
                         out_dir="m6_outputs"):
    """
    Main function M3 (AI Agent) will call when it routes a task to Module 6.

    Returns a dict that gets forwarded to Module 7 (Integration):
        {
            'fused_image_path': str,
            'change_mask_path': str or None,
            'change_percent': float or None,
            'summary': str,
            'confidence': float
        }
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result = {}

    optical = load_image(optical_path)
    sar = load_image(sar_path, grayscale=True)
    base_gray = cv2.cvtColor(optical, cv2.COLOR_BGR2GRAY)
    sar_aligned = coregister(base_gray, sar)

    fused = fuse_optical_sar(optical, sar_aligned)
    fused_path = f"{out_dir}/fused_t1.png"
    cv2.imwrite(fused_path, fused)
    result['fused_image_path'] = fused_path

    if optical_t2_path and sar_t2_path:
        optical_t2 = load_image(optical_t2_path)
        sar_t2 = load_image(sar_t2_path, grayscale=True)
        base_gray_t2 = cv2.cvtColor(optical_t2, cv2.COLOR_BGR2GRAY)
        sar_t2_aligned = coregister(base_gray_t2, sar_t2)
        fused_t2 = fuse_optical_sar(optical_t2, sar_t2_aligned)

        change_mask, change_pct = detect_change(fused, fused_t2)
        change_path = f"{out_dir}/change_mask.png"
        cv2.imwrite(change_path, change_mask)

        result['change_mask_path'] = change_path
        result['change_percent'] = change_pct
        result['summary'] = f"{change_pct}% of the area shows change between the two time periods."
    else:
        result['change_mask_path'] = None
        result['change_percent'] = None
        result['summary'] = "Single time-step optical-SAR fusion completed (no change detection)."

    result['confidence'] = 0.85  # placeholder; replace with a real model-based score if available
    return result


if __name__ == "__main__":
    # Example usage — replace with your actual file paths
    output = analyze_optical_sar(
        optical_path="optical_t1.jpg",
        sar_path="sar_t1.jpg",
        optical_t2_path="optical_t2.jpg",   # optional
        sar_t2_path="sar_t2.jpg"            # optional
    )
    print(output)