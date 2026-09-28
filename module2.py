import numpy as np
import cv2
from pathlib import Path
 
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_FILE_SIZE_MB = 25
STANDARD_SIZE = (512, 512)  # (width, height) - all downstream modules expect this
 
 
# ---------- 1. Validation ----------
 
def validate_image(path):
    """
    Checks the file exists, has a valid extension, isn't too large,
    and can actually be opened as an image. Raises ValueError if invalid.
    """
    p = Path(path)
 
    if not p.exists():
        raise ValueError(f"File not found: {path}")
 
    if p.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {p.suffix}")
 
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
 
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Could not read image (corrupt or unsupported): {path}")
 
    return img
 
 
# ---------- 2. Resize ----------
 
def resize_standard(img, size=STANDARD_SIZE):
    """Resizes image to a standard size so all downstream modules get consistent input."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
 
 
# ---------- 3. Preprocessing ----------
 
def denoise(img):
    """Removes sensor noise while preserving edges."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                            templateWindowSize=7, searchWindowSize=21)
 
 
def enhance_contrast(img):
    """
    Applies CLAHE (adaptive histogram equalization) on the luminance channel.
    Satellite images are often dull/hazy - this makes features more visible.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
 
    merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
 
 
def normalize_pixels(img):
    """Scales pixel values to use the full 0-255 range."""
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)
 
 
# ---------- 4. Main entry point -> Output to M3/M4/M5/M6 ----------
 
def process_image(input_path, out_dir="m2_outputs"):
    """
    Main function M1 calls after a user uploads an image.
 
    Returns a dict forwarded to M3 (and eventually M4/M5/M6):
        {
            'processed_image_path': str,
            'width': int,
            'height': int,
            'status': str
        }
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
 
    img = validate_image(input_path)
    img = resize_standard(img)
    img = denoise(img)
    img = enhance_contrast(img)
    img = normalize_pixels(img)
 
    out_path = f"{out_dir}/processed.png"
    cv2.imwrite(out_path, img)
 
    return {
        "processed_image_path": out_path,
        "width": img.shape[1],
        "height": img.shape[0],
        "status": "success"
    }
 
 
if __name__ == "__main__":
    # Example usage - replace with an actual uploaded file path
    result = process_image("optical_t1.jpg")
    print(result)
 


