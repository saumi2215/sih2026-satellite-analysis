import numpy as np
import cv2
from pathlib import Path
 
 
# ---------- 1. Load / basic I/O ----------
 
def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img
 
 
def resize_to_match(img, target_shape):
    return cv2.resize(img, (target_shape[1], target_shape[0]))
 
 
# ---------- 2. Preprocessing ----------
 
def preprocess(img):
    """Convert to grayscale and blur to reduce noise before comparison."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred
 
 
# ---------- 3. Change Detection ----------
 
def detect_change_regions(img_t1, img_t2, min_area=200, threshold=30):
    """
    Compares two images and returns bounding boxes of regions that changed.
    min_area filters out tiny noise contours.
    """
    gray1 = preprocess(img_t1)
    gray2 = preprocess(img_t2)
 
    if gray1.shape != gray2.shape:
        gray2 = resize_to_match(gray2, gray1.shape)
 
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)  # fill small gaps
 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
 
    return boxes, thresh
 
 
# ---------- 4. Grounding (basic object/blob detection in a single image) ----------
 
def detect_objects_basic(img, min_area=300):
    """
    Very basic 'grounding' placeholder: finds stand-out blobs/edges in a single image
    using edge detection + contours. For production accuracy, replace with a
    pretrained detector (e.g. YOLO) here.
    """
    gray = preprocess(img)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
 
    return boxes
 
 
# ---------- 5. Highlighting ----------
 
def draw_boxes(img, boxes, color=(0, 0, 255), label="Change"):
    """Draws bounding boxes with labels on a copy of the image."""
    out = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f"{label} {i+1}", (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
 
 
# ---------- 6. Main entry point -> Output to Module 7 ----------
 
def run_grounding_change_detection(task, image_path, image_t2_path=None, out_dir="m5_outputs"):
    """
    Main function M3 (AI Agent) calls when it routes a task to Module 5.
 
    task: string like "detect changes" or "find objects" (used to decide which mode)
 
    Returns a dict forwarded to Module 7:
        {
            'highlighted_image_path': str,
            'detected_regions': list of (x, y, w, h),
            'num_detections': int,
            'summary': str,
            'confidence': float
        }
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result = {}
 
    img1 = load_image(image_path)
 
    if image_t2_path:
        # Change detection mode (two images given)
        img2 = load_image(image_t2_path)
        boxes, _ = detect_change_regions(img1, img2)
        highlighted = draw_boxes(img1, boxes, color=(0, 0, 255), label="Change")
        summary = f"{len(boxes)} changed region(s) detected between the two images."
    else:
        # Grounding mode (single image, find objects)
        boxes = detect_objects_basic(img1)
        highlighted = draw_boxes(img1, boxes, color=(0, 255, 0), label="Object")
        summary = f"{len(boxes)} object/region(s) detected in the image."
 
    out_path = f"{out_dir}/highlighted.png"
    cv2.imwrite(out_path, highlighted)
 
    result['highlighted_image_path'] = out_path
    result['detected_regions'] = boxes
    result['num_detections'] = len(boxes)
    result['summary'] = summary
    result['confidence'] = 0.8  # placeholder; replace with model-based score if available
 
    return result
 
 
if __name__ == "__main__":
    # Example usage - change detection (two images)
    output = run_grounding_change_detection(
        task="detect changes",
        image_path="optical_t1.jpg",
        image_t2_path="optical_t2.jpg"   # omit this arg for single-image grounding mode
    )
    print(output)