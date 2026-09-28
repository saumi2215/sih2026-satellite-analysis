import numpy as np
import cv2

# Fake optical image (color)
optical = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
cv2.imwrite("optical_t1.jpg", optical)

# Fake SAR image (grayscale)
sar = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
cv2.imwrite("sar_t1.jpg", sar)

# Fake second time-period optical image
optical_t2 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
cv2.imwrite("optical_t2.jpg", optical_t2)

print("Test images created!")