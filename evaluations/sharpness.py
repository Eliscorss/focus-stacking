"""Independent sharpness metrics for a fused image, without dataset traversal."""

import cv2
import numpy as np


def _prepare(image, roi=None):
    """Normalize by dtype range and exclude the derivative kernel's outer border."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("A nonempty image array is required.")
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError("Expected a grayscale or three-channel BGR image.")
    if image.dtype in (np.uint8, np.uint16):
        normalized = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    elif np.issubdtype(image.dtype, np.floating):
        normalized = image.astype(np.float32)
    else:
        raise ValueError("Expected uint8, uint16, or normalized floating-point pixels.")
    if not np.isfinite(normalized).all() or normalized.min() < 0 or normalized.max() > 1:
        raise ValueError("Floating-point grayscale values must be finite and in [0, 1].")
    # Convert after normalization to avoid bit-depth-dependent grayscale rounding.
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else normalized
    height, width = gray.shape
    if min(height, width) < 3:
        raise ValueError("Evaluation requires an image of at least 3 x 3 pixels.")
    if roi is None:
        x, y, w, h = 0, 0, width, height
    else:
        if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) for v in roi):
            raise ValueError("ROI must contain four integers: x y width height.")
        x, y, w, h = roi
        if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
            raise ValueError("ROI must lie entirely inside the cropped fused image.")
    region = (slice(max(y, 1), min(y + h, height - 1)),
              slice(max(x, 1), min(x + w, width - 1)))
    if gray[region].size == 0:
        raise ValueError("ROI contains no pixels after excluding the image border.")
    return gray, region


def tenengrad(image, *, roi=None):
    """Return mean squared Sobel gradient magnitude, without thresholding."""
    gray, region = _prepare(image, roi)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3,
                   borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3,
                   borderType=cv2.BORDER_REFLECT_101)
    return float(np.mean(gx[region] ** 2 + gy[region] ** 2, dtype=np.float64))


def average_gradient(image, *, roi=None):
    """Return the average gradient (AG) of Kumar, SIViP 7:1125-1143 (2013), Sec. 3.3.

    AG = sum_ij sqrt((f(i,j) - f(i+1,j))^2 + (f(i,j) - f(i,j+1))^2) / mn, with
    forward differences and no division by 2 inside the root, as in the paper.
    The paper uses 8-bit gray values; here f is the normalized [0, 1] grayscale
    image, so values are the paper's scale divided by 255. The paper does not
    state how the last row and column are handled; the mean is taken over the
    same region as Tenengrad, which excludes the outer border, so every
    evaluated pixel has both forward neighbours.
    """
    gray, region = _prepare(image, roi)
    # Index [i, j] of both difference arrays refers to pixel (i, j) of the image.
    down = gray[:-1, :-1] - gray[1:, :-1]
    right = gray[:-1, :-1] - gray[:-1, 1:]
    magnitude = np.sqrt(down ** 2 + right ** 2)
    # The region ends at height - 1 and width - 1, so it fits the smaller arrays.
    return float(np.mean(magnitude[region], dtype=np.float64))


def evaluate(image, *, roi=None):
    """Return separate sharpness scores; do not combine them or evaluate input sequences."""
    return {"tenengrad": tenengrad(image, roi=roi),
            "average_gradient": average_gradient(image, roi=roi)}
