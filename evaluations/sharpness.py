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


def laplacian_variance(image, *, roi=None):
    """Return population variance of signed Laplacian responses (ksize=3)."""
    gray, region = _prepare(image, roi)
    response = cv2.Laplacian(gray, cv2.CV_32F, ksize=3,
                             borderType=cv2.BORDER_REFLECT_101)
    return float(np.var(response[region], dtype=np.float64, ddof=0))


def evaluate(image, *, roi=None):
    """Return separate scores; do not combine them or evaluate input sequences."""
    return {"tenengrad": tenengrad(image, roi=roi),
            "laplacian_variance": laplacian_variance(image, roi=roi)}
