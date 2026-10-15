"""Local Laplacian focus selection with label smoothing and color reconstruction."""

import cv2
import numpy as np


def smooth_source_map(source_map, image_count, size=5):
    """Apply one majority-vote pass; keep the original label when votes tie."""
    if size == 1:
        return source_map.copy()
    result = np.zeros_like(source_map)
    best_votes = np.full(source_map.shape, -1, dtype=np.float32)
    for index in range(image_count):
        votes = cv2.boxFilter(
            (source_map == index).astype(np.float32), -1, (size, size),
            normalize=False, borderType=cv2.BORDER_REPLICATE,
        )
        replace = (votes > best_votes) | (
            (votes == best_votes) & (source_map == index)
        )
        result[replace] = index
        best_votes[replace] = votes[replace]
    return result


def stack(images, *, gray_images=None, window_size=21, smooth_size=5):
    """Fuse aligned uint8/uint16 BGR images without modifying their arrays."""
    for name, size in (("window_size", window_size), ("smooth_size", smooth_size)):
        if not isinstance(size, int) or size < 1 or size % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer.")
    if len(images) == 0:
        raise ValueError("At least one color image is required.")
    first = images[0]
    if not isinstance(first, np.ndarray) or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Images must be three-channel BGR arrays.")
    if first.dtype not in (np.uint8, np.uint16):
        raise ValueError("Color images must use uint8 or uint16 pixels.")
    for image in images:
        if image.shape != first.shape or image.dtype != first.dtype:
            raise ValueError("All color images must have the same shape and dtype.")
    if min(first.shape[:2]) < window_size:
        raise ValueError("The image must be at least as large as the focus window.")
    if gray_images is not None and len(gray_images) != len(images):
        raise ValueError("Each color image must have a corresponding grayscale image.")

    # Keep only the best score and label instead of allocating an N x H x W volume.
    best_score = np.full(first.shape[:2], -np.inf, dtype=np.float32)
    source_map = np.zeros(first.shape[:2], dtype=np.int32)
    pixel_max = float(np.iinfo(first.dtype).max)
    for index, image in enumerate(images):
        gray = (cv2.cvtColor(image.astype(np.float32) / pixel_max, cv2.COLOR_BGR2GRAY)
                if gray_images is None else gray_images[index])
        if (gray.shape != first.shape[:2] or gray.dtype != np.float32
                or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1):
            raise ValueError("Grayscale copies must be finite float32 arrays in [0, 1].")
        response = cv2.Laplacian(gray, cv2.CV_32F, ksize=3,
                                 borderType=cv2.BORDER_REFLECT_101)
        score = cv2.boxFilter(np.abs(response), -1, (window_size, window_size),
                              normalize=True, borderType=cv2.BORDER_REFLECT_101)
        replace = score > best_score
        best_score[replace] = score[replace]
        source_map[replace] = index

    source_map = smooth_source_map(source_map, len(images), smooth_size)
    fused = np.empty_like(first)
    for index, image in enumerate(images):
        mask = source_map == index
        fused[mask] = image[mask]
    return fused
