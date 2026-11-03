"""GFDF guided-filter multi-focus fusion after Qiu et al. (2019).

Reference: X. Qiu, M. Li, L. Zhang, X. Yuan, "Guided filter-based multi-focus
image fusion through focus region detection", Signal Processing: Image
Communication 72:35-46, 2019, doi:10.1016/j.image.2018.12.004. Checked against
the authors' MATLAB code, bitname/Multi-focus-image-fusion-GFDF@40c2f25
(main.m, guidedfilter_LKH.m); for two images paper and code agree.

Paper rules: rough focus map |gray - mean_w(gray)| (Eqs. 1-4), guided-filter
refinement (Eqs. 5-6), initial decision map (Eq. 7), small-region removal
(Sec. 2.3), initial fused image (Eq. 8), guided-filter final decision map
(Eq. 9), and weighted fusion (Eq. 10).
Engineering adaptations: N frames use one joint per-pixel argmax instead of
the paper's pairwise cascade (Sec. 3.4.5), with ties kept by the earlier frame;
small regions join the most frequent label on their outer ring; one weight map
per selected frame is clipped at 0 and renormalized; the two guided-filter
stages have separate parameters; decisions use the shared grayscale copies and
weights are applied to all BGR channels.
"""

import math

import cv2
import numpy as np


# Adjustable defaults. stack() keyword overrides take precedence over these
# values; run_benchmark.py keeps no separate copy.
DEFAULT_PARAMETERS = {
    # Side w of the square mean filter for the rough focus map (Eqs. 1-4), in
    # pixels, with replicated borders. Paper Sec. 3.3 and code: w=7; the paper
    # reports good results for 5-15. Positive odd integer. Larger windows
    # respond to coarser structure and give wider focus bands; smaller ones
    # follow fine texture and noise.
    "mean_window": 7,
    # Guided-filter radius r for refining the focus map (Eqs. 5-6); the window
    # is (2r+1) x (2r+1) pixels. Paper and code: r=5; the paper sees artifacts
    # above 15. Integer >= 1. Larger radii smooth the focus map over wider
    # areas; smaller ones keep it closer to the rough map.
    "focus_radius": 5,
    # Guided-filter regularization eps for the focus map, in variance units of
    # the [0, 1] grayscale (a variance cannot exceed 0.25). Paper and code:
    # eps=0.3; the paper finds it has little effect. Float > 0. Larger values
    # make the filter act more like a plain box mean; smaller ones follow the
    # guide's edges more closely.
    "focus_eps": 0.3,
    # Small-region threshold R as a fraction of H*W (Sec. 2.3); 8-connected
    # regions with fewer than ceil(R*H*W) pixels are relabeled. Paper and code:
    # R=0.01. 0 <= R < 1; 0 disables removal. Larger values remove more
    # isolated decisions but may also erase narrow in-focus bands; smaller ones
    # keep more noise-driven islands.
    "small_region_ratio": 0.01,
    # Guided-filter radius r for the final weight maps (Eq. 9), same meaning
    # and range as focus_radius. The code shares r=5 between both stages.
    # Larger radii widen the blending band at focus boundaries; smaller ones
    # give harder transitions closer to the decision map.
    "weight_radius": 5,
    # Guided-filter eps for the final weight maps, same scale and range as
    # focus_eps. The code shares eps=0.3 between both stages. Smaller values
    # let the transitions follow edges of the initial fused image more closely.
    "weight_eps": 0.3,
}

# Scores are rounded so floating-point noise cannot break exact ties, for
# example between flat regions. This is a numerical rule, not a tuning value.
_SCORE_DECIMALS = 12


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite number greater than 0.")
    return float(value)


def resolve_parameters(shape, **overrides):
    """Merge overrides, validate them, and add the area threshold for this image size."""
    unknown = sorted(set(overrides) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise TypeError(f"Unknown guided filter parameters: {', '.join(unknown)}")
    values = {**DEFAULT_PARAMETERS, **overrides}

    mean_window = _positive_int("mean_window", values["mean_window"])
    if mean_window % 2 == 0:
        raise ValueError("mean_window must be a positive odd integer.")
    ratio = values["small_region_ratio"]
    if (not isinstance(ratio, (int, float)) or isinstance(ratio, bool)
            or not math.isfinite(ratio) or not 0 <= ratio < 1):
        raise ValueError("small_region_ratio must be at least 0 and below 1.")
    height, width = shape[:2]
    if height < 1 or width < 1:
        raise ValueError("The image must not be empty.")
    return {"mean_window": mean_window,
            "focus_radius": _positive_int("focus_radius", values["focus_radius"]),
            "focus_eps": _positive_float("focus_eps", values["focus_eps"]),
            "small_region_ratio": float(ratio),
            "weight_radius": _positive_int("weight_radius", values["weight_radius"]),
            "weight_eps": _positive_float("weight_eps", values["weight_eps"]),
            # Same rounding as the code's ceil(R*H*W).
            "min_area_pixels": math.ceil(ratio * height * width)}


def guided_filter(guide, src, radius, eps):
    """Gray guided filter of He et al., as guidedfilter_LKH (Eqs. 5-6, 9).

    Box means divide by the number of pixels inside the image, so borders need
    no padding values.
    """
    guide = np.asarray(guide, dtype=np.float64)
    src = np.asarray(src, dtype=np.float64)
    size = (2 * radius + 1, 2 * radius + 1)
    counts = cv2.boxFilter(np.ones(guide.shape), -1, size, normalize=False,
                           borderType=cv2.BORDER_CONSTANT)

    def mean(values):
        return cv2.boxFilter(values, -1, size, normalize=False,
                             borderType=cv2.BORDER_CONSTANT) / counts

    mean_guide, mean_src = mean(guide), mean(src)
    a = ((mean(guide * src) - mean_guide * mean_src)
         / (mean(guide * guide) - mean_guide * mean_guide + eps))
    b = mean_src - a * mean_guide
    return mean(a) * guide + mean(b)


def focus_map(gray, parameters):
    """Rough focus map refined by the guided filter (Eqs. 1-6), rounded for ties."""
    gray = gray.astype(np.float64)
    window = parameters["mean_window"]
    rough = np.abs(gray - cv2.blur(gray, (window, window), borderType=cv2.BORDER_REPLICATE))
    refined = guided_filter(gray, rough, parameters["focus_radius"], parameters["focus_eps"])
    return np.round(refined, _SCORE_DECIMALS)


def remove_small_regions(labels, count, min_area):
    """Relabel 8-connected regions below min_area; returns a new label map.

    Labels are handled in frame order 0..count-1 on the updated map. A small
    region takes the most frequent label on its outer 8-neighbor ring, the
    smaller label on ties. For two labels this equals the code's bwareaopen on
    the decision map followed by bwareaopen on its complement.
    """
    labels = labels.copy()
    if min_area <= 1:
        return labels
    height, width = labels.shape
    ring_kernel = np.ones((3, 3), dtype=np.uint8)
    for label in range(count):
        mask = (labels == label).astype(np.uint8)
        components, regions, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for region in range(1, components):
            x, y, w, h, area = stats[region]
            if area >= min_area:
                continue
            # The ring lies within the bounding box grown by one pixel.
            y0, y1 = max(y - 1, 0), min(y + h + 1, height)
            x0, x1 = max(x - 1, 0), min(x + w + 1, width)
            inside = regions[y0:y1, x0:x1] == region
            ring = cv2.dilate(inside.astype(np.uint8), ring_kernel).astype(bool) & ~inside
            neighbors = labels[y0:y1, x0:x1][ring]
            # Only a region covering the whole image has no ring.
            if neighbors.size:
                labels[y0:y1, x0:x1][inside] = np.bincount(neighbors, minlength=count).argmax()
    return labels


def stack(images, *, gray_images=None, **overrides):
    """Fuse aligned uint8/uint16 BGR images without modifying their arrays."""
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
    if gray_images is not None and len(gray_images) != len(images):
        raise ValueError("Each color image must have a corresponding grayscale image.")
    parameters = resolve_parameters(first.shape, **overrides)
    pixel_max = float(np.iinfo(first.dtype).max)

    def gray_of(index):
        gray = (cv2.cvtColor(images[index].astype(np.float32) / pixel_max, cv2.COLOR_BGR2GRAY)
                if gray_images is None else gray_images[index])
        if (gray.shape != first.shape[:2] or gray.dtype != np.float32
                or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1):
            raise ValueError("Grayscale copies must be finite float32 arrays in [0, 1].")
        return gray

    # Pass 1 keeps only the running best score and its frame (Eq. 7 for N frames).
    best_score, labels = None, np.zeros(first.shape[:2], dtype=np.int32)
    for index in range(len(images)):
        score = focus_map(gray_of(index), parameters)
        if index == 0:
            best_score = score
            continue
        # Strict comparison keeps the earlier frame on ties.
        replace = score > best_score
        best_score[replace] = score[replace]
        labels[replace] = index
    del best_score
    labels = remove_small_regions(labels, len(images), parameters["min_area_pixels"])

    # Eq. 8 for N frames: each pixel takes the gray value of its selected frame.
    present = np.flatnonzero(np.bincount(labels.ravel(), minlength=len(images)))
    fused_gray = np.zeros(first.shape[:2], dtype=np.float64)
    for index in present:
        selected = labels == index
        fused_gray[selected] = gray_of(index)[selected]

    # Pass 2 (Eqs. 9-10): one guided-filtered weight map per selected frame.
    # Frames absent from the label map have exactly zero weight and are skipped.
    numerator = np.zeros(first.shape, dtype=np.float64)
    denominator = np.zeros(first.shape[:2], dtype=np.float64)
    for index in present:
        weight = np.maximum(guided_filter(fused_gray, labels == index,
                                          parameters["weight_radius"],
                                          parameters["weight_eps"]), 0)
        numerator += weight[..., None] * images[index]
        denominator += weight
    # The unclipped weights sum to 1, so clipped sums are at least 1 up to rounding.
    if not np.isfinite(denominator).all() or denominator.min() <= 0:
        raise RuntimeError("Fusion weights do not have a finite positive sum.")
    result = numerator / denominator[..., None]
    return np.rint(np.clip(result, 0, pixel_max)).astype(first.dtype)
