"""Laplacian pyramid multi-focus fusion after Wang and Chang (2011).

Reference: W. Wang, F. Chang, "A Multi-focus Image Fusion Method Based on
Laplacian Pyramid", Journal of Computers 6(12):2559-2566, 2011,
doi:10.4304/jcp.6.12.2559-2566.

Paper rules: 5x5 binomial Reduce/Expand (Eqs. 1-9), maximum weighted region
energy on detail levels (Eqs. 13-14), and the deviation-and-entropy rule on the
top level (Eqs. 10-12).
Engineering adaptations: decisions use the shared grayscale copies and are
applied to all BGR channels; the two-image rules are extended to N frames by
argmax with ties kept by the earlier frame; levels are capped by image size.
"""

import math

import cv2
import numpy as np


# Adjustable defaults. stack() keyword overrides take precedence over these
# values; run_benchmark.py keeps no separate copy.
DEFAULT_PARAMETERS = {
    # Requested number of Reduce steps N; the pyramid has N detail levels plus
    # one top level. Paper: unspecified (Fig. 2 illustrates three levels on
    # 256x256 images). Integer >= 0; capped so the top level keeps at least
    # min_top_size pixels per side. More levels handle wider defocus blur, but
    # top-level averaging then covers larger image areas.
    "levels": 5,
    # Smallest allowed top-level side in pixels. Engineering guard, integer
    # >= 1 and at least as large as both windows below. Larger values reduce
    # the effective number of levels on small inputs.
    "min_top_size": 8,
    # Weight matrix omega for region energy on detail levels (Eq. 13), applied
    # to squared grayscale coefficients and normalized to sum 1 before use.
    # Paper: unspecified; Fig. 3 shows a 3x3 region. 5x5 binomial chosen from
    # the energy_size 1/3/5/7 sweep on Kro_DEMO_10 view1-7. Its normalized
    # values happen to equal the 2-D generating kernel, but the paper does not
    # prescribe that for omega. Odd square, finite, non-negative,
    # positive sum. Larger windows give steadier decisions and less background
    # noise but wider transition bands and halos; smaller ones follow noise more.
    # binomial_weights(n) and box_weights(n) below build n x n matrices.
    "energy_weights": ((1, 4, 6, 4, 1), (4, 16, 24, 16, 4), (6, 24, 36, 24, 6),
                       (4, 16, 24, 16, 4), (1, 4, 6, 4, 1)),
    # Odd square window J x K, in top-level pixels, for local deviation and
    # entropy (Eqs. 10-11). Paper: unspecified; a top_window 3/5/7 sweep on
    # Kro_DEMO_10 showed almost no change in scores or images, so 3 is kept.
    # A 3x3 window holds only nine
    # samples, so entropy often ties and the rule falls back to averaging;
    # larger windows give steadier statistics with coarser decisions.
    "top_window": 3,
    # Number of gray levels L for local entropy on the [0, 1] grayscale top
    # level (Eq. 11). Integer >= 2; 256 matches 8-bit gray levels. A 16/64/256
    # sweep on Kro_DEMO_10 showed almost no difference, so 256 is kept. Fewer levels
    # ignore small brightness differences and produce more ties.
    "entropy_bins": 256,
}

# Scores are rounded so floating-point noise cannot break exact ties, for
# example between flat regions. This is a numerical rule, not a tuning value.
_SCORE_DECIMALS = 12


def _odd_window(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer.")
    return value


def binomial_weights(size):
    """Odd size x size binomial matrix; size 5 gives the default [1 4 6 4 1] outer product."""
    _odd_window("Energy window size", size)
    row = [math.comb(size - 1, k) for k in range(size)]
    return tuple(tuple(a * b for b in row) for a in row)


def box_weights(size):
    """Odd size x size matrix of ones, i.e. an unweighted region sum."""
    _odd_window("Energy window size", size)
    return ((1,) * size,) * size


def resolve_parameters(shape, **overrides):
    """Merge overrides, validate them, and cap levels for this image size."""
    unknown = sorted(set(overrides) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise TypeError(f"Unknown Laplacian pyramid parameters: {', '.join(unknown)}")
    values = {**DEFAULT_PARAMETERS, **overrides}

    requested = values["levels"]
    if not isinstance(requested, int) or isinstance(requested, bool) or requested < 0:
        raise ValueError("levels must be a non-negative integer.")
    min_top_size = values["min_top_size"]
    if not isinstance(min_top_size, int) or isinstance(min_top_size, bool) or min_top_size < 1:
        raise ValueError("min_top_size must be a positive integer.")
    weights = np.asarray(values["energy_weights"], dtype=np.float64)
    if (weights.ndim != 2 or weights.shape[0] != weights.shape[1] or weights.shape[0] % 2 == 0
            or not np.isfinite(weights).all() or weights.min() < 0 or weights.sum() <= 0):
        raise ValueError("energy_weights must be an odd square, finite, non-negative "
                         "matrix with a positive sum.")
    top_window = _odd_window("top_window", values["top_window"])
    bins = values["entropy_bins"]
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("entropy_bins must be an integer of at least 2.")
    if max(weights.shape[0], top_window) > min_top_size:
        raise ValueError("min_top_size must be at least as large as both windows.")

    height, width = shape[:2]
    if min(height, width) < min_top_size:
        raise ValueError(f"The image must be at least {min_top_size} pixels per side.")
    levels = 0
    # pyrDown rounds odd sizes up, so every level is computed exactly as built.
    while levels < requested and min((height + 1) // 2, (width + 1) // 2) >= min_top_size:
        height, width = (height + 1) // 2, (width + 1) // 2
        levels += 1
    return {"requested_levels": requested, "levels": levels, "top_size_hw": [height, width],
            "min_top_size": min_top_size, "energy_weights": weights.tolist(),
            "top_window": top_window, "entropy_bins": bins}


def build_pyramid(image, levels):
    """Return float32 [LP0, ..., LPN]; LPN is the Gaussian top level (Eq. 7)."""
    current = image.astype(np.float32)
    pyramid = []
    for _ in range(levels):
        smaller = cv2.pyrDown(current, borderType=cv2.BORDER_REFLECT_101)
        # dstsize restores odd sizes; pyrUp supports only the reflect-101 border.
        expanded = cv2.pyrUp(smaller, dstsize=(current.shape[1], current.shape[0]))
        pyramid.append(current - expanded)
        current = smaller
    pyramid.append(current)
    return pyramid


def reconstruct(pyramid):
    """Invert build_pyramid by repeated Expand and addition (Eqs. 8-9)."""
    current = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        current = level + cv2.pyrUp(current, dstsize=(level.shape[1], level.shape[0]))
    return current


def region_energy(level, weights):
    """Weighted local sum of squared coefficients (Eq. 13); weights sum to 1."""
    energy = cv2.filter2D(np.square(level, dtype=np.float64), cv2.CV_64F, weights,
                          borderType=cv2.BORDER_REFLECT_101)
    return np.round(energy, _SCORE_DECIMALS)


def local_deviation(top, window):
    """Local gray-level variance over a window (Eq. 10)."""
    values = top.astype(np.float64)
    mean = cv2.boxFilter(values, -1, (window, window), normalize=True,
                         borderType=cv2.BORDER_REFLECT_101)
    mean_square = cv2.boxFilter(values * values, -1, (window, window), normalize=True,
                                borderType=cv2.BORDER_REFLECT_101)
    return np.round(np.maximum(mean_square - mean * mean, 0), _SCORE_DECIMALS)


def local_entropy(top, window, bins):
    """Local gray-level entropy in bits over a window (Eq. 11)."""
    levels = np.clip(np.floor(top.astype(np.float64) * bins), 0, bins - 1).astype(np.int32)
    entropy = np.zeros(top.shape, dtype=np.float64)
    for value in np.unique(levels):
        probability = cv2.boxFilter((levels == value).astype(np.float64), -1, (window, window),
                                    normalize=True, borderType=cv2.BORDER_REFLECT_101)
        present = probability > 0
        entropy[present] -= probability[present] * np.log2(probability[present])
    return np.round(entropy, _SCORE_DECIMALS)


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
    levels = parameters["levels"]
    weights = np.asarray(parameters["energy_weights"], dtype=np.float64)
    weights /= weights.sum()
    window, bins = parameters["top_window"], parameters["entropy_bins"]
    pixel_max = float(np.iinfo(first.dtype).max)

    # Keep only running winners per level instead of every frame's pyramid.
    best_energy, fused = [None] * levels, [None] * (levels + 1)
    best_deviation = best_entropy = deviation_top = entropy_top = None
    for index, image in enumerate(images):
        gray = (cv2.cvtColor(image.astype(np.float32) / pixel_max, cv2.COLOR_BGR2GRAY)
                if gray_images is None else gray_images[index])
        if (gray.shape != first.shape[:2] or gray.dtype != np.float32
                or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1):
            raise ValueError("Grayscale copies must be finite float32 arrays in [0, 1].")
        # Grayscale pyramids decide; color pyramids supply all three channels.
        gray_pyramid = build_pyramid(gray, levels)
        color_pyramid = build_pyramid(image, levels)

        for level in range(levels):
            energy = region_energy(gray_pyramid[level], weights)
            if index == 0:
                best_energy[level], fused[level] = energy, color_pyramid[level]
                continue
            # Strict comparison keeps the earlier frame on ties, as Eq. 14 keeps A.
            replace = energy > best_energy[level]
            best_energy[level][replace] = energy[replace]
            fused[level][replace] = color_pyramid[level][replace]

        deviation = local_deviation(gray_pyramid[levels], window)
        entropy = local_entropy(gray_pyramid[levels], window, bins)
        top = color_pyramid[levels]
        if index == 0:
            best_deviation, deviation_top = deviation, top
            best_entropy, entropy_top = entropy, top.copy()
        else:
            replace = deviation > best_deviation
            best_deviation[replace] = deviation[replace]
            deviation_top[replace] = top[replace]
            replace = entropy > best_entropy
            best_entropy[replace] = entropy[replace]
            entropy_top[replace] = top[replace]
        del gray_pyramid, color_pyramid

    # Eq. 12 for N frames: when one frame leads both D and E this equals that
    # frame, otherwise it averages the deviation and entropy winners.
    fused[levels] = (deviation_top + entropy_top) / 2
    result = reconstruct(fused)
    return np.rint(np.clip(result, 0, pixel_max)).astype(first.dtype)
