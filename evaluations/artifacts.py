"""Gradient overshoot: share of pixels whose fused edge is stronger than in every source.

Literature basis: the artifact indicator AM(i,j) in Eq. 21 of Kumar, SIViP
7:1125-1143 (2013), doi:10.1007/s11760-012-0361-x, which follows Petrovic and
Xydeas (ICCV 2005): AM = 1 where the fused edge strength exceeds that of every
source image, marking gradients that no source contains (seams, halos, ringing,
amplified noise). Only this indicator is used; the (1 - Q) and perceptual
weights of the full N_m^{AB/F} measure are not computed.

Engineering adaptations, not from the paper: N source frames instead of two; a
relative factor and an absolute floor tolerate noise and interpolation; the
per-pixel source maximum is dilated to tolerate sub-pixel registration
residuals. Edge strength is the 3 x 3 Sobel magnitude on the same normalized
grayscale and evaluation region as Tenengrad. Lower values are better.
"""

import math

import cv2
import numpy as np

from evaluations import sharpness

DEFAULT_PARAMETERS = {
    # Relative margin k: a pixel overshoots only if its fused edge strength exceeds
    # (1 + k) times the strongest source edge strength at that position.
    # Dimensionless, >= 0. Initial 0.1 (10 percent) is an engineering choice, not a
    # literature value. Larger values ignore milder sharpening; smaller values flag
    # more pixels, including interpolation and rounding differences.
    "overshoot_factor": 0,
    # Absolute floor eps, added after the relative margin, in Sobel magnitude units
    # of the [0, 1] grayscale image (a full black-to-white step gives 4). >= 0.
    # Initial 0.02 is about the response to a step of one to two 8-bit gray levels;
    # engineering choice. Larger values ignore weak overshoot in flat regions;
    # smaller values let flat-region noise count as overshoot.
    "overshoot_floor": 0,
    # Side of the square maximum filter applied to the per-pixel source maximum, in
    # pixels; positive odd integer, 1 disables it. Initial 3 tolerates about one
    # pixel of registration residual. Larger values tolerate more misalignment but
    # can hide seams next to strong edges; 1 is strictest.
    "source_max_window": 1,
}


def resolve_parameters(**overrides):
    """Merge explicit overrides into the file defaults and validate the result."""
    unknown = set(overrides) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown gradient overshoot parameters: {', '.join(sorted(unknown))}")
    parameters = {**DEFAULT_PARAMETERS, **overrides}
    for key in ("overshoot_factor", "overshoot_floor"):
        value = parameters[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be a finite number >= 0.")
        parameters[key] = float(value)
    window = parameters["source_max_window"]
    if (isinstance(window, bool) or not isinstance(window, (int, np.integer))
            or window < 1 or window % 2 == 0):
        raise ValueError("source_max_window must be a positive odd integer.")
    parameters["source_max_window"] = int(window)
    return parameters


def _edge_strength(gray):
    """Return the 3 x 3 Sobel gradient magnitude, with the same border rule as Tenengrad."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return cv2.magnitude(gx, gy)


def evaluate(fused, gray_images, *, roi=None, **overrides):
    """Return the gradient overshoot ratio of a fused image against its aligned sources.

    fused is the fused BGR image; gray_images are the shared float32 [0, 1]
    grayscale copies of the aligned source frames. Inputs are not modified.
    """
    parameters = resolve_parameters(**overrides)
    # Reuse the sharpness preparation so the fused grayscale and the region match
    # Tenengrad exactly; its normalization equals the benchmark's shared copies.
    fused_gray, region = sharpness._prepare(fused, roi)
    if not gray_images:
        raise ValueError("Gradient overshoot needs at least one source grayscale image.")
    # Accumulate the per-pixel source maximum one frame at a time.
    source_max = None
    for gray in gray_images:
        if (not isinstance(gray, np.ndarray) or gray.dtype != np.float32
                or gray.shape != fused_gray.shape):
            raise ValueError("Source grayscale copies must be float32 arrays matching the fused image.")
        if not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
            raise ValueError("Source grayscale copies must be finite and in [0, 1].")
        strength = _edge_strength(gray)
        if source_max is None:
            source_max = strength
        else:
            np.maximum(source_max, strength, out=source_max)
    window = parameters["source_max_window"]
    if window > 1:
        source_max = cv2.dilate(source_max, np.ones((window, window), np.uint8))
    limit = (1.0 + parameters["overshoot_factor"]) * source_max[region] + parameters["overshoot_floor"]
    overshoot = _edge_strength(fused_gray)[region] > limit
    return {"gradient_overshoot_ratio": float(np.mean(overshoot, dtype=np.float64))}
