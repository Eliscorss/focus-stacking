"""Single entry point for SG3 registration, fusion, evaluation, and reporting."""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from evaluations import sharpness
from methods import laplacian, method_a, method_b


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
VIEWS = ("flank", "rake", "face")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
METHODS = {"laplacian": laplacian.stack,
           "method_a": method_a.stack, "method_b": method_b.stack}
UNIMPLEMENTED = {"method_a", "method_b"}
CSV_FIELDS = ("tool", "view", "method", "image_count", "tenengrad",
              "laplacian_variance", "fusion_seconds", "status", "error")


def natural_key(path):
    """Sort filenames numerically, with a deterministic tie breaker."""
    parts = tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                  for part in re.split(r"(\d+)", path.name))
    return parts, path.name


def resolve_input(value):
    """Accept only a tool or a known view directly inside the raw-data root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        parts = path.relative_to(RAW_DIR.resolve()).parts
    except ValueError as exc:
        raise ValueError("Input must be inside data/raw/<tool>[/<view>].") from exc
    if not path.is_dir():
        raise ValueError(f"Input directory does not exist: {path}")
    if len(parts) == 1:
        return parts[0], [path / view for view in VIEWS]
    if len(parts) == 2 and parts[1] in VIEWS:
        return parts[0], [path]
    raise ValueError("Specify data/raw/<tool> or its flank, rake, or face directory.")


def load_images(directory):
    """Read one sequence without modifying files or silently discarding bad frames."""
    if not directory.is_dir():
        raise ValueError(f"View directory does not exist: {directory}")
    if any(path.is_dir() for path in directory.iterdir()):
        raise ValueError("Nested sequences are not supported; keep one sequence per view.")
    files = sorted((path for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                   key=natural_key)
    if not files:
        raise ValueError(f"No supported images found in {directory}")
    images = []
    for path in files:
        if not path.resolve().is_relative_to(RAW_DIR.resolve()):
            raise ValueError(f"Image resolves outside data/raw: {path.name}")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot decode image: {path.name}")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"Expected uint8/uint16 BGR color image: {path.name}")
        if images and (image.shape != images[0].shape or image.dtype != images[0].dtype):
            raise ValueError(f"Image shape or dtype differs from the first frame: {path.name}")
        images.append(image)
    return images, files


def to_gray_images(images):
    """Create shared float32 grayscale copies normalized by the fixed dtype range."""
    return [cv2.cvtColor(image.astype(np.float32) / float(np.iinfo(image.dtype).max),
                         cv2.COLOR_BGR2GRAY) for image in images]


def _registration_gray(image, max_size):
    """Use a smaller uint8 copy for feature estimation, retaining full output resolution."""
    height, width = image.shape[:2]
    ratio = min(1.0, max_size / max(height, width))
    if ratio < 1:
        image = cv2.resize(image, (round(width * ratio), round(height * ratio)),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint16:
        gray = np.rint(gray.astype(np.float32) / 257.0).astype(np.uint8)
    return gray


def _scale_translation(source, target):
    """Fit target = scale * source + translation, with no rotation or shear."""
    centered_source = source - source.mean(axis=0)
    centered_target = target - target.mean(axis=0)
    denominator = np.sum(centered_source ** 2)
    if denominator < 1:
        raise ValueError("Matched features have insufficient spatial spread.")
    scale = float(np.sum(centered_source * centered_target) / denominator)
    translation = target.mean(axis=0) - scale * source.mean(axis=0)
    return scale, translation


def _match_transform(source_features, target_features, resize_scale):
    """Estimate a checked scale/translation transform from two feature sets."""
    points, descriptors = source_features
    target_points, target_descriptors = target_features
    if (descriptors is None or target_descriptors is None
            or len(descriptors) < 6 or len(target_descriptors) < 6):
        raise ValueError("Insufficient features for reliable registration.")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors, target_descriptors, k=2)
    matches = [pair[0] for pair in pairs
               if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < 6:
        raise ValueError("Fewer than six reliable feature matches.")
    source = np.float64([points[m.queryIdx].pt for m in matches])
    target = np.float64([target_points[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(0)
    estimate, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.99,
    )
    if estimate is None or inliers is None or np.count_nonzero(inliers) < 6:
        raise ValueError("RANSAC could not find a reliable alignment.")
    keep = inliers.ravel().astype(bool)
    # RANSAC finds robust matches; refitting removes its optional rotation.
    source, target = source[keep], target[keep]
    scale, translation = _scale_translation(source, target)
    errors = np.linalg.norm(scale * source + translation - target, axis=1)
    keep = errors <= 3.0
    if np.count_nonzero(keep) < 6:
        raise ValueError("Scale/translation model does not fit the matches.")
    source, target = source[keep], target[keep]
    scale, translation = _scale_translation(source, target)
    residual = float(np.median(np.linalg.norm(scale * source + translation - target, axis=1)))
    if not 0.5 <= scale <= 2.0 or residual > 3.0:
        raise ValueError("Implausible scale or excessive registration residual.")
    matrix = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float64)
    matrix[:, 2] = translation / resize_scale
    return matrix, len(source), residual


def align_images(images, *, max_size=1600):
    """Align to the middle frame, with a checked adjacent-frame fallback."""
    if not images:
        raise ValueError("Registration requires at least one image.")
    height, width = images[0].shape[:2]
    if any(image.shape != images[0].shape or image.dtype != images[0].dtype for image in images):
        raise ValueError("Registration images must share shape and dtype.")
    reference_index = len(images) // 2
    reference = _registration_gray(images[reference_index], max_size)
    resize_scale = np.array([reference.shape[1] / width, reference.shape[0] / height])
    sift = cv2.SIFT_create(nfeatures=6000)
    features = {reference_index: sift.detectAndCompute(reference, None)}
    transforms = [None] * len(images)
    transforms[reference_index] = {
        "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "inliers": None,
        "median_residual_registration_pixels": None, "matched_to_index": reference_index,
        "direct_match_error": None,
    }
    for index in sorted(range(len(images)), key=lambda i: abs(i - reference_index)):
        if index == reference_index:
            continue
        features[index] = sift.detectAndCompute(_registration_gray(images[index], max_size), None)
        adjacent = index + 1 if index < reference_index else index - 1
        anchors = [reference_index] if adjacent == reference_index else [reference_index, adjacent]
        errors = []
        for anchor in anchors:
            try:
                local, inlier_count, residual = _match_transform(features[index], features[anchor], resize_scale)
                anchor_matrix = np.vstack((transforms[anchor]["matrix"], [0, 0, 1]))
                matrix = (anchor_matrix @ np.vstack((local, [0, 0, 1])))[:2]
                if not 0.5 <= matrix[0, 0] <= 2.0:
                    raise ValueError("Composed scale is outside the accepted range.")
                transforms[index] = {
                    "matrix": matrix.tolist(), "inliers": inlier_count,
                    "median_residual_registration_pixels": residual, "matched_to_index": anchor,
                    "direct_match_error": errors[0] if errors else None,
                }
                break
            except ValueError as exc:
                errors.append(f"anchor {anchor}: {exc}")
        if transforms[index] is None:
            raise ValueError(f"Frame {index}: " + "; ".join(errors))

    common_mask = np.ones((height, width), dtype=bool)
    for transform in transforms:
        # Linear interpolation must never mix valid image data with black padding.
        mask = cv2.warpAffine(np.ones((height, width), dtype=np.float32),
                              np.asarray(transform["matrix"]), (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        common_mask &= mask >= 1.0 - 1e-6

    rows = np.flatnonzero(common_mask.any(axis=1))
    columns = np.flatnonzero(common_mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("Registered images have no common valid area.")
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    if (not common_mask[y0:y1, x0:x1].all()
            or (y1 - y0) * (x1 - x0) < 0.25 * height * width):
        raise ValueError("Common valid area is irregular or below 25 percent of the frame.")
    aligned = []
    for image, transform in zip(images, transforms):
        warped = cv2.warpAffine(image, np.asarray(transform["matrix"]), (width, height),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        aligned.append(warped[y0:y1, x0:x1].copy())
    return aligned, {"reference_index": reference_index,
                     "crop_xywh": [x0, y0, x1 - x0, y1 - y0],
                     "transforms": transforms}


def save_image(path, image):
    """Write lossless PNG atomically, including paths with non-ASCII characters."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Could not encode {path.name}")
    temporary = path.with_suffix(".png.tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


def save_chart(path, rows, title):
    """Plot separate metric axes and annotate unavailable methods without zero bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    try:
        for axis, metric, label in zip(axes, ("tenengrad", "laplacian_variance"),
                                       ("Tenengrad", "Laplacian variance")):
            valid = [(i, row) for i, row in enumerate(rows) if row["status"] == "success"]
            axis.bar([i for i, _ in valid], [row[metric] for _, row in valid],
                     color="#3679a8", width=0.55)
            for i, row in valid:
                axis.annotate(f"{row[metric]:.4g}", (i, row[metric]),
                              xytext=(0, 5), textcoords="offset points", ha="center")
            for i, row in enumerate(rows):
                if row["status"] != "success":
                    axis.text(i, 0.08, row["status"].replace("_", "\n"),
                              transform=axis.get_xaxis_transform(), ha="center", fontsize=9)
            if not valid:
                axis.text(0.5, 0.65, "No valid scores", transform=axis.transAxes, ha="center")
            axis.set_xticks(range(len(rows)), [row["method"] for row in rows])
            axis.set_xlim(-0.6, len(rows) - 0.4)
            axis.set_ylim(bottom=0)
            axis.margins(y=0.22)
            axis.set_title(label)
            axis.set_ylabel("Score")
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        figure.suptitle(title)
        temporary = path.with_suffix(".png.tmp")
        figure.savefig(temporary, format="png", dpi=160)
        temporary.replace(path)
    finally:
        plt.close(figure)


def run_view(tool, directory, selected, args):
    """Prepare a view once and reuse read-only arrays across all selected methods."""
    output = RESULTS_DIR / tool / directory.name
    resolved_output = output.resolve()
    if (RESULTS_DIR.resolve() != RESULTS_DIR.absolute()
            or not resolved_output.is_relative_to(RESULTS_DIR.resolve())
            or resolved_output.is_relative_to(RAW_DIR.resolve())):
        raise ValueError("Output must stay within results and outside raw data.")
    output.mkdir(parents=True, exist_ok=True)
    info = {"started_utc": datetime.now(timezone.utc).isoformat(),
            "input_directory": str(directory.relative_to(PROJECT_ROOT)),
            "input_files": [], "methods": selected,
            "parameters": {"window_size": args.window_size, "smooth_size": args.smooth_size,
                           "registration_max_size": args.registration_max_size,
                           "roi_xywh_in_cropped_image": args.roi},
            "versions": {"numpy": np.__version__, "opencv": cv2.__version__},
            "registration": None}
    image_count = 0
    stage = "load"
    preparation_error = None
    start = perf_counter()
    try:
        images, files = load_images(directory)
        image_count = len(images)
        info["input_files"] = [path.name for path in files]
        stage = "registration"
        aligned, info["registration"] = align_images(images, max_size=args.registration_max_size)
        del images
        stage = "grayscale"
        grays = to_gray_images(aligned)
        stage = "evaluation_roi"
        sharpness._prepare(grays[0], args.roi)
        for array in [*aligned, *grays]:
            array.setflags(write=False)
    except Exception as exc:
        preparation_error = f"{stage}: {type(exc).__name__}: {exc}"
    info["preparation_seconds"] = perf_counter() - start
    info["preparation_error"] = preparation_error

    rows = []
    for method in selected:
        row = dict.fromkeys(CSV_FIELDS, "")
        row.update(tool=tool, view=directory.name, method=method, image_count=image_count)
        image_path = output / f"{method}.png"
        # Remove only this selected method's old generated image to prevent stale results.
        if image_path.exists():
            if not image_path.resolve().is_relative_to(RESULTS_DIR.resolve()):
                raise ValueError("Existing result image resolves outside results.")
            image_path.unlink()
        if method in UNIMPLEMENTED:
            row.update(status="not_implemented", error=f"{method} is not implemented.")
        elif preparation_error:
            row.update(status="failed", error=preparation_error)
        else:
            stage = "fusion"
            start = perf_counter()
            try:
                fused = METHODS[method](aligned, gray_images=grays,
                                        window_size=args.window_size, smooth_size=args.smooth_size)
                row["fusion_seconds"] = perf_counter() - start
                if not isinstance(fused, np.ndarray) or fused.shape != aligned[0].shape or fused.dtype != aligned[0].dtype:
                    raise ValueError("Fusion must return an image with the input shape and dtype.")
                stage = "evaluation"
                scores = sharpness.evaluate(fused, roi=args.roi)
                if not all(np.isfinite(value) for value in scores.values()):
                    raise ValueError("Evaluation returned non-finite scores.")
                stage = "save_image"
                save_image(image_path, fused)
                row.update(scores, status="success")
                del fused
            except NotImplementedError as exc:
                row.update(status="not_implemented", error=f"{stage}: {exc}")
            except Exception as exc:
                if stage == "fusion":
                    row["fusion_seconds"] = perf_counter() - start
                row.update(status="failed", error=f"{stage}: {type(exc).__name__}: {exc}")
        rows.append(row)
        print(f"{tool}/{directory.name}: {method}: {row['status']}"
              + (f" ({row['error']})" if row["error"] else f" ({row['fusion_seconds']:.3f} s)"))

    temporary = output / "benchmark.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "benchmark.csv")
    # Keep reproducibility details out of the requested compact CSV.
    temporary = output / "run_info.json.tmp"
    temporary.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "run_info.json")
    save_chart(output / "metrics_comparison.png", rows, f"{tool} / {directory.name}")
    return any(row["status"] == "success" for row in rows), preparation_error is not None or any(
        row["status"] == "failed" for row in rows)


def main(argv=None):
    """Run one tool or view; return a nonzero exit code for actual failures."""
    parser = argparse.ArgumentParser(description="SG3 focus stacking benchmark")
    parser.add_argument("--input", required=True, help="data/raw/<tool> or data/raw/<tool>/<view>")
    parser.add_argument("--method", choices=["all", *METHODS], default="all")
    parser.add_argument("--window-size", type=int, default=21, help="Odd local focus window size")
    parser.add_argument("--smooth-size", type=int, default=5, help="Odd majority window; 1 disables smoothing")
    parser.add_argument("--registration-max-size", type=int, default=1600,
                        help="Maximum feature-estimation dimension; output stays full resolution")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Evaluation rectangle in cropped output coordinates; default: full valid area")
    args = parser.parse_args(argv)
    if any(size < 1 or size % 2 == 0 for size in (args.window_size, args.smooth_size)):
        parser.error("Window sizes must be positive odd integers.")
    if args.registration_max_size < 64:
        parser.error("Registration maximum size must be at least 64.")
    try:
        tool, directories = resolve_input(args.input)
    except ValueError as exc:
        parser.error(str(exc))
    selected = list(METHODS) if args.method == "all" else [args.method]
    any_success, any_failure = False, False
    for directory in directories:
        try:
            success, failed = run_view(tool, directory, selected, args)
            any_success |= success
            any_failure |= failed
        except Exception as exc:
            any_failure = True
            print(f"{tool}/{directory.name}: reporting failed: {type(exc).__name__}: {exc}")
    return 1 if any_failure or not any_success else 0


if __name__ == "__main__":
    raise SystemExit(main())
