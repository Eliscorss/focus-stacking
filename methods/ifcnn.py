"""IFCNN-MAX multi-focus fusion after Zhang et al. (2020).

Reference: Y. Zhang, Y. Liu, P. Sun, H. Yan, X. Zhao, L. Zhang, "IFCNN: A
general image fusion framework based on convolutional neural network",
Information Fusion 54:99-118, 2020, doi:10.1016/j.inffus.2019.07.011, checked
against Sec. 2.2, Eq. 1 and Fig. 1, and against the authors' PyTorch code and
README errata, uzeful/IFCNN@7dd4a2a (Code/model.py, Code/IFCNN_Main.py). The
weights are the authors' released IFCNN-MAX.pth, used unchanged and not
retrained.

Paper rules (Sec. 2.2): CONV1 is ResNet101's first layer (64 kernels 7x7,
ImageNet-trained, frozen) with stride 1 and padding 3; CONV2 and CONV3 have 64
kernels 3x3, stride 1, padding 1, BN and ReLU; CONV4 has 3 kernels 1x1 with
padding 0 and, per the authors' README errata, stride 1; no BN or activation
after CONV1 and CONV4; no downsampling. Eq. 1 fuses the CONV2 features of all
N >= 2 inputs element-wise, with the maximum for multi-focus images; the
running maximum over frames used here is exactly Eq. 1 for any N.
From the code, where the paper is silent: replicate padding, BN before ReLU,
RGB input scaled to [0, 1] and normalized with ImageNet mean/std, output
denormalized and clamped to [0, 1].
Engineering adaptations: BGR input and output; scaling by the dtype range, so uint16 is accepted (untested on real
16-bit data, the model was trained on 8-bit images); rounding to the nearest
level instead of the authors' truncation; tiled inference with a margin equal
to the receptive-field radius, which reproduces whole-image inference up to
float rounding.
Check: with the authors' truncation and their lossless IV/MD inputs, outputs
match their published IFCNN-MAX results within one gray level (at most 0.012
percent of pixels differ). Their Lytro JPEG results differ more (mean 0.5,
up to 24 levels at few pixels, more in chroma than luma), presumably because
today's JPEG decoders upsample 4:2:0 chroma differently from the authors'
2019 software; that environment cannot be rebuilt to confirm it.
"""

import hashlib
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Adjustable defaults. stack() and load_model() keyword overrides take
# precedence over these values; run_benchmark.py keeps no separate copy. The
# network itself has no tunable parameters: all of its values come from the
# weight file.
DEFAULT_PARAMETERS = {
    # Weight file, relative to this methods/ directory unless absolute. It is
    # third-party and not committed; README explains where to download it.
    # Its SHA-256 must equal WEIGHTS_SHA256 below.
    "weights_path": "weights/IFCNN-MAX.pth",
    # PyTorch CUDA device such as "cuda" or "cuda:0". The method runs on a GPU
    # only and raises an error when CUDA is unavailable, so timings never
    # silently come from a CPU run.
    "device": "cuda",
    # Side of the square output tile in pixels, integer >= 1. Each tile is
    # computed with RECEPTIVE_RADIUS extra real pixels per side, so the result
    # does not depend on this value up to float rounding (cuDNN may pick other
    # kernels per size; on Kro_DEMO_10 at most 28 of 3.9 million values moved
    # by one level). Measured on an RTX 4060 Laptop for five 2592x1944 frames:
    # 1024 peaks at 1.05 GiB in 0.98 s, 2048 at 3.9 GiB in 1.02 s, so larger
    # tiles mainly cost memory.
    "tile_size": 1024,
}

# Fixed facts of the released model, not tuning values.
WEIGHTS_SHA256 = "6022b44c88a2f0367ba9b08fcdf0d663745d9a966fca0ee1f41867797b6ff53f"
WEIGHTS_URL = ("https://github.com/uzeful/IFCNN/raw/"
               "1d281720468bb0c317957b6bd26929ae45a9dfb5/Code/snapshots/IFCNN-MAX.pth")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Per-side reach of one output pixel: CONV1 3 + CONV2 1 + CONV3 1 + CONV4 0.
RECEPTIVE_RADIUS = 5

_MODELS = {}


class _ConvBlock(nn.Module):
    """Replicate padding 1, 3x3 convolution without bias, BN, ReLU (CONV2, CONV3)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        return F.relu(self.bn(self.conv(F.pad(x, (1, 1, 1, 1), mode="replicate"))))


class IFCNN(nn.Module):
    """IFCNN layers with the authors' parameter names, split at the fusion point."""

    def __init__(self):
        super().__init__()
        # ResNet101's first layer with stride 1 and no padding; the trained values
        # are stored in the weight file, so ResNet101 itself is never loaded.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, bias=False)
        self.conv2 = _ConvBlock()
        self.conv3 = _ConvBlock()
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, bias=True)

    def extract(self, x):
        """Features of one normalized RGB input, before fusion."""
        return self.conv2(self.conv1(F.pad(x, (3, 3, 3, 3), mode="replicate")))

    def reconstruct(self, features):
        """Normalized RGB output from fused features."""
        return self.conv4(self.conv3(features))


def _weights_file(weights_path):
    path = Path(weights_path)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_device(device):
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError("device must be a CUDA device such as 'cuda' or 'cuda:0'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; IFCNN runs on a GPU only. "
                           "Install a CUDA build of PyTorch (see README).")
    return torch.device(device)


def resolve_parameters(shape, **overrides):
    """Merge overrides, validate them, and add the tiling used for this image size."""
    unknown = sorted(set(overrides) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise TypeError(f"Unknown IFCNN parameters: {', '.join(unknown)}")
    values = {**DEFAULT_PARAMETERS, **overrides}
    tile = values["tile_size"]
    if not isinstance(tile, int) or isinstance(tile, bool) or tile < 1:
        raise ValueError("tile_size must be a positive integer.")
    if not isinstance(values["weights_path"], (str, os.PathLike)):
        raise ValueError("weights_path must be a path.")
    if not isinstance(values["device"], str) or not values["device"].startswith("cuda"):
        raise ValueError("device must be a CUDA device such as 'cuda' or 'cuda:0'.")
    height, width = shape[:2]
    if height < 1 or width < 1:
        raise ValueError("The image must not be empty.")
    return {"weights_path": str(values["weights_path"]), "device": values["device"],
            "tile_size": tile, "tile_margin_pixels": RECEPTIVE_RADIUS,
            "tiles": math.ceil(height / tile) * math.ceil(width / tile),
            "fusion_rule": "element-wise maximum of CONV2 features over all frames"}


def load_model(weights_path=None, device=None):
    """Load, verify and warm up the model once per process; return (model, info).

    Sets PyTorch to deterministic algorithms without TF32 for the whole process.
    Later calls with the same file and device return the cached model.
    """
    weights_path = DEFAULT_PARAMETERS["weights_path"] if weights_path is None else weights_path
    device = DEFAULT_PARAMETERS["device"] if device is None else device
    path = _weights_file(weights_path)
    torch_device = _check_device(device)
    key = (str(path), str(torch_device))
    if key in _MODELS:
        return _MODELS[key]

    start = perf_counter()
    if not path.is_file():
        raise FileNotFoundError(f"IFCNN weights not found at {path}. Download {WEIGHTS_URL} "
                                "and save it there (see README).")
    checksum = _sha256(path)
    if checksum != WEIGHTS_SHA256:
        raise ValueError(f"IFCNN weights have SHA-256 {checksum}, expected {WEIGHTS_SHA256}.")

    # cuBLAS needs this before its first use when determinism is enforced.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # TF32 would lower the precision of float32 convolutions on recent GPUs.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # weights_only=True reads tensors without executing code from the file.
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = IFCNN()
    model.load_state_dict(state, strict=True)
    model.eval().to(torch_device)
    loaded = perf_counter()

    # One full-size tile initializes CUDA and cuDNN outside the fusion timing.
    side = DEFAULT_PARAMETERS["tile_size"] + 2 * RECEPTIVE_RADIUS
    with torch.inference_mode():
        model.reconstruct(model.extract(torch.zeros((1, 3, side, side), device=torch_device)))
    torch.cuda.synchronize(torch_device)
    info = {"weights_file": str(path), "weights_sha256": checksum, "weights_url": WEIGHTS_URL,
            "device": str(torch_device), "device_name": torch.cuda.get_device_name(torch_device),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "deterministic": True, "tf32": False,
            "load_seconds": loaded - start, "warmup_seconds": perf_counter() - loaded}
    _MODELS[key] = (model, info)
    return _MODELS[key]


def _fuse_float(images, parameters):
    """Return the fused BGR image as float32 in [0, dtype max], before rounding."""
    model, _ = load_model(parameters["weights_path"], parameters["device"])
    device = next(model.parameters()).device
    first = images[0]
    height, width = first.shape[:2]
    pixel_max = float(np.iinfo(first.dtype).max)
    tile, margin = parameters["tile_size"], RECEPTIVE_RADIUS
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    result = np.empty(first.shape, dtype=np.float32)

    with torch.inference_mode():
        for y0 in range(0, height, tile):
            for x0 in range(0, width, tile):
                y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
                # Real neighbors absorb the padding of interior tile edges; at image
                # edges the network's own replicate padding acts as on the whole image.
                ey0, ey1 = max(y0 - margin, 0), min(y1 + margin, height)
                ex0, ex1 = max(x0 - margin, 0), min(x1 + margin, width)
                fused = None
                for image in images:
                    rgb = np.ascontiguousarray(image[ey0:ey1, ex0:ex1, ::-1], dtype=np.float32)
                    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0)
                    features = model.extract((x / pixel_max - mean) / std)
                    # The maximum is order independent, so ties need no rule.
                    fused = features if fused is None else torch.maximum(fused, features, out=fused)
                    del features
                output = (model.reconstruct(fused) * std + mean).clamp_(0, 1) * pixel_max
                del fused
                core = output[0, :, y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
                result[y0:y1, x0:x1] = core.permute(1, 2, 0).cpu().numpy()[..., ::-1]
    return result


def stack(images, *, gray_images=None, **overrides):
    """Fuse aligned uint8/uint16 BGR images without modifying their arrays.

    gray_images is accepted for the common interface but unused: IFCNN works
    on the color images directly.
    """
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
    parameters = resolve_parameters(first.shape, **overrides)
    result = _fuse_float(images, parameters)
    if not np.isfinite(result).all():
        raise RuntimeError("IFCNN produced non-finite values.")
    return np.rint(result).astype(first.dtype)
