"""Regression tests for fusion, registration, metrics, and benchmark reporting."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import run_benchmark as benchmark
from evaluations import sharpness
from methods import laplacian


def texture(size=256):
    """Make repeatable textured BGR data with enough features for registration."""
    rng = np.random.default_rng(42)
    image = rng.integers(30, 220, (size, size, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    for _ in range(50):
        x, y = rng.integers(10, size - 10, 2)
        color = tuple(int(value) for value in rng.integers(0, 256, 3))
        cv2.circle(image, (int(x), int(y)), 4, color, -1)
    return image


class FusionTests(unittest.TestCase):
    def test_complementary_focus_and_input_immutability(self):
        original = texture()
        blurred = cv2.GaussianBlur(original, (15, 15), 3)
        first, second = original.copy(), original.copy()
        first[:, 128:] = blurred[:, 128:]
        second[:, :128] = blurred[:, :128]
        images = [first, second]
        copies = [image.copy() for image in images]
        grays = benchmark.to_gray_images(images)
        gray_copies = [gray.copy() for gray in grays]
        for array in [*images, *grays]:
            array.setflags(write=False)
        fused = laplacian.stack(images, gray_images=grays)
        np.testing.assert_array_equal(fused[20:-20, 20:100], original[20:-20, 20:100])
        np.testing.assert_array_equal(fused[20:-20, 156:-20], original[20:-20, 156:-20])
        for image, copy in zip(images, copies):
            np.testing.assert_array_equal(image, copy)
        for gray, copy in zip(grays, gray_copies):
            np.testing.assert_array_equal(gray, copy)

    def test_smoothing_removes_isolated_label(self):
        labels = np.zeros((11, 11), dtype=np.int32)
        labels[5, 5] = 2
        smoothed = laplacian.smooth_source_map(labels, 3, 5)
        self.assertFalse(smoothed.any())
        self.assertEqual(labels[5, 5], 2)

    def test_equal_focus_keeps_first_color_source(self):
        first = np.full((30, 30, 3), (20, 40, 60), dtype=np.uint8)
        second = np.full((30, 30, 3), (80, 100, 120), dtype=np.uint8)
        np.testing.assert_array_equal(laplacian.stack([first, second]), first)

    def test_uint16_preserves_precision(self):
        image = texture(64).astype(np.uint16) * 257
        np.testing.assert_array_equal(laplacian.stack([image]), image)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            laplacian.stack([])
        with self.assertRaises(ValueError):
            laplacian.stack([texture(64)], window_size=20)
        with self.assertRaises(ValueError):
            laplacian.stack([texture(64), texture(80)])


class MetricTests(unittest.TestCase):
    def test_linear_ramp_has_known_gradient_energy(self):
        ramp = np.tile(np.arange(40, dtype=np.float32) / 100, (30, 1))
        self.assertAlmostEqual(sharpness.tenengrad(ramp), 0.0064, places=6)
        self.assertAlmostEqual(sharpness.laplacian_variance(ramp), 0, places=10)

    def test_blur_reduces_both_metrics(self):
        image = texture()
        blurred = cv2.GaussianBlur(image, (15, 15), 3)
        for key, value in sharpness.evaluate(image).items():
            self.assertGreater(value, sharpness.evaluate(blurred)[key])

    def test_equivalent_bit_depths_and_roi(self):
        image = texture(64)
        roi = (8, 8, 40, 40)
        scores = sharpness.evaluate(image, roi=roi)
        high_depth = sharpness.evaluate(image.astype(np.uint16) * 257, roi=roi)
        for key in scores:
            self.assertAlmostEqual(scores[key], high_depth[key], places=5)
        with self.assertRaises(ValueError):
            sharpness.evaluate(image, roi=(60, 60, 20, 20))
        with self.assertRaises(ValueError):
            sharpness.evaluate(np.full((10, 10), np.nan, dtype=np.float32))


class RegistrationTests(unittest.TestCase):
    def test_known_scale_translation_and_common_crop(self):
        reference = texture()
        transform = np.float64([[1.04, 0, -3], [0, 1.04, 2]])
        moved = cv2.warpAffine(reference, transform, (256, 256))
        copies = [moved.copy(), reference.copy()]
        aligned, info = benchmark.align_images([moved, reference], max_size=160)
        inverse = cv2.invertAffineTransform(transform)
        estimated = np.asarray(info["transforms"][0]["matrix"])
        self.assertAlmostEqual(estimated[0, 0], inverse[0, 0], delta=0.01)
        np.testing.assert_allclose(estimated[:, 2], inverse[:, 2], atol=1.0)
        self.assertEqual(estimated[0, 1], 0)
        self.assertEqual(estimated[1, 0], 0)
        self.assertEqual(aligned[0].shape, aligned[1].shape)
        np.testing.assert_array_equal(moved, copies[0])
        np.testing.assert_array_equal(reference, copies[1])
        error = np.mean(np.abs(aligned[0][10:-10, 10:-10].astype(float)
                               - aligned[1][10:-10, 10:-10].astype(float)))
        self.assertLess(error, 20)

    def test_untextured_sequence_fails_explicitly(self):
        image = np.full((64, 64, 3), 100, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Insufficient features"):
            benchmark.align_images([image, image.copy()])

    def test_adjacent_fallback_composes_to_reference(self):
        image = texture(64)
        local = np.float64([[1, 0, 1], [0, 1, 0]])
        direct = np.float64([[1, 0, 2], [0, 1, 0]])
        # Processing order is frame 1, frame 3, then frame 0, with frame 2 as reference.
        with patch.object(benchmark, "_match_transform", side_effect=[
                (direct, 8, 0.2), (direct, 8, 0.2), ValueError("Large focus difference"),
                (local, 8, 0.2)]):
            _, info = benchmark.align_images([image] * 4)
        self.assertEqual(info["transforms"][0]["matched_to_index"], 1)
        self.assertEqual(info["transforms"][0]["matrix"][0][2], 3)
        self.assertIn("Large focus difference", info["transforms"][0]["direct_match_error"])


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        # Synthetic inputs are created under results, never in the real raw-data tree.
        real_results = benchmark.RESULTS_DIR.resolve()
        real_results.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="_tests_", dir=real_results)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(real_results))
        self.addCleanup(self.temporary.cleanup)
        for name, path in (("PROJECT_ROOT", self.root), ("RAW_DIR", self.root / "data/raw"),
                           ("RESULTS_DIR", self.root / "results")):
            patcher = patch.object(benchmark, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tool = benchmark.RAW_DIR / "test_tool"
        for view in benchmark.VIEWS:
            folder = self.tool / view
            folder.mkdir(parents=True)
            benchmark.save_image(folder / "frame_10.png", texture(64))
            benchmark.save_image(folder / "frame_2.png", texture(64))

    def run_cli(self, *args):
        with redirect_stdout(io.StringIO()):
            return benchmark.main(args)

    def test_tool_traversal_csv_placeholders_and_raw_hashes(self):
        before = {p: hashlib.sha256(p.read_bytes()).digest() for p in self.tool.rglob("*.png")}
        self.assertEqual(self.run_cli("--input", str(self.tool)), 0)
        for view in benchmark.VIEWS:
            output = benchmark.RESULTS_DIR / "test_tool" / view
            with (output / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(benchmark.CSV_FIELDS))
                rows = list(reader)
            self.assertEqual([r["status"] for r in rows],
                             ["success", "not_implemented", "not_implemented"])
            self.assertEqual(rows[1]["tenengrad"], "")
            self.assertFalse((output / "method_a.png").exists())
            self.assertIsNotNone(cv2.imdecode(np.fromfile(output / "metrics_comparison.png", np.uint8), 1))
            info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["input_files"], ["frame_2.png", "frame_10.png"])
        after = {p: hashlib.sha256(p.read_bytes()).digest() for p in before}
        self.assertEqual(before, after)

    def test_single_view_and_single_method_chart(self):
        self.assertEqual(self.run_cli("--input", str(self.tool / "face"), "--method", "laplacian"), 0)
        self.assertTrue((benchmark.RESULTS_DIR / "test_tool/face/metrics_comparison.png").is_file())
        self.assertFalse((benchmark.RESULTS_DIR / "test_tool/flank").exists())

    def test_missing_view_records_failure_and_continues(self):
        with patch.object(benchmark, "VIEWS", ("flank", "rake", "face", "missing")):
            self.assertEqual(self.run_cli("--input", str(self.tool), "--method", "laplacian"), 1)
        output = benchmark.RESULTS_DIR / "test_tool/missing"
        with (output / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["tenengrad"], "")
        self.assertIn("load:", row["error"])
        self.assertTrue((output / "metrics_comparison.png").exists())

    def test_rerun_failure_removes_stale_selected_image(self):
        args = ("--input", str(self.tool / "face"), "--method", "laplacian")
        self.assertEqual(self.run_cli(*args), 0)
        with patch.dict(benchmark.METHODS, {"laplacian": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("test failure"))}):
            self.assertEqual(self.run_cli(*args), 1)
        output = benchmark.RESULTS_DIR / "test_tool/face"
        self.assertFalse((output / "laplacian.png").exists())
        with (output / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "failed")
        self.assertIn("fusion:", row["error"])
        self.assertGreaterEqual(float(row["fusion_seconds"]), 0)

    def test_outside_input_and_bad_roi_are_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.resolve_input(self.root)
        self.assertEqual(self.run_cli("--input", str(self.tool / "face"),
                                     "--method", "laplacian", "--roi", "60", "60", "20", "20"), 1)


if __name__ == "__main__":
    unittest.main()
