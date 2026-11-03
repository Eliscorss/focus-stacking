"""Regression tests for fusion, registration, metrics, and benchmark reporting."""

import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import run_benchmark as benchmark
from evaluations import sharpness
from methods import guided_filter, laplacian, laplacian_pyramid


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


def mean_error(image, reference, region):
    return np.abs(image[region].astype(float) - reference[region].astype(float)).mean()


class PyramidTests(unittest.TestCase):
    def test_round_trip_with_odd_sizes(self):
        image = np.random.default_rng(0).uniform(0, 255, (61, 83, 3)).astype(np.float32)
        pyramid = laplacian_pyramid.build_pyramid(image, 4)
        self.assertEqual([level.shape[:2] for level in pyramid],
                         [(61, 83), (31, 42), (16, 21), (8, 11), (4, 6)])
        self.assertLess(pyramid[0].min(), 0)
        np.testing.assert_allclose(laplacian_pyramid.reconstruct(pyramid), image, atol=1e-3)

    def test_parameters_cap_levels_and_reject_invalid_values(self):
        parameters = laplacian_pyramid.resolve_parameters((20, 30, 3))
        self.assertEqual(parameters["requested_levels"], laplacian_pyramid.DEFAULT_PARAMETERS["levels"])
        self.assertEqual(parameters["levels"], 1)
        self.assertEqual(parameters["top_size_hw"], [10, 15])
        self.assertEqual(laplacian_pyramid.resolve_parameters((20, 30), levels=0)["levels"], 0)
        with self.assertRaises(TypeError):
            laplacian_pyramid.resolve_parameters((64, 64), window_size=21)
        for bad in ({"levels": -1}, {"top_window": 4}, {"entropy_bins": 1},
                    {"energy_weights": ((1, 1), (1, 1))}, {"energy_weights": ((0,),)},
                    {"min_top_size": 2}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                laplacian_pyramid.resolve_parameters((64, 64), **bad)
        with self.assertRaises(ValueError):
            laplacian_pyramid.resolve_parameters((7, 64))

    def test_energy_weight_builders(self):
        self.assertEqual(laplacian_pyramid.binomial_weights(1), ((1,),))
        self.assertEqual(laplacian_pyramid.binomial_weights(3), ((1, 2, 1), (2, 4, 2), (1, 2, 1)))
        self.assertEqual(laplacian_pyramid.binomial_weights(5),
                         laplacian_pyramid.DEFAULT_PARAMETERS["energy_weights"])
        self.assertEqual(laplacian_pyramid.binomial_weights(5)[2], (6, 24, 36, 24, 6))
        self.assertEqual(laplacian_pyramid.box_weights(3), ((1, 1, 1),) * 3)
        for build in (laplacian_pyramid.binomial_weights, laplacian_pyramid.box_weights):
            for bad in (0, 4, -3):
                with self.subTest(build=build.__name__, bad=bad), self.assertRaises(ValueError):
                    build(bad)

    def test_region_measures_match_known_values(self):
        flat = np.full((9, 9), 0.25, dtype=np.float32)
        self.assertTrue(np.all(laplacian_pyramid.local_deviation(flat, 3) == 0))
        self.assertTrue(np.all(laplacian_pyramid.local_entropy(flat, 3, 256) == 0))
        weights = np.full((3, 3), 1 / 9)
        np.testing.assert_allclose(laplacian_pyramid.region_energy(flat, weights), 0.0625)
        checker = (np.indices((9, 9)).sum(axis=0) % 2).astype(np.float32)
        expected = -(5 / 9 * np.log2(5 / 9) + 4 / 9 * np.log2(4 / 9))
        self.assertAlmostEqual(laplacian_pyramid.local_entropy(checker, 3, 256)[4, 4], expected, places=9)
        self.assertAlmostEqual(laplacian_pyramid.local_deviation(checker, 3)[4, 4], 20 / 81, places=9)

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
        fused = laplacian_pyramid.stack(images, gray_images=grays)
        self.assertEqual((fused.shape, fused.dtype), (original.shape, original.dtype))
        for region in (np.s_[20:-20, 20:100], np.s_[20:-20, 156:-20]):
            self.assertLess(mean_error(fused, original, region), 0.5)
            self.assertGreater(mean_error(blurred, original, region), 20)
        for image, copy in zip(images, copies):
            np.testing.assert_array_equal(image, copy)
        for gray, copy in zip(grays, gray_copies):
            np.testing.assert_array_equal(gray, copy)

    def test_multi_frame_bands_are_combined_jointly(self):
        original = texture(240)
        blurred = cv2.GaussianBlur(original, (15, 15), 3)
        images = []
        for band in range(3):
            image = blurred.copy()
            image[:, band * 80:(band + 1) * 80] = original[:, band * 80:(band + 1) * 80]
            images.append(image)
        fused = laplacian_pyramid.stack(images)
        for band in range(3):
            region = np.s_[16:-16, band * 80 + 16:(band + 1) * 80 - 16]
            self.assertLess(mean_error(fused, original, region), 0.5)

    def test_single_input_odd_and_uint16_are_preserved(self):
        odd = texture(64)[:61, :63].copy()
        np.testing.assert_array_equal(laplacian_pyramid.stack([odd]), odd)
        high_depth = texture(64).astype(np.uint16) * 257
        fused = laplacian_pyramid.stack([high_depth])
        self.assertEqual(fused.dtype, np.uint16)
        np.testing.assert_array_equal(fused, high_depth)

    def test_flat_tie_keeps_first_color_source(self):
        first = np.full((30, 30, 3), (20, 40, 60), dtype=np.uint8)
        second = np.full((30, 30, 3), (80, 100, 120), dtype=np.uint8)
        np.testing.assert_array_equal(laplacian_pyramid.stack([first, second]), first)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            laplacian_pyramid.stack([])
        with self.assertRaises(ValueError):
            laplacian_pyramid.stack([texture(64), texture(80)])
        with self.assertRaises(ValueError):
            laplacian_pyramid.stack([texture(64)[:5, :5].copy()])
        with self.assertRaises(TypeError):
            laplacian_pyramid.stack([texture(64)], window_size=21)


def reference_box_sum(values, radius):
    """Window sums truncated at the image border, via cumulative sums like He's boxfilter.m."""
    height, width = values.shape
    total = np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    rows, columns = np.arange(height), np.arange(width)
    top, bottom = np.clip(rows - radius, 0, height), np.clip(rows + radius + 1, 0, height)
    left, right = np.clip(columns - radius, 0, width), np.clip(columns + radius + 1, 0, width)
    return (total[np.ix_(bottom, right)] - total[np.ix_(top, right)]
            - total[np.ix_(bottom, left)] + total[np.ix_(top, left)])


def reference_guided_filter(guide, src, radius, eps):
    """Direct translation of guidedfilter_LKH.m."""
    counts = reference_box_sum(np.ones(guide.shape), radius)
    mean_i = reference_box_sum(guide, radius) / counts
    mean_p = reference_box_sum(src, radius) / counts
    cov_ip = reference_box_sum(guide * src, radius) / counts - mean_i * mean_p
    var_i = reference_box_sum(guide * guide, radius) / counts - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return (reference_box_sum(a, radius) / counts * guide
            + reference_box_sum(b, radius) / counts)


def bwareaopen(mask, min_area):
    """MATLAB bwareaopen with 8-connectivity: drop regions with fewer than min_area pixels."""
    count, regions, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[regions]


def reference_gfdf(first, second):
    """Literal two-image main.m: B wins ties and the final weights are not clipped."""
    w, r, eps, ratio = 7, 5, 0.3, 0.01
    grays = [cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2GRAY).astype(np.float64)
             for image in (first, second)]
    scores = []
    for gray in grays:
        # imfilter(gray, fspecial('average', w), 'replicate')
        padded = np.pad(gray, w // 2, mode="edge")
        mean = reference_box_sum(padded, w // 2)[w // 2:-(w // 2), w // 2:-(w // 2)] / (w * w)
        scores.append(reference_guided_filter(gray, np.abs(gray - mean), r, eps))
    initial = scores[0] > scores[1]
    area = int(np.ceil(ratio * initial.size))
    decision = ~bwareaopen(~bwareaopen(initial, area), area)
    fused_gray = np.where(decision, grays[0], grays[1])
    weight = reference_guided_filter(fused_gray, decision.astype(np.float64), r, eps)
    fused = weight[..., None] * first + (1 - weight[..., None]) * second
    return fused, weight, scores, initial, decision


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()


class GuidedFilterTests(unittest.TestCase):
    def test_filter_identities_and_border_normalization(self):
        rng = np.random.default_rng(1)
        guide = rng.uniform(0, 1, (13, 17))
        np.testing.assert_allclose(guided_filter.guided_filter(guide, np.full((13, 17), 0.7), 5, 0.3),
                                   0.7, atol=1e-12)
        labels = rng.integers(0, 3, (13, 17))
        total = sum(guided_filter.guided_filter(guide, labels == k, 3, 0.01) for k in range(3))
        np.testing.assert_allclose(total, 1, atol=1e-12)
        # Radius 5 exceeds the 5x7 image, so every window is truncated at the border.
        small_guide, small_src = rng.uniform(0, 1, (5, 7)), rng.uniform(0, 1, (5, 7))
        for radius in (1, 2, 5):
            with self.subTest(radius=radius):
                np.testing.assert_allclose(
                    guided_filter.guided_filter(small_guide, small_src, radius, 0.1),
                    reference_guided_filter(small_guide, small_src, radius, 0.1), atol=1e-12)

    def test_matches_literal_two_image_reference(self):
        original = texture(192)
        blurred = cv2.GaussianBlur(original, (9, 9), 2)
        first, second = original.copy(), original.copy()
        first[:, 96:] = blurred[:, 96:]
        second[:, :96] = blurred[:, :96]
        # Islands on both sides that end up below ceil(0.01 * 192 * 192) = 369 pixels.
        first[80:98, 140:158] = original[80:98, 140:158]
        second[80:98, 140:158] = blurred[80:98, 140:158]
        second[120:138, 30:48] = original[120:138, 30:48]
        first[120:138, 30:48] = blurred[120:138, 30:48]
        fused, weight, scores, initial, decision = reference_gfdf(first, second)
        # Preconditions: no score ties, both islands removed, weights within [0, 1].
        self.assertGreater(np.abs(scores[0] - scores[1]).min(), 1e-9)
        self.assertTrue(initial[89, 149] and not decision[89, 149])
        self.assertTrue(not initial[129, 39] and decision[129, 39])
        self.assertGreater(weight.min(), -1e-12)
        self.assertLess(weight.max(), 1 + 1e-12)
        expected = np.rint(np.clip(fused, 0, 255)).astype(np.uint8)
        np.testing.assert_array_equal(guided_filter.stack([first, second]), expected)

    def test_complementary_focus_and_input_immutability(self):
        original = texture()
        blurred = cv2.GaussianBlur(original, (15, 15), 3)
        first, second = original.copy(), original.copy()
        first[:, 128:] = blurred[:, 128:]
        second[:, :128] = blurred[:, :128]
        images = [first, second]
        grays = benchmark.to_gray_images(images)
        before = [array_hash(array) for array in [*images, *grays]]
        for array in [*images, *grays]:
            array.setflags(write=False)
        fused = guided_filter.stack(images, gray_images=grays)
        self.assertEqual((fused.shape, fused.dtype), (original.shape, original.dtype))
        for region in (np.s_[20:-20, 20:100], np.s_[20:-20, 156:-20]):
            self.assertLess(mean_error(fused, original, region), 0.5)
            self.assertGreater(mean_error(blurred, original, region), 20)
        self.assertEqual([array_hash(array) for array in [*images, *grays]], before)

    def test_multi_frame_bands_are_combined_jointly(self):
        original = texture(240)
        blurred = cv2.GaussianBlur(original, (15, 15), 3)
        images = []
        for band in range(3):
            image = blurred.copy()
            image[:, band * 80:(band + 1) * 80] = original[:, band * 80:(band + 1) * 80]
            images.append(image)
        fused = guided_filter.stack(images)
        for band in range(3):
            region = np.s_[16:-16, band * 80 + 16:(band + 1) * 80 - 16]
            self.assertLess(mean_error(fused, original, region), 0.5)

    def test_single_input_odd_tiny_and_uint16(self):
        odd = texture(64)[:61, :63].copy()
        np.testing.assert_array_equal(guided_filter.stack([odd]), odd)
        high_depth = texture(64).astype(np.uint16) * 257
        np.testing.assert_array_equal(guided_filter.stack([high_depth]), high_depth)
        fused = guided_filter.stack([high_depth, cv2.GaussianBlur(high_depth, (5, 5), 1)])
        self.assertEqual((fused.shape, fused.dtype), (high_depth.shape, np.uint16))
        rng = np.random.default_rng(3)
        tiny = [rng.integers(0, 256, (5, 7, 3), dtype=np.uint8) for _ in range(3)]
        fused = guided_filter.stack(tiny)
        self.assertEqual((fused.shape, fused.dtype), (tiny[0].shape, np.uint8))

    def test_flat_and_identical_inputs_keep_first_frame(self):
        first = np.full((30, 30, 3), (20, 40, 60), dtype=np.uint8)
        second = np.full((30, 30, 3), (80, 100, 120), dtype=np.uint8)
        np.testing.assert_array_equal(guided_filter.stack([first, second, second]), first)
        image = texture(64)
        np.testing.assert_array_equal(guided_filter.stack([image, image.copy()]), image)

    def test_clipped_weights_are_valid_on_degenerate_guides(self):
        rng = np.random.default_rng(2)
        labels = rng.integers(0, 3, (24, 31))
        checker = (np.indices((24, 31)).sum(axis=0) % 2).astype(np.float64)
        for guide in (np.zeros((24, 31)), np.ones((24, 31)), checker):
            weights = [np.maximum(guided_filter.guided_filter(guide, labels == k, 5, 0.3), 0)
                       for k in range(3)]
            total = sum(weights)
            self.assertTrue(np.isfinite(total).all())
            self.assertGreater(total.min(), 0)
            normalized = [weight / total for weight in weights]
            for weight in normalized:
                self.assertGreaterEqual(weight.min(), 0)
                self.assertLessEqual(weight.max(), 1)
            np.testing.assert_allclose(sum(normalized), 1, atol=1e-12)

    def test_small_regions_join_ring_majority(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[:, :5] = 1
        labels[4:6, 4:6] = 2   # Ring: six pixels of label 0, six of label 1.
        labels[1, 8] = 1       # Ring: label 0 only.
        copy = labels.copy()
        cleaned = guided_filter.remove_small_regions(labels, 3, 20)
        np.testing.assert_array_equal(labels, copy)
        expected = copy.copy()
        expected[4:6, 4:6] = 0
        expected[1, 8] = 0
        np.testing.assert_array_equal(cleaned, expected)
        np.testing.assert_array_equal(guided_filter.remove_small_regions(labels, 3, 0), copy)
        self.assertEqual(guided_filter.resolve_parameters((10, 10), small_region_ratio=0)
                         ["min_area_pixels"], 0)

    def test_parameters_and_invalid_input(self):
        parameters = guided_filter.resolve_parameters((100, 100, 3))
        self.assertEqual(parameters["min_area_pixels"], 100)
        for key, value in guided_filter.DEFAULT_PARAMETERS.items():
            self.assertEqual(parameters[key], value)
        with self.assertRaises(TypeError):
            guided_filter.resolve_parameters((64, 64), window_size=21)
        with self.assertRaises(TypeError):
            guided_filter.stack([texture(64)], window_size=21)
        for bad in ({"mean_window": 4}, {"mean_window": 0}, {"focus_radius": 0},
                    {"weight_radius": 2.5}, {"focus_radius": True}, {"focus_eps": 0},
                    {"weight_eps": -1}, {"focus_eps": float("nan")},
                    {"small_region_ratio": 1}, {"small_region_ratio": -0.1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                guided_filter.resolve_parameters((64, 64), **bad)
        with self.assertRaises(ValueError):
            guided_filter.stack([])
        with self.assertRaises(ValueError):
            guided_filter.stack([texture(64), texture(80)])
        with self.assertRaises(ValueError):
            guided_filter.stack([texture(64)], gray_images=[np.zeros((64, 64), dtype=np.float64)])


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
        self.views = ("view1", "view2", "view10")
        for view in reversed(self.views):
            folder = self.tool / view
            folder.mkdir(parents=True)
            benchmark.save_image(folder / "frame_10.png", texture(64))
            benchmark.save_image(folder / "frame_2.png", texture(64))

    def run_cli(self, *args):
        with redirect_stdout(io.StringIO()):
            return benchmark.main(args)

    def test_tool_traversal_csv_and_raw_hashes(self):
        before = {p: hashlib.sha256(p.read_bytes()).digest() for p in self.tool.rglob("*.png")}
        self.assertEqual(self.run_cli("--input", str(self.tool)), 0)
        for view in self.views:
            output = benchmark.RESULTS_DIR / "test_tool" / view
            with (output / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(benchmark.CSV_FIELDS))
                rows = list(reader)
            self.assertEqual([r["method"] for r in rows], list(benchmark.METHODS))
            self.assertEqual([r["status"] for r in rows], ["success"] * 3)
            for method in benchmark.METHODS:
                self.assertTrue((output / f"{method}.png").is_file())
            self.assertIsNotNone(cv2.imdecode(np.fromfile(output / "metrics_comparison.png", np.uint8), 1))
            info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["input_files"], ["frame_2.png", "frame_10.png"])
            self.assertEqual(info["parameters"]["laplacian"], {"window_size": 21, "smooth_size": 5})
            effective = info["parameters"]["laplacian_pyramid"]
            self.assertEqual(effective["requested_levels"], laplacian_pyramid.DEFAULT_PARAMETERS["levels"])
            self.assertLessEqual(effective["levels"], effective["requested_levels"])
            effective = info["parameters"]["guided_filter"]
            self.assertEqual(effective["focus_radius"], guided_filter.DEFAULT_PARAMETERS["focus_radius"])
            self.assertEqual(effective["min_area_pixels"], math.ceil(0.01 * 64 * 64))
        after = {p: hashlib.sha256(p.read_bytes()).digest() for p in before}
        self.assertEqual(before, after)

    def test_placeholder_method_is_reported_without_image(self):
        with patch.object(benchmark, "UNIMPLEMENTED", {"guided_filter"}):
            self.assertEqual(self.run_cli("--input", str(self.tool / "view1")), 0)
        output = benchmark.RESULTS_DIR / "test_tool/view1"
        rows = self.read_rows(output)
        self.assertEqual([r["status"] for r in rows], ["success", "not_implemented", "success"])
        self.assertEqual(rows[1]["tenengrad"], "")
        self.assertFalse((output / "guided_filter.png").exists())
        info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
        self.assertNotIn("guided_filter", info["parameters"])

    def test_single_view_and_single_method_chart(self):
        self.assertEqual(self.run_cli("--input", str(self.tool / "view2"), "--method", "laplacian"), 0)
        self.assertTrue((benchmark.RESULTS_DIR / "test_tool/view2/metrics_comparison.png").is_file())
        self.assertFalse((benchmark.RESULTS_DIR / "test_tool/view1").exists())

    def test_empty_view_records_failure_and_continues(self):
        (self.tool / "view3").mkdir()
        self.assertEqual(self.run_cli("--input", str(self.tool), "--method", "laplacian"), 1)
        output = benchmark.RESULTS_DIR / "test_tool/view3"
        with (output / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["tenengrad"], "")
        self.assertIn("load:", row["error"])
        self.assertTrue((output / "metrics_comparison.png").exists())
        self.assertTrue((benchmark.RESULTS_DIR / "test_tool/view10/laplacian.png").exists())

    def test_numbered_view_discovery_and_relative_input(self):
        for name in ("flank", "rake", "face", "view0", "view01", "view_1", "View4", "notes"):
            (self.tool / name).mkdir()
        (self.tool / "view5").write_text("Not a directory.", encoding="utf-8")
        tool, views = benchmark.resolve_input("data/raw/test_tool")
        self.assertEqual(tool, "test_tool")
        self.assertEqual([path.name for path in views], list(self.views))
        self.assertEqual(benchmark.resolve_input("data/raw/test_tool/view10"),
                         ("test_tool", [self.tool / "view10"]))

    def test_old_and_invalid_view_names_are_rejected(self):
        for name in ("flank", "rake", "face", "view0", "view01", "view_1", "View4"):
            with self.subTest(name=name):
                folder = self.tool / name
                folder.mkdir()
                with self.assertRaisesRegex(ValueError, "view<N>"):
                    benchmark.resolve_input(folder)
        old_tool = benchmark.RAW_DIR / "old_tool"
        (old_tool / "face").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "No view<N> directories"):
            benchmark.resolve_input(old_tool)

    def test_rerun_failure_removes_stale_selected_image(self):
        args = ("--input", str(self.tool / "view1"), "--method", "laplacian")
        self.assertEqual(self.run_cli(*args), 0)
        with patch.dict(benchmark.METHODS, {"laplacian": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("test failure"))}):
            self.assertEqual(self.run_cli(*args), 1)
        output = benchmark.RESULTS_DIR / "test_tool/view1"
        self.assertFalse((output / "laplacian.png").exists())
        with (output / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "failed")
        self.assertIn("fusion:", row["error"])
        self.assertGreaterEqual(float(row["fusion_seconds"]), 0)

    def read_rows(self, output):
        with (output / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_pyramid_sweep_writes_separate_variants(self):
        view = benchmark.RESULTS_DIR / "test_tool/view1"
        self.assertEqual(self.run_cli("--input", str(self.tool / "view1"),
                                      "--pyramid-sweep", "levels", "1", "3"), 0)
        output = view / "sweep_levels"
        names = ["laplacian", "guided_filter", "laplacian_pyramid_levels1", "laplacian_pyramid_levels3"]
        rows = self.read_rows(output)
        self.assertEqual([row["method"] for row in rows], names)
        self.assertEqual([row["status"] for row in rows], ["success"] * 4)
        for name in names:
            self.assertTrue((output / f"{name}.png").is_file())
        info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["sweep"], {"parameter": "levels", "values": [1, 3]})
        for value in (1, 3):
            effective = info["parameters"][f"laplacian_pyramid_levels{value}"]
            self.assertEqual(effective["requested_levels"], value)
            self.assertEqual(effective["top_window"],
                             laplacian_pyramid.DEFAULT_PARAMETERS["top_window"])
        # The regular benchmark folder receives no sweep output.
        self.assertFalse((view / "benchmark.csv").exists())

        self.assertEqual(self.run_cli("--input", str(self.tool / "view1"), "--method",
                                      "laplacian_pyramid", "--pyramid-sweep", "levels", "2"), 0)
        self.assertEqual([row["method"] for row in self.read_rows(output)],
                         ["laplacian_pyramid_levels2"])
        self.assertEqual(sorted(path.name for path in output.glob("laplacian_pyramid_*.png")),
                         ["laplacian_pyramid_levels2.png"])

    def test_energy_sweep_records_effective_matrices(self):
        self.assertEqual(self.run_cli("--input", str(self.tool / "view1"), "--method",
                                      "laplacian_pyramid", "--pyramid-sweep", "energy_size", "1", "5"), 0)
        output = benchmark.RESULTS_DIR / "test_tool/view1/sweep_energy_size"
        names = ["laplacian_pyramid_energy_size1", "laplacian_pyramid_energy_size5"]
        self.assertEqual([row["method"] for row in self.read_rows(output)], names)
        info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["sweep"], {"parameter": "energy_size", "values": [1, 5]})
        self.assertEqual(info["parameters"][names[0]]["energy_weights"], [[1.0]])
        self.assertEqual(info["parameters"][names[1]]["energy_weights"][0], [1.0, 4.0, 6.0, 4.0, 1.0])
        self.assertEqual(info["parameters"][names[1]]["top_window"],
                         laplacian_pyramid.DEFAULT_PARAMETERS["top_window"])

        self.assertEqual(self.run_cli("--input", str(self.tool / "view1"), "--method",
                                      "laplacian_pyramid", "--pyramid-sweep", "energy_box", "3"), 0)
        info = json.loads((output.parent / "sweep_energy_box/run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["parameters"]["laplacian_pyramid_energy_box3"]["energy_weights"],
                         [[1.0] * 3] * 3)

    def test_invalid_pyramid_sweeps_are_rejected(self):
        view = str(self.tool / "view1")
        # energy_size 9 exceeds the default min_top_size of 8.
        for sweep in (["energy_weights", "3"], ["levels"], ["levels", "x"], ["levels", "3", "3"],
                      ["top_window", "4"], ["levels", "-1"], ["energy_size", "4"],
                      ["energy_size", "9"], ["energy_box", "0"]):
            with self.subTest(sweep=sweep), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.run_cli("--input", view, "--pyramid-sweep", *sweep)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli("--input", view, "--method", "laplacian", "--pyramid-sweep", "levels", "3")

    def test_multiple_inputs_run_once_per_view(self):
        self.assertEqual(self.run_cli("--input", str(self.tool / "view2"), str(self.tool / "view1"),
                                      str(self.tool / "view2"), "--method", "laplacian"), 0)
        for view in ("view1", "view2"):
            self.assertEqual(len(self.read_rows(benchmark.RESULTS_DIR / "test_tool" / view)), 1)
        self.assertFalse((benchmark.RESULTS_DIR / "test_tool/view10").exists())

    def test_outside_input_and_bad_roi_are_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.resolve_input(self.root)
        self.assertEqual(self.run_cli("--input", str(self.tool / "view1"),
                                     "--method", "laplacian", "--roi", "60", "60", "20", "20"), 1)


if __name__ == "__main__":
    unittest.main()
