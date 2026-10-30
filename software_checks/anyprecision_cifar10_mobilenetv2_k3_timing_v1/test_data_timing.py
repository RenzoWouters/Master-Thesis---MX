"""CPU-only checks; mock fixtures are never reachable by the production runner."""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import data_support as data
from timing_analysis import summarize_timing


class _MockOfficialSplit(Dataset):
    """Only a unittest substitute for already checksum-audited split structure."""

    def __init__(self, root, train, transform, download=False):
        if download:
            raise AssertionError("Mock dataset must never download.")
        self.root, self.train, self.transform = root, train, transform
        self.size = 50_000 if train else 10_000
        self.data = SimpleNamespace(shape=(self.size, 32, 32, 3))
        self.targets = [index % 10 for index in range(self.size)]
        self.classes = [str(index) for index in range(10)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = Image.fromarray(np.full((32, 32, 3), index % 256, dtype=np.uint8))
        return self.transform(image), self.targets[index]


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _mock_files(self):
        folder = self.root / data.DATASET_FOLDER
        folder.mkdir()
        expected = {}
        for i, name in enumerate(data.EXPECTED_MD5):
            payload = f"fixture {i} {name}".encode()
            (folder / name).write_bytes(payload)
            expected[name] = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        return expected

    def test_discovery_accepts_dataset_folder_itself(self):
        folder = self.root / "data" / "cifar10" / data.DATASET_FOLDER
        folder.mkdir(parents=True)
        self.assertEqual(data.find_data_root(self.root), folder.parent)
        self.assertEqual(data.find_data_root(self.root, folder), folder.parent)

    def test_missing_discovery_is_read_only_and_explicit(self):
        with self.assertRaisesRegex(data.CIFAR10DataError, "--data-root"):
            data.find_data_root(self.root)
        self.assertEqual(list(self.root.iterdir()), [])
        proposed = data.find_data_root(self.root, allow_missing=True)
        self.assertEqual(proposed, self.root / "data" / "cifar10")
        self.assertFalse(proposed.exists())

    def test_audit_includes_metadata_and_changes_no_bytes(self):
        expected = self._mock_files()
        files = list((self.root / data.DATASET_FOLDER).iterdir())
        before = {path.name: path.read_bytes() for path in files}
        with patch.object(data, "EXPECTED_MD5", expected):
            result = data.audit_cifar10(self.root)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["files"]), 7)
        self.assertIn("batches.meta", {row["name"] for row in result["files"]})
        self.assertEqual(before, {path.name: path.read_bytes() for path in files})

    def test_corrupt_metadata_is_not_repaired_even_with_download(self):
        expected = self._mock_files()
        meta = self.root / data.DATASET_FOLDER / "batches.meta"
        meta.write_bytes(b"corrupt metadata fixture")
        before = meta.read_bytes()
        with patch.object(data, "EXPECTED_MD5", expected), patch.object(data, "CIFAR10") as dataset:
            with self.assertRaisesRegex(data.CIFAR10DataError, "batches.meta"):
                data._prepare_data(self.root, download=True)
            dataset.assert_not_called()
        self.assertEqual(meta.read_bytes(), before)

    def test_existing_valid_data_avoid_downloader(self):
        expected = self._mock_files()
        with patch.object(data, "EXPECTED_MD5", expected), patch.object(data, "CIFAR10") as dataset:
            result = data._prepare_data(self.root, download=True)
            dataset.assert_not_called()
        self.assertFalse(result["downloaded"])

    def test_missing_data_never_implicitly_download(self):
        with patch.object(data, "CIFAR10") as dataset:
            with self.assertRaises(data.CIFAR10DataError):
                data.make_loaders(self.root)
            dataset.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_official_normalization(self):
        _training, validation = data.make_transforms()
        rgb = np.array([0, 128, 255], dtype=np.uint8)
        image = Image.fromarray(np.tile(rgb, (32, 32, 1)))
        result = validation(image)
        expected = (torch.tensor([0., 128. / 255., 1.])
                    - torch.tensor([125.3, 123.0, 113.9]) / 255.) / (torch.tensor([63.0, 62.1, 66.7]) / 255.)
        torch.testing.assert_close(result[:, 0, 0], expected, rtol=1e-6, atol=1e-6)

    def test_seeded_prefix_and_balanced_validation_without_global_rng_change(self):
        before = torch.random.get_rng_state().clone()
        with patch.object(data, "audit_cifar10", return_value={"status": "PASS"}), patch.object(data, "CIFAR10", _MockOfficialSplit):
            train, validation, info = data.make_loaders(self.root)
            train_again, validation_again, info_again = data.make_loaders(self.root)
            different, _validation_other, info_other = data.make_loaders(self.root, seed=20261031)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(len(train), 100)
        self.assertEqual(len(validation), 8)
        self.assertEqual(len(set(train.dataset.indices)), 12_800)
        self.assertEqual(train.dataset.indices, train_again.dataset.indices)
        self.assertEqual(validation.dataset.indices, validation_again.dataset.indices)
        self.assertNotEqual(train.dataset.indices, different.dataset.indices)
        self.assertEqual(info["train_indices_sha256"], info_again["train_indices_sha256"])
        self.assertNotEqual(info["train_indices_sha256"], info_other["train_indices_sha256"])
        counts = {label: 0 for label in range(10)}
        for index in validation.dataset.indices:
            counts[validation.dataset.dataset.targets[index]] += 1
        self.assertEqual(counts, {label: 100 for label in range(10)})
        self.assertEqual(info["full_train_batches"], 391)
        self.assertEqual(info["full_validation_batches"], 79)
        self.assertEqual(validation.batch_sampler.drop_last, False)
        self.assertEqual(len(list(validation.batch_sampler)[-1]), 104)

    def test_unexpected_split_size_and_labels_stop_run(self):
        dataset = _MockOfficialSplit(str(self.root), True, lambda x: x)
        dataset.targets[0] = 10
        with self.assertRaisesRegex(data.CIFAR10DataError, "labels"):
            data._check_loaded_split(dataset, 50_000, "train")
        dataset = _MockOfficialSplit(str(self.root), True, lambda x: x)
        dataset.size = 49_999
        with self.assertRaisesRegex(data.CIFAR10DataError, "afmetingen"):
            data._check_loaded_split(dataset, 50_000, "train")


def _row(step, wall, *, compute=None, batch=128):
    return {"step": step, "batch_size": batch,
            "wall_s": float(wall), "compute_s": float(wall if compute is None else compute),
            "data_wait_s": 0.0, "h2d_s": 0.0, "precision_passes": 5}


class TimingTests(unittest.TestCase):
    def test_end_to_end_epoch_estimate_no_second_precision_multiplier(self):
        training = [_row(i + 1, 50 if i < 10 else 2) for i in range(100)]
        validation = [_row(1, 30)] + [_row(i + 2, 1) for i in range(6)] + [_row(8, .8, batch=104)]
        result = summarize_timing(training, validation, checkpoint_seconds=2)
        self.assertEqual(result["training"]["eligible_rows"], 90)
        self.assertEqual(result["validation"]["eligible_rows"], 6)
        self.assertEqual(result["validation"]["partial_rows_not_used_for_forecast"], 1)
        self.assertEqual(result["per_epoch"]["training_seconds"], 391 * 2)
        self.assertEqual(result["per_epoch"]["validation_seconds"], 79)
        self.assertEqual(result["per_epoch"]["total_seconds"], 863)
        self.assertAlmostEqual(result["projections"]["400"]["total_hours"], 863 * 400 / 3600)
        self.assertTrue(result["estimate_complete"])

    def test_synthetic_compute_excludes_data_validation_and_checkpoints(self):
        result = summarize_timing([_row(1, 10, compute=1.5), _row(2, 10, compute=1.5)],
                                  warmup_train_steps=0, checkpoint_seconds=5, synthetic=True)
        self.assertEqual(result["kind"], "synthetic_compute_only")
        self.assertEqual(result["per_epoch"]["training_seconds"], 391 * 1.5)
        self.assertIsNone(result["per_epoch"]["total_seconds"])
        self.assertIsNone(result["projections"]["400"]["total_hours"])
        self.assertIsNone(result["projections"]["400"]["checkpoint_hours"])
        self.assertFalse(result["estimate_complete"])

    def test_no_eligible_rows_does_not_invent_estimates(self):
        result = summarize_timing([_row(1, 1)], [])
        self.assertFalse(result["training"]["available"])
        self.assertIsNone(result["projections"]["5"]["training_hours"])
        self.assertIsNone(result["projections"]["5"]["total_hours"])

    def test_quantile_scenarios_are_descriptive(self):
        result = summarize_timing([_row(i + 1, t) for i, t in enumerate((1, 2, 3))],
                                  [_row(1, 1), _row(2, 1)], warmup_train_steps=0,
                                  warmup_validation_steps=0)
        self.assertAlmostEqual(result["training"]["stats"]["wall_s"]["p10"], 1.2)
        self.assertAlmostEqual(result["training"]["stats"]["wall_s"]["p90"], 2.8)
        self.assertAlmostEqual(result["per_epoch"]["scenario_low_seconds"], 391 * 1.2 + 79)
        self.assertIn("not a confidence interval", result["scenario_definition"])

    def test_invalid_measurements_are_rejected(self):
        for field, value in [("wall_s", float("nan")), ("compute_s", -1),
                             ("data_wait_s", float("inf")), ("precision_passes", 1)]:
            row = _row(1, 1)
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_timing([row], warmup_train_steps=0)


def run_tests():
    stream = io.StringIO()
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(DataTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(TimingTests),
    ])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL",
              "tests_run": result.testsRun, "output": stream.getvalue(),
              "scope": "CPU fixtures only; no real dataset download or training"}
    if not result.wasSuccessful():
        raise AssertionError(report["output"])
    return report


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
