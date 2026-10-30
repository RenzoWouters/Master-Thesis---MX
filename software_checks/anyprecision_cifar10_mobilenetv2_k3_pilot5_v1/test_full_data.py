"""CPU fixtures for complete-epoch coverage and restartable data order.

No CIFAR images are downloaded or loaded by these tests.  The mock split is
available only through unittest patching; production always audits real data.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

import data_support as data


def _audit_fixture(root="fixture"):
    return {
        "status": "PASS", "root": str(root), "dataset_folder": str(root),
        "files": [{"name": name, "bytes": 1, "expected_md5": digest,
                   "actual_md5": digest, "matches": True}
                  for name, digest in data.EXPECTED_MD5.items()],
        "downloaded": False, "modified_existing_data": False,
    }


class _IndexSplit(Dataset):
    """Each returned value exposes the dataset index, without image work."""
    reads = 0
    events = []

    def __init__(self, root, train, transform, download=False):
        if download:
            raise AssertionError("The fixture never downloads.")
        type(self).events.append("train" if train else "validation")
        self.size = data.TRAIN_SIZE if train else data.VALIDATION_SIZE
        self.data = SimpleNamespace(shape=(self.size, 32, 32, 3))
        self.targets = [i % 10 for i in range(self.size)]
        self.classes = [str(i) for i in range(10)]
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        type(self).reads += 1
        return index, self.targets[index]


def _collect_indices(loader):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pin_memory.*")
        batches = [indices.tolist() for indices, _targets in loader]
    return [index for batch in batches for index in batch], [len(batch) for batch in batches]


class FullDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        _IndexSplit.reads = 0
        _IndexSplit.events = []

    def tearDown(self):
        self.temporary.cleanup()

    def _loaders(self):
        with patch.object(data, "_prepare_data", return_value=_audit_fixture(self.root)), \
                patch.object(data, "CIFAR10", _IndexSplit):
            return data.make_full_loaders(self.root)

    def _files(self, root, *, metadata=True):
        folder = root / data.DATASET_FOLDER
        folder.mkdir(parents=True)
        expected = {}
        for name in data.EXPECTED_MD5:
            if not metadata and name == "batches.meta":
                continue
            payload = ("test only: " + name).encode("ascii")
            (folder / name).write_bytes(payload)
            expected[name] = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        return expected

    def test_each_official_example_once_per_epoch_and_partial_batches_kept(self):
        train, validation, info, train_rng, val_rng = self._loaders()
        train_indices, train_batches = _collect_indices(train)
        val_indices, val_batches = _collect_indices(validation)
        self.assertEqual(len(train_indices), 50_000)
        self.assertEqual(sorted(train_indices), list(range(50_000)))
        self.assertEqual(val_indices, list(range(10_000)))
        self.assertEqual((len(train_batches), train_batches[-1]), (391, 80))
        self.assertEqual((len(val_batches), val_batches[-1]), (79, 16))
        self.assertEqual(info["train_examples"], 50_000)
        self.assertEqual(info["validation_examples"], 10_000)
        self.assertFalse(train.drop_last)
        self.assertFalse(validation.drop_last)
        self.assertIs(train.generator, train_rng)
        self.assertIs(validation.generator, val_rng)

    def test_epoch_shuffle_changes_and_restored_loader_generators_replay_next_epoch(self):
        train, validation, _info, train_rng, val_rng = self._loaders()
        first_train, _ = _collect_indices(train)
        _collect_indices(validation)
        saved_train_rng, saved_val_rng = train_rng.get_state().clone(), val_rng.get_state().clone()
        next_train, _ = _collect_indices(train)
        next_validation, _ = _collect_indices(validation)
        next_train_rng, next_val_rng = train_rng.get_state().clone(), val_rng.get_state().clone()
        self.assertNotEqual(first_train, next_train)

        resumed_train, resumed_val, _info, resumed_train_rng, resumed_val_rng = self._loaders()
        resumed_train_rng.set_state(saved_train_rng)
        resumed_val_rng.set_state(saved_val_rng)
        actual_train, _ = _collect_indices(resumed_train)
        actual_val, _ = _collect_indices(resumed_val)
        self.assertEqual(next_train, actual_train)
        self.assertEqual(next_validation, actual_val)
        self.assertTrue(torch.equal(next_train_rng, resumed_train_rng.get_state()))
        self.assertTrue(torch.equal(next_val_rng, resumed_val_rng.get_state()))

    def test_construction_audits_before_loading_and_consumes_no_global_rng_or_images(self):
        before_torch = torch.get_rng_state().clone()
        before_numpy = np.random.get_state()
        before_python = random.getstate()

        def audit(root, *, download):
            self.assertFalse(download)
            _IndexSplit.events.append("audit")
            return _audit_fixture(root)

        with patch.object(data, "_prepare_data", side_effect=audit), \
                patch.object(data, "CIFAR10", _IndexSplit):
            data.make_full_loaders(self.root)
        self.assertEqual(_IndexSplit.events, ["audit", "train", "validation"])
        self.assertEqual(_IndexSplit.reads, 0)
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])

    def test_discovery_uses_complete_new_copy_before_legacy_missing_metadata(self):
        legacy = self.root / "data" / "cifar10"
        complete = self.root / "data" / "cifar10_mobilenet_complete"
        self._files(legacy, metadata=False)
        self._files(complete)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(data.find_data_root(self.root), complete)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_complete_legacy_copy_beats_an_incomplete_preferred_copy(self):
        preferred = self.root / "data" / "cifar10_mobilenet_complete"
        legacy = self.root / "data" / "cifar10"
        self._files(preferred, metadata=False)
        self._files(legacy)
        self.assertEqual(data.find_data_root(self.root), legacy)

    def test_explicit_incomplete_root_does_not_fall_back_or_get_repaired(self):
        incomplete = self.root / "data" / "cifar10"
        self._files(incomplete, metadata=False)
        complete = self.root / "data" / "cifar10_mobilenet_complete"
        self._files(complete)
        self.assertEqual(data.find_data_root(self.root, incomplete), incomplete)
        with patch.object(data, "CIFAR10") as downloader:
            with self.assertRaises(data.CIFAR10DataError):
                data.make_full_loaders(incomplete, download=True)
            downloader.assert_not_called()
        self.assertFalse((incomplete / data.DATASET_FOLDER / "batches.meta").exists())
        absent = self.root / "explicit_missing"
        with self.assertRaises(data.CIFAR10DataError):
            data.find_data_root(self.root, absent)
        self.assertEqual(data.find_data_root(self.root, absent, allow_missing=True), absent)
        self.assertFalse(absent.exists())

    def test_dataset_identity_is_content_based_and_independent_of_location(self):
        source = self.root / "source"
        copy = self.root / "copy"
        checksums = self._files(source)
        self._files(copy)
        with patch.object(data, "EXPECTED_MD5", checksums):
            first = data._dataset_identity(data.audit_cifar10(source))
            second = data._dataset_identity(data.audit_cifar10(copy))
        self.assertEqual(first, second)
        self.assertEqual(len(first["files"]), 7)
        self.assertNotIn(str(source), json.dumps(first))
        self.assertNotIn(str(copy), json.dumps(second))
        without_hash = {key: value for key, value in first.items() if key != "sha256"}
        digest = hashlib.sha256(json.dumps(without_hash, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(first["sha256"], digest)

    def test_dataset_identity_rejects_incomplete_or_unverified_audits(self):
        for audit in ({"status": "PASS"}, {"status": "FAIL", "files": []}):
            with self.assertRaises(data.CIFAR10DataError):
                data._dataset_identity(audit)
        invalid = _audit_fixture()
        invalid["files"][0]["actual_md5"] = "0" * 32
        with self.assertRaises(data.CIFAR10DataError):
            data._dataset_identity(invalid)

    def test_workers_must_be_zero_before_any_data_action(self):
        with patch.object(data, "_prepare_data") as prepare:
            for workers in (1, 2, -1, True, 0.0):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    data.make_full_loaders(self.root, workers=workers)
            prepare.assert_not_called()


def run_tests():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FullDataTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL",
              "tests_run": result.testsRun, "output": stream.getvalue(),
              "scope": "CPU fixtures; complete data coverage and epoch-boundary generator replay; no real CIFAR download or training"}
    if not result.wasSuccessful():
        raise AssertionError(report["output"])
    return report


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
