"""Small read-only dataset fixtures; no ImageNet or CUDA is required."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import tempfile
import time
from unittest.mock import patch


def _expect_error(call, contains):
    import data_support as d
    try:
        call()
    except d.ImageNetDataError as exc:
        assert contains in str(exc), (contains, str(exc))
        return str(exc)
    raise AssertionError("Expected ImageNetDataError: " + contains)


def _tree_digest(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def _write_fixture(root, class_count=3, train_per_class=12, val_per_class=6):
    from PIL import Image
    classes = [f"n{index + 1:08d}" for index in range(class_count)]
    for split, count in (("train", train_per_class), ("val", val_per_class)):
        for class_index, name in enumerate(classes):
            folder = root / split / name
            folder.mkdir(parents=True)
            for index in range(count):
                # Different spatial sizes and colors exercise PIL RGB conversion,
                # crop/resize and all classes without a large fixture.
                image = Image.new("RGB", (35 + index, 41 + class_index),
                                  color=(class_index * 70, index * 17 % 256, 91))
                image.save(folder / f"image_{index:03d}.JPEG")
    return classes


def _selection_tests():
    import numpy as np
    import torch
    import data_support as d
    torch_before = torch.get_rng_state().clone()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    targets = [label for label in range(7) for _ in range(11)]
    first = d._select_indices(1001, targets, 7, 256, 5, 101)
    repeated = d._select_indices(1001, targets, 7, 256, 5, 101)
    changed = d._select_indices(1001, targets, 7, 256, 5, 102)
    assert first == repeated and first != changed
    train, val = first
    oracle = torch.randperm(1001, generator=torch.Generator().manual_seed(101))[:256].tolist()
    assert train == oracle and len(set(train)) == 256
    assert len(set(val)) == 35
    assert Counter(targets[index] for index in val) == {label: 5 for label in range(7)}
    assert [targets[index] for index in val] != sorted(targets[index] for index in val)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    np_after = np.random.get_state()
    assert np_after[0] == numpy_before[0] and np.array_equal(np_after[1], numpy_before[1])
    assert np_after[2:] == numpy_before[2:]
    _expect_error(lambda: d._select_indices(2, targets, 7, 3, 5, 101), "Onvoldoende")
    _expect_error(lambda: d._select_indices(100, [0, 0, 1], 2, 50, 2, 101), "Te weinig")
    _expect_error(lambda: d._select_indices(100, [0, 2], 2, 50, 1, 101), "Ongeldig")
    return {"status": "PASS", "training_matches_full_randperm_prefix": True,
            "no_repeated_indices": True, "same_seed_repeats_selection": True,
            "changed_seed_changes_selection": True, "exact_validation_class_balance": True,
            "validation_not_class_order_prefix": True, "global_RNGs_unchanged": True}


def _structural_failure_tests(root):
    import data_support as d
    cases = {}
    cases["missing_root"] = _expect_error(lambda: d.audit_imagenet(root / "missing"), "ImageNet-map ontbreekt")
    archive_only = root / "archive_only"
    archive_only.mkdir()
    (archive_only / "ILSVRC2012_img_train.tar").write_bytes(b"fixture")
    cases["archive_not_prepared"] = _expect_error(lambda: d.audit_imagenet(archive_only), "archieven")
    flat = root / "flat"
    (flat / "train" / "n00000001").mkdir(parents=True)
    (flat / "val").mkdir()
    (flat / "val" / "ILSVRC2012_val_00000001.JPEG").write_bytes(b"fixture")
    cases["flat_validation"] = _expect_error(lambda: d.audit_imagenet(flat), "losse afbeeldingen")
    mismatched = root / "mismatched"
    (mismatched / "train" / "n00000001").mkdir(parents=True)
    (mismatched / "val" / "n00000002").mkdir(parents=True)
    cases["mismatched_labels"] = _expect_error(lambda: d.audit_imagenet(mismatched), "WNID-klassen")
    invalid = root / "invalid_class"
    (invalid / "train" / "cat").mkdir(parents=True)
    (invalid / "val" / "cat").mkdir(parents=True)
    cases["invalid_class_name"] = _expect_error(lambda: d.audit_imagenet(invalid), "onverwachte klasmappen")
    return {"status": "PASS", "rejected": sorted(cases)}


def _traversal_order_test(root):
    from torchvision import datasets
    import data_support as d
    fixture = root / "nested_order"
    class_root = fixture / "n00000001"
    for folder in ("a/sub", "a-other", "b"):
        path = class_root / folder
        path.mkdir(parents=True)
        (path / "image.JPEG").write_bytes(b"metadata-only fixture")
    actual = [str(path) for path in d._image_files(class_root)]
    reference = [path for path, _ in datasets.ImageFolder(str(fixture)).samples]
    assert actual == reference
    return {"status": "PASS", "nested_path_order_matches_ImageFolder": True}


def _metadata_collate(batch):
    """Test-only IPC payload: no shared tensor storage or local socket needed."""
    import torch
    return {"labels": [int(label) for _, label in batch],
            "shapes": [tuple(image.shape) for image, _ in batch],
            "all_float32_finite": all(image.dtype == torch.float32 and bool(torch.isfinite(image).all())
                                      for image, _ in batch)}


def _fixture_loader_tests(root, *, include_spawn):
    import numpy as np
    import torch
    from torchvision import datasets
    import data_support as d

    fixture = root / "prepared"
    classes = _write_fixture(fixture)
    before = _tree_digest(fixture)
    tiny_manifest = d.audit_imagenet(fixture, strict=False)
    assert tiny_manifest["class_count"] == 3 and tiny_manifest["train_count"] == 36
    assert tiny_manifest["val_count"] == 18
    assert tiny_manifest["classes"] == classes
    _expect_error(lambda: d.audit_imagenet(fixture, strict=True), "Verwacht 1000 klassen")
    _expect_error(lambda: d.build_loaders(fixture, workers=0, audit_manifest=tiny_manifest), "strikte inventaris")

    # Reduce constants only inside the test fixture; no such controls are
    # exposed by the public runner or loader API.
    constants = {"EXPECTED_CLASSES": 3, "EXPECTED_TRAIN": 36, "EXPECTED_VAL": 18,
                 "EXPECTED_VAL_PER_CLASS": 6, "TRAIN_BATCH_SIZE": 4, "TRAIN_BATCHES": 3,
                 "TRAIN_SAMPLES": 12, "VAL_PER_CLASS": 2, "VAL_SAMPLES": 6, "VAL_BATCH_SIZE": 4}
    with patch.multiple(d, **constants):
        manifest = d.audit_imagenet(fixture, strict=True)
        rng_before = torch.get_rng_state().clone()
        train, val, metadata = d.build_loaders(fixture, workers=0, seed=303, audit_manifest=manifest)
        assert torch.equal(torch.get_rng_state(), rng_before), "Loader construction consumed model RNG."
        assert len(train.dataset) == 12 and len(train) == 3
        assert len(val.dataset) == 6 and len(val) == 2
        assert Counter(val.dataset.targets) == {0: 2, 1: 2, 2: 2}
        assert metadata["train_samples"] == 12 and metadata["validation_samples"] == 6
        assert metadata["train_indices_sha256"] != metadata["validation_indices_sha256"]
        assert not hasattr(train.dataset, "dataset"), "Workers must not retain the full source ImageFolder."
        assert len(train.dataset.samples) == 12

        # Check exact same samples and transform behavior against independent
        # full ImageFolder objects with a reset random-augmentation RNG.
        train_transform, val_transform = d._transforms()
        source_train = datasets.ImageFolder(str(fixture / "train"), transform=train_transform,
                                             loader=datasets.folder.pil_loader)
        source_val = datasets.ImageFolder(str(fixture / "val"), transform=val_transform,
                                           loader=datasets.folder.pil_loader)
        train_indices, val_indices = d._select_indices(36, source_val.targets, 3, 12, 2, 303)
        state = torch.get_rng_state().clone()
        actual, label_actual = train.dataset[0]
        torch.set_rng_state(state)
        expected, label_expected = source_train[train_indices[0]]
        assert torch.equal(actual, expected) and label_actual == label_expected
        actual_val, val_label = val.dataset[0]
        expected_val, expected_label = source_val[val_indices[0]]
        assert torch.equal(actual_val, expected_val) and val_label == expected_label
        assert tuple(actual.shape) == (3, 224, 224) and actual.dtype == torch.float32
        assert torch.isfinite(actual).all() and torch.isfinite(actual_val).all()

        # A whole bounded iteration yields exactly its agreed amount. The
        # iterator is exhausted instead of leaking a prefetched batch201.
        train_batches = list(train)
        assert [len(labels) for _, labels in train_batches] == [4, 4, 4]
        val_batches = list(val)
        assert [len(labels) for _, labels in val_batches] == [4, 2]
        assert sum(len(labels) for _, labels in val_batches) == 6
        del train_batches, val_batches
        restored = pickle.loads(pickle.dumps(train.dataset))
        assert restored.samples == train.dataset.samples and restored.class_to_idx == train.dataset.class_to_idx
        callback = pickle.loads(pickle.dumps(d.seed_worker))
        assert callback is d.seed_worker

        stale = copy.deepcopy(manifest)
        stale["splits"]["train"]["relative_paths_sha256"] = "0" * 64
        _expect_error(lambda: d.build_loaders(fixture, workers=0, audit_manifest=stale), "paden of volgorde")
        wrong_counts = copy.deepcopy(manifest)
        wrong_counts["splits"]["val"]["per_class_counts"][classes[0]] = 7
        _expect_error(lambda: d.build_loaders(fixture, workers=0, audit_manifest=wrong_counts), "klasaantallen")
        other_root = copy.deepcopy(manifest)
        other_root["root"] = str(root / "elsewhere")
        _expect_error(lambda: d.build_loaders(fixture, workers=0, audit_manifest=other_root), "verschilt")

        # A changed val count is rejected even when both splits keep the same
        # class set; failure does not repair or reorganize the dataset.
        extra = fixture / "val" / classes[0] / "extra.JPEG"
        extra.write_bytes((fixture / "val" / classes[0] / "image_000.JPEG").read_bytes())
        _expect_error(lambda: d.audit_imagenet(fixture, strict=True), "verwacht 6 afbeeldingen")
        extra.unlink()
        broken = fixture / "train" / classes[0] / "image_000.JPEG"
        original = broken.read_bytes()
        broken.write_bytes(b"not-an-image")
        _expect_error(lambda: d.audit_imagenet(fixture, strict=True), "niet worden gedecodeerd")
        broken.write_bytes(original)

        spawned = False
        if include_spawn:
            # Spawn is deliberately used on Linux too, covering the same
            # pickling boundary used by the Windows laptop.
            train_spawn, val_spawn = d._make_loaders(train.dataset, val.dataset, workers=1, seed=303,
                                                     train_batch_size=4, val_batch_size=4)
            # Only the test collator is replaced. This sandbox disallows the
            # Unix socket used by Torch tensor-sharing IPC; Windows production
            # loaders keep the ordinary tensor collator unchanged.
            train_spawn.collate_fn = _metadata_collate
            val_spawn.collate_fn = _metadata_collate
            train_results, val_results = list(train_spawn), list(val_spawn)
            train_total = sum(len(item["labels"]) for item in train_results)
            val_total = sum(len(item["labels"]) for item in val_results)
            assert train_total == 12 and val_total == 6
            assert all(item["all_float32_finite"] for item in train_results + val_results)
            assert all(shape == (3, 224, 224) for item in train_results + val_results for shape in item["shapes"])
            spawned = True
        assert _tree_digest(fixture) == before, "The dataset fixture was modified by audit or loading."
    return {"status": "PASS", "bounded_train_loader_exhaustion": True,
            "stratified_validation_and_partial_last_batch": True,
            "transforms_equal_independent_ImageFolder": True,
            "strict_public_dataset_requirements_enforced": True,
            "stale_path_count_and_root_manifests_rejected": True,
            "corrupt_decode_and_wrong_validation_counts_rejected": True,
            "dataset_files_unchanged": True, "selected_dataset_pickle_roundtrip": True,
            "spawn_worker_iteration_with_metadata_only_test_collator": "PASS" if spawned else "NOT_RUN_IN_THIS_INVOCATION",
            "spawn_tensor_IPC": "NOT_TESTED: sandbox blocks the Unix resource-sharing socket; production default collator is unchanged."}


def run_tests(*, include_spawn: bool = False):
    import numpy as np
    import torch
    if not __debug__:
        raise RuntimeError("Data-zelftests vereisen assertions; gebruik geen python -O.")
    began = time.monotonic()
    torch_state, py_state, np_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    try:
        with tempfile.TemporaryDirectory(prefix="r18-realdata-tests-") as temporary:
            root = Path(temporary)
            result = {"status": "PASS", "scope": "Small generated image fixtures and CPU-only data-loader tests; no ImageNet/GPU benchmark.",
                      "selection": _selection_tests(),
                      "structure": _structural_failure_tests(root),
                      "traversal": _traversal_order_test(root),
                      "loaders": _fixture_loader_tests(root, include_spawn=include_spawn)}
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(py_state)
        np.random.set_state(np_state)
    result["caller_global_RNG_restored"] = True
    result["seconds"] = time.monotonic() - began
    return result


if __name__ == "__main__":
    print(json.dumps(run_tests(include_spawn=True), indent=2))
