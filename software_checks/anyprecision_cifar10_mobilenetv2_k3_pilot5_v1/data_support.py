"""Read-only CIFAR-10 discovery, timing subsets and complete epoch loaders.

Only ``download=True`` permits obtaining the official CIFAR-10 archive. Existing
dataset files are never repaired or overwritten by this module. The production
loader always audits the official checksums before torchvision unpickles data.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10


DATASET_FOLDER = "cifar-10-batches-py"
BATCH_SIZE = 128
TRAIN_SIZE = 50_000
VALIDATION_SIZE = 10_000
MEAN = tuple(x / 255.0 for x in (125.3, 123.0, 113.9))
STD = tuple(x / 255.0 for x in (63.0, 62.1, 66.7))
EXPECTED_MD5 = {
    **dict(CIFAR10.train_list), **dict(CIFAR10.test_list),
    CIFAR10.meta["filename"]: CIFAR10.meta["md5"],
}


class CIFAR10DataError(RuntimeError):
    """Missing or invalid official data; no training may proceed."""


def _normalise_root(path) -> Path:
    result = Path(path).expanduser().resolve()
    return result.parent if result.name == DATASET_FOLDER else result


def find_data_root(project_root, explicit=None, *, allow_missing=False) -> Path:
    """Find a dataset root without searching virtual environments or results.

    ``explicit`` may be either the root containing ``cifar-10-batches-py`` or
    that directory itself. ``allow_missing`` is only intended for an explicitly
    requested official download; it does not create any directories.
    """
    project_root = Path(project_root).expanduser().resolve()
    if explicit is not None:
        candidates = [_normalise_root(explicit)]
    else:
        candidates = [project_root / "data" / "cifar10_mobilenet_complete",
                      project_root / "data" / "cifar10", project_root / "data"]
        for name in ("Any-Precision-DNNs-master", "Any-Precision-DNNs", "any-precision-nets"):
            candidates.append(project_root / name / "data" / "cifar10")
            for parent in ("third_party", "external", "sources"):
                candidates.append(project_root / parent / name / "data" / "cifar10")
    candidates = list(dict.fromkeys(_normalise_root(path) for path in candidates))
    # Prefer a structurally complete copy over an older six-batch copy that
    # lacks batches.meta.  Integrity is still checked before any unpickling;
    # discovery itself does not modify files or read the dataset into memory.
    for candidate in candidates:
        if all((candidate / DATASET_FOLDER / name).is_file() for name in EXPECTED_MD5):
            return candidate
    for candidate in candidates:
        if (candidate / DATASET_FOLDER).is_dir():
            return candidate
    if allow_missing:
        # Retain the original download default when no dataset exists at all.
        return candidates[0] if explicit is not None else project_root / "data" / "cifar10"
    searched = "\n  ".join(str(path) for path in candidates)
    raise CIFAR10DataError(
        "CIFAR-10 niet gevonden. Geef --data-root met de map waarin "
        "cifar-10-batches-py staat (of met die map zelf). Gebruik alleen "
        "--download als de officiele dataset nog moet worden opgehaald. "
        "Er is niets gedownload of gewijzigd. Gecontroleerde locaties:\n  " + searched
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cifar10(data_root) -> dict:
    """Check all five train batches, the test batch and label metadata."""
    root = _normalise_root(data_root)
    folder = root / DATASET_FOLDER
    if not folder.is_dir():
        raise CIFAR10DataError(
            f"CIFAR-10-map ontbreekt: {folder}. Geef --data-root of gebruik "
            "--download voor een nieuwe officiele download."
        )
    files = []
    problems = []
    for name, expected in EXPECTED_MD5.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{name}: ontbreekt")
            continue
        actual = _md5(path)
        files.append({"name": name, "bytes": path.stat().st_size,
                      "expected_md5": expected, "actual_md5": actual,
                      "matches": actual == expected})
        if actual != expected:
            problems.append(f"{name}: MD5 komt niet overeen met officiele CIFAR-10")
    if problems:
        raise CIFAR10DataError(
            "CIFAR-10-controle mislukt: " + "; ".join(problems) + ". "
            "Bestaande data worden niet gerepareerd of overschreven. "
            "Kies een andere bestaande datasetmap of een nieuwe lege map voor --download."
        )
    return {"status": "PASS", "root": str(root), "dataset_folder": str(folder),
            "integrity": "official torchvision CIFAR10 MD5, including metadata",
            "files": files, "modified_existing_data": False}


def _prepare_data(data_root, *, download: bool) -> dict:
    root = _normalise_root(data_root)
    if (root / DATASET_FOLDER).exists():
        # Never let torchvision's automatic extraction overwrite partial data.
        result = audit_cifar10(root)
        result["downloaded"] = False
        return result
    if not download:
        return audit_cifar10(root)  # Raises the actionable missing-data error.
    archive = root / CIFAR10.filename
    if archive.exists() and (not archive.is_file() or _md5(archive) != CIFAR10.tgz_md5):
        raise CIFAR10DataError(
            f"Bestaand archief is niet de officiele CIFAR-10-download: {archive}. "
            "Dit bestand wordt niet overschreven; kies een nieuwe lege --data-root."
        )
    # torchvision validates the official archive before extraction. This is the
    # only branch allowed to create dataset files, and needs --download upstream.
    CIFAR10(root=str(root), train=True, download=True)
    result = audit_cifar10(root)
    result["downloaded"] = True
    return result


def make_transforms():
    """Use the order and normalization of upstream datasets/data.py."""
    training = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    validation = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    return training, validation


def _seed_loader_worker(_worker_id):
    # DataLoader derives torch.initial_seed from its dedicated generator.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _check_loaded_split(dataset, expected_size: int, split: str):
    if len(dataset) != expected_size or tuple(dataset.data.shape) != (expected_size, 32, 32, 3):
        raise CIFAR10DataError(f"Onverwachte {split}-afmetingen: vereist {expected_size} beelden van 32x32 RGB.")
    counts = Counter(int(label) for label in dataset.targets)
    expected_count = expected_size // 10
    if counts != Counter({label: expected_count for label in range(10)}):
        raise CIFAR10DataError(f"Onjuiste {split}-labels of klasseaantallen: {dict(counts)}.")
    if len(dataset.classes) != 10:
        raise CIFAR10DataError(f"De {split}-split moet exact tien klassen hebben.")
    return counts


def _index_hash(indices) -> str:
    payload = json.dumps(list(map(int, indices)), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def make_loaders(data_root, seed=20261030, workers=0, download=False,
                 train_steps=100, validation_examples=1000):
    """Return 100 real training batches and a balanced evaluation timing subset.

    Both subsets come from the complete, audited official splits. No synthetic
    fallback is available here. The full epoch still means 50,000 train and
    10,000 test images; the test subset is used only for a timing estimate.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a nonnegative integer.")
    if isinstance(train_steps, bool) or not isinstance(train_steps, int) or not 1 <= train_steps <= TRAIN_SIZE // BATCH_SIZE:
        raise ValueError("train_steps must select a nonempty prefix of full 128-image batches.")
    if (isinstance(validation_examples, bool) or not isinstance(validation_examples, int)
            or not 10 <= validation_examples <= VALIDATION_SIZE or validation_examples % 10):
        raise ValueError("validation_examples must be a multiple of 10 between 10 and 10,000.")
    root = _normalise_root(data_root)
    audit = _prepare_data(root, download=bool(download))
    train_transform, validation_transform = make_transforms()
    training = CIFAR10(root=str(root), train=True, transform=train_transform, download=False)
    validation = CIFAR10(root=str(root), train=False, transform=validation_transform, download=False)
    train_counts = _check_loaded_split(training, TRAIN_SIZE, "train")
    validation_counts = _check_loaded_split(validation, VALIDATION_SIZE, "test")

    permutation_rng = torch.Generator().manual_seed(int(seed))
    train_indices = torch.randperm(TRAIN_SIZE, generator=permutation_rng)[:train_steps * BATCH_SIZE].tolist()
    subset_rng = torch.Generator().manual_seed(int(seed) + 1)
    val_indices = []
    per_class = validation_examples // 10
    targets = torch.tensor(validation.targets, dtype=torch.long)
    for label in range(10):
        pool = torch.nonzero(targets == label, as_tuple=False).flatten()
        positions = torch.randperm(len(pool), generator=subset_rng)[:per_class]
        val_indices.extend(pool[positions].tolist())
    ordering = torch.randperm(len(val_indices), generator=subset_rng).tolist()
    val_indices = [val_indices[position] for position in ordering]

    common = {"batch_size": BATCH_SIZE, "num_workers": workers,
              "pin_memory": True, "shuffle": False, "drop_last": False,
              "worker_init_fn": _seed_loader_worker,
              "persistent_workers": workers > 0}
    train_loader = DataLoader(Subset(training, train_indices),
                              generator=torch.Generator().manual_seed(int(seed) + 2), **common)
    val_loader = DataLoader(Subset(validation, val_indices),
                            generator=torch.Generator().manual_seed(int(seed) + 3), **common)
    info = {**audit, "seed": int(seed), "workers": workers, "batch_size": BATCH_SIZE,
            "full_train_examples": TRAIN_SIZE, "full_validation_examples": VALIDATION_SIZE,
            "full_train_batches": (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE,
            "full_validation_batches": (VALIDATION_SIZE + BATCH_SIZE - 1) // BATCH_SIZE,
            "train_examples": len(train_indices), "train_batches": len(train_loader),
            "validation_examples": len(val_indices), "validation_batches": len(val_loader),
            "train_class_counts": dict(sorted(train_counts.items())),
            "full_validation_class_counts": dict(sorted(validation_counts.items())),
            "validation_subset_class_counts": {label: per_class for label in range(10)},
            "train_indices_sha256": _index_hash(train_indices),
            "validation_indices_sha256": _index_hash(val_indices),
            "train_first_indices": train_indices[:20], "validation_first_indices": val_indices[:20],
            "train_order": "prefix of seeded permutation of all 50,000 official training examples; no replacement",
            "validation_selection": "100 per class by default; seeded without replacement from official test split",
            "normalization_mean": list(MEAN), "normalization_std": list(STD),
            "training_transforms": ["RandomHorizontalFlip", "RandomCrop(32,padding=4)", "ToTensor", "Normalize"],
            "validation_transforms": ["ToTensor", "Normalize"],
            "augmentation_rng": "worker torch seed; dedicated DataLoader generators, no bit-exact resume claim",
            "scope": "timing probe only, not a validation score for model selection"}
    return train_loader, val_loader, info


def _dataset_identity(audit: dict) -> dict:
    """Root-independent identity of the checksum-audited official dataset."""
    rows = audit.get("files", [])
    if audit.get("status") != "PASS" or len(rows) != len(EXPECTED_MD5):
        raise CIFAR10DataError("Volledige CIFAR-10-integriteitscontrole ontbreekt.")
    by_name = {row.get("name"): row for row in rows}
    if set(by_name) != set(EXPECTED_MD5):
        raise CIFAR10DataError("Datasetidentiteit bevat niet de zeven officiële CIFAR-10-bestanden.")
    for name, expected in EXPECTED_MD5.items():
        row = by_name[name]
        if not row.get("matches") or row.get("actual_md5") != expected or row.get("expected_md5") != expected:
            raise CIFAR10DataError("Onbevestigde datasetidentiteit voor " + name)
    identity = {
        "schema_version": 1,
        "dataset": "CIFAR-10-python",
        "train_examples": TRAIN_SIZE,
        "validation_examples": VALIDATION_SIZE,
        "files": [{"name": name, "md5": by_name[name]["actual_md5"]}
                  for name in sorted(EXPECTED_MD5)],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**identity, "sha256": hashlib.sha256(encoded).hexdigest()}


def make_full_loaders(data_root, seed=20261030, workers=0, download=False):
    """Load all official train/test examples, with stateful epoch shuffling.

    The returned generators must be saved after each completed epoch and
    restored before creating the next loader iterator.  Workers are fixed at
    zero so augmentation RNG state lives in the main process.  Constructing
    these loaders does not consume a global RNG or iterate over any images.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers != 0:
        raise ValueError("De volledige vijf-epochproef gebruikt exact workers=0 voor hervatten per epoch.")
    root = _normalise_root(data_root)
    audit = _prepare_data(root, download=bool(download))
    identity = _dataset_identity(audit)
    train_transform, validation_transform = make_transforms()
    training = CIFAR10(root=str(root), train=True, transform=train_transform, download=False)
    validation = CIFAR10(root=str(root), train=False, transform=validation_transform, download=False)
    train_counts = _check_loaded_split(training, TRAIN_SIZE, "train")
    validation_counts = _check_loaded_split(validation, VALIDATION_SIZE, "test")

    train_generator = torch.Generator().manual_seed(int(seed) + 2)
    val_generator = torch.Generator().manual_seed(int(seed) + 3)
    common = {"batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": True,
              "drop_last": False, "persistent_workers": False,
              "worker_init_fn": _seed_loader_worker}
    train_loader = DataLoader(training, shuffle=True, generator=train_generator, **common)
    val_loader = DataLoader(validation, shuffle=False, generator=val_generator, **common)
    info = {
        **audit, "identity": identity, "seed": int(seed), "workers": 0,
        "batch_size": BATCH_SIZE, "validation_batch_size": BATCH_SIZE,
        "full_train_examples": TRAIN_SIZE, "full_validation_examples": VALIDATION_SIZE,
        "full_train_batches": len(train_loader), "full_validation_batches": len(val_loader),
        "train_examples": len(training), "train_batches": len(train_loader),
        "validation_examples": len(validation), "validation_batches": len(val_loader),
        "train_last_batch": TRAIN_SIZE % BATCH_SIZE or BATCH_SIZE,
        "validation_last_batch": VALIDATION_SIZE % BATCH_SIZE or BATCH_SIZE,
        "train_class_counts": dict(sorted(train_counts.items())),
        "full_validation_class_counts": dict(sorted(validation_counts.items())),
        "train_drop_last": False, "validation_drop_last": False,
        "train_order": "New seeded random permutation of all 50,000 examples each epoch; no replacement",
        "validation_selection": "All 10,000 official test examples in fixed dataset order",
        "normalization_mean": list(MEAN), "normalization_std": list(STD),
        "training_transforms": ["RandomHorizontalFlip", "RandomCrop(32,padding=4)", "ToTensor", "Normalize"],
        "validation_transforms": ["ToTensor", "Normalize"],
        "augmentation_rng": "Main-process torch RNG (workers=0); checkpoint global RNG and both loader generators at epoch boundary",
        "scope": "Five complete training epochs with full official test-set evaluation; no timing subset",
    }
    return train_loader, val_loader, info, train_generator, val_generator
