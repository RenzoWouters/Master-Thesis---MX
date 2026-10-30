"""Read-only ImageNet inventory and bounded real-data diagnostic loaders.

The public path requires complete, prepared ILSVRC2012 train/validation splits.
The selected 51,200 training and 5,000 validation images are a diagnostic, not
an epoch or a final ImageNet result. No dataset download or mutation occurs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Callable


EXPECTED_CLASSES = 1000
EXPECTED_TRAIN = 1_281_167
EXPECTED_VAL = 50_000
EXPECTED_VAL_PER_CLASS = 50
TRAIN_BATCH_SIZE = 256
TRAIN_BATCHES = 200
TRAIN_SAMPLES = TRAIN_BATCH_SIZE * TRAIN_BATCHES
VAL_PER_CLASS = 5
VAL_SAMPLES = EXPECTED_CLASSES * VAL_PER_CLASS
VAL_BATCH_SIZE = 64
WNID = re.compile(r"n[0-9]{8}\Z")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageNetDataError(RuntimeError):
    """Dataset is absent, unprepared, incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImageNetDataError(message)


def _classes(split: Path) -> list[str]:
    with os.scandir(split) as entries:
        names = sorted(entry.name for entry in entries if entry.is_dir())
    invalid = [name for name in names if WNID.fullmatch(name) is None]
    _require(not invalid,
             f"{split.name}: onverwachte klasmappen {invalid[:5]}. Verwacht uitsluitend "
             "WordNet-klassen zoals n01440764; --imagenet-root moet train en val bevatten.")
    return names


def _image_files(class_dir: Path):
    """ImageFolder traversal order, with a check against directory cycles."""
    visited: set[str] = set()
    rows = []
    for current, directories, filenames in os.walk(class_dir, followlinks=True):
        resolved = os.path.normcase(os.path.realpath(current))
        _require(resolved not in visited,
                 f"Herhaalde of cyclische mapkoppeling onder {class_dir}.")
        visited.add(resolved)
        directories.sort()
        rows.append((current, filenames))
    # ImageFolder sorts os.walk rows, not only each directory's children.
    # The distinction matters for nested names such as a/sub and a-other.
    for current, filenames in sorted(rows):
        for filename in sorted(filenames):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                yield Path(current) / filename


def _decode_samples(paths: list[Path]) -> list[dict[str, Any]]:
    from PIL import Image
    records = []
    for path in paths:
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                _require(rgb.width > 0 and rgb.height > 0, f"Lege afbeelding: {path}")
                records.append({"relative_path": path.name, "width": rgb.width,
                                "height": rgb.height, "mode_after_conversion": "RGB"})
        except Exception as exc:
            raise ImageNetDataError(f"Afbeelding kan niet worden gedecodeerd: {path}: {exc}") from exc
    return records


def audit_imagenet(root: str | Path, *, strict: bool = True,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Inventory without writing dataset files.

    ``strict=False`` exists only for tiny software-test fixtures. Public loader
    construction always requires a complete strict manifest and full splits.
    """
    root = Path(root).expanduser().resolve()
    _require(root.is_dir(),
             f"ImageNet-map ontbreekt: {root}. Geef --imagenet-root met een bestaande, "
             "voorbereide ILSVRC2012-map met train/<wnid>/ en val/<wnid>/. "
             "De proef met echte data kan pas daarna starten; er wordt niets gedownload.")
    train, val = root / "train", root / "val"
    if not train.is_dir() or not val.is_dir():
        archives = sorted(path.name for path in root.iterdir()
                          if path.is_file() and path.name.lower().endswith((".tar", ".tar.gz", ".tgz", ".zip")))
        detail = f" Gevonden archieven: {archives[:5]}; die zijn nog niet voorbereid als ImageFolder." if archives else ""
        raise ImageNetDataError(f"{root} moet train en val bevatten, beide met klasmappen n########."
                                + detail + " Dit pakket pakt geen data uit en verplaatst geen bestanden.")
    for split in (train, val):
        loose = [path.name for path in split.iterdir()
                 if path.is_file() and path.name.lower().endswith(IMAGE_EXTENSIONS)]
        _require(not loose,
                 f"{split.name} bevat losse afbeeldingen buiten klasmappen, bijvoorbeeld {loose[:3]}. "
                 "Bereid de officiële indeling train/<wnid>/ en val/<wnid>/ vooraf voor; "
                 "validatielabels worden niet geraden.")
    classes_train, classes_val = _classes(train), _classes(val)
    _require(classes_train and classes_val, "train of val bevat geen geldige klasmappen.")
    _require(classes_train == classes_val,
             "De gesorteerde WNID-klassen van train en val verschillen. "
             f"Alleen train: {sorted(set(classes_train)-set(classes_val))[:5]}; "
             f"alleen val: {sorted(set(classes_val)-set(classes_train))[:5]}.")
    if strict:
        _require(len(classes_train) == EXPECTED_CLASSES,
                 f"Verwacht {EXPECTED_CLASSES} klassen; gevonden {len(classes_train)}. "
                 "Deze korte proef selecteert zelf beelden uit de volledige ImageNet-dataset.")
    mapping = {name: index for index, name in enumerate(classes_train)}
    mapping_digest = hashlib.sha256(json.dumps(mapping, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
    splits = {}
    sample_classes = {0, len(classes_train) // 2, len(classes_train) - 1}
    for name, split, expected in (("train", train, EXPECTED_TRAIN), ("val", val, EXPECTED_VAL)):
        fingerprint, paths_fingerprint = hashlib.sha256(), hashlib.sha256()
        counts, sampled = {}, []
        total_count, total_bytes = 0, 0
        if progress:
            progress(f"ImageNet-inventaris {name}: {len(classes_train)} klassen controleren; dataset blijft ongewijzigd.")
        for index, class_name in enumerate(classes_train):
            count, first, last = 0, None, None
            for path in _image_files(split / class_name):
                size = path.stat().st_size
                _require(size > 0, f"Bestand met nul bytes: {path}")
                relative = path.relative_to(split).as_posix()
                fingerprint.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\n")
                paths_fingerprint.update(relative.encode("utf-8") + b"\n")
                count += 1
                total_bytes += size
                first = path if first is None else first
                last = path
            _require(count > 0, f"Lege klasse: {split / class_name}")
            if strict and name == "val":
                _require(count == EXPECTED_VAL_PER_CLASS,
                         f"val/{class_name}: verwacht {EXPECTED_VAL_PER_CLASS} afbeeldingen, gevonden {count}.")
            counts[class_name] = count
            total_count += count
            if index in sample_classes:
                if first is not None:
                    sampled.append(first)
                if last is not None and last != first:
                    sampled.append(last)
            if progress and ((index + 1) % 100 == 0 or index + 1 == len(classes_train)):
                progress(f"ImageNet-inventaris {name}: {index + 1}/{len(classes_train)} klassen; "
                         f"{total_count:,} afbeeldingen.")
        if strict:
            _require(total_count == expected,
                     f"{name}: verwacht {expected:,} afbeeldingen, gevonden {total_count:,}. "
                     "De complete ILSVRC2012-splits zijn nodig; de proef kiest daarna de beperkte subset.")
        decoded = _decode_samples(sampled)
        for record, path in zip(decoded, sampled):
            record["relative_path"] = path.relative_to(split).as_posix()
        splits[name] = {"path": str(split), "count": total_count, "expected_count": expected,
                        "file_bytes": total_bytes, "filelist_sha256": fingerprint.hexdigest(),
                        "relative_paths_sha256": paths_fingerprint.hexdigest(),
                        "per_class_counts": counts, "decode_samples": decoded}
    return {
        "schema_version": 1, "dataset": "ImageNet-1k / ILSVRC2012", "root": str(root),
        "strict": strict, "class_count": len(classes_train), "classes": classes_train,
        "class_to_idx_sha256": mapping_digest, "train_count": splits["train"]["count"],
        "val_count": splits["val"]["count"], "splits": splits,
        "checks": {"matching_class_order": True, "nonempty_classes": True,
                   "expected_full_counts": strict, "sample_decode_passed": True},
        "image_extensions": list(IMAGE_EXTENSIONS),
        "note": "Structuur, aantallen, paden/groottes en enkele decodes gecontroleerd; geen volledige "
                "inhoudshashes of onafhankelijke labelverificatie. Datasetbestanden blijven ongewijzigd.",
    }


def seed_worker(worker_id: int) -> None:
    """Module-level callback; compatible with Windows multiprocessing spawn."""
    import numpy as np
    import torch
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SelectedImageDataset:
    """Small map-style dataset; workers receive only the selected sample paths."""
    def __init__(self, source, indices):
        self.samples = [source.samples[index] for index in indices]
        self.targets = [target for _, target in self.samples]
        self.classes = list(source.classes)
        self.class_to_idx = dict(source.class_to_idx)
        self.transform = source.transform
        self.loader = source.loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _select_indices(train_count: int, val_targets: list[int], class_count: int,
                    train_samples: int, val_per_class: int, seed: int):
    """Explicit independent RNG streams; never consume the model's global RNG."""
    import torch
    _require(0 < train_samples <= train_count, "Onvoldoende trainingsbeelden voor de vaste prefix.")
    _require(class_count > 0 and val_per_class > 0, "Ongeldige validatieselectie.")
    train_generator = torch.Generator().manual_seed(seed)
    train_indices = torch.randperm(train_count, generator=train_generator)[:train_samples].tolist()
    by_class = [[] for _ in range(class_count)]
    for index, target in enumerate(val_targets):
        _require(0 <= target < class_count, "Ongeldig validatieklasselabel.")
        by_class[target].append(index)
    val_generator = torch.Generator().manual_seed(seed + 1)
    val_indices = []
    for candidates in by_class:
        _require(len(candidates) >= val_per_class, "Te weinig validatiebeelden in een klasse.")
        choices = torch.randperm(len(candidates), generator=val_generator)[:val_per_class].tolist()
        val_indices.extend(candidates[index] for index in choices)
    # Mix the balanced set so partial progress is not a prefix of class IDs.
    order = torch.randperm(len(val_indices), generator=val_generator).tolist()
    val_indices = [val_indices[index] for index in order]
    _require(len(set(train_indices)) == train_samples, "Trainingselectie bevat herhalingen.")
    _require(len(set(val_indices)) == class_count * val_per_class, "Validatieselectie bevat herhalingen.")
    return train_indices, val_indices


def _index_digest(indices: list[int]) -> str:
    return hashlib.sha256(("\n".join(map(str, indices)) + "\n").encode("ascii")).hexdigest()


def _sample_path_digest(samples, split: Path) -> str:
    fingerprint = hashlib.sha256()
    for path, _ in samples:
        fingerprint.update(Path(path).relative_to(split).as_posix().encode("utf-8") + b"\n")
    return fingerprint.hexdigest()


def _transforms():
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    return train, val


def _make_loaders(train_data, val_data, *, workers: int, seed: int,
                  train_batch_size: int = TRAIN_BATCH_SIZE, val_batch_size: int = VAL_BATCH_SIZE):
    import torch
    _require(isinstance(workers, int) and workers >= 0, "Aantal dataloader-workers moet minstens nul zijn.")
    common = {"num_workers": workers, "pin_memory": True, "drop_last": False,
              "persistent_workers": False, "worker_init_fn": seed_worker}
    if workers:
        common.update(prefetch_factor=2, multiprocessing_context="spawn", timeout=120)
    train_generator = torch.Generator().manual_seed(seed + 2)
    val_generator = torch.Generator().manual_seed(seed + 3)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size,
                                               shuffle=False, generator=train_generator, **common)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size,
                                             shuffle=False, generator=val_generator, **common)
    return train_loader, val_loader


def build_loaders(root: str | Path, *, workers: int = 4, seed: int = 20261009,
                  audit_manifest: dict[str, Any] | None = None,
                  progress: Callable[[str], None] | None = None):
    """Return bounded train/validation loaders and JSON-safe selection metadata.

    Global batch256 and 200 complete train batches are fixed. Validation uses
    5 samples per each of 1,000 classes with frozen-BN eval batches of at most64.
    The full training permutation is generated once, then only its first51,200
    entries are retained. No batch201 can be prefetched by this bounded loader.
    """
    from torchvision import datasets
    root = Path(root).expanduser().resolve()
    manifest = audit_manifest if audit_manifest is not None else audit_imagenet(root, strict=True, progress=progress)
    _require(manifest.get("strict") is True and manifest.get("class_count") == EXPECTED_CLASSES and
             manifest.get("train_count") == EXPECTED_TRAIN and manifest.get("val_count") == EXPECTED_VAL,
             "De publieke loaders vereisen een geslaagde strikte inventaris van volledige ImageNet-splits.")
    _require(Path(manifest["root"]).resolve() == root, "De ImageNet-map verschilt van de gecontroleerde dataset.")
    train_transform, val_transform = _transforms()
    if progress:
        progress("ImageFolder-index opbouwen en vergelijken met de gecontroleerde paden.")
    train_source = datasets.ImageFolder(str(root / "train"), transform=train_transform,
                                        loader=datasets.folder.pil_loader)
    val_source = datasets.ImageFolder(str(root / "val"), transform=val_transform,
                                      loader=datasets.folder.pil_loader)
    _require(train_source.class_to_idx == val_source.class_to_idx,
             "ImageFolder-klassevolgorde verschilt tussen train en val.")
    _require(train_source.classes == manifest["classes"] and
             len(train_source.classes) == EXPECTED_CLASSES and
             all(WNID.fullmatch(name) for name in train_source.classes),
             "ImageFolder-klassen zijn gewijzigd sinds de datasetcontrole.")
    _require(len(train_source) == EXPECTED_TRAIN and len(val_source) == EXPECTED_VAL,
             "ImageFolder-aantallen zijn gewijzigd sinds de datasetcontrole.")
    for name, source in (("train", train_source), ("val", val_source)):
        _require(_sample_path_digest(source.samples, root / name) == manifest["splits"][name]["relative_paths_sha256"],
                 f"ImageFolder-paden of volgorde zijn gewijzigd sinds de controle: {name}.")
        counts = [0] * EXPECTED_CLASSES
        for _, target in source.samples:
            counts[target] += 1
        _require({name: counts[index] for index, name in enumerate(source.classes)} ==
                 manifest["splits"][name]["per_class_counts"],
                 f"ImageFolder-klasaantallen zijn gewijzigd sinds de controle: {name}.")
    train_indices, val_indices = _select_indices(len(train_source), val_source.targets,
                                                EXPECTED_CLASSES, TRAIN_SAMPLES, VAL_PER_CLASS, seed)
    train_data = SelectedImageDataset(train_source, train_indices)
    val_data = SelectedImageDataset(val_source, val_indices)
    val_counts = [0] * EXPECTED_CLASSES
    train_counts = [0] * EXPECTED_CLASSES
    for target in val_data.targets:
        val_counts[target] += 1
    for target in train_data.targets:
        train_counts[target] += 1
    _require(val_counts == [VAL_PER_CLASS] * EXPECTED_CLASSES, "Validatie is niet exact vijf beelden per klasse.")
    del train_source, val_source
    train_loader, val_loader = _make_loaders(train_data, val_data, workers=workers, seed=seed,
                                            train_batch_size=TRAIN_BATCH_SIZE, val_batch_size=VAL_BATCH_SIZE)
    _require(len(train_loader) == TRAIN_BATCHES and len(train_data) == TRAIN_SAMPLES,
             "De trainingsloader bevat niet precies 200 complete batches.")
    _require(len(val_loader) == math.ceil(VAL_SAMPLES / VAL_BATCH_SIZE), "Onverwachte validatielengte.")
    metadata = {
        "dataset": "ImageNet-1k / ILSVRC2012", "root": str(root),
        "training_scope": "First 51,200 entries of one seeded random permutation of the complete training split; no replacement.",
        "validation_scope": "Seeded stratified subset: five images from each of 1,000 validation classes, shuffled once.",
        "train_samples": TRAIN_SAMPLES, "train_batches": TRAIN_BATCHES, "train_batch_size": TRAIN_BATCH_SIZE,
        "validation_samples": VAL_SAMPLES, "validation_batches": len(val_loader), "validation_batch_size": VAL_BATCH_SIZE,
        "validation_images_per_class": VAL_PER_CLASS, "source_train_samples": EXPECTED_TRAIN,
        "source_validation_samples": EXPECTED_VAL, "complete_epoch": False,
        "final_ImageNet_accuracy": False, "workers": workers,
        "prefetch_factor": 2 if workers else None, "persistent_workers": False,
        "multiprocessing_context": "spawn" if workers else None,
        "worker_timeout_seconds": 120 if workers else None,
        "pin_memory": True,
        "rng_seeds": {"train_permutation": seed, "validation_selection": seed + 1,
                      "train_worker_generator": seed + 2, "validation_worker_generator": seed + 3},
        "train_indices_sha256": _index_digest(train_indices), "validation_indices_sha256": _index_digest(val_indices),
        "train_selected_paths_sha256": _sample_path_digest(train_data.samples, root / "train"),
        "validation_selected_paths_sha256": _sample_path_digest(val_data.samples, root / "val"),
        "train_selected_class_count_min": min(train_counts), "train_selected_class_count_max": max(train_counts),
        "validation_selected_class_count_min": min(val_counts), "validation_selected_class_count_max": max(val_counts),
        "train_transform": str(train_transform), "validation_transform": str(val_transform),
        "selection_rng_isolated_from_model": True,
        "note": "The selection digest identifies sample order, not image contents. Worker count is part of the runtime "
                "configuration; random augmentations are not promised identical across worker counts. No dataset files are changed.",
    }
    if progress:
        progress("Data klaar: 51.200 trainingsbeelden in 200 batches; 5.000 validatiebeelden in 79 batches.")
    return train_loader, val_loader, metadata
