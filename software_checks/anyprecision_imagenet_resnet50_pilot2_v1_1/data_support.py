"""Read-only ImageNet-1k inventory and the original ImageFolder data recipe.

No download, extraction, move, label inference, or dataset cache is performed.
The inventory verifies structure/counts; its path/size digest is not an image
content hash and does not establish that labels are semantically correct.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Callable


EXPECTED_CLASSES = 1000
EXPECTED_TRAIN = 1_281_167
EXPECTED_VAL = 50_000
EXPECTED_VAL_PER_CLASS = 50
WNID = re.compile(r"n[0-9]{8}\Z")
# torchvision.datasets.folder.IMG_EXTENSIONS, including the historical formats.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageNetDataError(RuntimeError):
    """Dataset is absent, unprepared, incomplete, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImageNetDataError(message)


def _classes(split: Path) -> list[str]:
    with os.scandir(split) as entries:
        names = sorted(entry.name for entry in entries if entry.is_dir())
    invalid = [name for name in names if WNID.fullmatch(name) is None]
    _require(not invalid,
             f"{split.name}: onverwachte klasmappen {invalid[:5]}. "
             "Verwacht uitsluitend WordNet-klassen zoals n01440764; "
             "wijs --imagenet-root naar de map die train en val bevat.")
    return names


def _image_files(class_dir: Path):
    """Yield in ImageFolder order while refusing recursive directory cycles."""
    visited: set[str] = set()
    for current, directories, filenames in os.walk(class_dir, followlinks=True):
        resolved = os.path.normcase(os.path.realpath(current))
        _require(resolved not in visited,
                 f"Herhaalde of cyclische mapkoppeling gevonden onder {class_dir}.")
        visited.add(resolved)
        directories.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                yield Path(current) / filename


def _decode_samples(paths: list[Path]) -> list[dict[str, Any]]:
    from PIL import Image

    records: list[dict[str, Any]] = []
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
    """Inventory train/val without modifying either split.

    ``strict=False`` is only for tiny local validation fixtures. The public
    runner always uses strict=True and build_loaders requires full ImageNet.
    """
    root = Path(root).expanduser().resolve()
    _require(root.is_dir(),
             f"ImageNet-map ontbreekt: {root}. De synthetische check heeft geen data nodig. "
             "Voor de echte pilot: geef --imagenet-root met een bestaande, voorbereide "
             "ILSVRC2012-map met train/<wnid>/ en val/<wnid>/. Er wordt niets gedownload.")
    train = root / "train"
    val = root / "val"
    if not train.is_dir() or not val.is_dir():
        archives = sorted(p.name for p in root.iterdir()
                          if p.is_file() and p.name.lower().endswith((".tar", ".tar.gz", ".tgz", ".zip")))
        detail = (f" Gevonden archieven: {archives[:5]}; archieven zijn nog geen ImageFolder-dataset."
                  if archives else "")
        raise ImageNetDataError(
            f"{root} moet zowel train als val bevatten, beide met klasmappen n########."
            + detail + " Het testpakket pakt data niet uit en verplaatst geen bestanden.")

    classes_train, classes_val = _classes(train), _classes(val)
    if not classes_val:
        raw_count = sum(1 for p in val.iterdir()
                        if p.is_file() and p.name.lower().endswith(IMAGE_EXTENSIONS))
        _require(not raw_count,
                 f"val bevat {raw_count} losse afbeeldingen zonder klasmappen. "
                 "Gebruik de officiële ILSVRC2012-validatielabels om vooraf "
                 "val/<wnid>/<afbeelding> voor te bereiden. Deze controle raadt geen labels.")
    _require(classes_train and classes_val, "train of val bevat geen geldige klasmappen.")
    _require(classes_train == classes_val,
             "De gesorteerde WNID-klassen van train en val verschillen. "
             f"Alleen train: {sorted(set(classes_train)-set(classes_val))[:5]}; "
             f"alleen val: {sorted(set(classes_val)-set(classes_train))[:5]}.")
    if strict:
        _require(len(classes_train) == EXPECTED_CLASSES,
                 f"Verwacht {EXPECTED_CLASSES} klassen; gevonden {len(classes_train)}. "
                 "Een subset is niet toegestaan voor deze ImageNet-pilot.")
    mapping = {name: index for index, name in enumerate(classes_train)}
    map_digest = hashlib.sha256(json.dumps(mapping, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
    splits: dict[str, Any] = {}
    sample_indices = {0, len(classes_train) // 2, len(classes_train) - 1}
    for split_name, split, expected in [("train", train, EXPECTED_TRAIN),
                                        ("val", val, EXPECTED_VAL)]:
        fingerprint = hashlib.sha256()
        per_class: dict[str, int] = {}
        total_count, total_bytes = 0, 0
        sampled: list[Path] = []
        for index, class_name in enumerate(classes_train):
            count = 0
            first: Path | None = None
            last: Path | None = None
            for path in _image_files(split / class_name):
                size = path.stat().st_size
                _require(size > 0, f"Bestand met nul bytes: {path}")
                relative = path.relative_to(split).as_posix()
                fingerprint.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\n")
                count += 1
                total_bytes += size
                if first is None:
                    first = path
                last = path
            _require(count > 0, f"Lege klasse: {split / class_name}")
            if strict and split_name == "val":
                _require(count == EXPECTED_VAL_PER_CLASS,
                         f"val/{class_name}: verwacht 50 afbeeldingen, gevonden {count}. "
                         "Controleer de officiële validatie-indeling.")
            per_class[class_name] = count
            total_count += count
            if index in sample_indices:
                if first is not None:
                    sampled.append(first)
                if last is not None and last != first:
                    sampled.append(last)
            if progress is not None and ((index + 1) % 200 == 0 or index + 1 == len(classes_train)):
                progress(f"ImageNet-inventaris {split_name}: {index + 1}/{len(classes_train)} "
                         f"klassen; {total_count:,} afbeeldingen.")
        if strict:
            _require(total_count == expected,
                     f"{split_name}: verwacht {expected:,} afbeeldingen, gevonden {total_count:,}. "
                     "Alleen de volledige ILSVRC2012-train- en validatiesplits zijn toegestaan.")
        decoded = _decode_samples(sampled)
        for record, path in zip(decoded, sampled):
            record["relative_path"] = path.relative_to(split).as_posix()
        splits[split_name] = {
            "path": str(split), "count": total_count, "expected_count": expected,
            "file_bytes": total_bytes, "filelist_sha256": fingerprint.hexdigest(),
            "per_class_counts": per_class, "decode_samples": decoded,
        }
    return {
        "schema_version": 1, "dataset": "ImageNet-1k / ILSVRC2012",
        "root": str(root), "strict": strict, "class_count": len(classes_train),
        "classes": classes_train, "class_to_idx_sha256": map_digest,
        "train_count": splits["train"]["count"], "val_count": splits["val"]["count"],
        "splits": splits,
        "checks": {"matching_class_order": True, "nonempty_classes": True,
                   "expected_full_counts": strict, "sample_decode_passed": True},
        "image_extensions": list(IMAGE_EXTENSIONS),
        "note": ("Alleen structuur, aantallen, pad/grootte-inventaris en enkele decodes gecontroleerd. "
                 "Geen volledige inhoudshashes of onafhankelijke verificatie van labels. "
                 "Er zijn geen datasetbestanden gewijzigd."),
        "sources": ["https://ftp.image-net.org/download",
                    "https://image-net.org/challenges/LSVRC/2013/2013-downloads.php"],
    }


def seed_worker(worker_id: int) -> None:
    """Module-level callback, picklable under Windows DataLoader spawn."""
    import numpy as np
    import torch

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(root: str | Path, batch_size: int = 256, workers: int = 4,
                  seed: int = 20261009, val_batch_size: int = 64,
                  loader_generator_state: Any = None):
    """Build full ImageNet loaders after audit_imagenet(root, strict=True).

    Evaluation batch64 only partitions independent eval samples; BN is frozen.
    Training batch256 is always retained. The engine handles the four replicas.
    """
    import torch
    from torchvision import datasets, transforms
    from torchvision.transforms import InterpolationMode

    _require(batch_size == 256, "Trainbatch moet 256 blijven in dit pakket.")
    _require(workers >= 0, "Aantal dataloader-workers moet minstens nul zijn.")
    _require(val_batch_size > 0, "Validatiebatch moet positief zijn.")
    root = Path(root).expanduser().resolve()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    # Explicit PIL loading matches ImageFolder's default without an accimage backend.
    train_data = datasets.ImageFolder(str(root / "train"), transform=train_transform,
                                     loader=datasets.folder.pil_loader)
    val_data = datasets.ImageFolder(str(root / "val"), transform=val_transform,
                                   loader=datasets.folder.pil_loader)
    _require(train_data.class_to_idx == val_data.class_to_idx,
             "ImageFolder-klassevolgorde veranderde sinds de data-audit.")
    _require(len(train_data.classes) == EXPECTED_CLASSES
             and all(WNID.fullmatch(name) for name in train_data.classes),
             "ImageFolder vereist precies 1.000 geldige WNID-klassen.")
    _require(len(train_data) == EXPECTED_TRAIN and len(val_data) == EXPECTED_VAL,
             "ImageFolder-aantallen wijken af van de volledige ImageNet-splits.")

    generator = torch.Generator()
    if loader_generator_state is None:
        generator.manual_seed(seed)
    else:
        generator.set_state(loader_generator_state.cpu())
    # Separate validation generator means validation worker creation cannot advance
    # the training sampler's RNG. Its transforms are deterministic.
    validation_generator = torch.Generator().manual_seed(seed + 1)
    common = {"num_workers": workers, "pin_memory": True,
              "drop_last": False, "persistent_workers": False,
              "worker_init_fn": seed_worker}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, generator=generator, **common)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size,
                                             shuffle=False, generator=validation_generator, **common)
    return train_loader, val_loader, generator
