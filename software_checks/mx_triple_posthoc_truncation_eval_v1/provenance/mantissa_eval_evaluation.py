"""Read-only paired inference of H/A/B/C; raw per-image outputs are retained.

This module never constructs an optimizer, calls train(), backward(), or
recalibrates BatchNorm. A/B/C use precisely the same trained low bank.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import dual_adapter as adapter
import resume_core as core
import svhn_support as support
import truncation as tr

require = tr.require


def read_export(path: Path, expected_sha256: str) -> dict:
    """Exports are tensors and plain containers, never arbitrary pickle classes."""
    marker = path.with_suffix(path.suffix + '.sha256')
    require(path.is_file() and marker.is_file(), 'MX-export of checksum ontbreekt: ' + str(path))
    digest = core.file_sha256(path)
    require(digest == expected_sha256 == marker.read_text(encoding='ascii').strip(),
            'MX-export faalt checksum of komt niet overeen met run_report.json: ' + path.name)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(type(payload) is dict, 'MX-export is geen woordenboek.')
    return payload


def assemble_paths(models, source: dict, direct: dict, source_digest: str,
                   out: Path, role: str, device) -> tuple[dict, dict, list, list]:
    """All conversion work precedes inference. Derived files are actually saved
    and reloaded. Nothing calls a master quantizer to produce B or C.
    """
    check = tr.check_source_pair(source, direct)
    derived = {}; exports = {}
    for key, method in [('B', 'rounded_from_source'), ('C', 'mantissa_truncation')]:
        payload = tr.derive_payload(source, method, source_digest)
        path = out / f'{role}_{key}_{method}_E2M1.pt'
        saved = core.atomic_save(path, payload)
        reloaded = read_export(path, saved['sha256'])
        require(not core.differences(payload, reloaded), 'Afgeleide export verandert na opslaan/herladen.')
        derived[key] = reloaded
        exports[key] = {**saved, 'method': method, 'source_file_sha256': source_digest,
                        'master_weights_read_to_derive': False, 'source_scale_bytes_copied': True,
                        'saved_and_reloaded': True}
    pairs, counts = tr.weight_diagnostics(source, direct, derived['B'], derived['C'])
    paths = {
        tr.PATHS[0]: adapter.load_exported_model(models, source, device),
        tr.PATHS[1]: adapter.load_exported_model(models, direct, device),
        tr.PATHS[2]: tr.load_derived_model(models, derived['B'], device),
        tr.PATHS[3]: tr.load_derived_model(models, derived['C'], device),
    }
    for model in paths.values():
        require(not {n + '.weight' for n in adapter.TARGET_LAYERS}.intersection(dict(model.named_parameters())),
                'Gekwantiseerde laag bevat onverwacht een FP32-master.')
    return paths, {**check, 'derived_exports': exports}, pairs, counts


def frozen_state_check(model, key: str) -> str:
    require(all(not m.training for m in model.modules()), 'Model of BN staat niet in eval(): ' + key)
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Gradient actief: ' + key)
    require(all(m.abit == 32 for m in model.modules() if hasattr(m, 'abit')), 'A32 gewijzigd: ' + key)
    expected = 6 if key == tr.PATHS[0] else 4
    require(model.weight_mode == expected, 'Verkeerde gewichtsmodus.')
    for name in adapter.BN_LAYERS:
        bank = model.get_submodule(name)
        require(bank.weight_mode == expected and adapter.BANK[bank.weight_mode] == ('32' if expected == 6 else '4'),
                'Verkeerde BN-bank in ' + key)
    return support.legacy.state_digest(model.state_dict())


def paired_outputs(reference: dict, other: dict, labels: np.ndarray) -> dict:
    r, o = reference['predictions'], other['predictions']
    rc, oc = r == labels, o == labels
    changed = r != o
    wrong_to_correct = int((~rc & oc).sum())
    correct_to_wrong = int((rc & ~oc).sum())
    n = len(labels)
    error = other['logits'].astype(np.float64) - reference['logits'].astype(np.float64)
    return {'n': n, 'reference_correct1': int(rc.sum()), 'other_correct1': int(oc.sum()),
            'accuracy_difference_pp': 100 * (int(oc.sum()) - int(rc.sum())) / n,
            'prediction_changes': int(changed.sum()), 'wrong_to_correct': wrong_to_correct,
            'correct_to_wrong': correct_to_wrong,
            'both_correct': int((rc & oc).sum()), 'both_wrong': int((~rc & ~oc).sum()),
            'wrong_to_other_wrong': int((~rc & ~oc & changed).sum()),
            'net_correct_change': wrong_to_correct - correct_to_wrong,
            'mean_abs_logit_difference': float(np.abs(error).mean()),
            'max_abs_logit_difference': float(np.abs(error).max())}


def evaluate_paths(paths: dict, loader, expected_n: int, device, out: Path,
                   role: str, epoch: int, log) -> tuple[list[dict], list[dict], dict]:
    require(tuple(paths) == tr.PATHS, 'Onvolledige H/A/B/C-evaluatie.')
    states = {key: frozen_state_check(model, key) for key, model in paths.items()}
    storage = {key: {'logits': [], 'predictions': [], 'per_image_ce': [], 'correct5': 0}
               for key in paths}
    indices, labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for i, (x, y, idx) in enumerate(loader):
            require(tuple(x.shape[1:]) == (3, 40, 40) and x.dtype == torch.float32, 'Afwijkende invoer.')
            require(bool(torch.isfinite(x).all()), 'Niet-eindige invoer.')
            indices.append(idx.cpu().numpy().astype(np.int64, copy=False))
            labels.append(y.cpu().numpy().astype(np.int64, copy=False))
            x, y = x.to(device), y.to(device)
            for key, model in paths.items():
                logits = model(x)
                require(tuple(logits.shape) == (y.shape[0], 10) and bool(torch.isfinite(logits).all()),
                        'Ongeldige logits in ' + key)
                # Same top-k convention as upstream accuracy(); ties are not
                # silently resolved differently with a separate argmax call.
                top5 = logits.topk(5, dim=1, largest=True, sorted=True).indices
                row = storage[key]
                row['logits'].append(logits.cpu().numpy())
                row['predictions'].append(top5[:, 0].cpu().numpy())
                row['per_image_ce'].append(F.cross_entropy(logits, y, reduction='none').cpu().numpy())
                row['correct5'] += int((top5 == y[:, None]).any(dim=1).sum())
            if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(loader):
                log(f'{role}: batch {i + 1}/{len(loader)}; vier paden, geen updates.')
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    ids, yy = np.concatenate(indices), np.concatenate(labels)
    require(len(ids) == expected_n and np.array_equal(ids, np.arange(expected_n)),
            'Niet exact eenmaal de volledige testset in oorspronkelijke volgorde.')
    npz_data = {'indices': ids, 'labels': yy}
    records = []
    for key, result in storage.items():
        for field in ('logits', 'predictions', 'per_image_ce'):
            result[field] = np.concatenate(result[field])
            require(np.isfinite(result[field]).all(), 'Niet-eindige opgeslagen uitvoer.')
            npz_data[key + '__' + field] = result[field]
        ncorrect = int((result['predictions'] == yy).sum())
        bit = 6 if key == tr.PATHS[0] else 4
        records.append({'role': role, 'epoch': epoch, 'path': key, 'weight_bits': bit,
                        'activation_bits': 32, 'bn_bank_key': adapter.BANK[bit],
                        'n': expected_n, 'correct1': ncorrect, 'correct5': result['correct5'],
                        'top1_percent': 100 * ncorrect / expected_n,
                        'top5_percent': 100 * result['correct5'] / expected_n,
                        'hard_cross_entropy': float(result['per_image_ce'].astype(np.float64).mean()),
                        'optimizer_steps': 0, 'batchnorm_recalibrated': False})
    paired = []
    for ref, other in [(1, 3), (1, 2), (2, 3), (0, 1)]:
        rr, oo = tr.PATHS[ref], tr.PATHS[other]
        row = {'role': role, 'epoch': epoch, 'comparison': oo[0] + '_minus_' + rr[0],
               'reference_path': rr, 'other_path': oo,
               **paired_outputs(storage[rr], storage[oo], yy)}
        require(row['net_correct_change'] == row['other_correct1'] - row['reference_correct1'], 'Onjuiste gepaarde telling.')
        paired.append(row)
    outputs = out / f'{role}_per_image_outputs.npz'
    with outputs.open('xb') as handle:
        np.savez_compressed(handle, **npz_data)
    for key, model in paths.items():
        require(frozen_state_check(model, key) == states[key], 'Model- of BN-staat gewijzigd tijdens evaluatie.')
    details = {'all_model_and_bn_states_unchanged': True, 'all_models_no_grad': True,
               'low_paths_all_use_same_trained_bank4': True, 'six_FP32_master_parameters_absent_all_paths': True,
               'optimizer_steps': 0, 'n_test': expected_n,
               'label_sha256': hashlib.sha256(yy.tobytes()).hexdigest(),
               'sample_order_sha256': hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest(),
               'paired_evaluation_seconds_not_a_speed_benchmark': seconds,
               'per_image_file': outputs.name, 'per_image_sha256': core.file_sha256(outputs),
               'per_image_data_in_feedback': True}
    return records, paired, details
