"""Paired frozen inference on eight paths from the same archived checkpoint.

Only stored MX exports are converted. No optimizer, backward, training-mode
forward, BatchNorm recalibration, master quantizer or scale correction exists
in this evaluator. All paths are measured again on each identical test batch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import joint_adapter as adapter
import resume_core as core
import svhn_support as support
import truncation as tr

require = support.require
COMPARISONS = (
    ('direct8', 'direct6'), ('direct8', 'direct4'), ('direct6', 'direct4'),
    ('direct4', 'round64'), ('direct4', 'mantissa64'), ('round64', 'mantissa64'),
    ('direct6', 'round86'), ('direct6', 'exponent86'), ('round86', 'exponent86'),
    ('direct4', 'combined864'), ('mantissa64', 'combined864'),
)


def read_export(path: Path, expected_sha256: str) -> dict:
    marker = path.with_suffix(path.suffix + '.sha256')
    require(path.is_file() and marker.is_file(), 'MX-export of checksum ontbreekt: ' + str(path))
    digest = core.file_sha256(path)
    require(digest == expected_sha256 == marker.read_text(encoding='ascii').strip(),
            'MX-export faalt checksum of rapportidentiteit: ' + path.name)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(type(payload) is dict, 'MX-export is geen woordenboek.')
    return payload


def assemble_paths(models, direct: dict, source_hashes: dict, out: Path,
                   role: str, device) -> tuple[dict, dict, list, list]:
    """Persist/reload every derived payload before reconstructing its model."""
    check = tr.check_source_set(direct)
    derived, exports = {}, {}
    for key in tr.PATHS:
        if key.startswith('direct'):
            continue
        spec = tr.PATH_SPEC[key]
        source_bit = spec['source_bits']
        payload = tr.derive_payload(direct[source_bit], key, source_hashes[source_bit])
        path = out / f'{role}_{key}_derived.pt'
        saved = core.atomic_save(path, payload)
        reloaded = read_export(path, saved['sha256'])
        require(not core.differences(payload, reloaded), 'Afgeleide export gewijzigd na herladen.')
        derived[key] = reloaded
        exports[key] = {**saved, 'source_file_sha256': source_hashes[source_bit],
                        'source_weight_bits': source_bit, 'target_weight_bits': spec['target_bits'],
                        'source_scale_bytes_copied': True,
                        'master_weights_read_to_derive': False, 'saved_and_reloaded': True}
    weights, counts = tr.weight_diagnostics(direct, derived, device)
    paths = {}
    for key in tr.PATHS:
        if key.startswith('direct'):
            paths[key] = adapter.load_exported_model(models, direct[tr.PATH_SPEC[key]['target_bits']], device)
        else:
            paths[key] = tr.load_derived_model(models, derived[key], device)
        require(not {n + '.weight' for n in adapter.TARGET_LAYERS}.intersection(dict(paths[key].named_parameters())),
                'Een MX-laag bevat onverwacht een FP32-master: ' + key)
    return paths, {**check, 'derived_exports': exports}, weights, counts


def frozen_state_check(model, key: str) -> str:
    require(all(not m.training for m in model.modules()), 'Model of BN staat niet in eval(): ' + key)
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Gradient actief: ' + key)
    require(all(m.abit == 32 for m in model.modules() if hasattr(m, 'abit')), 'A32 gewijzigd: ' + key)
    expected = tr.PATH_SPEC[key]['target_bits']
    require(model.weight_mode == expected, 'Verkeerde gewichtsmodus: ' + key)
    for name in adapter.BN_LAYERS:
        bank = model.get_submodule(name)
        require(bank.weight_mode == expected and adapter.BANK[bank.weight_mode] == adapter.BANK[expected],
                'Verkeerde getrainde BN-bank in ' + key)
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
            'correct_to_wrong': correct_to_wrong, 'both_correct': int((rc & oc).sum()),
            'both_wrong': int((~rc & ~oc).sum()),
            'wrong_to_other_wrong': int((~rc & ~oc & changed).sum()),
            'net_correct_change': wrong_to_correct - correct_to_wrong,
            'mean_abs_logit_difference': float(np.abs(error).mean()),
            'max_abs_logit_difference': float(np.abs(error).max())}


def evaluate_paths(paths: dict, loader, expected_n: int, device, out: Path,
                   role: str, epoch: int, log) -> tuple[list[dict], list[dict], dict]:
    require(tuple(paths) == tr.PATHS, 'Onvolledige achtpadenevaluatie.')
    states = {key: frozen_state_check(model, key) for key, model in paths.items()}
    storage = {key: {'logits': [], 'predictions': [], 'per_image_ce': [], 'correct5': 0}
               for key in paths}
    indices, labels = [], []
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
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
                # Same top-k convention as upstream accuracy, including ties.
                top5 = logits.topk(5, dim=1, largest=True, sorted=True).indices
                row = storage[key]
                row['logits'].append(logits.cpu().numpy())
                row['predictions'].append(top5[:, 0].cpu().numpy())
                row['per_image_ce'].append(F.cross_entropy(logits, y, reduction='none').cpu().numpy())
                row['correct5'] += int((top5 == y[:, None]).any(dim=1).sum())
            if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(loader):
                log(f'{role}: batch {i + 1}/{len(loader)}; acht paden; uitsluitend evaluatie.')
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    require(bool(indices), 'Lege testset.')
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
        spec = tr.PATH_SPEC[key]
        bit = spec['target_bits']
        records.append({'role': role, 'epoch': epoch, 'path': key, 'weight_bits': bit,
                        'weight_format': adapter.FORMATS[bit], 'source_weight_bits': spec['source_bits'],
                        'family': spec['family'], 'direct_baseline': spec['direct_baseline'],
                        'activation_bits': 32, 'bn_bank_key': adapter.BANK[bit],
                        'n': expected_n, 'correct1': ncorrect, 'correct5': result['correct5'],
                        'top1_percent': 100 * ncorrect / expected_n,
                        'top5_percent': 100 * result['correct5'] / expected_n,
                        'hard_cross_entropy': float(result['per_image_ce'].astype(np.float64).mean()),
                        'optimizer_steps': 0, 'batchnorm_recalibrated': False})
    paired = []
    for rr, oo in COMPARISONS:
        row = {'role': role, 'epoch': epoch, 'comparison': oo + '_minus_' + rr,
               'reference_path': rr, 'other_path': oo,
               **paired_outputs(storage[rr], storage[oo], yy)}
        require(row['net_correct_change'] == row['other_correct1'] - row['reference_correct1'],
                'Onjuiste gepaarde telling.')
        paired.append(row)
    outputs = out / f'{role}_per_image_outputs.npz'
    with outputs.open('xb') as handle:
        np.savez_compressed(handle, **npz_data)
    for key, model in paths.items():
        require(frozen_state_check(model, key) == states[key], 'Model- of BN-staat gewijzigd tijdens evaluatie.')
    details = {'all_model_and_bn_states_unchanged': True, 'all_models_no_grad': True,
               'all_paths_use_frozen_target_precision_bn': True,
               'six_FP32_master_parameters_absent_all_paths': True,
               'optimizer_steps': 0, 'batchnorm_recalibrated': False, 'n_test': expected_n,
               'label_sha256': hashlib.sha256(yy.tobytes()).hexdigest(),
               'sample_order_sha256': hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest(),
               'paired_evaluation_seconds_not_a_speed_benchmark': seconds,
               'per_image_file': outputs.name, 'per_image_sha256': core.file_sha256(outputs),
               'per_image_data_in_feedback': True}
    return records, paired, details
