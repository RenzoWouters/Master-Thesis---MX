"""Paired frozen inference on nine paths from one archived checkpoint group.

Only stored MX exports are converted. No optimizer, backward, training-mode
forward, BatchNorm recalibration, master quantizer or learned scale exists
in this evaluator. All paths are measured again on each identical test batch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

import joint_adapter as adapter
import resume_core as core
import svhn_support as support
import scale64 as tr

require = support.require
COMPARISONS = tr.COMPARISONS


def read_export(path: Path, expected_sha256: str) -> dict:
    marker = path.with_suffix(path.suffix + '.sha256')
    require(path.is_file() and marker.is_file(), 'MX-export of checksum ontbreekt: ' + str(path))
    digest = core.file_sha256(path)
    require(digest == expected_sha256 == marker.read_text(encoding='ascii').strip(),
            'MX-export faalt checksum of rapportidentiteit: ' + path.name)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(type(payload) is dict, 'MX-export is geen woordenboek.')
    return payload


def assemble_paths(models, payloads: dict, source_hashes: dict, out: Path,
                   role: str, device, training_kind: str) -> tuple[dict, dict, list, list]:
    """Validate B-trained source exports and persist/reload all six A/B/C payloads."""
    check = tr.check_source_set(payloads, training_kind, source_hashes)
    derived, exports = {}, {}
    for key in tr.DERIVED_PATHS:
        spec = tr.PATH_SPEC[key]
        payload = tr.derive_payload(payloads[8], key, source_hashes[8], training_kind)
        path = out / f'{role}_{key}_derived.pt'
        saved = core.atomic_save(path, payload)
        reloaded = read_export(path, saved['sha256'])
        tr.validate_derived(reloaded)
        require(not core.differences(payload, reloaded), 'Afgeleide export gewijzigd na herladen.')
        derived[key] = reloaded
        exports[key] = {**saved, 'source_file_sha256': source_hashes[8],
            'source_weight_bits': 8, 'target_weight_bits': spec['target_bits'],
            'training_kind': training_kind, 'family': spec['family'],
            'scale_exponent_offset': spec['scale_exponent_offset'],
            'scale_factor': 1 if spec['family'] == 'A' else 64,
            'source_scale_bytes_copied': spec['family'] == 'A',
            'master_weights_read_to_derive': False, 'saved_and_reloaded': True,
            'tensor_scale_reselection': False, 'batchnorm_recalibrated': False}
    weights, counts = tr.weight_diagnostics(payloads, derived, device)
    paths = {}
    for key in tr.PATHS:
        if key in tr.DERIVED_PATHS:
            paths[key] = tr.load_derived_model(models, derived[key], device)
        else:
            paths[key] = tr.load_trained_model(models, payloads[tr.PATH_SPEC[key]['target_bits']], device, training_kind)
        require(not {n + '.weight' for n in adapter.TARGET_LAYERS}.intersection(dict(paths[key].named_parameters())),
                'Een MX-laag bevat onverwacht een FP32-master: ' + key)
    for bit in (6, 4):
        require(frozen_state_check(paths['trained' + str(bit)], 'trained' + str(bit))
                == frozen_state_check(paths['B' + str(bit)], 'B' + str(bit)),
                'B-reconstructie heeft andere bevroren model-/BN-staat dan de opgeslagen trainingsmodus.')
    return paths, {**check, 'derived_exports': exports,
        'B_same_frozen_state_as_trained': True,
        'one_E4_source_for_all_derived': True, 'A_source_scales_unchanged': True,
        'BC_source_scale_exponent_offset': 6, 'B_element_codes_equal_A': True,
        'C_normalized_rounding_input': 'decoded_E4_elements_div64',
        'mantissa_shift_after_each_E2M3_path': True,
        'original_auxiliary_state_preserved': True, 'no_tensor_scale_reselection': True}, weights, counts


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
    require(tuple(paths) == tr.PATHS, 'Onvolledige negenpadenevaluatie.')
    states = {key: frozen_state_check(model, key) for key, model in paths.items()}
    for bit in (6, 4):
        require(states['B' + str(bit)] == states['trained' + str(bit)],
                'B en de opgeslagen trainingsmodus hebben niet dezelfde model-/BN-staat.')
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
                log(f'{role}: batch {i + 1}/{len(loader)}; negen paden; uitsluitend evaluatie.')
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
                        'scale_exponent_offset': spec['scale_exponent_offset'],
                        'activation_bits': 32, 'bn_bank_key': adapter.BANK[bit],
                        'n': expected_n, 'correct1': ncorrect, 'correct5': result['correct5'],
                        'top1_percent': 100 * ncorrect / expected_n,
                        'top5_percent': 100 * result['correct5'] / expected_n,
                        'hard_cross_entropy': float(result['per_image_ce'].astype(np.float64).mean()),
                        'optimizer_steps': 0, 'batchnorm_recalibrated': False,
                        'classes_predicted': int(np.count_nonzero(np.bincount(result['predictions'], minlength=10))),
                        'dominant_predicted_class': int(np.bincount(result['predictions'], minlength=10).argmax()),
                        'dominant_prediction_fraction': float(np.bincount(result['predictions'], minlength=10).max() / expected_n)})
    paired = []
    for rr, oo in COMPARISONS:
        row = {'role': role, 'epoch': epoch, 'comparison': oo + '_minus_' + rr,
               'reference_path': rr, 'other_path': oo,
               **paired_outputs(storage[rr], storage[oo], yy)}
        require(row['net_correct_change'] == row['other_correct1'] - row['reference_correct1'],
                'Onjuiste gepaarde telling.')
        paired.append(row)
    reproduction = {}
    for bit in (6, 4):
        baseline, converted = storage['trained' + str(bit)], storage['B' + str(bit)]
        paired_row = paired_outputs(baseline, converted, yy)
        exact = np.array_equal(baseline['logits'], converted['logits'])
        close = bool(np.allclose(baseline['logits'], converted['logits'], rtol=1e-4, atol=1e-5))
        require(close, 'B-logits wijken materieel af van de identieke opgeslagen trainingsmodus.')
        ce_error = np.abs(converted['per_image_ce'].astype(np.float64)
                          - baseline['per_image_ce'].astype(np.float64))
        reproduction[str(bit)] = {
            'logits_exact': exact, 'logits_allclose': close,
            'logit_rtol': 1e-4, 'logit_atol': 1e-5,
            'predictions_equal': paired_row['prediction_changes'] == 0,
            'prediction_changes': paired_row['prediction_changes'],
            'correct1_difference': paired_row['net_correct_change'],
            'max_abs_logit_difference': paired_row['max_abs_logit_difference'],
            'ce_exact': bool(np.array_equal(baseline['per_image_ce'], converted['per_image_ce'])),
            'max_abs_ce_difference': float(ce_error.max())}
        if not exact:
            warnings.warn(f'{role}/B{bit}: dezelfde modelstaat, maar niet-bitexacte herhaalde logits; '
                          f'maximaal verschil={paired_row["max_abs_logit_difference"]:.8g}, '
                          f'gewijzigde voorspellingen={paired_row["prediction_changes"]}. '
                          'De oorspronkelijke CUDA-backend garandeert geen bitexacte herhaling.', RuntimeWarning)
    outputs = out / f'{role}_per_image_outputs.npz'
    with outputs.open('xb') as handle:
        np.savez_compressed(handle, **npz_data)
    for key, model in paths.items():
        require(frozen_state_check(model, key) == states[key], 'Model- of BN-staat gewijzigd tijdens evaluatie.')
    details = {'B_same_run_reproduction': reproduction,
               'all_model_and_bn_states_unchanged': True, 'all_models_no_grad': True,
               'all_paths_use_frozen_target_precision_bn': True,
               'six_FP32_master_parameters_absent_all_paths': True,
               'optimizer_steps': 0, 'batchnorm_recalibrated': False, 'n_test': expected_n,
               'label_sha256': hashlib.sha256(yy.tobytes()).hexdigest(),
               'sample_order_sha256': hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest(),
               'paired_evaluation_seconds_not_a_speed_benchmark': seconds,
               'per_image_file': outputs.name, 'per_image_sha256': core.file_sha256(outputs),
               'per_image_data_in_feedback': True}
    return records, paired, details
