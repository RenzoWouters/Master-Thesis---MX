"""Read-only numerical and SVHN probe of mantissa-MSB removal in standard E2M1.

The real-model command consumes one pinned compact export. It never reads
FP32 masters, trains, changes BN statistics, or rewrites research inputs.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import warnings
import zipfile

import mantissa_rules as rules

ROOT = Path(__file__).resolve().parent
REVISION = 'mx_mantissa_msb_exponent_probe_v1'
VARIANTS = ('source8', 'source6', 'lsb_trunc', 'rne', 'msb_bestexp')
BITS = {'source8': 8, 'source6': 6, 'lsb_trunc': 4, 'rne': 4, 'msb_bestexp': 4}
LABELS = {'source8': 'E4M3: ongewijzigde bron', 'source6': 'E2M3: ongewijzigde bron',
          'lsb_trunc': 'LSB verwijderen (bestaand)', 'rne': 'Afronding naar E2M1',
          'msb_bestexp': 'MSB verwijderen + optimale exponent'}
PROTOCOL = {
    'schema': 'mx-mantissa-msb-exponent-probe-protocol-v1',
    'stage': 'exploratory_numerical_and_posthoc_only',
    'source': 'pinned epoch100 compact E4 export from independent E2 scale freeze-after50 run',
    'model': 'CNN-8 / svhnq expand=2', 'activation_bits': 32,
    'block_size': 32, 'output_format': 'standard MX E2M1',
    'mantissa_bit_names': 'm2=magnitude 1/2, m1=1/4, m0=1/8 for normal E2M3 elements',
    'lsb_trunc': 'retain sign, exponent field E, and m2; delete m1,m0',
    'rne': 'nearest E2M1 value to the SAME E2M3 code, ties to even, saturating',
    'msb_bestexp': 'retain sign and m0, choose Eprime in 0..3 to minimize absolute source error',
    'msb_ties': 'minimize abs(Eprime-E), then choose smallest Eprime',
    'msb_equivalent_bit_rule': 'Eprime=min(3,E+int((mantissa&7)==6)); target_mantissa=m0',
    'optimization_scope': 'smallest per-element source error under fixed m0; NOT accuracy optimality',
    'shared_scales': 'copy existing E2 E8M0 bytes unchanged for all three E2M1 alternatives',
    'bn_policy': 'same saved bank4, fixed evaluation mode for every E2M1 alternative; no recalibration',
    'other_modes': 'E4 and E2M3 are loaded as unchanged controls',
    'training_performed': False, 'optimizer_created': False,
    'dataset': 'full SVHN official test, no train or extra loading',
    'batch_size': 128, 'shuffle': False, 'workers': 0,
    'error_reference': 'decoded E2M3 source, not unavailable FP32 master',
    'numerical_population': 'uniform code patterns with scale1, not observed weights',
    'accuracy_population': 'actual 26032 SVHN test images; only populated after evaluate',
    'hardware_claim': 'bit operations described; no measured area, latency or energy',
    'limitations': [
        'Parent weights and BN were trained with LSB truncation; posthoc results do not rule out QAT adaptation.',
        'One selected model and dataset; no general superiority or inferiority claim.',
        'Exponent tie policy is fixed before network evaluation.',
        'No test labels are used to select the conversion rule.',
        'The removed mantissa bits are still needed during conversion.',
        'Inference uses FP32 convolutions with decoded weights; its wall time is not low-bit hardware timing.'
    ],
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project_root():
    if ROOT.parent.name == 'software_checks':
        return ROOT.parent.parent
    return Path.cwd().resolve()


def load_reference():
    return json.loads((ROOT / 'REFERENCE.json').read_text(encoding='utf-8'))


def verify_package():
    path = ROOT / 'PACKAGE_SHA256.json'
    require(path.is_file(), 'PACKAGE_SHA256.json ontbreekt; pak het volledige experiment opnieuw uit.')
    expected = json.loads(path.read_text(encoding='utf-8'))
    for name, digest in expected.items():
        require((ROOT / name).is_file() and sha256(ROOT / name) == digest,
                'Pakketbestand ontbreekt of is gewijzigd: ' + name)
    return {'files_checked': len(expected), 'manifest_sha256': sha256(path)}


def numeric_stage(out):
    write_json(out / 'numeric_summary.json', rules.numeric_summary())
    write_csv(out / 'codebook_mapping.csv', rules.numeric_rows())
    equivalent = all(
        rules.map_code(code, 'msb_bestexp') ==
        (((code >> 5) << 3) | (min(3, ((code >> 3) & 3) + int((code & 7) == 6)) << 1) | (code & 1))
        for code in range(64)
    )
    require(equivalent, 'De eenvoudige MSB-bitregel verschilt van de exhaustieve optimale mapping.')
    hardware = {
        'all64_bit_rule_equals_exhaustive_optimum': equivalent,
        'sign_preserved': True, 'shared_scale_preserved': True,
        'msb_rule': 'Eprime=min(3,E+[m2 AND m1 AND NOT m0]); mprime=m0',
        'rne_rule': 'keep m2; increment=m1 AND (m0 OR m2); saturate magnitude code to7',
        'runtime_search_or_float_log_needed': False,
        'new_per_weight_metadata_bits': 0,
        'compact_low_code_bits': 4,
        'removed_bits_read_during_conversion': True,
        'measured_hardware_area': None, 'measured_hardware_energy': None,
        'interpretation': 'Logic description only; no synthesized hardware or bandwidth saving measured.'
    }
    write_json(out / 'hardware_logic.json', hardware)
    return hardware


def import_runtime(device_name):
    legacy = ROOT / 'vendor' / 'legacy'
    upstream = ROOT / 'vendor' / 'upstream'
    sys.path[:0] = [str(legacy), str(upstream)]
    import torch
    import numpy as np
    import torchvision
    import scipy
    models = importlib.import_module('models')
    data = importlib.import_module('datasets.data')
    adapter = importlib.import_module('trunc_adapter')
    for module, allowed in [(models, upstream), (data, upstream), (adapter, legacy)]:
        require(Path(module.__file__).resolve().is_relative_to(allowed.resolve()),
                'Importconflict voor ' + module.__name__ + '; start de runner in een nieuw Pythonproces.')
    require(torch.__version__.split('+')[0] == '2.10.0', 'Gebruik de bestaande PyTorch 2.10.0-omgeving; niets upgraden.')
    require(torchvision.__version__.split('+')[0] == '0.25.0', 'Gebruik de bestaande torchvision 0.25.0-omgeving.')
    device = torch.device('cuda:0' if device_name == 'cuda' else device_name)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA is niet beschikbaar. Gebruik je bestaande .venv-train.')
        torch.cuda.set_device(device)
    # Retain the parent's backend policy, including the installed TF32 defaults.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(20261009)
    np.random.seed(20261009)
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
        'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'platform': platform.platform(),
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'cuda_build': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'matmul_fp32_precision': str(torch.backends.cuda.matmul.fp32_precision),
        'cudnn_conv_fp32_precision': str(torch.backends.cudnn.conv.fp32_precision),
        'AMP': False, 'torch_compile': False}
    return SimpleNamespace(torch=torch, np=np, torchvision=torchvision, device=device,
                           models=models, data=data, a=adapter, environment=environment)


def recode_tensor(codes, method, torch):
    require(codes.dtype == torch.uint8 and bool((codes <= 63).all()), 'Verwacht geldige E2M3-codebytes.')
    c = codes.to(torch.int64)
    sign, e, m = (c >> 5) << 3, (c >> 3) & 3, c & 7
    if method == 'lsb_trunc':
        target = c >> 2
    elif method == 'rne':
        base = (e << 1) | (m >> 2)
        increment = ((m >> 1) & 1) & ((m & 1) | (m >> 2))
        target = sign | (base + increment).clamp_max(7)
    elif method == 'msb_bestexp':
        target = sign | ((e + (m == 6).long()).clamp_max(3) << 1) | (m & 1)
    else:
        raise ValueError('Onbekende omzetting: ' + str(method))
    return target.to(torch.uint8)


def audit_tensor_rules(rt):
    torch, a = rt.torch, rt.a
    codes = torch.arange(64, dtype=torch.uint8, device=rt.device)
    for method in rules.METHODS:
        converted = recode_tensor(codes, method, torch)
        expected = torch.tensor(rules.lookup(method), dtype=torch.uint8, device=rt.device)
        require(torch.equal(converted, expected), 'Tensorregel wijkt af van exacte referentie: ' + method)
        for bit, values in [(6, codes), (4, converted)]:
            decoded = a.codec(bit).decode_elements(values).cpu()
            wanted = torch.tensor([float(rules.decode_code(int(c), bit)) for c in values.cpu().tolist()], dtype=decoded.dtype)
            require(torch.equal(decoded, wanted), 'Bestaande codec en rationale referentie verschillen.')
        require(torch.equal(codes.long() >> 5, converted.long() >> 3), 'Teken is gewijzigd.')
    return {'all64_codes_match_exact_reference': True, 'all_methods_match_existing_decoders': True,
            'sign_codes_preserved': True, 'device': str(rt.device)}


def locate_source(explicit, reference):
    if explicit is not None:
        path = explicit.expanduser().resolve()
        require(path.is_file(), 'Bronexport ontbreekt: ' + str(path))
        return path
    old_dir = reference['source_run_directory']
    run_name = old_dir.replace('\\', '/').rstrip('/').split('/')[-1]
    candidates = [Path(old_dir) / reference['source_file'],
        project_root() / 'mx_runs' / 'mx_triple_independent_e2_scale_freeze50_reference100' / run_name / reference['source_file']]
    matches = []
    for path in candidates:
        if path.is_file() and path.resolve() not in matches:
            matches.append(path.resolve())
    require(len(matches) == 1,
        'De vastgelegde epoch100-bronexport is niet eenduidig gevonden. Geef --source-export met het pad naar ' + reference['source_file'])
    return matches[0]


def verify_source(path, reference, rt):
    require(path.stat().st_size == reference['source_bytes'], 'Bronexport heeft een andere bestandsgrootte.')
    digest = sha256(path)
    require(digest == reference['source_sha256'], 'Dit is niet de vastgelegde epoch100-bronexport; SHA-256 verschilt.')
    payload = rt.a.load_verified_payload(path)
    rt.a.validate_source(payload)
    p = payload['provenance']
    require(p.get('source_epoch') == 100 and p.get('checkpoint_sha256') == reference['checkpoint_sha256'],
            'Bronexport verwijst niet naar het bedoelde epoch100-checkpoint.')
    require(p.get('synthetic_fixture_only') is False and p.get('absolute_E2_scale_frozen_at_this_checkpoint') is True,
            'Geen echte voltooide freeze50-run als bron.')
    return payload


def build_variants(payload, rt):
    torch, a = rt.torch, rt.a
    models, chains = {}, {}
    for name in a.TARGET_LAYERS:
        source = a.codec(8).MXTensor.from_payload(payload['mx_layers'][name], rt.device)
        chains[name] = a.code_chain(source, payload['lower_scale_codes'][name].to(rt.device))
    for variant in VARIANTS:
        bit = BITS[variant]
        model = a.load_exported_model(rt.models, payload, rt.device, bit=bit)
        if bit == 4:
            for name in a.TARGET_LAYERS:
                middle = chains[name][6]
                encoded = a.codec(4).MXTensor(recode_tensor(middle.codes, variant, torch),
                                              middle.scale_codes.clone(), middle.shape)
                encoded.validate()
                require(torch.equal(encoded.scale_codes, middle.scale_codes), 'E2-shared scale gewijzigd.')
                if variant == 'lsb_trunc':
                    require(torch.equal(encoded.codes, chains[name][4].codes), 'Baseline verschilt van bestaande truncatie.')
                    require(torch.equal(model.get_submodule(name).decoded_weight, encoded.decode()),
                            'Baselinegewichten verschillen van oorspronkelijke exportloader.')
                with torch.no_grad():
                    model.get_submodule(name).decoded_weight.copy_(encoded.decode())
        require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Model is niet bevroren.')
        for name in a.BN_LAYERS:
            require(model.get_submodule(name).weight_mode == bit, 'Verkeerde BN-bankselectie.')
        models[variant] = model
    return models, chains


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def weight_diagnostics(chains, rt):
    torch, a = rt.torch, rt.a
    rows, frequencies, totals = [], [], {}
    for method in rules.METHODS:
        totals[method] = {'n': 0, 'abs_sum': 0., 'square_sum': 0., 'max_abs': 0.,
                          'zero_target': 0, 'zero_source': 0, 'exact': 0, 'exp_changed': 0, 'source_square_sum': 0.}
    for name, chain in chains.items():
        middle = chain[6]
        source = middle.decode().to(torch.float64)
        logical_codes = middle.codes.reshape(middle.shape[0], -1)[:, :math.prod(middle.shape[1:])].reshape(-1)
        counts = torch.bincount(logical_codes.long(), minlength=64).cpu().tolist()
        for code, count in enumerate(counts):
            frequencies.append({'layer': name, 'source_code': code, 'source_bits': f'{code:06b}', 'count': count})
        for method in rules.METHODS:
            new_codes = recode_tensor(middle.codes, method, torch)
            encoded = a.codec(4).MXTensor(new_codes, middle.scale_codes.clone(), middle.shape)
            target = encoded.decode().to(torch.float64)
            target_codes = new_codes.reshape(middle.shape[0], -1)[:, :math.prod(middle.shape[1:])].reshape(-1)
            difference = target - source
            n = source.numel()
            abs_sum, square_sum = difference.abs().sum().item(), difference.square().sum().item()
            max_abs = difference.abs().max().item()
            exp_changed = int((((logical_codes.long() >> 3) & 3) != ((target_codes.long() >> 1) & 3)).sum())
            vals = {'n': n, 'abs_sum': abs_sum, 'square_sum': square_sum, 'max_abs': max_abs,
                    'zero_target': int((target == 0).sum()), 'zero_source': int((source == 0).sum()),
                    'exact': int((target == source).sum()), 'exp_changed': exp_changed,
                    'source_square_sum': source.square().sum().item()}
            total = totals[method]
            for key, value in vals.items():
                total[key] = max(total[key], value) if key == 'max_abs' else total[key] + value
            rows.append(_weight_row(name, method, vals))
    for method, vals in totals.items():
        rows.append(_weight_row('ALL_MX_LAYERS', method, vals))
    return rows, frequencies


def _weight_row(layer, method, values):
    n = values['n']
    return {'layer': layer, 'method': method, 'n_weights': n,
            'mae_vs_e2m3': values['abs_sum'] / n, 'mse_vs_e2m3': values['square_sum'] / n,
            'rmse_vs_e2m3': math.sqrt(values['square_sum'] / n), 'max_abs_error': values['max_abs'],
            'normalized_mse': values['square_sum'] / values['source_square_sum'] if values['source_square_sum'] else None,
            'zero_source': values['zero_source'], 'zero_target': values['zero_target'],
            'exact_values': values['exact'], 'exponent_fields_changed': values['exp_changed'],
            'padding_excluded': True, 'shared_scale_unchanged': True}


def load_test_data(explicit, source_path, reference, rt):
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    else:
        sibling = source_path.parent / 'dataset_manifest.json'
        if sibling.is_file():
            manifest = json.loads(sibling.read_text(encoding='utf-8'))
            candidates.append(Path(manifest['files']['root']))
        candidates.extend([project_root() / 'data' / 'svhn', Path(reference['dataset']['files']['root'])])
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))
    matching = [p for p in candidates if (p / 'test_32x32.mat').is_file()]
    require(bool(matching), 'SVHN-testbestand ontbreekt. Geef --data-root met de map die test_32x32.mat bevat.')
    root = matching[0]
    path = root / 'test_32x32.mat'
    expected = reference['dataset']['files']['files']['test_32x32.mat']
    require(path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'],
            'SVHN-testbestand wijkt af van de gecontroleerde dataset; niets is gewijzigd of gedownload.')
    rt.data.data_paths['svhn'] = str(root)
    transform = rt.data.get_transform('svhn', 'val')
    require(repr(transform) == reference['dataset']['transforms']['test'], 'Testtransformatie wijkt af van de referentie.')
    dataset = rt.data.get_dataset('svhn', 'val', transform, download=False)
    expected_selection = reference['dataset']['selection']['test']
    require(dataset.split == 'test' and len(dataset) == expected_selection['n'], 'Onjuiste SVHN-testselectie.')
    labels = rt.np.asarray(dataset.labels, dtype=rt.np.int64)
    require(hashlib.sha256(labels.tobytes()).hexdigest() == expected_selection['label_sha256'], 'Testlabels of hun volgorde verschillen.')
    require(rt.np.bincount(labels, minlength=10).tolist() == expected_selection['labels_histogram'], 'Klassenhistogram verschilt.')
    return dataset, {'path': str(path), 'sha256': expected['sha256'], 'n': len(dataset),
                     'transform': repr(transform), 'label_sha256': expected_selection['label_sha256'],
                     'downloaded': False, 'train_loaded': False, 'extra_loaded': False}


def evaluate_dataset(models, dataset, rt, out, log):
    torch = rt.torch
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False,
        num_workers=0, pin_memory=rt.device.type == 'cuda', drop_last=False)
    before = {name: state_digest(model) for name, model in models.items()}
    totals = {name: {'n': 0, 'correct1': 0, 'correct5': 0, 'loss_sum': 0.} for name in VARIANTS}
    paired = {name: {'variant': name, 'baseline': 'lsb_trunc', 'n': 0, 'both_correct': 0,
                     'baseline_correct_only': 0, 'variant_correct_only': 0, 'both_wrong': 0,
                     'prediction_disagreements': 0} for name in ('rne', 'msb_bestexp')}
    offset = 0
    started = time.perf_counter()
    fields = ['index', 'label', *VARIANTS]
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for batch_index, (inputs, labels) in enumerate(loader):
                require(inputs.shape[1:] == (3, 40, 40) and inputs.dtype == torch.float32, 'Onjuiste modelinvoer.')
                inputs = inputs.to(rt.device, non_blocking=rt.device.type == 'cuda')
                labels = labels.to(rt.device, non_blocking=rt.device.type == 'cuda')
                predicted = {}
                for name in VARIANTS:
                    logits = models[name](inputs)
                    require(logits.shape == (labels.numel(), 10) and bool(torch.isfinite(logits).all()), 'Niet-eindige of onjuiste logits: ' + name)
                    top5 = logits.topk(5, dim=1, largest=True, sorted=True).indices
                    predicted[name] = top5[:, 0]
                    t = totals[name]
                    t['n'] += labels.numel()
                    t['correct1'] += int((top5[:, 0] == labels).sum())
                    t['correct5'] += int((top5 == labels[:, None]).any(dim=1).sum())
                    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
                    require(bool(torch.isfinite(loss)), 'Niet-eindige testloss: ' + name)
                    t['loss_sum'] += loss.item()
                base_ok = predicted['lsb_trunc'] == labels
                for name, counts in paired.items():
                    variant_ok = predicted[name] == labels
                    counts['n'] += labels.numel()
                    counts['both_correct'] += int((base_ok & variant_ok).sum())
                    counts['baseline_correct_only'] += int((base_ok & ~variant_ok).sum())
                    counts['variant_correct_only'] += int((~base_ok & variant_ok).sum())
                    counts['both_wrong'] += int((~base_ok & ~variant_ok).sum())
                    counts['prediction_disagreements'] += int((predicted[name] != predicted['lsb_trunc']).sum())
                cpu_preds = {name: value.cpu().tolist() for name, value in predicted.items()}
                for j, label in enumerate(labels.cpu().tolist()):
                    writer.writerow({'index': offset + j, 'label': label,
                                     **{name: value[j] for name, value in cpu_preds.items()}})
                offset += labels.numel()
                if batch_index == 0 or (batch_index + 1) % 25 == 0 or offset == len(dataset):
                    log(f'Testbeelden: {offset}/{len(dataset)}; vijf varianten, geen training.')
    require(offset == len(dataset), 'Niet alle testvoorbeelden zijn verwerkt.')
    require(before == {name: state_digest(model) for name, model in models.items()}, 'Modelstaat of BN is gewijzigd tijdens evaluatie.')
    rows = []
    for name in VARIANTS:
        t = totals[name]
        require(t['n'] == len(dataset), 'Onvolledige precisie-evaluatie.')
        rows.append({'variant': name, 'label': LABELS[name], 'weight_bits': BITS[name],
            'weight_format': {8: 'E4M3', 6: 'E2M3', 4: 'E2M1'}[BITS[name]],
            'n': t['n'], 'correct1': t['correct1'], 'correct5': t['correct5'],
            'top1_percent': 100. * t['correct1'] / t['n'], 'top5_percent': 100. * t['correct5'] / t['n'],
            'loss': t['loss_sum'] / t['n']})
    for counts in paired.values():
        require(sum(counts[k] for k in ['both_correct', 'baseline_correct_only', 'variant_correct_only', 'both_wrong']) == len(dataset), 'Paired counts zijn onvolledig.')
    write_csv(out / 'accuracy.csv', rows)
    write_csv(out / 'paired_predictions.csv', list(paired.values()))
    return rows, {'model_and_bn_state_unchanged': True, 'examples': offset,
                  'evaluation_wall_seconds': time.perf_counter() - started, 'real_training_performed': False}


def check_source_models(payload, rt):
    models, chains = build_variants(payload, rt)
    torch = rt.torch
    inputs = torch.linspace(-1., 1., steps=3 * 3 * 40 * 40, device=rt.device).reshape(3, 3, 40, 40)
    before = {name: state_digest(model) for name, model in models.items()}
    with torch.inference_mode():
        for name, model in models.items():
            result = model(inputs)
            require(result.shape == (3, 10) and bool(torch.isfinite(result).all()), 'Synthetische forward mislukt: ' + name)
    require(before == {name: state_digest(model) for name, model in models.items()}, 'Synthetische check wijzigt modelstaat.')
    return {'synthetic_images': 3, 'variants': len(models), 'all_outputs_finite': True,
            'model_and_bn_state_unchanged': True, 'accuracy_measured': False}


def make_feedback(out):
    target = out / 'feedback_bundle.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path != target and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(out).as_posix())
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['numeric', 'check', 'evaluate'])
    parser.add_argument('--source-export', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, help='Nieuwe, nog niet bestaande resultatenmap.')
    parser.add_argument('--device', default='cuda', help='Standaard cuda; CPU alleen expliciet, zonder stille fallback.')
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output if args.output else project_root() / 'mx_runs' / REVISION / (stamp + '_' + args.command)).resolve()
    require(not out.exists(), 'Resultatenmap bestaat al; kies een nieuwe map. Er wordt niets overschreven.')
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-mantissa-msb-exponent-probe-report-v1', 'package_revision': REVISION,
        'command': args.command, 'started_at': now(), 'status': 'RUNNING', 'protocol': PROTOCOL,
        'output_directory': str(out), 'network_evaluated': False,
        'synthetic_only': args.command == 'check', 'training_performed': False,
        'warnings': [], 'limitations': PROTOCOL['limitations']}
    log_path = out / 'run.log'
    def log(message):
        print(message, flush=True)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(now() + ' ' + message + '\n')
    old_warning = warnings.showwarning
    def record_warning(message, category, filename, lineno, file=None, line=None):
        text = category.__name__ + ': ' + str(message)
        report['warnings'].append(text)
        log('WAARSCHUWING: ' + text)
    warnings.showwarning = record_warning
    exit_code = 0
    started = time.perf_counter()
    try:
        report['package_verification'] = verify_package()
        reference = load_reference()
        write_json(out / 'protocol.json', PROTOCOL)
        write_json(out / 'REFERENCE.json', reference)
        report['hardware_logic'] = numeric_stage(out)
        log('NUMERIEK PASS: alle 64 broncodes, drie omzettingen; optimale beperkte mapping exact gecontroleerd.')
        if args.command != 'numeric':
            rt = import_runtime(args.device)
            report['environment'] = rt.environment
            report['tensor_rule_checks'] = audit_tensor_rules(rt)
            source_path = locate_source(args.source_export, reference)
            payload = verify_source(source_path, reference, rt)
            report['source'] = {'path': str(source_path), 'sha256': reference['source_sha256'], 'epoch': 100,
                                'checkpoint_sha256': reference['checkpoint_sha256'], 'compact_source_only': True}
            log('BRON PASS: oorspronkelijke compacte epoch100-export en opgeslagen E2-schalen geverifieerd.')
            if args.command == 'check':
                report['forward_checks'] = check_source_models(payload, rt)
                log('CHECK PASS: vijf varianten met synthetische invoer; geen SVHN-accuracy gemeten.')
            else:
                dataset, dataset_info = load_test_data(args.data_root, source_path, reference, rt)
                report['dataset'] = dataset_info
                models, chains = build_variants(payload, rt)
                weight_rows, frequency_rows = weight_diagnostics(chains, rt)
                write_csv(out / 'weight_error.csv', weight_rows)
                write_csv(out / 'source_code_histogram.csv', frequency_rows)
                rows, audit = evaluate_dataset(models, dataset, rt, out, log)
                report.update(network_evaluated=True, synthetic_only=False, evaluation_audit=audit, accuracy=rows)
                discrepancies = []
                for row in rows:
                    if row['variant'] in ('source8', 'source6', 'lsb_trunc'):
                        expected = reference['expected_metrics'][str(row['weight_bits'])]
                        for key in ('n', 'correct1', 'correct5'):
                            if row[key] != expected[key]:
                                discrepancies.append({'variant': row['variant'], 'field': key,
                                    'historical': expected[key], 'observed': row[key]})
                report['historical_reference_differences'] = discrepancies
                if discrepancies:
                    report['warnings'].append('De opnieuw gemeten baselineaantallen verschillen van de historische trainingsevaluatie; controleer deze verschillen voordat je de onderzoeksconclusie trekt.')
                require(sha256(Path(dataset_info['path'])) == dataset_info['sha256'], 'Datasetbestand is gewijzigd.')
                report['dataset_unchanged'] = True
                for row in rows:
                    log(f"{row['label']}: {row['correct1']}/{row['n']} = {row['top1_percent']:.6f}%")
            require(sha256(source_path) == reference['source_sha256'], 'Bronexport is gewijzigd.')
            report['source_unchanged'] = True
        report['status'] = 'PASS_WITH_WARNINGS' if report['warnings'] else 'PASS'
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        (out / 'error_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        log('FOUT: ' + report['error'])
        exit_code = 1
    finally:
        report['ended_at'] = now()
        report['session_wall_seconds'] = time.perf_counter() - started
        write_json(out / 'run_report.json', report)
        warnings.showwarning = old_warning
        # Feedback includes the exact new evaluator, separate from research inputs.
        for name in ['run_probe.py', 'mantissa_rules.py', 'REFERENCE.json', 'PACKAGE_SHA256.json']:
            if (ROOT / name).is_file():
                (out / 'runner_source').mkdir(exist_ok=True)
                shutil.copy2(ROOT / name, out / 'runner_source' / name)
        try:
            from probe_notebook import write_notebook, create_figures
            write_notebook(out)
            if (out / 'numeric_summary.json').is_file():
                create_figures(out)
        except Exception as exc:
            report['warnings'].append('Notebook/figuren: ' + type(exc).__name__ + ': ' + str(exc))
            if report['status'] == 'PASS':
                report['status'] = 'PASS_WITH_WARNINGS'
            write_json(out / 'run_report.json', report)
            log('WAARSCHUWING: ' + report['warnings'][-1])
        report['ended_at'] = now()
        report['session_wall_seconds'] = time.perf_counter() - started
        report['session_timing_excludes_feedback_zip'] = True
        write_json(out / 'run_report.json', report)
        bundle = make_feedback(out)
        log('Resultaten: ' + str(out))
        log('Stuur dit bestand terug: ' + str(bundle))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
