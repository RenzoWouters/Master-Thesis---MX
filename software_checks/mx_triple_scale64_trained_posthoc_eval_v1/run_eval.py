"""Frozen A/B/C controls on the completed scale64/B-trained MX model.

check uses synthetic code/model fixtures. evaluate measures the scale64-trained
reference at its archived E4-best42 and epoch100. Earlier direct/A measurements
are copied as historical results for comparison, never reevaluated or pooled.
A: naive exponent+mantissa at original scale. B: same codes, scale times64.
C: RNE from stored E4 elements/64, scale times64; then identical mantissa shift.
No training, master quantization, learned scale, BN calibration or reselection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
import svhn_support as s

ROOT = Path(__file__).resolve().parent
NOTEBOOK = 'MX_triple_scale64_trained_posthoc_resultaten.ipynb'
TRAININGS = ('scale64',)
ROLES = ('best_E4M3_test', 'last_epoch')


def dependencies(out: Path, log) -> None:
    for command, name in [('freeze', 'pip_freeze.txt'), ('check', 'pip_check.txt')]:
        result = subprocess.run([sys.executable, '-m', 'pip', command], capture_output=True,
                                text=True, encoding='utf-8', errors='replace')
        (out / name).write_text(result.stdout + result.stderr, encoding='utf-8')
        s.require(result.returncode == 0, 'pip ' + command + ' mislukt; zie ' + name + '. Niets upgraden.')
        if command == 'check': log('pip check: ' + result.stdout.strip())


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def safe_basename(name) -> str:
    s.require(type(name) is str and name not in ('', '.', '..') and Path(name).name == name
              and not any(c in name for c in '/\\:'), 'Onveilig bestandsveld in referentierapport.')
    return name


def inspect_reference(ref: Path, rt, training_kind: str) -> tuple[dict, dict, list]:
    """Validate the complete archived contract before any MX tensor is opened."""
    import resume_core as core
    s.require(training_kind in TRAININGS, 'Onbekend type brontraining.')
    required = ['run_report.json', 'selected_metrics.csv', 'epoch_metrics.csv',
                'checkpoint_selection.json', 'training_summary.json']
    s.require(ref.is_dir() and all((ref / n).is_file() for n in required),
              'Verwacht de volledige oorspronkelijke runmap, niet alleen de feedbackbundel: ' + training_kind)
    report = json.loads((ref / 'run_report.json').read_text(encoding='utf-8'))
    expected_schema = 'mx-triple-scale64-reference100-report-v1'
    s.require(report.get('schema') == expected_schema and report.get('status') == 'PASS'
              and report.get('command') in ('run', 'resume'), 'Verkeerde of onvoltooide brontraining: ' + training_kind)
    pinned = json.loads((ROOT / 'EXPECTED_REFERENCE.json').read_text(encoding='utf-8'))
    s.require(core.file_sha256(ref / 'run_report.json') == pinned['run_report_sha256']
              and report.get('checkpoint_selection') == pinned['checkpoint_selection']
              and report.get('mx_exports') == pinned['mx_exports']
              and report.get('initial_model_sha256') == pinned['initial_model_sha256'],
              'Dit pakket verwacht de afgesproken scale64-run met E4M3-best42 en epoch100; bronrapport of exports verschillen.')
    s.require(report.get('synthetic_fixture_only') is False and report.get('full_training_completed') is True
              and report.get('completed_epoch') == 100 and report.get('optimizer_steps') == 57300
              and report.get('training_forward_passes') == 171900,
              'Verwacht een echte volledige honderd-epochtraining met drie MX-modi.')
    summary = json.loads((ref / 'training_summary.json').read_text(encoding='utf-8'))
    s.require(summary == report.get('training_summary') and summary.get('full_100_epoch_training_completed') is True
              and summary.get('completed_epochs') == 100 and summary.get('reference_end_epoch') == 100,
              'Voltooiingsrapporten zijn inconsistent.')
    expected = json.loads((ROOT / (training_kind.upper() + '_REFERENCE_PROTOCOL.json')).read_text(encoding='utf-8'))
    differences = core.differences(report['config'], expected)
    s.require(not differences, 'Bronprotocol wijkt af: ' + ', '.join(differences[:10]))
    known = json.loads((ROOT / 'provenance' / (training_kind + '_reference100_PACKAGE_SHA256.json')).read_text(encoding='utf-8'))
    s.require(report['package_sha256'] == known, 'Onbekende trainingspakketversie: ' + training_kind)
    names = ['mx_format.py', 'mx_e2m1.py', 'mx_e4m3.py', 'mx_adapter.py', 'joint_adapter.py',
             'joint_support.py', 'svhn_support.py', 'smoke_support.py', 'resume_core.py']
    names += ['trunc_adapter.py', 'trunc_support.py', 'posthoc_truncation.py']
    for name in names:
        s.require(report['package_sha256'][name] == core.file_sha256(ROOT / name), 'Numerieke broncode wijkt af: ' + name)
    for key in ('python', 'torch', 'torchvision', 'numpy', 'scipy', 'pillow', 'cuda_build', 'cudnn', 'device_name'):
        s.require(report['environment'][key] == rt.environment[key],
                  'Uitvoeringsomgeving wijkt af bij ' + key + '; gebruik dezelfde .venv-train en GPU.')
    for key in ('cudnn_benchmark', 'cudnn_deterministic', 'deterministic_algorithms',
                'matmul_fp32_precision', 'cudnn_conv_fp32_precision', 'AMP', 'torch_compile'):
        s.require(report['execution_settings'][key] == rt.settings[key], 'Backend wijkt af bij ' + key)
    rows = csv_rows(ref / 'epoch_metrics.csv')
    keys = {(e, phase, bit) for e in range(1, 101) for phase in ('train', 'test') for bit in (4, 6, 8)}
    actual = {(int(r['epoch']), r['phase'], int(r['weight_bits'])) for r in rows}
    s.require(len(rows) == 600 and actual == keys, 'Onvolledige epoch-/modusgeschiedenis.')
    formats = {8: 'MXFP8_E4M3', 6: 'MXFP6_E2M3', 4: 'MXFP4_E2M1'}
    banks = {8: '8', 6: '32', 4: '4'}
    for r in rows:
        bit, epoch, n = int(r['weight_bits']), int(r['epoch']), int(r['n'])
        n_expected = 73257 if r['phase'] == 'train' else 26032
        c1, c5 = int(r['correct1']), int(r['correct5'])
        lr = expected['lr']
        for milestone in expected['scheduler_milestones']:
            if milestone < epoch: lr *= expected['scheduler_gamma']
        s.require(n == n_expected and 0 <= c1 <= c5 <= n and int(r['activation_bits']) == 32
                  and r['weight_format'] == formats[bit] and r['bn_bank_key'] == banks[bit], 'Onjuiste referentiemodus of dekking.')
        s.require(math.isfinite(float(r['objective_loss'])) and abs(float(r['lr']) - lr) < 1e-15
                  and abs(float(r['top1_percent']) - 100*c1/n) < 1e-4
                  and abs(float(r['count_top1_percent']) - 100*c1/n) < 1e-7,
                  'Onjuiste referentiescore of LR.')
    high = sorted((r for r in rows if r['phase'] == 'test' and int(r['weight_bits']) == 8), key=lambda r: int(r['epoch']))
    best = max(high, key=lambda r: float(r['top1_percent']))
    selection = report['checkpoint_selection']
    s.require(selection == json.loads((ref / 'checkpoint_selection.json').read_text(encoding='utf-8')), 'Checkpointselectierapporten verschillen.')
    s.require(selection['best']['epoch'] == int(best['epoch']) and selection['last']['epoch'] == 100
              and selection.get('selection_mode') == 'MXFP8_E4M3/A32'
              and selection.get('all_modes_from_same_selected_checkpoint') is True,
              'Vaste E4M3-best/epoch100-selectie wijkt af.')
    s.require(abs(float(selection['best']['selection_score_E4M3']) - float(best['top1_percent'])) < 1e-10,
              'Geselecteerde E4M3-score wijkt af.')
    selected = csv_rows(ref / 'selected_metrics.csv')
    s.require(len(selected) == 6 and {(r['role'], int(r['weight_bits'])) for r in selected}
              == {(role, bit) for role in ROLES for bit in (4, 6, 8)}, 'Onvolledige geselecteerde metrics.')
    for r in selected:
        epoch = selection['best' if r['role'] == ROLES[0] else 'last']['epoch']
        s.require(int(r['epoch']) == epoch and int(r['n']) == 26032, 'Verkeerd geselecteerd eindpunt.')
        h = next(x for x in rows if x['phase'] == 'test' and int(x['epoch']) == epoch and x['weight_bits'] == r['weight_bits'])
        s.require(r == {'role': r['role'], **h}, 'Geselecteerde metrics verschillen van de historie.')
    fingerprints = {n: core.file_sha256(ref / n) for n in required}
    for role in ROLES:
        epoch = selection['best' if role == ROLES[0] else 'last']['epoch']
        items = report['mx_exports'][role]
        for bit in (8, 6, 4):
            item = items[str(bit)]; name = safe_basename(item['file']); path = ref / name
            expected_truncation = bit != 8
            s.require(item['decoded_weights_exact'] is True and item['six_fp32_masters_absent'] is True
                      and item['truncation'] is expected_truncation and item['epoch'] == epoch and item['weight_bits'] == bit,
                      'Bronexport heeft een afwijkend gewichtspad.')
            s.require(item['source_weight_bits'] == 8 and item['source_file_sha256'] == items['8']['sha256']
                      and item['source_scale_bytes_copied'] is (bit == 8)
                      and item['scale_exponent_offset'] == (0 if bit == 8 else 6)
                      and item['scale_factor'] == (1 if bit == 8 else 64)
                      and item['shared_scale_compensation'] is (bit != 8)
                      and item['learned_shared_scale'] is False and item['master_weights_read_to_derive'] is False
                      and item['master_weights_read_to_encode_source'] is (bit == 8)
                      and item['saved_and_reloaded'] is True
                      and item['is_source_export'] is (bit == 8) and item['derived_from_saved_source'] is (bit != 8),
                      'Opgeslagen scale64-export is niet via de verwachte B-route uit de opgeslagen E4-bron afgeleid.')
            s.require(path.is_file() and path.stat().st_size == item['bytes'], 'Export ontbreekt of heeft verkeerde grootte: ' + name)
            marker = path.with_suffix(path.suffix + '.sha256')
            s.require(marker.is_file(), 'Exportchecksum ontbreekt: ' + name)
            fingerprints[name], fingerprints[marker.name] = core.file_sha256(path), core.file_sha256(marker)
            s.require(fingerprints[name] == item['sha256'] == marker.read_text(encoding='ascii').strip(), 'Exporthash wijkt af: ' + name)
    return report, fingerprints, selected


def prepare_test_only(args, modules, reference: dict, out: Path, log):
    import torch
    data = args.data_root.resolve()
    path = data / 'test_32x32.mat'
    s.require(path.is_file(), 'test_32x32.mat ontbreekt; gebruik de bestaande data/svhn-map.')
    info = s.hashes(path)
    expected = reference['dataset_files']['files']['test_32x32.mat']
    s.require(all(info[k] == expected[k] for k in info), 'Testdata wijken af van de referentie.')
    s.require(info['md5'] == modules.data.SVHNMIX.split_list['test'][2], 'Testdata falen auteurs-MD5.')
    modules.data.data_paths['svhn'] = str(data)
    transform = modules.data.get_transform('svhn', 'val')
    s.require(repr(transform) == reference['transforms']['test'], 'Andere testpreprocessing.')
    dataset = modules.data.get_dataset('svhn', 'val', transform, download=False)
    s.require(dataset.split == 'test', 'Verkeerde split.')
    selection = s.validate_dataset(dataset, 26032, 'SVHN test')
    s.require(selection == reference['dataset_selection']['test'], 'Testlabels of volgorde verschillen.')
    loader = torch.utils.data.DataLoader(s.IndexedDataset(dataset), batch_size=128,
                shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    log('DATA PASS: uitsluitend test=26032; 204 batches; train en extra niet ingelezen.')
    s.write_json(out / 'dataset_manifest.json', {'test_file': {'path': str(path), **info},
                 'selection': selection, 'transform': repr(transform), 'train_read': False, 'extra_read': False})
    return loader, path, info


def attach_prior_results(reference: dict, out: Path) -> dict:
    """Copy immutable prior observations, with their original provenance.

    These rows are for cross-training descriptive plots only. They are not
    fresh measurements and are never paired with this run's per-image outputs.
    """
    import resume_core as core
    source = ROOT / 'prior_posthoc'
    manifest = json.loads((source / 'REFERENCE.json').read_text(encoding='utf-8'))
    s.require(manifest.get('schema') == 'mx-triple-scale64-prior-posthoc-reference-v1'
              and manifest.get('synthetic_fixture_only') is False,
              'Ongeldige herkomst van de eerdere post-hocresultaten.')
    expected_files = {'metrics.csv', 'paired_metrics.csv', 'run_report.json'}
    s.require(set(manifest['files_sha256']) == expected_files, 'Onvolledige historische vergelijking.')
    for name, digest in manifest['files_sha256'].items():
        s.require(core.file_sha256(source / name) == digest, 'Historisch vergelijkingsbestand gewijzigd: ' + name)
    old = json.loads((source / 'run_report.json').read_text(encoding='utf-8'))
    known = json.loads((ROOT / 'provenance' / 'previous_posthoc_PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    s.require(old.get('schema') == 'mx-triple-scale64-posthoc-report-v1'
              and old.get('status') == 'PASS' and old.get('command') == 'evaluate'
              and old.get('synthetic_fixture_only') is False
              and old.get('training_kinds') == ['direct', 'truncation']
              and old.get('package_sha256') == known
              and old.get('input_files_unchanged') is True
              and old.get('test_file_unchanged') is True,
              'Eerdere A/B/C-evaluatie is niet de afgesproken voltooide proef.')
    s.require(len(csv_rows(source / 'metrics.csv')) == len(old['metrics']) == 36
              and len(csv_rows(source / 'paired_metrics.csv')) == len(old['paired_metrics']) == 48,
              'Historische A/B/C-tabellen zijn onvolledig.')
    metadata_by_kind = manifest['reference_training_metadata']
    s.require(set(metadata_by_kind) == {'direct', 'truncation'}, 'Historische trainingsherkomst ontbreekt.')
    for kind, metadata in metadata_by_kind.items():
        for key in ('initial_model_sha256', 'dataset_selection', 'transforms', 'environment', 'execution_settings'):
            s.require(reference[key] == metadata[key], 'Historische en nieuwe training verschillen bij ' + kind + '/' + key)
        s.require(reference['dataset_files']['files'] == metadata['dataset_files']['files'],
                  'De drie trainingen gebruiken niet dezelfde datasetbestanden.')
    destination = out / 'prior_posthoc'
    destination.mkdir(exist_ok=False)
    for name in sorted(expected_files | {'REFERENCE.json'}):
        shutil.copy2(source / name, destination / name)
    return {'directory': 'prior_posthoc', 'source_bundle_sha256': manifest['source_bundle_sha256'],
            'files_sha256': manifest['files_sha256'], 'training_kinds': ['direct', 'truncation'],
            'historical_measurements_reused': True, 'new_model_evaluations': 0,
            'cross_training_per_image_pairs_computed': False,
            'same_initial_model_dataset_preprocessing_environment_and_backend': True}


def evaluate(args, out: Path, report: dict, rt, modules, log) -> None:
    import evaluation as ev
    import scale64 as tr
    import resume_core as core
    inputs = {'scale64': args.scale64_run.resolve()}
    references, befores, selected_rows = {}, {}, {}
    report['references'], report['roles'] = {}, {}
    report['path_specs'] = tr.PATH_SPEC
    report['synthetic_fixture_only'] = False
    for kind, ref in inputs.items():
        s.require(not out.is_relative_to(ref), 'Uitvoer mag niet in een oorspronkelijke runmap staan.')
        reference, before, selected = inspect_reference(ref, rt, kind)
        s.require(report['source']['workcopy_files_sha256'] == reference['source']['workcopy_files_sha256'], 'Modelbron wijkt af van de trainingsbron.')
        references[kind], befores[kind], selected_rows[kind] = reference, before, selected
        report['references'][kind] = {'directory': str(ref), 'report_sha256': before['run_report.json'],
            'checkpoint_selection': reference['checkpoint_selection'], 'checkpoint_files_opened': False,
            'FP32_master_weights_opened': False, 'source_export_hashes': reference['mx_exports'], 'contract_matches': True}
        copies = out / 'reference_inputs' / kind; copies.mkdir(parents=True)
        for name in ('run_report.json', 'selected_metrics.csv', 'epoch_metrics.csv', 'checkpoint_selection.json', 'training_summary.json'):
            shutil.copy2(ref / name, copies / name)
    first = references['scale64']
    report['prior_posthoc'] = attach_prior_results(first, out)
    report['same_initial_model_and_dataset_across_training_kinds'] = True
    loader, test_path, test_before = prepare_test_only(args, modules, first, out, log)
    n_test = first['config']['test_examples']
    all_metrics, all_pairs, all_weights, all_counts, reproduction = [], [], [], [], []
    for kind, ref in inputs.items():
        reference = references[kind]; report['roles'][kind] = {}
        group_out = out / kind; group_out.mkdir()
        for role in ROLES:
            exports = reference['mx_exports'][role]
            payloads = {bit: ev.read_export(ref / exports[str(bit)]['file'], exports[str(bit)]['sha256']) for bit in (8, 6, 4)}
            hashes = {bit: exports[str(bit)]['sha256'] for bit in (8, 6, 4)}
            selection = reference['checkpoint_selection']['best' if role == ROLES[0] else 'last']
            epoch, checkpoint_sha = selection['epoch'], selection['sha256']
            for payload in payloads.values():
                s.require(payload['provenance']['source_epoch'] == epoch
                          and payload['provenance']['checkpoint_sha256'] == checkpoint_sha
                          and payload['provenance']['synthetic_fixture_only'] is False,
                          'Bronexport heeft afwijkende checkpointprovenance.')
            log(f'EVALUATIE: {kind}/{role}, epoch {epoch}; bronmodi en A/B/C; opgeslagen doel-BN bevroren.')
            paths, checks, weight_rows, code_rows = ev.assemble_paths(modules.models, payloads, hashes, group_out, role, rt.device, kind)
            metrics, pairs, details = ev.evaluate_paths(paths, loader, n_test, rt.device, group_out, role, epoch, log)
            s.require(details['label_sha256'] == first['dataset_selection']['test']['label_sha256'], 'Labels na evaluatie wijken af.')
            details['per_image_file'] = kind + '/' + details['per_image_file']
            checks.update(details)
            report['roles'][kind][role] = {'epoch': epoch, 'checkpoint_sha256': checkpoint_sha, **checks}
            all_metrics.extend({'training': kind, **r} for r in metrics)
            all_pairs.extend({'training': kind, **r} for r in pairs)
            all_weights.extend({'training': kind, 'role': role, 'epoch': epoch, **r} for r in weight_rows)
            all_counts.extend({'training': kind, 'role': role, 'epoch': epoch, **r} for r in code_rows)
            for r in metrics:
                log(f'  {r["path"]}: {r["top1_percent"]:.4f}%; correct={r["correct1"]}/{r["n"]}; CE={r["hard_cross_entropy"]:.6f}')
            for key, bit in [('source8', 8), ('trained6', 6), ('trained4', 4)]:
                new = next(r for r in metrics if r['path'] == key)
                old = next(r for r in selected_rows[kind] if r['role'] == role and int(r['weight_bits']) == bit)
                delta = new['correct1'] - int(old['correct1'])
                reproduction.append({'training': kind, 'role': role, 'epoch': epoch, 'path': key,
                    'historical_correct1': int(old['correct1']), 'reevaluated_correct1': new['correct1'],
                    'difference_correct1': delta, 'accuracy_difference_pp': 100*delta/n_test,
                    'historical_ce': float(old['objective_loss']), 'reevaluated_ce': new['hard_cross_entropy'],
                    'correct1_matches_history': delta == 0})
                if delta:
                    warnings.warn(f'{kind}/{role}/{key}: {delta:+d} correcte voorspellingen tegenover het trainingsrapport. '
                                  'Gebruik de hier opnieuw gemeten bronmodus; de oorspronkelijke CUDA-backend is niet bitexact.', RuntimeWarning)
            # Preserve partial values in the failure report without overwriting
            # any file; aggregate CSVs are created exactly once after all groups.
            report['metrics'], report['paired_metrics'] = all_metrics, all_pairs
            del paths, payloads
    s.require(len(all_metrics) == 18 and len(all_pairs) == 24 and len(reproduction) == 6
              and len(all_weights) == 216 and len(all_counts) == 108,
              'Onvolledige vergelijking van de twee scale64-checkpoints.')
    for name, rows in [('metrics.csv', all_metrics), ('paired_metrics.csv', all_pairs),
                       ('weight_pair_metrics.csv', all_weights), ('layer_code_metrics.csv', all_counts),
                       ('historical_reproduction.csv', reproduction)]:
        s.write_csv(out / name, rows)
    for kind, fingerprints in befores.items():
        for name, digest in fingerprints.items():
            s.require(core.file_sha256(inputs[kind] / name) == digest, 'Oorspronkelijk bestand gewijzigd: ' + kind + '/' + name)
    s.require(s.hashes(test_path) == test_before, 'Testbestand gewijzigd.')
    report.update(input_files_unchanged=True, input_sha256=befores, test_file_unchanged=True,
        train_and_extra_read=False, original_checkpoint_selection_unchanged=True,
        historical_correct_counts_all_match=all(r['correct1_matches_history'] for r in reproduction),
        metrics=all_metrics, paired_metrics=all_pairs, evaluated_checkpoint_groups=2,
        derived_models_all_evaluated_from_saved_and_reloaded_codes=True)
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n',
                              'SAVE_FIGURES = False  # Zet op True om PNG/PDF in een nieuwe analysemap te bewaren.\n']
    s.write_json(out / NOTEBOOK, notebook)
    with (ROOT / 'notebook_support.py').open('rb') as src, (out / 'notebook_support.py').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    log('EVALUATIE VOLTOOID: scale64-training, E4M3-best42 en epoch100, A/B/C en opgeslagen modi; geen updates.')


def feedback(out: Path) -> None:
    names = ['run_report.json', 'run.log', 'warnings.log', 'pip_freeze.txt', 'pip_check.txt',
             'source_manifest.json', 'compatibility.patch', 'dataset_manifest.json', 'scale64_check_report.json',
             'metrics.csv', 'paired_metrics.csv', 'weight_pair_metrics.csv', 'layer_code_metrics.csv',
             'historical_reproduction.csv', NOTEBOOK, 'notebook_support.py']
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as z:
        for name in names:
            if (out / name).is_file(): z.write(out / name, name)
        for kind in TRAININGS:
            for path in sorted((out / kind).glob('*_per_image_outputs.npz')):
                z.write(path, kind + '/' + path.name)
        if (out / 'reference_inputs').is_dir():
            for path in sorted((out / 'reference_inputs').rglob('*')):
                if path.is_file(): z.write(path, path.relative_to(out).as_posix())
        if (out / 'prior_posthoc').is_dir():
            for path in sorted((out / 'prior_posthoc').iterdir()):
                if path.is_file(): z.write(path, path.relative_to(out).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'evaluate'])
    parser.add_argument('--scale64-run', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_triple_scale64_trained_posthoc_eval'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    args = parser.parse_args()
    if (args.command == 'evaluate') != (args.scale64_run is not None):
        parser.error('evaluate vereist --scale64-run; check gebruikt geen getrainde bronbestanden.')
    for ref in (args.scale64_run,):
        if ref is not None and args.output_root.resolve().is_relative_to(ref.resolve()):
            parser.error('Gebruik een nieuwe uitvoermap buiten de oorspronkelijke runmap.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-triple-scale64-trained-posthoc-report-v1', 'status': 'RUNNING',
              'package_revision': ROOT.name, 'command': args.command, 'started_at': s.now(),
              'output_directory': str(out), 'argv': sys.argv, 'warnings': [], 'optimizer_steps': 0,
              'batchnorm_recalibration': False, 'new_checkpoint_selection': False,
              'activation_bits': 32, 'shared_scale_compensation': True, 'learned_shared_scale': False,
              'training_kinds': list(TRAININGS), 'scale_exponent_offset': 6, 'scale_factor': 64,
              'synthetic_checks_only': args.command == 'check', 'synthetic_fixture_only': args.command == 'check',
              'limitations': [
                  'New measurements only on the B-trained model; earlier direct/A results are archived historical observations.',
                  'Never mix checkpoints, FP32 auxiliary state or BN state across training histories.',
                  'Stored E4-best and epoch100 are fixed; no new selection, fitting or training occurs.',
                  'A copies E4 scales unchanged; B/C add6 to the E8M0 exponent (times64), with overflow rejected.',
                  'C rounds only decoded E4 elements divided by64; it never reads original FP32 master weights.',
                  'Every E2M1 route keeps the same mantissa shift after its corresponding E2M3 route.',
                  'All target BN banks stay frozen; no recalibration, so a scale intervention can change BN compatibility.',
                  'The fixed factor64 is a range-derived control, not a fitted optimum or a repair of lost exponent information.',
                  'One training seed; the two checkpoints of one run are not independent repetitions.',
                  'FP32 decoded buffers emulate MX; no native-kernel speed or energy claim.',
                  'Six quantized FP32 masters are absent; FP32 exception layers and BN auxiliary state remain.',
                  'Low accuracy is an outcome, not a software FAIL; historical mismatches are reported.',
                  'CUDA repeatability is not bit-exact under the original backend policy.',
              ]}
    log = s.legacy.Log(out / 'run.log'); started = time.perf_counter(); code = 1
    def on_warning(message, category, filename, lineno, file=None, line=None):
        entry = {'category': category.__name__, 'message': str(message), 'file': filename, 'line': lineno}
        report['warnings'].append(entry); log('WAARSCHUWING: ' + str(message))
        with (out / 'warnings.log').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('default'); warnings.showwarning = on_warning
            log('RESULTATENMAP: ' + str(out))
            log('POST-HOC A/B/C: scale64-training, E4M3-best42 en epoch100; A32; bevroren BN; geen training.')
            report['package_sha256'] = s.verify_package()
            source, info = s.legacy.prepare_source(out); report['source'] = info
            dependencies(out, log)
            rt = s.runtime(20261009, args.device)
            report.update(environment=rt.environment, execution_settings=rt.settings, interpreter=sys.executable)
            modules = s.import_source(source)
            log('GPU: ' + rt.environment['device_name'])
            if args.command == 'check':
                import test_scale64
                report['tests'] = test_scale64.run_tests(modules, rt.device, out, log)
                s.write_json(out / 'scale64_check_report.json', report['tests'])
                report['notebook_dependencies'] = {}
                for name in ('pandas', 'matplotlib', 'ipykernel'):
                    try: report['notebook_dependencies'][name] = metadata.version(name)
                    except metadata.PackageNotFoundError: report['notebook_dependencies'][name] = 'MISSING; notebook only'
                log('CHECK PASS: bitbewerkingen, vaste schaaloffset, afrondingscontrole, B-bronformaat en bevroren reconstructie.')
            else:
                evaluate(args, out, report, rt, modules, log)
            s.verify_package()
            s.require(all(s.legacy.sha256(source / name) == digest for name, digest in info['workcopy_files_sha256'].items()),
                      'Bronwerkcopy gewijzigd tijdens de proef.')
            report['source_unchanged'] = True; report['status'] = 'PASS'; code = 0
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report.update(error=repr(exc), traceback=traceback.format_exc())
        log(report['status'] + ': ' + str(exc)); log(report['traceback'])
        log('Bestaande bronresultaten worden niet overschreven. Deel deze feedbackbundel.')
    finally:
        report['ended_at'] = s.now(); report['session_wall_seconds'] = time.perf_counter() - started
        log('EINDSTATUS: ' + report['status']); log('Rapport: ' + str(out / 'run_report.json'))
        log.handle.close(); s.write_json(out / 'run_report.json', report); feedback(out)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
