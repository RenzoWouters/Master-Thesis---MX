"""Frozen post-hoc exponent AND mantissa truncation on the same MX checkpoints.

check: synthetic code/model checks only. evaluate: eight paired paths on the
full SVHN test split at the archived E4M3-best and epoch100 endpoints. No new
training, optimizer, BN calibration, scale correction or checkpoint selection.
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
NOTEBOOK = 'MX_triple_posthoc_truncatie_resultaten.ipynb'
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


def inspect_reference(ref: Path, rt) -> tuple[dict, dict, list]:
    """Validate the complete training contract and provenance before tensor load."""
    import resume_core as core
    required = ['run_report.json', 'selected_metrics.csv', 'epoch_metrics.csv',
                'checkpoint_selection.json', 'training_summary.json']
    s.require(ref.is_dir() and all((ref / n).is_file() for n in required),
              'Verwacht de volledige oorspronkelijke _joint_run-map, niet alleen de feedbackbundel.')
    report = json.loads((ref / 'run_report.json').read_text(encoding='utf-8'))
    s.require(report.get('schema') == 'mx-weights-triple-direct-reference100-report-v1'
              and report.get('status') == 'PASS' and report.get('command') in ('run', 'resume'),
              'Verwacht een geslaagde gezamenlijke directe E4M3/E2M3/E2M1-referentie.')
    s.require(report.get('synthetic_fixture_only') is False and report.get('full_training_completed') is True
              and report.get('completed_epoch') == 100 and report.get('optimizer_steps') == 57300
              and report.get('training_forward_passes') == 171900,
              'Geen echte volledige honderd-epochreferentie met drie directe gewichtsmodi.')
    summary = json.loads((ref / 'training_summary.json').read_text(encoding='utf-8'))
    s.require(summary == report.get('training_summary') and summary.get('full_100_epoch_training_completed') is True,
              'Voltooiingsrapporten zijn inconsistent.')
    expected = json.loads((ROOT / 'REFERENCE_PROTOCOL.json').read_text(encoding='utf-8'))
    differences = core.differences(report['config'], expected)
    s.require(not differences, 'Referentieprotocol wijkt af: ' + ', '.join(differences[:10]))
    known_package = json.loads((ROOT / 'provenance/reference100_PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    s.require(report['package_sha256'] == known_package, 'Onbekende trainingspakketversie.')
    for name in ('mx_format.py', 'mx_e2m1.py', 'mx_e4m3.py', 'mx_adapter.py', 'joint_adapter.py',
                 'joint_support.py', 'svhn_support.py', 'smoke_support.py', 'resume_core.py'):
        s.require(report['package_sha256'][name] == core.file_sha256(ROOT / name),
                  'Herbruikte numerieke code wijkt af: ' + name)
    for key in ('python', 'torch', 'torchvision', 'numpy', 'scipy', 'pillow', 'cuda_build', 'cudnn', 'device_name'):
        s.require(report['environment'][key] == rt.environment[key],
                  'Uitvoeringsomgeving wijkt af bij ' + key + '; gebruik dezelfde .venv-train en GPU.')
    for key in ('cudnn_benchmark', 'cudnn_deterministic', 'deterministic_algorithms',
                'matmul_fp32_precision', 'cudnn_conv_fp32_precision', 'AMP', 'torch_compile'):
        s.require(report['execution_settings'][key] == rt.settings[key], 'Backend wijkt af bij ' + key)
    epoch_rows = csv_rows(ref / 'epoch_metrics.csv')
    keys = {(e, phase, bit) for e in range(1, 101) for phase in ('train', 'test') for bit in (4, 6, 8)}
    actual = {(int(row['epoch']), row['phase'], int(row['weight_bits'])) for row in epoch_rows}
    s.require(len(epoch_rows) == 600 and actual == keys, 'Onvolledige epoch-/modusgeschiedenis.')
    s.require(all(math.isfinite(float(row['top1_percent'])) and 0 <= float(row['top1_percent']) <= 100
                  and int(row['activation_bits']) == 32 for row in epoch_rows), 'Ongeldige referentiemetrics.')
    high = sorted((row for row in epoch_rows if row['phase'] == 'test' and int(row['weight_bits']) == 8),
                  key=lambda row: int(row['epoch']))
    best = max(high, key=lambda row: float(row['top1_percent']))
    selection = report['checkpoint_selection']
    s.require(selection == json.loads((ref / 'checkpoint_selection.json').read_text(encoding='utf-8')),
              'Checkpointselectierapporten verschillen.')
    s.require(selection['best']['epoch'] == int(best['epoch']) and selection['last']['epoch'] == 100
              and selection.get('selection_mode') == 'MXFP8_E4M3/A32'
              and selection.get('all_modes_from_same_selected_checkpoint') is True,
              'Vaste E4M3-best/epoch100-selectie wijkt af.')
    selected = csv_rows(ref / 'selected_metrics.csv')
    s.require(len(selected) == 6 and {(row['role'], int(row['weight_bits'])) for row in selected}
              == {(role, bit) for role in ROLES for bit in (4, 6, 8)}, 'Onvolledige geselecteerde metrics.')
    for row in selected:
        epoch = selection['best' if row['role'] == ROLES[0] else 'last']['epoch']
        s.require(int(row['epoch']) == epoch and int(row['n']) == 26032, 'Verkeerd geselecteerd eindpunt.')
        history = next(x for x in epoch_rows if x['phase'] == 'test' and int(x['epoch']) == epoch
                       and x['weight_bits'] == row['weight_bits'])
        s.require(row == {'role': row['role'], **history}, 'Geselecteerde metrics verschillen van de historie.')
    fingerprints = {name: core.file_sha256(ref / name) for name in required}
    for role in ROLES:
        epoch = selection['best' if role == ROLES[0] else 'last']['epoch']
        for bit in (8, 6, 4):
            item = report['mx_exports'][role][str(bit)]
            name = safe_basename(item['file'])
            path = ref / name
            s.require(item['decoded_weights_exact'] is True and item['six_fp32_masters_absent'] is True
                      and item['truncation'] is False and item['epoch'] == epoch and item['weight_bits'] == bit,
                      'Bronexport is niet de afgesproken directe export.')
            s.require(path.is_file() and path.stat().st_size == item['bytes'], 'Export ontbreekt of heeft verkeerde grootte: ' + name)
            marker = path.with_suffix(path.suffix + '.sha256')
            s.require(marker.is_file(), 'Exportchecksum ontbreekt: ' + name)
            fingerprints[name] = core.file_sha256(path)
            fingerprints[marker.name] = core.file_sha256(marker)
            s.require(fingerprints[name] == item['sha256'] == marker.read_text(encoding='ascii').strip(),
                      'Exporthash wijkt af: ' + name)
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


def evaluate(args, out: Path, report: dict, rt, modules, log) -> None:
    import evaluation as ev
    import truncation as tr
    import resume_core as core
    ref = args.reference_run.resolve()
    s.require(not out.is_relative_to(ref), 'Uitvoer mag niet in de oorspronkelijke resultatenmap staan.')
    reference, before, selected = inspect_reference(ref, rt)
    s.require(report['source']['workcopy_files_sha256'] == reference['source']['workcopy_files_sha256'],
              'Modelbron wijkt af van de trainingsbron.')
    report['reference'] = {'directory': str(ref), 'report_sha256': before['run_report.json'],
                           'checkpoint_selection': reference['checkpoint_selection'],
                           'checkpoint_files_opened': False, 'FP32_master_weights_opened': False,
                           'source_export_hashes': reference['mx_exports'], 'contract_matches': True}
    report['path_specs'] = tr.PATH_SPEC
    report['synthetic_fixture_only'] = reference['synthetic_fixture_only']
    copies = out / 'reference_inputs'; copies.mkdir()
    for name in ('run_report.json', 'selected_metrics.csv', 'epoch_metrics.csv',
                 'checkpoint_selection.json', 'training_summary.json'):
        shutil.copy2(ref / name, copies / name)
    loader, test_path, test_before = prepare_test_only(args, modules, reference, out, log)
    n_test = reference['config']['test_examples']  # Production contract enforces exactly 26032.
    all_metrics, all_pairs, all_weights, all_counts, reproduction = [], [], [], [], []
    report['roles'] = {}
    for role in ROLES:
        exports = reference['mx_exports'][role]
        direct = {bit: ev.read_export(ref / exports[str(bit)]['file'], exports[str(bit)]['sha256']) for bit in (8, 6, 4)}
        hashes = {bit: exports[str(bit)]['sha256'] for bit in (8, 6, 4)}
        selection = reference['checkpoint_selection']['best' if role == ROLES[0] else 'last']
        epoch, checkpoint_sha = selection['epoch'], selection['sha256']
        for payload in direct.values():
            s.require(payload['provenance']['source_epoch'] == epoch
                      and payload['provenance']['checkpoint_sha256'] == checkpoint_sha
                      and payload['provenance']['synthetic_fixture_only'] is reference['synthetic_fixture_only'],
                      'Bronexport heeft afwijkende checkpointprovenance.')
        log(f'EVALUATIE: {role}, epoch {epoch}; acht paden; bevroren BN-bank van de doelprecisie.')
        paths, checks, weight_rows, count_rows = ev.assemble_paths(modules.models, direct, hashes, out, role, rt.device)
        metrics, pairs, details = ev.evaluate_paths(paths, loader, n_test, rt.device, out, role, epoch, log)
        s.require(details['label_sha256'] == reference['dataset_selection']['test']['label_sha256'],
                  'Labels na evaluatie wijken af.')
        checks.update(details)
        report['roles'][role] = {'epoch': epoch, 'checkpoint_sha256': checkpoint_sha, **checks}
        all_metrics.extend(metrics); all_pairs.extend(pairs)
        all_weights.extend({'role': role, 'epoch': epoch, **row} for row in weight_rows)
        all_counts.extend({'role': role, 'epoch': epoch, **row} for row in count_rows)
        for row in metrics:
            log(f'  {row["path"]}: {row["top1_percent"]:.4f}%; correct={row["correct1"]}/{row["n"]}; CE={row["hard_cross_entropy"]:.6f}')
        for key, bit in [('direct8', 8), ('direct6', 6), ('direct4', 4)]:
            new = next(row for row in metrics if row['path'] == key)
            old = next(row for row in selected if row['role'] == role and int(row['weight_bits']) == bit)
            delta = new['correct1'] - int(old['correct1'])
            reproduction.append({'role': role, 'epoch': epoch, 'path': key,
                'historical_correct1': int(old['correct1']), 'reevaluated_correct1': new['correct1'],
                'difference_correct1': delta, 'accuracy_difference_pp': 100 * delta / n_test,
                'historical_ce': float(old['objective_loss']), 'reevaluated_ce': new['hard_cross_entropy'],
                'correct1_matches_history': delta == 0})
            if delta:
                warnings.warn(f'{role}/{key}: {delta:+d} correcte voorspellingen versus het trainingsrapport. '
                              'Vergelijk truncaties met de hier opnieuw gemeten directe doelmodus. '
                              'De behouden CUDA-backend garandeert geen bitexacte herhaling.', RuntimeWarning)
        del paths, direct
    for name, rows in [('metrics.csv', all_metrics), ('paired_metrics.csv', all_pairs),
                       ('weight_pair_metrics.csv', all_weights), ('layer_code_metrics.csv', all_counts),
                       ('historical_reproduction.csv', reproduction)]:
        s.write_csv(out / name, rows)
    for name, digest in before.items():
        s.require(core.file_sha256(ref / name) == digest, 'Origineel invoerbestand gewijzigd: ' + name)
    s.require(s.hashes(test_path) == test_before, 'Testbestand gewijzigd.')
    report.update(input_files_unchanged=True, input_sha256=before, test_file_unchanged=True,
                  train_and_extra_read=False, original_checkpoint_selection_unchanged=True,
                  historical_correct_counts_all_match=all(row['correct1_matches_history'] for row in reproduction),
                  metrics=all_metrics, paired_metrics=all_pairs,
                  derived_models_all_evaluated_from_saved_and_reloaded_codes=True)
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n',
                              'SAVE_FIGURES = True  # Nieuwe analysemap; ruwe resultaten blijven behouden.\n']
    s.write_json(out / NOTEBOOK, notebook)
    with (ROOT / 'notebook_support.py').open('rb') as src, (out / 'notebook_support.py').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    log('EVALUATIE VOLTOOID: beide vaste checkpoints; exponent, mantissa en combinatie; geen updates.')


def feedback(out: Path) -> None:
    names = ['run_report.json', 'run.log', 'warnings.log', 'pip_freeze.txt', 'pip_check.txt',
             'source_manifest.json', 'compatibility.patch', 'dataset_manifest.json', 'truncation_check_report.json',
             'metrics.csv', 'paired_metrics.csv', 'weight_pair_metrics.csv', 'layer_code_metrics.csv',
             'historical_reproduction.csv', NOTEBOOK, 'notebook_support.py']
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as z:
        for name in names:
            if (out / name).is_file(): z.write(out / name, name)
        for path in sorted(out.glob('*_per_image_outputs.npz')): z.write(path, path.name)
        if (out / 'reference_inputs').is_dir():
            for path in sorted((out / 'reference_inputs').iterdir()):
                if path.is_file(): z.write(path, 'reference_inputs/' + path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'evaluate'])
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_triple_posthoc_truncation_eval'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    args = parser.parse_args()
    if (args.command == 'evaluate') != (args.reference_run is not None):
        parser.error('evaluate vereist --reference-run; check gebruikt geen getrainde bronbestanden.')
    if args.reference_run is not None and args.output_root.resolve().is_relative_to(args.reference_run.resolve()):
        parser.error('Gebruik een nieuwe uitvoermap buiten de referentiemap.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-triple-posthoc-truncation-eval-report-v1', 'status': 'RUNNING',
              'package_revision': ROOT.name, 'command': args.command, 'started_at': s.now(),
              'output_directory': str(out), 'argv': sys.argv, 'warnings': [], 'optimizer_steps': 0,
              'batchnorm_recalibration': False, 'new_checkpoint_selection': False,
              'activation_bits': 32, 'shared_scale_compensation': False, 'learned_shared_scale': False,
              'synthetic_checks_only': args.command == 'check', 'synthetic_fixture_only': args.command == 'check',
              'limitations': [
                  'The source was jointly trained for three direct MX modes, not for truncation.',
                  'Every comparison to a direct target uses that same trained target BN bank; no recalibration.',
                  'E4M3-best and epoch100 were already selected; no new selection or training occurs.',
                  'One training seed; the two checkpoints are not independent repeats.',
                  'All derived E8M0 scale bytes are copied unchanged from their respective MX source.',
                  'E4M3-source scales differ from direct E2 scales; round86 is fixed-scale local rounding and can saturate.',
                  'Exponent MSB truncation includes no exponent-bias correction or scale compensation.',
                  'FP32 decoded buffers emulate MX; no native-kernel speed or energy claim.',
                  'The six quantized FP32 masters are absent; FP32 exception layers and BN auxiliary state remain.',
                  'A low truncation accuracy is an outcome, not a software FAIL; historical direct mismatches are reported.',
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
            log('POST-HOC: drie directe modi, exponent- en mantissatruncatie op dezelfde checkpoints; A32; vaste bronschalen.')
            report['package_sha256'] = s.verify_package()
            source, info = s.legacy.prepare_source(out); report['source'] = info
            dependencies(out, log)
            rt = s.runtime(20261009, args.device)
            report.update(environment=rt.environment, execution_settings=rt.settings, interpreter=sys.executable)
            modules = s.import_source(source)
            log('GPU: ' + rt.environment['device_name'])
            if args.command == 'check':
                import test_truncation
                report['tests'] = test_truncation.run_tests(modules, rt.device, out, log)
                s.write_json(out / 'truncation_check_report.json', report['tests'])
                report['notebook_dependencies'] = {}
                for name in ('pandas', 'matplotlib', 'ipykernel'):
                    try: report['notebook_dependencies'][name] = metadata.version(name)
                    except metadata.PackageNotFoundError: report['notebook_dependencies'][name] = 'MISSING; notebook only'
                log('CHECK PASS: bitbewerkingen, vaste schalen, frozen modellen en opslaan/herladen.')
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
