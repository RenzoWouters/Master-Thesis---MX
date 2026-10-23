"""Evaluate real E2M3-code mantissa truncation on the frozen dual-MX reference.

check = synthetic numerical/integration tests. evaluate = four paired paths
on the complete SVHN test split for the already selected best and last roles.
There is no training command, optimizer, calibration, dataset download or
checkpoint selection in this runner. Original input files remain read-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.metadata as metadata
import json
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


def dependencies(out: Path, log) -> None:
    for command, name in [('freeze', 'pip_freeze.txt'), ('check', 'pip_check.txt')]:
        result = subprocess.run([sys.executable, '-m', 'pip', command], capture_output=True,
                                text=True, encoding='utf-8', errors='replace')
        (out / name).write_text(result.stdout + result.stderr, encoding='utf-8')
        s.require(result.returncode == 0, 'pip ' + command + ' mislukt; zie ' + name + '. Niets upgraden.')
        if command == 'check':
            log('pip check: ' + result.stdout.strip())


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def inspect_reference(ref: Path, rt) -> tuple[dict, dict, list]:
    """Validate protocol/provenance before loading any model exports."""
    import math
    import resume_core as core
    required = ['run_report.json', 'selected_metrics.csv', 'epoch_metrics.csv', 'checkpoint_selection.json']
    s.require(ref.is_dir() and all((ref / n).is_file() for n in required),
              'Geen volledige referentiemap; verwacht de originele _joint_run-map, niet een feedback-ZIP.')
    report = json.loads((ref / 'run_report.json').read_text(encoding='utf-8'))
    s.require(report.get('schema') == 'mx-weights-dual-reference100-report-v1' and report.get('status') == 'PASS',
              'Verwacht een geslaagde volledige tweemodus-MX-referentie.')
    s.require(report.get('synthetic_fixture_only') is False and report.get('full_training_completed') is True
              and report.get('completed_epoch') == 100 and report.get('optimizer_steps') == 57300,
              'Geen echte, volledige honderd-epochrun.')
    expected = json.loads((ROOT / 'REFERENCE_PROTOCOL.json').read_text(encoding='utf-8'))
    changes = core.differences(report['config'], expected)
    s.require(not changes, 'Referentieprotocol wijkt af: ' + ', '.join(changes[:10]))
    for key in ('python', 'torch', 'torchvision', 'numpy', 'scipy', 'pillow', 'cuda_build', 'cudnn', 'device_name'):
        s.require(report['environment'][key] == rt.environment[key],
                  'Uitvoeringsomgeving wijkt af bij ' + key + '; niet stilzwijgend als gelijke evaluatie behandelen.')
    for key in ('cudnn_benchmark', 'cudnn_deterministic', 'deterministic_algorithms',
                'matmul_fp32_precision', 'cudnn_conv_fp32_precision', 'AMP', 'torch_compile'):
        s.require(report['execution_settings'][key] == rt.settings[key], 'Backend wijkt af bij ' + key)
    for name in ('mx_format.py', 'mx_e2m1.py', 'mx_adapter.py', 'dual_adapter.py', 'svhn_support.py', 'smoke_support.py'):
        s.require(report['package_sha256'].get(name) == core.file_sha256(ROOT / name),
                  'Herbruikte code wijkt af van de trainingscode: ' + name)
    epoch_rows = csv_rows(ref / 'epoch_metrics.csv')
    expected_keys = {(e, p, b) for e in range(1, 101) for p in ('train', 'test') for b in (4, 6)}
    actual = {(int(r['epoch']), r['phase'], int(r['weight_bits'])) for r in epoch_rows}
    s.require(len(epoch_rows) == 400 and actual == expected_keys, 'Onvolledige epoch-/modusgeschiedenis.')
    s.require(all(math.isfinite(float(r['top1_percent'])) for r in epoch_rows), 'Niet-eindige referentiescores.')
    high_tests = sorted((r for r in epoch_rows if r['phase'] == 'test' and int(r['weight_bits']) == 6),
                        key=lambda r: int(r['epoch']))
    best = max(high_tests, key=lambda r: float(r['top1_percent']))
    selection = report['checkpoint_selection']
    s.require(selection == json.loads((ref / 'checkpoint_selection.json').read_text(encoding='utf-8')),
              'Checkpointselectierapporten zijn inconsistent.')
    s.require(selection['best']['epoch'] == int(best['epoch']) and selection['last']['epoch'] == 100,
              'Beste/laatste rol wijkt af van de bestaande selectieregel.')
    selected = csv_rows(ref / 'selected_metrics.csv')
    s.require(len(selected) == 4 and {(r['role'], int(r['weight_bits'])) for r in selected}
              == {(role, b) for role in ('best_E2M3_test', 'last_epoch') for b in (4, 6)},
              'Onvolledige geselecteerde referentiemetrics.')
    for r in selected:
        e = selection['best' if r['role'] == 'best_E2M3_test' else 'last']['epoch']
        s.require(int(r['epoch']) == e and int(r['n']) == 26032, 'Onjuist geselecteerd eindpunt.')
        historical = next(x for x in epoch_rows if x['phase'] == 'test' and int(x['epoch']) == e
                          and x['weight_bits'] == r['weight_bits'])
        s.require(r == {'role': r['role'], **historical}, 'Geselecteerde metrics stemmen niet overeen met epochhistorie.')
    fingerprints = {n: core.file_sha256(ref / n) for n in required}
    for role in ('best_E2M3_test', 'last_epoch'):
        for bit in ('6', '4'):
            item = report['mx_exports'][role][bit]
            name = item['file']
            s.require(type(name) is str and Path(name).name == name and not any(c in name for c in '/\\:'),
                      'Onveilig exportpad in rapport.')
            p = ref / name
            s.require(item['decoded_weights_exact'] is True and item['six_fp32_masters_absent'] is True
                      and item['truncation'] is False, 'Bronexport is niet de afgesproken directe export.')
            s.require(p.is_file() and p.stat().st_size == item['bytes'], 'Export ontbreekt of verkeerde grootte: ' + str(p))
            for q in (p, p.with_suffix(p.suffix + '.sha256')):
                s.require(q.is_file(), 'Exportchecksum ontbreekt: ' + str(q))
                fingerprints[q.name] = core.file_sha256(q)
            s.require(fingerprints[name] == item['sha256'], 'Exporthash wijkt af.')
    return report, fingerprints, selected


def prepare_test_only(args, modules, reference: dict, out: Path, log):
    import torch
    data = args.data_root.resolve()
    path = data / 'test_32x32.mat'
    s.require(path.is_file(), 'test_32x32.mat ontbreekt; gebruik de bestaande data/svhn-map. Geen automatische download.')
    info = s.hashes(path)
    expected = reference['dataset_files']['files']['test_32x32.mat']
    s.require(all(info[k] == expected[k] for k in info), 'Testdata wijken af van de trainingsreferentie.')
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
    log('DATA PASS: alleen test=26032; batch=128; 204 batches; train en extra niet gelezen.')
    s.write_json(out / 'dataset_manifest.json', {'test_file': {'path': str(path), **info},
                 'selection': selection, 'transform': repr(transform), 'train_read': False, 'extra_read': False})
    return loader, path, info


def evaluate(args, out: Path, report: dict, rt, modules, log) -> None:
    import torch
    import evaluation as ev
    import truncation as tr
    import resume_core as core
    ref = args.reference_run.resolve()
    s.require(not out.is_relative_to(ref), 'Uitvoer mag niet in de oorspronkelijke resultatenmap terechtkomen.')
    reference, before, selected = inspect_reference(ref, rt)
    report['reference'] = {'directory': str(ref), 'report_sha256': before['run_report.json'],
                           'checkpoint_selection': reference['checkpoint_selection'],
                           'checkpoint_files_opened': False, 'FP32_master_weights_opened': False,
                           'source_export_hashes': reference['mx_exports'], 'contract_matches': True}
    copies = out / 'reference_inputs'; copies.mkdir()
    for name in ('run_report.json', 'selected_metrics.csv', 'epoch_metrics.csv', 'checkpoint_selection.json'):
        shutil.copy2(ref / name, copies / name)
    loader, test_path, test_before = prepare_test_only(args, modules, reference, out, log)
    all_metrics, all_pairs, all_weights, all_counts, reproduction = [], [], [], [], []
    report['roles'] = {}
    for role in ('best_E2M3_test', 'last_epoch'):
        exports = reference['mx_exports'][role]
        high_info, low_info = exports['6'], exports['4']
        h = ev.read_export(ref / high_info['file'], high_info['sha256'])
        a = ev.read_export(ref / low_info['file'], low_info['sha256'])
        epoch = reference['checkpoint_selection']['best' if role == 'best_E2M3_test' else 'last']['epoch']
        cp_hash = reference['checkpoint_selection']['best' if role == 'best_E2M3_test' else 'last']['sha256']
        for payload in (h, a):
            s.require(payload['provenance']['source_epoch'] == epoch
                      and payload['provenance']['checkpoint_sha256'] == cp_hash
                      and payload['provenance']['synthetic_fixture_only'] is False,
                      'Bronexport heeft afwijkende of synthetische checkpointprovenance.')
        log(f'EVALUATIE: {role}, epoch {epoch}. A/B/C gebruiken alle de oorspronkelijke lage BN-bank.')
        paths, checks, weight_rows, count_rows = ev.assemble_paths(modules.models, h, a, high_info['sha256'],
                                                                  out, role, rt.device)
        metrics, pairs, details = ev.evaluate_paths(paths, loader, 26032, rt.device, out, role, epoch, log)
        s.require(details['label_sha256'] == reference['dataset_selection']['test']['label_sha256'],
                  'Labels na evaluatie wijken af.')
        checks.update(details)
        report['roles'][role] = {'epoch': epoch, **checks}
        all_metrics.extend(metrics); all_pairs.extend(pairs)
        all_weights.extend({'role': role, 'epoch': epoch, **r} for r in weight_rows)
        all_counts.extend({'role': role, 'epoch': epoch, **r} for r in count_rows)
        for r in metrics:
            log(f'  {r["path"]}: {r["top1_percent"]:.4f}%; correct={r["correct1"]}/{r["n"]}; CE={r["hard_cross_entropy"]:.6f}')
        for path, bit in [(tr.PATHS[0], 6), (tr.PATHS[1], 4)]:
            new = next(r for r in metrics if r['path'] == path)
            old = next(r for r in selected if r['role'] == role and int(r['weight_bits']) == bit)
            delta = new['correct1'] - int(old['correct1'])
            checkrow = {'role': role, 'epoch': epoch, 'path': path,
                        'historical_correct1': int(old['correct1']), 'reevaluated_correct1': new['correct1'],
                        'difference_correct1': delta, 'accuracy_difference_pp': 100 * delta / 26032,
                        'historical_ce': float(old['objective_loss']), 'reevaluated_ce': new['hard_cross_entropy'],
                        'correct1_matches_history': delta == 0}
            reproduction.append(checkrow)
            # No accuracy-based rejection or silent tuning. Same-file CPU
            # export and GPU forward/backend rounding may differ minutely.
            if delta:
                warnings.warn(f'{role}/{path}: {delta:+d} correcte voorspellingen versus trainingrapport. '
                              'Alle paden opnieuw gepaard gemeten; vergelijk de nieuwe A met C. '
                              'Herhaalbaarheid is niet bitexact gegarandeerd.', RuntimeWarning)
        del paths, h, a
    s.write_csv(out / 'metrics.csv', all_metrics)
    s.write_csv(out / 'paired_metrics.csv', all_pairs)
    s.write_csv(out / 'weight_pair_metrics.csv', all_weights)
    s.write_csv(out / 'layer_code_metrics.csv', all_counts)
    s.write_csv(out / 'historical_reproduction.csv', reproduction)
    for name, digest in before.items():
        s.require(core.file_sha256(ref / name) == digest, 'Origineel invoerbestand gewijzigd: ' + name)
    s.require(s.hashes(test_path) == test_before, 'Testbestand gewijzigd.')
    report.update(input_files_unchanged=True, input_sha256=before, test_file_unchanged=True,
                  train_and_extra_read=False, original_checkpoint_selection_unchanged=True,
                  historical_correct_counts_all_match=all(r['correct1_matches_history'] for r in reproduction),
                  metrics=all_metrics, paired_metrics=all_pairs,
                  derived_models_all_evaluated_from_saved_and_reloaded_codes=True)
    notebook = json.loads((ROOT / 'MX_mantissa_truncatie_resultaten.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            text = ''.join(cell['source'])
            text = text.replace('RUN_DIR = Path(".")  # RUN_DIRECTORY',
                                'RUN_DIR = Path(' + json.dumps(str(out)) + ')  # RUN_DIRECTORY')
            cell['source'] = text.splitlines(True)
    s.write_json(out / 'MX_mantissa_truncatie_resultaten.ipynb', notebook)
    log('EVALUATIE VOLTOOID: H/A/B/C voor beide vaste checkpoints; geen nieuwe selectie of training.')


def feedback(out: Path) -> None:
    names = ['run_report.json', 'run.log', 'warnings.log', 'pip_freeze.txt', 'pip_check.txt',
             'source_manifest.json', 'compatibility.patch', 'dataset_manifest.json',
             'metrics.csv', 'paired_metrics.csv', 'weight_pair_metrics.csv',
             'layer_code_metrics.csv', 'historical_reproduction.csv']
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as z:
        for name in names:
            if (out / name).is_file(): z.write(out / name, name)
        for p in out.glob('*_per_image_outputs.npz'):
            z.write(p, p.name)
        if (out / 'reference_inputs').is_dir():
            for p in (out / 'reference_inputs').iterdir():
                if p.is_file(): z.write(p, 'reference_inputs/' + p.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'evaluate'])
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_mantissa_truncation_eval'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    args = parser.parse_args()
    if (args.command == 'evaluate') != (args.reference_run is not None):
        parser.error('evaluate vereist --reference-run; check gebruikt geen getrainde bronbestanden.')
    if args.reference_run is not None:
        if args.output_root.resolve().is_relative_to(args.reference_run.resolve()):
            parser.error('Gebruik een nieuwe uitvoermap buiten de referentiemap.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-mantissa-truncation-evaluation-v1', 'status': 'RUNNING',
              'command': args.command, 'started_at': s.now(), 'output_directory': str(out),
              'argv': sys.argv, 'warnings': [], 'optimizer_steps': 0, 'batchnorm_recalibration': False,
              'new_checkpoint_selection': False, 'synthetic_checks_only': args.command == 'check',
              'limitations': ['E2M3 source was trained for direct quantization, not for truncation.',
                              'A/B/C use the same trained low BN bank; no adaptation or recalibration.',
                              'Best role was already test-selected; last epoch100 is the fixed endpoint.',
                              'One training seed; the two checkpoints are not independent repeats.',
                              'FP32 decoded execution buffers; no native MX speed or energy claim.',
                              'Source model exports omit six FP32 masters but retain FP32 exceptions and BN state.',
                              'B is a control, not code-only truncation. C uses code shift only.',
                              'A comes from the archived direct E2M1 export, not recomputation from a master.',
                              'CPU export versus training GPU arithmetic can affect historical reproduction.']}
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
            log('MANTISSATRUNCATIE: opgeslagen E2M3 -> E2M1; vaste schalen/A32/lage BN; geen training.')
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
                report['notebook_dependencies'] = {}
                for name in ('pandas', 'matplotlib', 'ipykernel'):
                    try: report['notebook_dependencies'][name] = metadata.version(name)
                    except metadata.PackageNotFoundError: report['notebook_dependencies'][name] = 'MISSING; notebook only'
                log('CHECK PASS: codebitshift, schalen, signed zero, twee BN-banken en opslag/herladen.')
            else:
                evaluate(args, out, report, rt, modules, log)
            s.verify_package()
            s.require(all(s.legacy.sha256(source / n) == d for n, d in info['workcopy_files_sha256'].items()),
                      'Bronwerkcopy gewijzigd tijdens de proef.')
            report['source_unchanged'] = True; report['status'] = 'PASS'; code = 0
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report.update(error=repr(exc), traceback=traceback.format_exc())
        log(report['status'] + ': ' + str(exc)); log(report['traceback'])
        log('Geen bestaande onderzoekscode of bronresultaten overschreven. Deel de feedbackbundel.')
    finally:
        report['ended_at'] = s.now(); report['session_wall_seconds'] = time.perf_counter() - started
        log('EINDSTATUS: ' + report['status']); log('Rapport: ' + str(out / 'run_report.json'))
        log.handle.close(); s.write_json(out / 'run_report.json', report); feedback(out)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
