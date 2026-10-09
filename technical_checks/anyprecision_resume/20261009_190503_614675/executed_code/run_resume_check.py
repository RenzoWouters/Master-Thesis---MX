"""Compare uninterrupted training with save -> process exit -> resume.

Runs only the tiny epoch-boundary check, NEVER the 400-epoch reference training.
Default uses existing, checksum-verified CIFAR-10 files, without any download.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
WORKER_PATH = ROOT / 'resume_worker.py'
import run_pilot as pilot
support = pilot.support
require = support.require

TEST_CONFIG = {
    'purpose': 'technical_epoch_boundary_resume_check',
    'seed': 20261009, 'epochs': 3, 'train_per_class': 2, 'val_per_class': 2,
    'batch_size': 8, 'bits': [1, 2, 4, 8, 32], 'model': 'resnet20q', 'expand': 5,
    'optimizer': 'Adam', 'lr': 0.001, 'weight_decay': 0.0,
    'scheduler_milestones': [1, 2], 'scheduler_gamma': 0.1,
    'workers': 0, 'drop_last': False,
    'rng_policy': 'continuous streams restored at epoch boundary; no per-epoch reseeding',
    'scheduler_note': 'ARTIFICIAL TEST milestones, NOT the 400-epoch research schedule',
}


def verify_package() -> dict[str, str]:
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        require(support.sha256(ROOT / name) == expected,
                'Pakketbestand is gewijzigd: ' + name + '; gebruik het oorspronkelijke controlepakket.')
    return manifest


def start_worker(label: str, argv: list[str], env: dict[str, str], log) -> dict[str, Any]:
    log('\n' + label)
    started = datetime.now().astimezone().isoformat()
    t0 = time.perf_counter()
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', env=env,
                          cwd=str(ROOT), bufsize=1) as child:
        try:
            require(child.stdout is not None, 'Geen subprocess-uitvoer.')
            for line in child.stdout:
                log(line.rstrip('\r\n'))
            rc = child.wait()
        except BaseException:
            # Never leave a hidden training child running after Ctrl+C in parent.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            raise
    return {'label': label, 'pid': child.pid, 'returncode': rc, 'argv': argv,
            'started_at': started, 'ended_at': datetime.now().astimezone().isoformat(),
            'seconds_diagnostic_only': time.perf_counter() - t0}


def compare_outputs(out: Path) -> dict[str, Any]:
    import torch
    import resume_core as core
    torch.set_num_threads(2)
    a1 = core.load_verified(out / 'uninterrupted' / 'epoch_001.pt')
    b1 = core.load_verified(out / 'before_stop' / 'epoch_001.pt')
    prefix_diff = core.differences(a1, b1)
    prefix_ok = not prefix_diff
    del a1, b1
    a = core.load_verified(out / 'uninterrupted' / 'epoch_003.pt')
    b = core.load_verified(out / 'after_resume' / 'epoch_003.pt')
    component_diffs = {name: core.differences(a[name], b[name]) for name in
                       ['contract', 'model', 'optimizer', 'scheduler', 'rng',
                        'trace', 'validation_logits', 'completed_epoch', 'next_epoch', 'optimizer_steps']}
    whole_diff = core.differences(a, b)
    uninterrupted_trace = json.loads((out / 'uninterrupted' / 'trace.json').read_text(encoding='utf-8'))
    resumed_trace = json.loads((out / 'after_resume' / 'trace.json').read_text(encoding='utf-8'))
    phase_reports = {name: json.loads((out / name / 'worker_report.json').read_text(encoding='utf-8'))
                     for name in ['uninterrupted', 'before_stop', 'after_resume']}
    checks = {
        'same_checkpoint_before_stop': prefix_ok,
        'same_final_model_including_batchnorm': not component_diffs['model'],
        'same_final_optimizer': not component_diffs['optimizer'],
        'same_final_scheduler': not component_diffs['scheduler'],
        'same_final_rng_including_loader_generators': not component_diffs['rng'],
        'same_training_history_losses_orders_and_augmented_inputs': not component_diffs['trace'],
        'same_final_validation_logits_all_precisions': not component_diffs['validation_logits'],
        'same_entire_checkpoint_content': not whole_diff,
        'same_exported_json_history': uninterrupted_trace == resumed_trace == a['trace'] == b['trace'],
        'new_process_restoration_checks_passed': all(phase_reports['after_resume']['restore_checks'].values()),
        'resume_started_at_epoch_2': phase_reports['after_resume']['first_epoch_this_process'] == 2,
        'nine_optimizer_steps_per_path': a['optimizer_steps'] == b['optimizer_steps'] == 9,
        'three_completed_mini_epochs': a['completed_epoch'] == b['completed_epoch'] == 3,
        'expected_lr_sequence': [e['lr_used'] for e in a['trace']['epochs']] == [0.001, 0.0001, 0.00001],
        'all_worker_statuses_pass': all(p['status'] == 'PASS' for p in phase_reports.values()),
        'all_sources_unchanged': all(p['source_unchanged'] for p in phase_reports.values()),
    }
    different_tensors = {}
    for key in component_diffs['model'][:10]:
        name = key.removeprefix('/')
        if name in a['model'] and name in b['model']:
            x, y = a['model'][name], b['model'][name]
            different_tensors[name] = {'shape_a': list(x.shape), 'shape_b': list(y.shape)}
            if x.shape == y.shape:
                different_tensors[name]['max_abs_difference'] = float((x.to(torch.float64) - y.to(torch.float64)).abs().max())
    result = {'checks': checks, 'all_checks_pass': all(checks.values()),
        'prefix_differences': prefix_diff[:20],
        'component_difference_paths': {k: v[:20] for k, v in component_diffs.items()},
        'whole_checkpoint_difference_count': len(whole_diff),
        'different_model_tensors': different_tensors,
        'lr_used_per_mini_epoch': [e['lr_used'] for e in a['trace']['epochs']],
        'optimizer_steps_per_path': a['optimizer_steps'],
        'environment': phase_reports['uninterrupted']['environment'],
        'synthetic_fixture_only': bool(phase_reports['uninterrupted'].get('synthetic_fixture_only')),
        'warnings': {name: p['warnings'] for name, p in phase_reports.items()},
        'final_model_sha256': a['trace']['epochs'][-1]['model_sha256'],
        'comparison_rule': 'exact tensor values/dtypes/shapes and Python values; NOT binary .pt file bytes'}
    return result


def run_check(args, out: Path, report: dict, log) -> None:
    report['package_sha256'] = verify_package()
    log('ANY-PRECISION HERVATCONTROLE - geen referentietraining, geen MX')
    log('A: 3 mini-epochs in een proces. B: 1 mini-epoch, proces stopt, nieuw proces doet 2 en 3.')
    log('20 trainingsbeelden, 20 controlebeelden; batch 8; drie optimizerstappen per mini-epoch.')
    log('Learning-rateverlagingen na mini-epochs 1 en 2 zijn alleen voor deze technische controle.')
    require(args.data_root.is_dir(), 'Bestaande datamap ontbreekt; er wordt NIETS gedownload.')
    source, info = pilot.prepare_source(out)
    report['source'] = info
    source_manifest = out / 'source_manifest.json'
    support.json_write(out / 'test_config.json', TEST_CONFIG)
    report['config'] = TEST_CONFIG
    # Keep exact executed code beside each report. No datasets/checkpoints copied.
    code_snapshot = out / 'executed_code'
    code_snapshot.mkdir()
    for name in ['run_resume_check.py', 'resume_worker.py', 'resume_core.py',
                 'run_pilot.py', 'smoke_support.py', 'PACKAGE_SHA256.json']:
        shutil.copyfile(ROOT / name, code_snapshot / name)
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(TEST_CONFIG['seed']), PYTHONUNBUFFERED='1',
               PYTHONIOENCODING='utf-8', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    base = [sys.executable, '-B', str(WORKER_PATH), '--config', str(out / 'test_config.json'),
            '--source', str(source), '--source-manifest', str(source_manifest),
            '--data-root', str(args.data_root.resolve()), '--device', args.device]
    phases = [
        ('[1/3] Ononderbroken controlepad A', 'uninterrupted', 3, None),
        ('[2/3] Controlepad B: trainen, checkpoint opslaan en Python afsluiten', 'before_stop', 1, None),
        ('[3/3] Controlepad B: nieuw Python-proces, laden en verder trainen', 'after_resume', 3,
         out / 'before_stop' / 'epoch_001.pt'),
    ]
    report['processes'] = []
    for label, directory, stop_epoch, resume in phases:
        command = base + ['--out', str(out / directory), '--stop-after-epoch', str(stop_epoch)]
        if resume:
            command += ['--resume', str(resume)]
        phase = start_worker(label, command, env, log)
        report['processes'].append(phase)
        require(phase['returncode'] == 0, f'{directory} mislukte; zie {directory}/worker_report.json.')
    report['comparison'] = compare_outputs(out)
    for check, passed in report['comparison']['checks'].items():
        log(('  OK   ' if passed else '  FAIL ') + check)
    require(report['comparison']['all_checks_pass'], 'Niet alle hervatcontroles zijn geslaagd.')
    require(verify_package() == report['package_sha256'], 'Pakketcode gewijzigd tijdens uitvoering.')
    report['checkpoint_storage_MiB'] = sum(p.stat().st_size for p in out.rglob('*.pt')) / 1024**2
    report['epoch_boundary_resume_tested'] = True
    log('Vergelijking voltooid. Er wordt GEEN lange training gestart.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('technical_checks/anyprecision_resume'))
    args = parser.parse_args()
    out = args.output_root.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    args.data_root = args.data_root.resolve()
    log = support.Log(out / 'run.log')
    report: dict[str, Any] = {'status': 'FAIL', 'purpose': 'small cross-process epoch-boundary resume check',
        'run_directory': str(out), 'started_at': datetime.now().astimezone().isoformat(),
        'epoch_boundary_resume_tested': False,
        'limitations': ['No full reference training or paper reproduction.',
            'Tiny subsets/batch and artificial LR milestones; not a training-time benchmark.',
            'No mid-epoch restart, no simulated power failure, no cross-device/version equivalence.',
            'Official test set not evaluated. No MX or stored-code truncation.',
            'Future reference runner must reuse and integrate the tested core.']}
    t0 = time.perf_counter()
    try:
        run_check(args, out, report, log)
        report['status'] = 'PASS'
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
        log('FOUT: ' + str(exc))
        log(report['traceback'])
    finally:
        try:
            freeze = subprocess.run([sys.executable, '-B', '-m', 'pip', 'freeze', '--all'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60, check=False)
            (out / 'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
            report['pip_freeze_returncode'] = freeze.returncode
        except Exception as exc:
            report['pip_freeze_error'] = str(exc)
        report['seconds_diagnostic_only'] = time.perf_counter() - t0
        report['ended_at'] = datetime.now().astimezone().isoformat()
        support.json_write(out / 'resume_check_report.json', report)
        log('\nSTATUS: ' + report['status'])
        log('Rapportmap: ' + str(out))
        log('Geen downloads. Geen pakketten geinstalleerd. Geen eerdere resultaten overschreven.')
        log.handle.close()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
