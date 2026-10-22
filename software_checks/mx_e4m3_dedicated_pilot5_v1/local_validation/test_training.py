"""Small synthetic CPU integration test, explicitly NOT an SVHN experiment.

Only this test redirects Tensor.cuda to CPU to exercise the unchanged authors'
CUDA forward. Production CLI remains CUDA-only. Two batches per mini-epoch.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch
import svhn_support as s
import single_support as t
import run_train as r
import resume_core as core


def main(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    logs = []
    log = logs.append
    prepared = out / 'source'
    prepared.mkdir()
    source, source_info = s.legacy.prepare_source(prepared)
    modules = s.import_source(source)
    generator = torch.Generator().manual_seed(4837)
    x = torch.rand(4, 3, 40, 40, generator=generator) * 2 - 1
    labels = torch.tensor([0, 1, 2, 3])
    datasets = {phase: torch.utils.data.TensorDataset(x.clone(), labels.clone())
                for phase in ('train', 'test')}

    def context(path):
        path.mkdir()
        rt = s.runtime(20261009, 'cpu', strict=False)
        c = r.config()
        # Fixture-only shorter epochs/LR transitions; production config untouched.
        c.update(train_examples=4, test_examples=4, batch_size=2,
                 scheduler_milestones=[2, 3, 4], purpose='SYNTHETIC_CPU_TEST_ONLY')
        model = t.build_model(modules, 'mx', rt.device)
        optimizer = modules.optimizer.get_optimizer_config(model, 'adam', c['lr'], 0.0)
        scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
        loaders = {phase: s.TrackedLoader(torch.utils.data.DataLoader(
            s.IndexedDataset(ds), batch_size=2, shuffle=phase == 'train',
            num_workers=0, pin_memory=False, drop_last=False))
            for phase, ds in datasets.items()}
        contract = {'config': c, 'environment': rt.environment, 'settings': rt.settings,
                    'source_files': source_info['workcopy_files_sha256'],
                    'synthetic_fixture_only': True}
        report = {'status': 'PASS', 'schema': 'mx-e4m3-dedicated-pilot5-report-v1',
                  'config': c, 'environment': rt.environment, 'execution_settings': rt.settings,
                  'warnings': [], 'synthetic_fixture_only': True,
                  'limitations': ['SYNTHETIC CPU TEST; no SVHN data or GPU timing']}
        return SimpleNamespace(rt=rt, c=c, source=source, source_info=source_info,
                               modules=modules, model=model, optimizer=optimizer,
                               scheduler=scheduler, loaders=loaders, contract=contract, report=report)

    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor

    def must_fail(callback):
        try:
            callback()
        except (RuntimeError, ValueError, FileExistsError):
            return
        raise AssertionError('Expected refusal did not occur')

    tests = {}
    with patch.object(torch.Tensor, 'cuda', cpu_cuda):
        continuous_path = out / 'continuous'
        continuous = context(continuous_path)
        trace = r.run_epochs(continuous, continuous_path, log, 5, install_signal_handler=False)
        final_state = core.load_verified(continuous_path / 'checkpoints/epoch_00005.pt')
        r.export_results(continuous, continuous_path, trace, log)
        s.write_json(continuous_path / 'run_report.json', continuous.report)
        r.feedback_bundle(continuous_path)
        tests['five_synthetic_epochs_10_updates'] = continuous.report['optimizer_steps'] == 10
        assert tests['five_synthetic_epochs_10_updates']
        assert continuous.report['training_summary']['pilot_completed']
        assert not continuous.report['training_summary']['full_100_epoch_training_completed']
        assert continuous.report['training_summary']['synthetic_fixture_only']
        assert len(list((continuous_path / 'checkpoints').glob('epoch_*.pt'))) == 5
        with (continuous_path / 'epoch_metrics.csv').open() as handle:
            assert len(list(csv.DictReader(handle))) == 10

        # Start from a real immutable epoch2 checkpoint, in a new context/process-like state.
        resume_path = out / 'resumed'
        resumed = context(resume_path)
        checkpoint = continuous_path / 'checkpoints/epoch_00002.pt'
        checkpoint_hash = core.file_sha256(checkpoint)
        resumed_trace = r.run_epochs(resumed, resume_path, log, 5, checkpoint, False)
        resumed_state = core.load_verified(resume_path / 'checkpoints/epoch_00005.pt')
        for key in ('model', 'optimizer', 'scheduler', 'rng'):
            errors = core.differences(final_state[key], resumed_state[key])
            assert not errors, (key, errors[:4])
        for a, b in zip(trace['epochs'], resumed_trace['epochs']):
            for key in ('train', 'test', 'model_sha256', 'lr_used', 'lr_next',
                        'best_epoch_so_far', 'weight_updates', 'gradient_checks'):
                assert a[key] == b[key], (a['epoch'], key)
            for phase in ('train', 'test'):
                assert a['coverage'][phase]['order_sha256'] == b['coverage'][phase]['order_sha256']
        assert core.file_sha256(checkpoint) == checkpoint_hash
        r.export_results(resumed, resume_path, resumed_trace, log)
        tests['resume_model_optimizer_scheduler_rng_metrics_exact_on_CPU'] = True
        tests['resume_covers_synthetic_LR_transitions'] = True

        wrong_contract = copy.deepcopy(resumed.contract)
        wrong_contract['config']['weight_format'] = 'MXFP6_E2M3'
        must_fail(lambda: core.restore_checkpoint(final_state, resumed.model, resumed.optimizer,
            resumed.scheduler, {}, resumed.rt.device, wrong_contract))
        must_fail(lambda: core.atomic_save(checkpoint, final_state))
        corrupt = out / 'corrupt.pt'
        corrupt.write_bytes(checkpoint.read_bytes() + b'bad')
        corrupt.with_suffix('.pt.sha256').write_text(checkpoint_hash + '\n', encoding='ascii')
        must_fail(lambda: core.load_verified(corrupt))
        tests['wrong_format_contract_corruption_and_overwrite_rejected'] = True

        stopped_path = out / 'stopped'
        stopped = context(stopped_path)
        (stopped_path / 'STOP_AFTER_EPOCH').touch()
        stopped_trace = r.run_epochs(stopped, stopped_path, log, 5, install_signal_handler=False)
        assert len(stopped_trace['epochs']) == 1
        r.export_results(stopped, stopped_path, stopped_trace, log)
        assert not stopped.report['training_summary']['pilot_completed']
        stopped.report['status'] = 'STOPPED_AT_EPOCH_BOUNDARY'
        s.write_json(stopped_path / 'run_report.json', stopped.report)
        tests['stop_marker_finishes_epoch_without_false_completion'] = True

    # Highest active-mode score only; a tie must keep the first epoch.
    rows = [{'epoch': i + 1, 'test': [{'top1_percent': score}]}
            for i, score in enumerate([10, 80, 80, 20, 50])]
    assert t.best_record(rows)['epoch'] == 2
    tests['best_selection_first_strict_maximum'] = True
    cli_results = {}
    for arguments in (['run', '--stop-after-epoch', '100'], ['run', '--device', 'cpu'],
                      ['run', '--mode', 'control'], ['resume']):
        result = subprocess.run([sys.executable, str(ROOT / 'run_train.py'), *arguments],
                                cwd=out, text=True, capture_output=True)
        assert result.returncode == 2, (arguments, result.stdout, result.stderr)
        cli_results[' '.join(arguments)] = {'exit_code': result.returncode,
                                         'reason': result.stderr.splitlines()[-1]}
    tests['production_CLI_limits_enforced'] = cli_results
    tests['no_production_output_created_by_rejected_CLI'] = not (out / 'mx_runs').exists()
    assert tests['no_production_output_created_by_rejected_CLI']
    assert r.config()['scheduler_milestones'] == [50, 75, 90]
    assert r.config()['train_examples'] == 73257 and r.config()['pilot_stop_epoch'] == 5
    assert all(core.file_sha256(source / name) == digest
               for name, digest in source_info['workcopy_files_sha256'].items())
    tests['production_config_and_source_unchanged'] = True
    report = {'status': 'PASS', 'synthetic_fixture_only': True,
              'device': 'CPU; test-only Tensor.cuda redirection',
              'environment': continuous.rt.environment, 'tests': tests,
              'not_a_SVHN_accuracy_or_GPU_timing_result': True}
    (out / 'integration_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'synthetic.log').write_text('\n'.join(logs) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
