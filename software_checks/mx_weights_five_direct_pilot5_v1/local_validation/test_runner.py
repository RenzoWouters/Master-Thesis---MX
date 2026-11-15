"""CPU-only synthetic runner checks. No SVHN data or research accuracy.

Run with the project's existing dependencies. All fixtures live in a fresh
temporary directory outside the experiment package. Tests use the real CNN-8.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
import torch
import joint_adapter as a
import joint_support as t
import run_train as runner
import svhn_support as s


def fixture(out):
    out.mkdir()
    torch.set_num_threads(2)
    rt = s.runtime(20261009, 'cpu', strict=False)
    source, source_info = s.legacy.prepare_source(out)
    for name in list(sys.modules):
        if name.split('.')[0] in ('models', 'datasets', 'optimizer', 'utils'):
            del sys.modules[name]
    modules = s.import_source(source)
    c = runner.config()
    c.update(train_examples=8, test_examples=7, batch_size=4, pin_memory=False)
    generator = torch.Generator().manual_seed(1701)
    loaders = {}
    for phase, size in [('train', 8), ('test', 7)]:
        data = torch.utils.data.TensorDataset(torch.randn(size, 3, 40, 40, generator=generator),
                                              torch.arange(size) % 10)
        loaders[phase] = s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(data),
            batch_size=4, shuffle=phase == 'train', num_workers=0, pin_memory=False))
    model = t.build_model(modules, 'joint', rt.device)
    optimizer = modules.optimizer.get_optimizer_config(model, 'adam', c['lr'], c['weight_decay'])
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
    report = {'config': c, 'status': 'DEVELOPMENT_FIXTURE', 'synthetic_fixture_only': True,
              'completed_epochs': 0}
    contract = {'config': c, 'synthetic_fixture_only': True, 'source_sha256': source_info['workcopy_files_sha256']}
    return SimpleNamespace(rt=rt, s=s, a=a, t=t, c=c, model=model, optimizer=optimizer,
        scheduler=scheduler, loaders=loaders, contract=contract, report=report,
        source=source, source_info=source_info, modules=modules)


def log(message):
    if message.startswith('CHECKPOINT'): print(message, flush=True)


def run_checks(output_root=None):
    root = Path(tempfile.mkdtemp(prefix='mx_five_runner_')) if output_root is None else Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'synthetic CPU runner checks; no SVHN accuracy', 'status': 'RUNNING'}
    full = fixture(root / 'continuous')
    assert runner.run_epochs(full, SimpleNamespace(resume=None, stop_after_epoch=5), root / 'continuous', log) == 5
    full_state = full.rt.core.load_verified(root / 'continuous/checkpoints/epoch_00005.pt')
    assert len(full_state['trace']['metrics']) == 50
    assert full_state['optimizer_steps'] == 10
    assert all(row['n'] in (8, 7) for row in full_state['trace']['metrics'])
    runner.export_results(full, root / 'continuous', log)
    report['five_epoch_endpoint_and_five_exports'] = 'PASS'

    stopped = fixture(root / 'stopped')
    assert runner.run_epochs(stopped, SimpleNamespace(resume=None, stop_after_epoch=2), root / 'stopped', log) == 2
    resumed = fixture(root / 'resumed')
    checkpoint = root / 'stopped/checkpoints/epoch_00002.pt'
    assert runner.run_epochs(resumed, SimpleNamespace(resume=checkpoint, stop_after_epoch=5), root / 'resumed', log) == 5
    resumed_state = resumed.rt.core.load_verified(root / 'resumed/checkpoints/epoch_00005.pt')
    for name in ('model', 'optimizer', 'scheduler', 'rng'):
        assert not full.rt.core.differences(full_state[name], resumed_state[name]), name
    assert not full.rt.core.differences(full_state['trace']['metrics'], resumed_state['trace']['metrics'])
    for epoch in range(1, 6):
        for prefix in ('train_order', 'test_order', 'test_predictions'):
            extension = 'npz' if prefix == 'test_predictions' else 'npy'
            name = f'{prefix}_epoch_{epoch:05d}.{extension}'
            assert (root / 'resumed' / name).is_file(), name
    report['continuous_vs_epoch2_resume_exact_cpu'] = 'PASS'
    report['all_earlier_evidence_copied_into_resume'] = 'PASS'

    try:
        bad = fixture(root / 'reject_completed')
        runner.run_epochs(bad, SimpleNamespace(resume=root / 'continuous/checkpoints/epoch_00005.pt',
                                               stop_after_epoch=5), root / 'reject_completed', log)
    except RuntimeError as exc:
        assert 'pilotepoch1..4' in str(exc)
    else:
        raise AssertionError('Completed checkpoint was accepted')
    incompatible = copy.deepcopy(resumed_state)
    incompatible['contract']['config']['loss_weights']['E4M3'] = 2.0
    try:
        resumed.rt.core.restore_checkpoint(incompatible, resumed.model, resumed.optimizer,
            resumed.scheduler, {}, resumed.rt.device, resumed.contract)
    except RuntimeError as exc:
        assert 'Hervatprotocol' in str(exc)
    else:
        raise AssertionError('Different loss protocol was accepted')
    report['incompatible_or_completed_resume_rejected'] = 'PASS'

    failed = fixture(root / 'save_failure')
    original_save = failed.rt.core.atomic_save
    def fail_epoch5(path, payload):
        if path.name == 'epoch_00005.pt': raise OSError('SIMULATED_CHECKPOINT_WRITE_FAILURE')
        return original_save(path, payload)
    failed.rt.core.atomic_save = fail_epoch5
    try:
        try:
            runner.run_epochs(failed, SimpleNamespace(resume=None, stop_after_epoch=5), root / 'save_failure', log)
        except OSError as exc:
            assert 'SIMULATED_CHECKPOINT' in str(exc)
        else:
            raise AssertionError('Simulated save failure was swallowed')
    finally:
        failed.rt.core.atomic_save = original_save
    assert failed.report['completed_epochs'] == 4
    with (root / 'save_failure/metrics.csv').open() as file:
        saved_metrics = list(csv.DictReader(file))
    assert len(saved_metrics) == 40 and max(int(r['epoch']) for r in saved_metrics) == 4
    assert (root / 'save_failure/metrics_epoch_00005.csv').is_file()
    assert not (root / 'save_failure/checkpoints/epoch_00005.pt').exists()
    report['save_failure_keeps_raw_evidence_without_false_completed_epoch'] = 'PASS'
    report['status'] = 'PASS'
    (root / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run_checks(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
