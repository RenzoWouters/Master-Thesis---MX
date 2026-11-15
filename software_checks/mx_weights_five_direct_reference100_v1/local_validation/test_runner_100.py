"""Real CNN-8, synthetic CPU-only 100-epoch runner checks; no SVHN accuracy."""
from __future__ import annotations

import ast
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
    c.update(train_examples=2, test_examples=2, batch_size=2, pin_memory=False)
    generator = torch.Generator().manual_seed(1701)
    loaders = {}
    for phase in ('train', 'test'):
        data = torch.utils.data.TensorDataset(torch.randn(2, 3, 40, 40, generator=generator), torch.arange(2))
        loaders[phase] = s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(data),
            batch_size=2, shuffle=phase == 'train', num_workers=0, pin_memory=False))
    model = t.build_model(modules, 'joint', rt.device)
    optimizer = modules.optimizer.get_optimizer_config(model, 'adam', c['lr'], c['weight_decay'])
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
    report = {'config': c, 'status': 'DEVELOPMENT_FIXTURE', 'synthetic_fixture_only': True, 'completed_epochs': 0}
    contract = {'config': c, 'synthetic_fixture_only': True, 'source_sha256': source_info['workcopy_files_sha256']}
    return SimpleNamespace(rt=rt, s=s, a=a, t=t, c=c, model=model, optimizer=optimizer,
        scheduler=scheduler, loaders=loaders, contract=contract, report=report,
        source=source, source_info=source_info, modules=modules)


def log(message):
    if message.startswith('CHECKPOINT') and any(f'epoch_{e:05d}' in message for e in (1, 50, 51, 75, 76, 90, 91, 99, 100)):
        print(message, flush=True)


def expect_resume_rejection(ctx, path, out, text):
    try:
        runner.run_epochs(ctx, SimpleNamespace(resume=path, stop_after_epoch=100), out, log)
    except RuntimeError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError('Invalid resume checkpoint accepted')


def run_checks(output_root=None, pilot_package=None):
    root = Path(tempfile.mkdtemp(prefix='mx_five_reference100_')) if output_root is None else Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'real CNN-8 with two synthetic training/test examples on CPU; no SVHN accuracy', 'status': 'RUNNING'}
    assert runner.END_EPOCH == 100
    assert runner.parse_args(['train']).resume is None
    assert runner.parse_args(['train']).stop_after_epoch == 100
    assert runner.config()['planned_epochs'] == 100
    assert runner.config()['checkpoint_selection_on_test'] is False
    if pilot_package is not None:
        original = ast.parse((Path(pilot_package) / 'joint_support.py').read_text())
        current = ast.parse((PACKAGE_ROOT / 'joint_support.py').read_text())
        old_nodes = {n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new_nodes = {n.name: n for n in current.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert old_nodes.keys() == new_nodes.keys()
        for name in old_nodes:
            if name != 'phase':
                assert ast.dump(old_nodes[name], include_attributes=False) == ast.dump(new_nodes[name], include_attributes=False), name
        old_phase = ast.get_source_segment((Path(pilot_package) / 'joint_support.py').read_text(), old_nodes['phase'])
        new_phase = ast.get_source_segment((PACKAGE_ROOT / 'joint_support.py').read_text(), new_nodes['phase'])
        expected = old_phase.replace('lr, out, log):', 'lr, out, log, total_epochs=5):').replace('{epoch}/5 ', '{epoch}/{total_epochs} ')
        assert expected == new_phase
        report['training_functions_unchanged_from_pilot_except_progress_label'] = 'PASS'

    full_out = root / 'continuous'
    full = fixture(full_out)
    assert runner.run_epochs(full, SimpleNamespace(resume=None, stop_after_epoch=100), full_out, log) == 100
    state = full.rt.core.load_verified(full_out / 'checkpoints/epoch_00100.pt')
    assert state['completed_epoch'] == 100 and state['next_epoch'] == 101
    assert len(state['trace']['metrics']) == 1000 and state['optimizer_steps'] == 100
    assert full.report['full_training_completed'] is True
    assert full.report['timing']['measured_100_epoch_seconds'] > 0
    assert full.report['timing']['estimated_100_epoch_seconds'] is None
    for epoch in state['trace']['epochs']:
        expected = 0.001 if epoch['epoch'] <= 50 else (0.0001 if epoch['epoch'] <= 75 else (0.00001 if epoch['epoch'] <= 90 else 0.000001))
        assert abs(epoch['lr_used'] - expected) < 1e-15, epoch['epoch']
    report['actual_100_epoch_runner_and_50_75_90_lr_boundaries'] = 'PASS'
    runner.export_results(full, full_out, log)
    for item in full.report['exports'].values():
        assert item['file'].startswith('exports/epoch00100_')
        assert (full_out / item['file']).is_file()
    report['five_epoch100_exports_and_relative_report_paths'] = 'PASS'

    for boundary in (50, 99):
        out = root / f'resumed_from_{boundary}'
        resumed = fixture(out)
        checkpoint = full_out / f'checkpoints/epoch_{boundary:05d}.pt'
        assert runner.run_epochs(resumed, SimpleNamespace(resume=checkpoint, stop_after_epoch=100), out, log) == 100
        final = resumed.rt.core.load_verified(out / 'checkpoints/epoch_00100.pt')
        for name in ('model', 'optimizer', 'scheduler', 'rng'):
            assert not full.rt.core.differences(state[name], final[name]), (boundary, name)
        assert not full.rt.core.differences(state['trace']['metrics'], final['trace']['metrics'])
        for epoch in (1, boundary, 100):
            assert (out / f'checkpoints/epoch_{epoch:05d}.pt').is_file()
            assert (out / f'test_predictions_epoch_{epoch:05d}.npz').is_file()
        report[f'resume_from_epoch{boundary}_matches_continuous_exact_cpu'] = 'PASS'

    invalid = fixture(root / 'reject_completed')
    expect_resume_rejection(invalid, full_out / 'checkpoints/epoch_00100.pt', root / 'reject_completed', 'referentie-epoch1..99')
    pilot = copy.deepcopy(full.rt.core.load_verified(full_out / 'checkpoints/epoch_00005.pt'))
    pilot['contract']['config']['schema'] = 'mx-five-direct-pilot5-protocol-v1'
    pilot['contract']['config']['package'] = 'mx_weights_five_direct_pilot5_v1'
    negative_dir = root / 'negative_pilot_fixture'
    negative_dir.mkdir()
    pilot_path = negative_dir / 'epoch_00005.pt'
    full.rt.core.atomic_save(pilot_path, pilot)
    invalid = fixture(root / 'reject_pilot')
    expect_resume_rejection(invalid, pilot_path, root / 'reject_pilot', 'geen pilotcheckpoint')
    report['completed_run_and_pilot_checkpoint_rejected'] = 'PASS'

    failed_out = root / 'save_failure'
    failed = fixture(failed_out)
    original_save = failed.rt.core.atomic_save
    def fail_epoch100(path, payload):
        if path.name == 'epoch_00100.pt': raise OSError('SIMULATED_CHECKPOINT_WRITE_FAILURE')
        return original_save(path, payload)
    failed.rt.core.atomic_save = fail_epoch100
    try:
        try:
            runner.run_epochs(failed, SimpleNamespace(resume=full_out / 'checkpoints/epoch_00099.pt', stop_after_epoch=100), failed_out, log)
        except OSError as exc:
            assert 'SIMULATED_CHECKPOINT' in str(exc)
        else:
            raise AssertionError('Simulated save failure was swallowed')
    finally:
        failed.rt.core.atomic_save = original_save
    assert failed.report['completed_epochs'] == 99
    with (failed_out / 'metrics.csv').open() as file: rows = list(csv.DictReader(file))
    assert len(rows) == 990 and max(int(r['epoch']) for r in rows) == 99
    assert (failed_out / 'metrics_epoch_00100.csv').is_file()
    assert not (failed_out / 'checkpoints/epoch_00100.pt').exists()
    report['epoch100_write_failure_does_not_publish_completed_endpoint'] = 'PASS'
    report['status'] = 'PASS'
    report['python_version'] = sys.version.split()[0]
    report['torch_version'] = str(torch.__version__)
    report['cuda_tested'] = False
    (root / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    run_checks(Path(sys.argv[1]) if len(sys.argv) > 1 else None,
               Path(sys.argv[2]) if len(sys.argv) > 2 else None)
