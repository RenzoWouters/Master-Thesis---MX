"""Generate real tiny CPU trajectories for continuation55->100 validation.

The unchanged dynamic runner executes all100 epochs, then the unchanged
five-epoch freeze runner produces the actual parent55 training checkpoint.
All images, accuracies and timings are synthetic software fixtures only.
"""
from __future__ import annotations
from contextlib import ExitStack, redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pilot_root = package / 'pilot_runtime'
    sys.path.insert(0, str(pilot_root))
    pilot = load_module('freeze100_fixture_pilot', pilot_root / 'run_train.py')
    helper = load_module('freeze100_fixture_original_context', pilot_root / 'local_validation/build_parent_fixture.py')
    testhelper = load_module('freeze100_fixture_pilot_helpers', pilot_root / 'local_validation/test_training_integration.py')
    s, t, a, core, torch, original = pilot.s, pilot.t, pilot.a, pilot.core, pilot.torch, pilot.ref
    torch.set_num_threads(2)
    dynamic = output / 'dynamic100'
    dynamic.mkdir()
    source, source_info = s.legacy.prepare_source(dynamic)
    modules = s.import_source(source)
    ctx = helper.make_context(dynamic, source, source_info, modules, s.verify_package())
    log = s.legacy.Log(dynamic / 'run.log')
    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor
    with (dynamic / 'stdout.log').open('x', encoding='utf-8') as stdout:
        with redirect_stdout(stdout), patch.object(torch.Tensor, 'cuda', cpu_cuda):
            history = original.run_epochs(ctx, dynamic, log, 100, install_signal_handler=False)
    log.handle.close()
    assert len(history['epochs']) == 100
    assert history['epochs'][-1]['optimizer_steps'] == 200
    write_json(dynamic / 'history.json', history)
    write_json(dynamic / 'parent_contract.json', ctx.contract)
    best = t.best_record(history['epochs'])
    report = ctx.report
    report.update(status='PASS', schema='mx-triple-learned-scale32-64-reference100-report-v1',
        completed_epoch=100, synthetic_fixture_only=True, full_training_completed=True,
        output_directory=str(dynamic), command='synthetic_fixture',
        selected_best_epoch=best['epoch'], selected_last_epoch=100,
        optimizer_steps=200, training_forward_passes=600,
        source_unchanged=True, dataset_files_unchanged=True)
    write_json(dynamic / 'run_report.json', report)
    s.write_csv(dynamic / 'epoch_metrics.csv', [row for rec in history['epochs'] for phase in ('train', 'test') for row in rec[phase]])
    s.write_csv(dynamic / 'scale_epoch_metrics.csv', [row for rec in history['epochs'] for row in rec['scale_epoch_metrics']])
    s.write_csv(dynamic / 'epoch_times.csv', [dict(epoch=rec['epoch'], **{key:rec[key] for key in
        ('train_seconds', 'test_seconds', 'checkpoint_seconds', 'other_seconds', 'total_epoch_seconds',
         'first_epoch_in_process', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes')}) for rec in history['epochs']])
    write_json(dynamic / 'training_summary.json', dict(schema='mx-triple-learned-scale32-64-reference100-completion-v1',
        completed_epochs=100, full_training_completed=True, full_100_epoch_training_completed=True,
        best_epoch_E4M3_A32=best['epoch'], last_epoch=100, synthetic_fixture_only=True,
        train_examples=3, test_examples=3, optimizer_steps=200))
    detail = lambda rec: dict(epoch=rec['epoch'], file=f"checkpoints/epoch_{rec['epoch']:05d}.pt",
                             sha256=core.file_sha256(Path(rec['checkpoint_path'])),
                             selection_score_E4M3=rec['test'][-1]['top1_percent'])
    write_json(dynamic / 'checkpoint_selection.json', dict(best=detail(best), last=detail(history['epochs'][-1]),
        selection_mode='MXFP8_E4M3/A32', all_modes_from_same_selected_checkpoint=True,
        selected_on_official_test=True, independent_test_estimate=False))
    s.write_csv(dynamic / 'selected_metrics.csv', [dict(role=role, **row) for role, rec in
        [('best_E4M3_test', best), ('last_epoch', history['epochs'][-1])] for row in rec['test']])
    write_json(dynamic / 'fixture.json', dict(source=str(source), source_info=source_info,
        synthetic_fixture_only=True, epochs_executed=100, optimizer_updates=200))

    evidence = testhelper.prepare_evidence(pilot, dynamic, output / 'parent50_reference')
    frozen = output / 'frozen_pilot55'
    frozen.mkdir()
    freeze_ctx = helper.make_context(frozen, source, source_info, modules, s.verify_package())
    # Keep the actual unchanged pilot package identity in the pilot contract.
    freeze_ctx.branch_package_hashes = pilot.verify_package()
    freeze_ctx.report.update(schema='mx-triple-freeze-offsets-e50-pilot5-report-v1',
                            status='RUNNING', synthetic_fixture_only=True)
    log = s.legacy.Log(frozen / 'run.log')
    with (frozen / 'stdout.log').open('x', encoding='utf-8') as stdout:
        with redirect_stdout(stdout), patch.object(torch.Tensor, 'cuda', cpu_cuda):
            frozen_history = pilot.run_epochs(freeze_ctx, frozen, log, 55,
                parent_checkpoint=dynamic / 'checkpoints/epoch_00050.pt', evidence=evidence,
                install_signal_handler=False)
            pilot.export_results(freeze_ctx, frozen, frozen_history, log, evidence)
    log.handle.close()
    freeze_ctx.report.update(status='PASS', source_unchanged=True, dataset_files_unchanged=True,
        parent_checkpoint_unchanged=True, output_directory=str(frozen), command='run',
        package_sha256=freeze_ctx.branch_package_hashes)
    write_json(frozen / 'run_report.json', freeze_ctx.report)
    pilot.feedback_bundle(frozen)
    parent55 = core.load_verified(frozen / 'checkpoints/epoch_00055.pt')
    assert parent55['optimizer_steps'] == 110
    assert parent55['contract'] == freeze_ctx.contract
    result = dict(status='PASS', synthetic_fixture_only=True, CUDA_tested=False,
        no_SVHN_data_loaded_or_downloaded=True, train_examples=3, test_examples=3, batch_size=2,
        dynamic_epochs_executed=100, frozen_pilot_epochs_executed=5,
        dynamic100_directory=str(dynamic), frozen_pilot55_directory=str(frozen),
        parent50_reference_directory=str(evidence['directory']), source=str(source), source_info=source_info,
        parent50_checkpoint=str(dynamic / 'checkpoints/epoch_00050.pt'),
        parent55_checkpoint=str(frozen / 'checkpoints/epoch_00055.pt'),
        parent55_checkpoint_sha256=core.file_sha256(frozen / 'checkpoints/epoch_00055.pt'),
        parent55_model_sha256=s.legacy.state_digest(parent55['model']),
        parent55_contract_sha256=core.json_digest(parent55['contract']),
        parent55_trace_sha256=core.json_digest(parent55['trace']),
        duration_seconds=time.perf_counter() - started)
    write_json(output / 'fixture.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: build_fixtures.py PACKAGE OUTPUT_DIRECTORY')
    main(*sys.argv[1:])
