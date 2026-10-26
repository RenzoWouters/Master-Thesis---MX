"""Real CPU integration tests for frozen continuation55->100.

All examples, timings and accuracies are explicitly synthetic. The production
training/restore/export functions are exercised without any SVHN data.
"""
from __future__ import annotations
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
from zipfile import ZipFile

sys.dont_write_bytecode = True


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(package, output, fixture_root, reference_root):
    package, output, fixture_root, reference_root = [Path(p).resolve() for p in
        (package, output, fixture_root, reference_root)]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sys.path.insert(0, str(package))
    r = load_module('freeze100_runner_integration', package / 'run_train.py')
    import continuation_core as cc
    import torch
    torch.set_num_threads(2)
    binding_names = ['run_train.py', 'continuation_core.py', 'continuation_bridge.py']
    code_binding = {name:r.core.file_sha256(package / name) for name in binding_names}
    helper = load_module('freeze100_test_original_context', package / 'pilot_runtime/local_validation/build_parent_fixture.py')
    old_tests = load_module('freeze100_export_independent_oracle', package / 'pilot_runtime/local_validation/test_training_integration.py')
    info = read_json(fixture_root / 'fixture.json')
    paths = read_json(reference_root / 'reference_paths.json')
    source = Path(info['source'])
    modules = r.s.import_source(source)
    evidence = cc.load_evidence(Path(paths['dynamic_run']), Path(paths['pilot_dir']),
        Path(paths['dynamic_dir']), paths['expected_pilot_sha'], paths['expected_dynamic_sha'])
    parent55 = Path(paths['parent_checkpoint55'])
    parent_state = r.core.load_verified(parent55)
    parent_sha = r.core.file_sha256(parent55)
    dynamic_history_sha = r.core.file_sha256(evidence['dynamic_run'] / 'history.json')
    immutable_pilot = {p.relative_to(package / 'pilot_runtime').as_posix():r.core.file_sha256(p)
                      for p in (package / 'pilot_runtime').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    package_id = {'synthetic_fixture_only_test_sha256':r.core.file_sha256(Path(__file__))}
    cases, main_cases, rejections = {}, {}, {}

    def fresh_context(report=None):
        ctx = helper.make_context(output, source, info['source_info'], modules, r.s.verify_package())
        assert not r.core.differences(ctx.contract, evidence['dynamic_contract'])
        ctx.branch_package_hashes = package_id
        ctx.report.update(schema='mx-triple-freeze-offsets-e50-reference100-report-v1', status='RUNNING',
                          limitations=['SYNTHETIC CPU FIXTURE ONLY'])
        if report is not None:
            report.update(ctx.report)
            ctx.report = report
        return ctx

    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor

    def no_lower_encoder(*args, **kwargs):
        raise AssertionError('Frozen training attempted direct lower quantization.')

    def run_case(name, stop=100, resume=None, marker=False):
        path = output / name
        path.mkdir()
        ctx = fresh_context()
        log = r.s.legacy.Log(path / 'run.log')
        if marker:
            (path / 'STOP_AFTER_EPOCH').touch()
        with (output / (name + '_stdout.log')).open('x', encoding='utf-8') as stdout:
            with redirect_stdout(stdout), ExitStack() as stack:
                stack.enter_context(patch.object(torch.Tensor, 'cuda', cpu_cuda))
                for bit in (4, 6):
                    stack.enter_context(patch.object(r.a.codec(bit), 'encode_tensor', no_lower_encoder))
                    stack.enter_context(patch.object(r.a.codec(bit), 'encode_elements', no_lower_encoder))
                trace = r.run_epochs(ctx, path, log, stop_after=stop, parent_checkpoint55=parent55,
                    evidence=evidence, resume=resume, install_signal_handler=False)
                r.export_results(ctx, path, trace, log, evidence)
        log.handle.close()
        ctx.report.update(status='PASS' if ctx.report['continuation_completed'] else 'STOPPED_AT_EPOCH_BOUNDARY',
            source_unchanged=True, dataset_files_unchanged=True, parent_checkpoint_unchanged=True,
            dynamic_history_unchanged=True)
        write_json(path / 'run_report.json', ctx.report)
        r.feedback_bundle(path)
        cases[name] = dict(status=ctx.report['status'], completed_epoch=trace['epochs'][-1]['epoch'],
            inherited_frozen_epochs=5, new_optimizer_updates_in_process=ctx.guard.steps,
            output_directory=str(path))
        return path, ctx, trace

    # Directly inspect restoration before the first new batch.
    restore_ctx = fresh_context()
    before_trace, done, steps = cc.initialize_continuation(restore_ctx, parent55, evidence, package_id)
    assert done == 55 and steps == 110
    for key, live in [('model', restore_ctx.model.state_dict()), ('optimizer', restore_ctx.optimizer.state_dict()),
                      ('scheduler', restore_ctx.scheduler.state_dict()),
                      ('rng', r.core.capture_rng({}, restore_ctx.rt.device))]:
        assert not r.core.differences(live, parent_state[key]), key
    assert not r.core.differences(before_trace['epochs'], evidence['pilot_history']['epochs'])
    restore_ctx.guard.close()

    path, ctx, trace = run_case('frozen56_to100')
    final = r.core.load_verified(path / 'checkpoints/epoch_00100.pt')
    assert ctx.report['continuation_completed'] is True
    assert [rec['epoch'] for rec in trace['epochs']] == list(range(51, 101))
    assert trace['epochs'][:5] == evidence['pilot_history']['epochs']
    assert ctx.guard.steps == 90 and final['optimizer_steps'] == 200
    assert final['scheduler']['last_epoch'] == 100
    assert final['scheduler']['_last_lr'][1] == 0
    lr_boundaries = {}
    for rec in trace['epochs']:
        ep = rec['epoch']
        expected = .001
        expected_next = .001
        for milestone in (50, 75, 90):
            if milestone < ep:
                expected *= .1
            if milestone <= ep:
                expected_next *= .1
        assert rec['lr_used'] == expected and rec['lr_next'] == expected_next, ep
        assert rec['scale_lr_used'] == rec['scale_lr_next'] == 0
        assert rec['optimizer_steps'] == ep * 2
        assert rec['scale_state'] == evidence['baseline']['scale_state']
        assert rec['freeze_checks']['all_theta_unchanged'] and rec['freeze_checks']['all_hard_offsets_unchanged']
        assert rec['freeze_checks']['checked_optimizer_steps'] == 2
        assert rec['batchnorm_counters_correct'] and rec['test_optimizer_steps'] == 0
        assert rec['gradient_checks']['optimizer_steps_in_phase'] == 2
        assert rec['gradient_checks']['detached_teacher_checks'] == 4
        assert rec['gradient_checks']['finite_gradient_checks'] == 2
        for bit in ('E4M3', 'E2M3', 'E2M1'):
            assert rec['gradient_checks']['forward_calls_' + bit] == 2
        for phase in ('train', 'test'):
            assert rec['coverage'][phase]['order_sha256'] == evidence['comparison'][ep]['coverage'][phase]['order_sha256']
            assert rec['coverage_historical_match'][phase]
            assert all(row['n'] == 3 and row['activation_bits'] == 32 for row in rec[phase])
        for row in rec['scale_epoch_metrics']:
            assert row['scale_lr'] == 0
            assert all(row[key] == 0 for key in ('theta_changed_blocks', 'theta_update_l2',
                'net_choice_changes', 'choice_flips_during_epoch', 'unique_blocks_switched_during_epoch',
                'projected_values_during_epoch'))
        if ep in (55, 56, 74, 75, 76, 89, 90, 91, 100):
            lr_boundaries[str(ep)] = dict(used=rec['lr_used'], next=rec['lr_next'], theta=rec['scale_lr_used'])
    assert any(not torch.equal(value, final['model'][name]) for name, value in parent_state['model'].items()
               if name.endswith('.weight'))
    for name in r.a.theta_state_keys():
        assert torch.equal(parent_state['model'][name], final['model'][name])
    selected = cc.select_records(trace, evidence)
    assert selected['last_epoch']['epoch'] == 100
    assert selected['best_E4M3_global']['epoch'] == 16
    assert selected['best_E4M3_postfreeze']['epoch'] == 51
    for role, rec in selected.items():
        selected_path, selected_state = cc.checkpoint_for_record(ctx, rec, evidence)
        expected_theta = {name:selected_state['model'][name + '.quantize_fn.theta'].tolist() for name in r.a.TARGET_LAYERS}
        old_tests.verify_exports(r, torch, path, ctx.report['mx_exports'][role], rec['epoch'], modules, expected_theta)
    assert r.core.differences(
        {name:r.core.load_verified(evidence['dynamic_run'] / 'checkpoints/epoch_00016.pt')['model'][name + '.quantize_fn.theta'] for name in r.a.TARGET_LAYERS},
        {name:parent_state['model'][name + '.quantize_fn.theta'] for name in r.a.TARGET_LAYERS})

    partial, _, partial_trace = run_case('stop74', stop=74)
    assert len(partial_trace['epochs']) == 24
    resumed, resumed_ctx, resumed_trace = run_case('resume74_to100', resume=partial / 'checkpoints/epoch_00074.pt')
    resumed_final = r.core.load_verified(resumed / 'checkpoints/epoch_00100.pt')
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        differences = r.core.differences(final[key], resumed_final[key])
        assert not differences, (key, differences[:5])
    for before, after in zip(trace['epochs'], resumed_trace['epochs']):
        for key in ('train', 'test', 'scale_state', 'scale_epoch_metrics', 'weight_updates', 'gradient_checks',
                    'model_sha256', 'lr_used', 'lr_next', 'scale_lr_used', 'scale_lr_next', 'freeze_checks',
                    'test_prediction_counts', 'test_label_counts'):
            assert before[key] == after[key], (before['epoch'], key)
    resumed89, _, _ = run_case('resume89_to100', resume=path / 'checkpoints/epoch_00089.pt')
    final89 = r.core.load_verified(resumed89 / 'checkpoints/epoch_00100.pt')
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        assert not r.core.differences(final[key], final89[key]), key

    # Pure selection controls: historical post-freeze epochs must never be
    # substituted for the newly frozen trajectory in its global selection.
    altered = copy.deepcopy(trace)
    altered_evidence = copy.deepcopy(evidence)
    for rec in altered_evidence['dynamic_history']['epochs']:
        rec['test'][-1]['top1_percent'] = 70.
    altered_evidence['dynamic_history']['epochs'][79]['test'][-1]['top1_percent'] = 99.
    for rec in altered['epochs']:
        rec['test'][-1]['top1_percent'] = 80.
    altered['epochs'][19]['test'][-1]['top1_percent'] = 95.  # epoch70
    choices = cc.select_records(altered, altered_evidence)
    assert choices['best_E4M3_global']['epoch'] == choices['best_E4M3_postfreeze']['epoch'] == 70
    assert choices['last_epoch']['epoch'] == 100

    def rejects(name, state):
        fresh = fresh_context()
        initial_hash = r.s.legacy.state_digest(fresh.model.state_dict())
        try:
            cc.verify_parent55_state(fresh, state, evidence)
        except (RuntimeError, AssertionError, ValueError) as exc:
            rejections[name] = dict(rejected=True, message=str(exc))
        else:
            raise AssertionError('Invalid parent accepted: ' + name)
        assert r.s.legacy.state_digest(fresh.model.state_dict()) == initial_hash
    bad = copy.deepcopy(parent_state); bad['completed_epoch'] = 54
    rejects('wrong_parent_epoch', bad)
    bad = copy.deepcopy(parent_state); bad['contract']['config']['seed'] += 1
    rejects('wrong_parent_contract', bad)
    bad = copy.deepcopy(parent_state)
    name = next(n for n, v in bad['model'].items() if n.endswith('.weight') and v.dtype == torch.float32)
    bad['model'][name].reshape(-1)[0] += 1
    rejects('wrong_parent_model', bad)
    bad = copy.deepcopy(parent_state); bad['trace']['epochs'][0]['lr_used'] = .2
    rejects('wrong_parent_trace', bad)
    bad = copy.deepcopy(parent_state); bad['optimizer_steps'] -= 1
    rejects('wrong_parent_optimizer_steps', bad)
    bad = copy.deepcopy(parent_state); bad['scheduler']['last_epoch'] = 54
    rejects('wrong_parent_scheduler_epoch', bad)
    bad = copy.deepcopy(parent_state); bad['optimizer']['param_groups'][1]['lr'] = .0001
    rejects('nonzero_parent_theta_lr', bad)
    bad_checksum = output / 'bad_checksum.pt'
    shutil.copyfile(parent55, bad_checksum)
    bad_checksum.with_suffix('.pt.sha256').write_text('0' * 64 + '\n')
    try:
        r.core.load_verified(bad_checksum)
    except RuntimeError as exc:
        rejections['wrong_file_checksum'] = dict(rejected=True, message=str(exc))
    else:
        raise AssertionError('Wrong file checksum accepted')
    corrupt = output / 'corrupt_dynamic'
    shutil.copytree(evidence['dynamic_dir'], corrupt)
    with (corrupt / 'epoch_metrics.csv').open('a') as handle:
        handle.write('\n')
    try:
        cc.load_evidence(evidence['dynamic_run'], evidence['pilot_dir'], corrupt,
            evidence['pilot_reference_sha256'], evidence['dynamic_reference_sha256'])
    except RuntimeError as exc:
        rejections['changed_historical_reference'] = dict(rejected=True, message=str(exc))
    else:
        raise AssertionError('Changed historical reference accepted')
    wrong_history = output / 'wrong_history'
    wrong_history.mkdir()
    shutil.copyfile(evidence['dynamic_run'] / 'history.json', wrong_history / 'history.json')
    with (wrong_history / 'history.json').open('a') as handle:
        handle.write('\n')
    try:
        cc.load_evidence(wrong_history, evidence['pilot_dir'], evidence['dynamic_dir'],
            evidence['pilot_reference_sha256'], evidence['dynamic_reference_sha256'])
    except RuntimeError as exc:
        rejections['changed_external_history'] = dict(rejected=True, message=str(exc))
    else:
        raise AssertionError('Changed original full100 history accepted')

    # A strict resume from this same program may not accept an edited contract.
    tampered_resume = output / 'wrong_resume.pt'
    payload = r.core.load_verified(partial / 'checkpoints/epoch_00074.pt')
    payload['contract']['config']['scale_lr'] = .0001
    r.core.atomic_save(tampered_resume, payload)
    fresh = fresh_context()
    try:
        cc.initialize_continuation(fresh, parent55, evidence, package_id, tampered_resume)
    except RuntimeError as exc:
        rejections['resume_changed_contract'] = dict(rejected=True, message=str(exc))
    else:
        raise AssertionError('Edited continuation contract accepted')

    path_rejections = {}
    for label, destination in [('package', package / 'new_result'),
        ('parent_pilot', parent55.parent.parent / 'new_result'),
        ('dynamic_reference', evidence['dynamic_run'] / 'new_result'),
        ('resume_run', partial / 'new_result'),
        ('dataset', output / 'dataset/new_result')]:
        try:
            cc.validate_output_location(destination, parent55, evidence['dynamic_run'],
                partial / 'checkpoints/epoch_00074.pt', output / 'dataset')
        except RuntimeError as exc:
            path_rejections[label] = dict(rejected=True, message=str(exc))
        else:
            raise AssertionError('Protected output location accepted: ' + label)
    cc.validate_output_location(output / 'safe_sibling', parent55, evidence['dynamic_run'],
        partial / 'last_checkpoint.pt', output / 'dataset')

    # The actual main() default100 path is checked with its supported soft-stop
    # marker, avoiding a duplicate full45-epoch run solely for CLI plumbing.
    def main_context(args, destination, report, log):
        fresh = fresh_context(report)
        (destination / 'STOP_AFTER_EPOCH').touch()
        return fresh
    def verify_inputs(fresh, log):
        assert r.core.file_sha256(parent55) == parent_sha
        assert r.core.file_sha256(evidence['dynamic_run'] / 'history.json') == dynamic_history_sha
        fresh.report.update(source_unchanged=True, dataset_files_unchanged=True,
            parent_checkpoint_unchanged=True, dynamic_history_unchanged=True)
    destination = output / 'main_default100_softstop'
    argv = [str(package / 'run_train.py'), 'run', '--parent-checkpoint', str(parent55),
            '--dynamic-run', str(evidence['dynamic_run']), '--output-root', str(destination)]
    with (output / 'main_default100_stdout.log').open('x', encoding='utf-8') as stdout:
        with redirect_stdout(stdout), ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(r, 'create_context', main_context))
            stack.enter_context(patch.object(r, 'verify_package', lambda:package_id))
            stack.enter_context(patch.object(r, 'load_evidence', lambda *args, **kwargs:evidence))
            stack.enter_context(patch.object(r, 'verify_unchanged', verify_inputs))
            stack.enter_context(patch.object(torch.Tensor, 'cuda', cpu_cuda))
            return_code = r.main()
    directories = [p for p in destination.iterdir() if p.is_dir()]
    assert len(directories) == 1
    main_report = read_json(directories[0] / 'run_report.json')
    assert return_code == 0 and main_report['status'] == 'STOPPED_AT_EPOCH_BOUNDARY', main_report.get('traceback')
    assert main_report['completed_epoch'] == 56 and not main_report['continuation_completed']
    assert main_report['config']['end_epoch'] == 100
    with ZipFile(directories[0] / 'feedback_bundle.zip') as archive:
        names = archive.namelist()
        assert {'run_report.json', 'history.json', 'freeze_summary.json', 'lineage.json',
                'pilot_reference/REFERENCE.json', 'dynamic_reference/history.json',
                r.NOTEBOOK, 'notebook_support.py'}.issubset(names)
        assert not any(n.endswith('.pt') for n in names)
    main_cases['default100_softstop56'] = dict(status=main_report['status'], exit_code=return_code,
        completed_epoch=56, planned_end_epoch=100, feedback_bundle_verified=True)

    invalid_cli = {}
    bad_args = [['run', '--stop-after-epoch', value] for value in ('0', '55', '101', '100.0')]
    bad_args += [['run', '--device', 'cpu'], ['run', '--synthetic'], ['resume'],
                ['run', '--checkpoint', str(parent55)], ['run', '--download']]
    for args in bad_args:
        result = subprocess.run([sys.executable, str(package / 'run_train.py'), *args],
            cwd=output, text=True, capture_output=True)
        assert result.returncode == 2, (args, result.stdout, result.stderr)
        invalid_cli[' '.join(args)] = dict(exit_code=2, message=result.stderr.splitlines()[-1])
    assert not (output / 'mx_runs').exists()
    assert r.core.file_sha256(parent55) == parent_sha
    assert r.core.file_sha256(evidence['dynamic_run'] / 'history.json') == dynamic_history_sha
    for name, digest in immutable_pilot.items():
        assert r.core.file_sha256(package / 'pilot_runtime' / name) == digest, name
    for name, digest in code_binding.items():
        assert r.core.file_sha256(package / name) == digest, 'Production code changed during validation: ' + name
    result = dict(status='PASS', synthetic_fixture_only=True, CUDA_tested=False,
        no_SVHN_data_loaded_or_downloaded=True, not_a_research_accuracy_or_GPU_timing_result=True,
        fixture=dict(dynamic_epochs=100, frozen_pilot_epochs=5, train_examples=3, test_examples=3, batch_size=2),
        code_binding=code_binding, test_file_sha256=r.core.file_sha256(Path(__file__)),
        parent55_checkpoint_sha256=parent_sha, duration_seconds=time.perf_counter() - started,
        cases=cases, main_cases=main_cases, lr_boundaries=lr_boundaries,
        selected_epochs={role:rec['epoch'] for role, rec in selected.items()},
        checks=dict(strict_parent55_model_optimizer_scheduler_rng_restored_without_intervention=True,
            only_epochs56_100_new_and_exact_pilot_prefix_preserved=True,
            all4080_theta_values_and_choices_fixed_after_every_update=True,
            three_precisions_A32_FP32_original_recursive_losses_preserved=True,
            original_absolute_Adam_BN_scheduler_positions_preserved=True,
            LR_transitions_after75_and90_correct=True, same_historical_data_orders_all50_frozen_epochs=True,
            CPU_resume74_exact_model_optimizer_scheduler_rng=True,
            CPU_resume89_exact_model_optimizer_scheduler_rng=True,
            global_pre16_and_inherited_post51_and_new_last100_exports_valid=True,
            historical_postfreeze_metrics_excluded_from_new_global_selection=True,
            independent_packed_choices_and_source_only_export_reconstruction=True,
            negative_inputs=rejections, protected_output_directories=path_rejections,
            default_main100_and_controlled_softstop56=True, invalid_cli=invalid_cli,
            parent_pilot_dynamic_history_and_inherited_runtime_unchanged=True))
    write_json(output / 'training_integration_report.json', result)
    print(json.dumps(dict(status='PASS', report=str(output / 'training_integration_report.json'),
                         seconds=result['duration_seconds']), indent=2))
    return result


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit('Usage: test_training_integration.py PACKAGE OUTPUT FIXTURES REFERENCES')
    main(*sys.argv[1:])
