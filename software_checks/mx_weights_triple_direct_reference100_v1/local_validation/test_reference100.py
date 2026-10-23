"""Real 100-epoch runner integration with two synthetic CPU images only.

Only this test substitutes the data/runtime context and redirects the authors'
Tensor.cuda calls to CPU. The actual CLI main, joint model, three passes, Adam,
50/75/90 scheduler, checkpoint/restore, exports and feedback generation execute.
The production CLI has no CPU or synthetic bypass. No SVHN data is loaded or
downloaded, and neither the scores nor timing are research results.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch
import joint_adapter as a
import joint_support as t
import resume_core as core
import run_train as r
import svhn_support as s

NOTEBOOK = 'MX_triple_direct_reference100_resultaten.ipynb'
MODES = (4, 6, 8)
FORMATS = {4: 'MXFP4_E2M1', 6: 'MXFP6_E2M3', 8: 'MXFP8_E4M3'}
TEACHERS = {4: 'MXFP6_E2M3', 6: 'MXFP8_E4M3', 8: None}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main(out: Path) -> dict:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    package_hashes = s.verify_package()
    production_config = r.config()
    assert production_config['schema'] == 'mx-weights-triple-direct-reference100-protocol-v1'
    assert production_config['planned_epochs'] == production_config['reference_end_epoch'] == 100
    assert 'pilot_stop_epoch' not in production_config
    assert production_config['seed'] == 20261009
    assert production_config['scheduler_milestones'] == [50, 75, 90]
    assert production_config['scheduler_gamma'] == .1 and production_config['lr'] == .001
    assert production_config['weight_bits'] == list(MODES) and tuple(a.MODES) == MODES
    assert production_config['training_mode_order'] == [8, 6, 4]
    assert production_config['test_mode_order'] == [4, 6, 8]
    assert production_config['passes_per_training_batch'] == 3
    assert production_config['optimizer_steps_per_training_batch'] == 1
    assert production_config['activation_bits'] == 32
    assert production_config['master_dtype'] == production_config['gradient_dtype'] == 'float32'
    assert production_config['adam_moments_dtype'] == 'float32'
    assert production_config['train_examples'] == 73257 and production_config['test_examples'] == 26032
    assert production_config['batch_size'] == 128 and not production_config['extra_used']
    assert production_config['distillation']
    for flag in ('truncation', 'mantissa_truncation', 'exponent_truncation',
                 'shared_scale_compensation', 'learned_shared_scale', 'extra_FP32_teacher_forward'):
        assert production_config[flag] is False, flag

    prepared = out / 'source'
    prepared.mkdir()
    source, source_info = s.legacy.prepare_source(prepared)
    modules = s.import_source(source)
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(2, 3, 40, 40, generator=generator) * 2 - 1
    labels = torch.tensor([0, 1])
    datasets = {phase: torch.utils.data.TensorDataset(images.clone(), labels.clone())
                for phase in ('train', 'test')}
    cases, contexts = {}, {}

    def fixture_context(args, path, report, log, check_only=False):
        assert not check_only, 'This integration fixture tests run/resume only.'
        assert args.stop_after_epoch == 100 and not args.download
        rt = s.runtime(production_config['seed'], 'cpu', strict=False)
        c = r.config()
        # Only fixture sizes change: production LR milestones and endpoint stay.
        c.update(train_examples=2, test_examples=2, batch_size=2,
                 purpose='SYNTHETIC_CPU_TRIPLE_DIRECT_REFERENCE100_INTEGRATION_ONLY')
        model = t.build_model(modules, 'joint', rt.device)
        optimizer = modules.optimizer.get_optimizer_config(model, 'adam', c['lr'], c['weight_decay'])
        scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
        case_name = path.parent.name
        case = cases[case_name]

        def count_update(opt, hook_args, hook_kwargs):
            case['optimizer_updates_in_process'] += 1

        optimizer.register_step_post_hook(count_update)
        loaders = {phase: s.TrackedLoader(torch.utils.data.DataLoader(
            s.IndexedDataset(dataset), batch_size=2, shuffle=phase == 'train',
            num_workers=0, pin_memory=False, drop_last=False))
            for phase, dataset in datasets.items()}
        assert all(len(loader) == 1 for loader in loaders.values())
        contract = {'config': c, 'environment': rt.environment, 'settings': rt.settings,
                    'source_files': source_info['workcopy_files_sha256'],
                    'package_sha256': package_hashes, 'synthetic_fixture_only': True}
        report.update(config=c, environment=rt.environment, execution_settings=rt.settings,
                      source=source_info, package_sha256=package_hashes,
                      contract_sha256=core.json_digest(contract), synthetic_fixture_only=True,
                      dataset_downloaded=False)
        report['limitations'].append(
            'SYNTHETIC CPU FIXTURE: two fake images, one minibatch; no SVHN accuracy or GPU timing.')
        ctx = SimpleNamespace(rt=rt, c=c, source=source, source_info=source_info,
                              modules=modules, model=model, optimizer=optimizer,
                              scheduler=scheduler, loaders=loaders, contract=contract, report=report)
        contexts[case_name] = ctx
        case['fresh_model_sha256_before_optional_restore'] = s.legacy.state_digest(model.state_dict())
        if case.get('stop_marker'):
            (path / 'STOP_AFTER_EPOCH').touch()
        return ctx

    def verify_fixture_unchanged(ctx, log):
        assert all(core.file_sha256(source / name) == digest
                   for name, digest in source_info['workcopy_files_sha256'].items())
        assert s.verify_package() == package_hashes
        assert all(torch.equal(dataset.tensors[0], images) and torch.equal(dataset.tensors[1], labels)
                   for dataset in datasets.values())
        ctx.report.update(source_unchanged=True, package_unchanged=True,
                          synthetic_input_tensors_unchanged=True)

    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor

    def run_case(name, checkpoint=None, stop_marker=False, expected_status='PASS'):
        cases[name] = {'optimizer_updates_in_process': 0, 'stop_marker': stop_marker}
        arguments = ['resume' if checkpoint else 'run', '--output-root', str(out / name)]
        if checkpoint is not None:
            arguments += ['--checkpoint', str(checkpoint)]
        print('SYNTHETIC triple-direct reference100 runner case: ' + name, flush=True)
        with (out / (name + '_stdout.log')).open('x', encoding='utf-8') as stdout:
            with redirect_stdout(stdout), patch.object(sys, 'argv', [str(ROOT / 'run_train.py'), *arguments]), \
                 patch.object(r, 'create_context', fixture_context), \
                 patch.object(r, 'verify_unchanged', verify_fixture_unchanged), \
                 patch.object(torch.Tensor, 'cuda', cpu_cuda):
                exit_code = r.main()
        paths = list((out / name).glob('*_joint_*'))
        assert len(paths) == 1, paths
        path = paths[0]
        report = json.loads((path / 'run_report.json').read_text(encoding='utf-8'))
        assert report['status'] == expected_status, (name, report.get('error'), report.get('traceback'))
        assert exit_code == (1 if expected_status == 'FAIL' else 0), (name, exit_code)
        assert report['synthetic_fixture_only'] and not report['dataset_downloaded']
        assert report['schema'] == 'mx-weights-triple-direct-reference100-report-v1'
        cases[name].update(status=report['status'], exit_code=exit_code, output_directory=str(path))
        return path, report

    continuous_path, continuous_report = run_case('continuous100')
    records = json.loads((continuous_path / 'history.json').read_text(encoding='utf-8'))['epochs']
    assert [record['epoch'] for record in records] == list(range(1, 101))
    assert continuous_report['optimizer_steps'] == 100
    assert continuous_report['training_forward_passes'] == 300
    assert cases['continuous100']['optimizer_updates_in_process'] == 100
    assert continuous_report['full_training_completed'] is True
    assert continuous_report['training_summary']['full_100_epoch_training_completed'] is True
    assert 'pilot_completed' not in continuous_report['training_summary']
    assert continuous_report['training_summary']['expected_optimizer_steps_at_reference_completion'] == 100
    assert continuous_report['training_summary']['expected_training_forward_passes_at_reference_completion'] == 300
    elapsed = sum(record['total_epoch_seconds'] for record in records)
    assert continuous_report['timing_estimate']['measured_100_epoch_seconds'] == elapsed
    assert continuous_report['timing_estimate']['measured_completed_epoch_seconds'] == elapsed
    assert continuous_report['timing_estimate']['full_100_epoch_training_completed'] is True
    complete_limits = ' '.join(continuous_report['timing_estimate']['limitations'])
    assert 'five epochs' not in complete_limits and 'All 100 epochs' in complete_limits
    assert 'LR transitions at 50/75/90' in complete_limits
    assert 'sum of all 100 recorded epoch durations' in continuous_report['timing_estimate']['interpretation']
    assert len(list((continuous_path / 'checkpoints').glob('epoch_*.pt'))) == 100
    assert len(read_csv(continuous_path / 'epoch_metrics.csv')) == 600
    assert len(read_csv(continuous_path / 'weight_updates.csv')) == 100 * 6 * 3
    assert len(list((continuous_path / 'sample_order').glob('*.npy'))) == 200

    # Independent schedule specification includes both sides of every boundary.
    phase_rates = [.001, .001 * .1, .001 * .1 * .1, .001 * .1 * .1 * .1]
    for (first, last), rate in zip([(1, 50), (51, 75), (76, 90), (91, 100)], phase_rates):
        assert all(record['lr_used'] == rate for record in records[first - 1:last])
    for milestone, next_rate in zip([50, 75, 90], phase_rates[1:]):
        assert records[milestone - 1]['lr_next'] == next_rate
        assert records[milestone]['lr_used'] == next_rate
    boundary_lrs = {str(epoch): {'lr_used': records[epoch - 1]['lr_used'],
                                 'lr_next': records[epoch - 1]['lr_next']}
                    for epoch in [1, 50, 51, 75, 76, 90, 91, 99, 100]}

    for record in records:
        epoch = record['epoch']
        assert record['optimizer_steps'] == epoch
        assert record['test_optimizer_steps'] == 0 and record['batchnorm_counters_correct']
        audit = record['gradient_checks']
        assert audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == 1
        assert audit['detached_teacher_checks'] == 2
        assert all(audit[key] == 1 for key in ('forward_calls_E4M3', 'forward_calls_E2M3', 'forward_calls_E2M1'))
        assert record['adam_checks']['all_active_adam_steps_correct']
        assert record['adam_checks']['moments_fp32_finite']
        assert record['adam_checks']['unused_bn_has_no_optimizer_state']
        for phase in ('train', 'test'):
            assert [row['weight_bits'] for row in record[phase]] == list(MODES)
            assert record['coverage'][phase]['n'] == 2 and record['coverage'][phase]['complete_once']
            for row in record[phase]:
                bit = row['weight_bits']
                assert row['n'] == 2 and row['activation_bits'] == 32
                assert row['weight_format'] == FORMATS[bit]
                assert row['bn_bank_key'] == a.BANK[bit]
                assert row['teacher_weight_format'] == (TEACHERS[bit] if phase == 'train' else None)
                assert row['objective_type'] == (
                    'recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy')
                assert abs(row['top1_percent'] - 100 * row['correct1'] / 2) < 1e-4
        assert {row['weight_bits'] for row in record['weight_updates']} == set(MODES)
        assert all(row['master_dtype'] == 'torch.float32' for row in record['weight_updates'])

    scores = [record['test'][-1]['top1_percent'] for record in records]
    expected_best = scores.index(max(scores)) + 1
    selection = continuous_report['checkpoint_selection']
    assert selection['best']['epoch'] == expected_best and selection['last']['epoch'] == 100
    assert selection['selection_mode'] == 'MXFP8_E4M3/A32'
    assert selection['all_modes_from_same_selected_checkpoint'] is True
    selected = read_csv(continuous_path / 'selected_metrics.csv')
    assert len(selected) == 6
    for role, epoch in [('best_E4M3_test', expected_best), ('last_epoch', 100)]:
        rows = [row for row in selected if row['role'] == role]
        assert sorted(int(row['weight_bits']) for row in rows) == list(MODES)
        assert {int(row['epoch']) for row in rows} == {epoch}
        for row in rows:
            metric = next(item for item in records[epoch - 1]['test'] if item['weight_bits'] == int(row['weight_bits']))
            assert float(row['top1_percent']) == metric['top1_percent']
        exports = continuous_report['mx_exports'][role]
        assert set(exports) == {'4', '6', '8'}
        for bit in MODES:
            detail = exports[str(bit)]
            assert detail['epoch'] == epoch and detail['weight_bits'] == bit
            assert detail['decoded_weights_exact'] and detail['six_fp32_masters_absent']
            assert detail['truncation'] is False and detail['full_test_forward_rerun'] is False
            export_path = continuous_path / detail['file']
            assert export_path.is_file() and core.file_sha256(export_path) == detail['sha256']
    assert len(list(continuous_path.glob('mx_direct_*.pt'))) == 6
    assert len(list(continuous_path.glob('mx_export_layers_*.csv'))) == 6
    final_checkpoint = continuous_path / 'checkpoints/epoch_00100.pt'
    final_state = core.load_verified(final_checkpoint)
    assert core.file_sha256(continuous_path / 'last_checkpoint.pt') == core.file_sha256(final_checkpoint)
    assert final_state['scheduler']['last_epoch'] == 100 and final_state['optimizer_steps'] == 100

    with ZipFile(continuous_path / 'feedback_bundle.zip') as archive:
        expected = {NOTEBOOK, 'notebook_support.py', 'training_summary.json', 'timing_estimate.json',
                    'run_report.json', 'checkpoint_selection.json', 'selected_metrics.csv'}
        expected.update(f'mx_export_layers_{bit}_{role}.csv' for bit in MODES for role in ('best', 'last'))
        assert expected.issubset(archive.namelist())
    notebook = json.loads((continuous_path / NOTEBOOK).read_text(encoding='utf-8'))
    parameter_cells = [cell for cell in notebook['cells'] if 'parameters' in cell.get('metadata', {}).get('tags', [])]
    assert len(parameter_cells) == 1 and str(continuous_path.resolve()) in ''.join(parameter_cells[0]['source'])
    assert all(not cell.get('outputs') and cell.get('execution_count') is None
               for cell in notebook['cells'] if cell['cell_type'] == 'code')
    assert core.file_sha256(continuous_path / 'notebook_support.py') == core.file_sha256(ROOT / 'notebook_support.py')

    checkpoint99 = continuous_path / 'checkpoints/epoch_00099.pt'
    checkpoint99_hash = core.file_sha256(checkpoint99)
    resumed_path, resumed_report = run_case('resume99_to100', checkpoint99)
    resumed_state = core.load_verified(resumed_path / 'checkpoints/epoch_00100.pt')
    assert cases['resume99_to100']['optimizer_updates_in_process'] == 1
    assert len(list((resumed_path / 'checkpoints').glob('epoch_*.pt'))) == 1
    assert all(resumed_report['restore_checks'].values())
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        differences = core.differences(final_state[key], resumed_state[key])
        assert not differences, (key, differences[:4])
    for before, after in zip(final_state['trace']['epochs'], resumed_state['trace']['epochs']):
        for key in ('train', 'test', 'model_sha256', 'lr_used', 'lr_next', 'best_epoch_so_far',
                    'weight_updates', 'gradient_checks'):
            assert before[key] == after[key], (before['epoch'], key)
        for phase in ('train', 'test'):
            assert before['coverage'][phase]['order_sha256'] == after['coverage'][phase]['order_sha256']
    assert core.file_sha256(checkpoint99) == checkpoint99_hash
    assert resumed_report['full_training_completed'] is True
    assert resumed_report['training_summary']['resumed'] is True
    assert resumed_report['initial_model_sha256'] == continuous_report['initial_model_sha256']

    stopped_path, stopped_report = run_case('early_stop', stop_marker=True,
                                          expected_status='STOPPED_AT_EPOCH_BOUNDARY')
    assert stopped_report['completed_epoch'] == 1
    assert cases['early_stop']['optimizer_updates_in_process'] == 1
    assert stopped_report['full_training_completed'] is False
    assert stopped_report['training_summary']['full_100_epoch_training_completed'] is False
    assert stopped_report['timing_estimate']['measured_100_epoch_seconds'] is None
    assert stopped_report['timing_estimate']['full_100_epoch_training_completed'] is False
    partial_limits = ' '.join(stopped_report['timing_estimate']['limitations'])
    assert 'Only completed epochs are observed' in partial_limits and 'comparable' in partial_limits
    assert 'not measured or remaining time' in stopped_report['timing_estimate']['interpretation']
    assert (stopped_path / 'checkpoints/epoch_00001.pt').is_file()
    assert len(read_csv(stopped_path / 'selected_metrics.csv')) == 6

    # Valid synthetic epoch5 payload carrying the former pilot contract: reject
    # it before restoring model state or performing updates. No user data used.
    old_pilot = core.load_verified(continuous_path / 'checkpoints/epoch_00005.pt')
    old_pilot['contract'] = copy.deepcopy(old_pilot['contract'])
    old_config = old_pilot['contract']['config']
    old_config['schema'] = 'mx-weights-triple-direct-pilot5-protocol-v1'
    old_config['pilot_stop_epoch'] = 5
    del old_config['reference_end_epoch']
    old_pilot_file = out / 'old_pilot_protocol_fixture.pt'
    core.atomic_save(old_pilot_file, old_pilot)
    old_pilot_hash = core.file_sha256(old_pilot_file)
    rejected_path, rejected_report = run_case('reject_old_pilot', old_pilot_file, expected_status='FAIL')
    assert 'Hervatprotocol/omgeving/bron verschilt' in rejected_report['error']
    assert cases['reject_old_pilot']['optimizer_updates_in_process'] == 0
    assert not list((rejected_path / 'checkpoints').glob('*.pt'))
    assert core.file_sha256(old_pilot_file) == old_pilot_hash
    assert s.legacy.state_digest(contexts['reject_old_pilot'].model.state_dict()) \
        == cases['reject_old_pilot']['fresh_model_sha256_before_optional_restore']

    completed_path, completed_report = run_case('reject_completed100', final_checkpoint, expected_status='FAIL')
    assert 'geen nieuwe referentie-epochs' in completed_report['error']
    assert cases['reject_completed100']['optimizer_updates_in_process'] == 0
    assert not list((completed_path / 'checkpoints').glob('*.pt'))

    # Lower formats peak elsewhere and E4M3 ties: choose its first maximum for all.
    competing_scores = [(10, 20, 30), (40, 50, 80), (99, 60, 80), (70, 100, 70), (95, 90, 79)]
    selection_fixture = [
        {'epoch': epoch, 'test': [{'weight_bits': bit, 'top1_percent': score}
                                 for bit, score in zip(MODES, scores)]}
        for epoch, scores in enumerate(competing_scores, 1)]
    assert t.best_record(selection_fixture)['epoch'] == 2

    cli_results = {}
    rejected_arguments = [['run', '--stop-after-epoch', value] for value in ['-1', '0', '5', '99', '101', '100.0']]
    rejected_arguments += [['run', '--device', 'cpu'], ['run', '--synthetic'],
                           ['run', '--mode', 'control'], ['resume'],
                           ['run', '--checkpoint', str(checkpoint99)],
                           ['run', '--warmstart', str(checkpoint99)]]
    for arguments in rejected_arguments:
        result = subprocess.run([sys.executable, str(ROOT / 'run_train.py'), *arguments],
                                cwd=out, text=True, capture_output=True)
        assert result.returncode == 2, (arguments, result.stdout, result.stderr)
        cli_results[' '.join(arguments)] = {'exit_code': result.returncode,
                                           'reason': result.stderr.splitlines()[-1]}
    assert not (out / 'mx_runs').exists()
    assert r.config() == production_config
    assert s.verify_package() == package_hashes
    assert len({case['fresh_model_sha256_before_optional_restore'] for case in cases.values()}) == 1
    assert core.file_sha256(final_checkpoint) == core.file_sha256(continuous_path / 'last_checkpoint.pt')

    report = {
        'status': 'PASS', 'synthetic_fixture_only': True,
        'scope': 'Real triple-direct reference100 main/model/loop/exports on two synthetic CPU images only',
        'device': 'CPU; test-only Tensor.cuda redirection',
        'no_SVHN_data_loaded_or_downloaded': True,
        'not_a_SVHN_accuracy_or_GPU_timing_result': True,
        'notebook_executed': False,
        'environment': contexts['continuous100'].rt.environment,
        'production_config': production_config,
        'fixture_changes': {'train_examples': 2, 'test_examples': 2, 'batch_size': 2},
        'package_sha256': package_hashes, 'source_sha256': source_info['workcopy_files_sha256'],
        'test_duration_seconds': time.perf_counter() - started,
        'total_synthetic_optimizer_updates': sum(case['optimizer_updates_in_process'] for case in cases.values()),
        'cases': cases, 'actual_LR_boundaries': boundary_lrs,
        'tests': {
            'real_main_default100_and_PASS_only_at100': True,
            '100_synthetic_epochs_100_updates_300_forwards_600_metrics': True,
            'unchanged_50_75_90_LR_schedule_all_four_phases': True,
            'three_forwards_two_teacher_checks_one_update_per_batch': True,
            'A32_FP32_master_Adam_and_three_BN_audits': True,
            'train_recursive_teacher_labels_test_all_hard_CE': True,
            'highest_E4M3_first_strict_maximum_not_lower_mode_peak': True,
            'six_selected_metric_rows_from_common_best_or_epoch100_checkpoint': True,
            'six_direct_exports_with_verified_decoded_weights': True,
            'feedback_and_unexecuted_notebook_copied': True,
            'resume99_to100_model_optimizer_scheduler_rng_exact_CPU': True,
            'measured100_time_only_after100': True,
            'early_stop_complete_epoch_without_false_PASS_or100_result': True,
            'old_pilot_contract_rejected_before_restore_and_updates': True,
            'completed100_resume_rejected_before_updates': True,
            'fresh_seed_state_identical_before_every_optional_restore': True,
            'production_config_package_source_and_fixture_data_unchanged': True,
            'production_CLI_refusals': cli_results,
            'rejected_CLI_created_no_production_output': True,
        },
    }
    report_path = out / 'reference100_validation_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(report_path),
                      'test_duration_seconds': report['test_duration_seconds'],
                      'total_synthetic_optimizer_updates': report['total_synthetic_optimizer_updates'],
                      'not_a_SVHN_accuracy_or_GPU_timing_result': True}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
