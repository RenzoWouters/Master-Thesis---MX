"""Focused CPU integration of the real reference100 runner on two fake images.

This is NOT an SVHN experiment or a production timing measurement. Only this
test supplies a synthetic context and redirects Tensor.cuda to CPU. The actual
runner entry point, 100-epoch loop, author forward, E4M3 model, Adam, unchanged
50/75/90 scheduler, checkpointing, export and feedback generation all run.
Production CLI has no CPU or synthetic-data option. No dataset is downloaded.
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
import resume_core as core
import run_train as r
import single_support as t
import svhn_support as s

NOTEBOOK = 'MX_E4M3_dedicated_reference100_resultaten.ipynb'


def main(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    package_hashes = s.verify_package()
    production_config = r.config()
    assert production_config['schema'] == 'mx-e4m3-dedicated-reference100-protocol-v1'
    assert production_config['planned_epochs'] == production_config['reference_end_epoch'] == 100
    assert 'pilot_stop_epoch' not in production_config
    assert production_config['seed'] == 20261009
    assert production_config['scheduler_milestones'] == [50, 75, 90]
    assert production_config['scheduler_gamma'] == .1 and production_config['lr'] == .001
    assert production_config['weight_format'] == 'MXFP8_E4M3'
    assert production_config['codec_revision'] == 'e4m3-exact-binary-v2'
    assert production_config['activation_bits'] == 32
    assert production_config['master_dtype'] == production_config['gradient_dtype'] == 'float32'
    assert production_config['adam_moments_dtype'] == 'float32'
    assert production_config['train_examples'] == 73257 and production_config['test_examples'] == 26032
    assert production_config['batch_size'] == 128 and not production_config['extra_used']
    assert not production_config['truncation'] and not production_config['distillation']

    prepared = out / 'source'
    prepared.mkdir()
    source, source_info = s.legacy.prepare_source(prepared)
    modules = s.import_source(source)
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(2, 3, 40, 40, generator=generator) * 2 - 1
    labels = torch.tensor([0, 1])
    datasets = {phase: torch.utils.data.TensorDataset(images.clone(), labels.clone())
                for phase in ('train', 'test')}
    cases = {}
    contexts = {}

    def fixture_context(args, path, report, log, check_only=False):
        assert not check_only, 'This fixture tests run/resume only.'
        assert args.stop_after_epoch == 100 and not args.download
        rt = s.runtime(production_config['seed'], 'cpu', strict=False)
        c = r.config()
        # Test-only dataset length; LR milestones and endpoint remain unchanged.
        c.update(train_examples=2, test_examples=2, batch_size=2,
                 purpose='SYNTHETIC_CPU_REFERENCE100_INTEGRATION_ONLY')
        model = t.build_model(modules, 'mx', rt.device)
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
        report['limitations'].append('SYNTHETIC CPU FIXTURE: two fake images; no SVHN accuracy or GPU timing.')
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
        assert all(torch.equal(ds.tensors[0], images) and torch.equal(ds.tensors[1], labels)
                   for ds in datasets.values())
        ctx.report.update(source_unchanged=True, package_unchanged=True,
                          synthetic_input_tensors_unchanged=True)

    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor

    def run_case(name, checkpoint=None, stop_marker=False, expected_status='PASS'):
        cases[name] = {'optimizer_updates_in_process': 0, 'stop_marker': stop_marker}
        arguments = ['resume' if checkpoint else 'run', '--output-root', str(out / name)]
        if checkpoint is not None:
            arguments += ['--checkpoint', str(checkpoint), '--stop-after-epoch', '100']
        print('reference100 fixture: ' + name, flush=True)
        # create_context and final dataset verification are test-only substitutes;
        # main/run_epochs/export_results/feedback_bundle execute without replacement.
        with (out / (name + '_stdout.log')).open('x', encoding='utf-8') as stdout:
            with redirect_stdout(stdout), patch.object(sys, 'argv', [str(ROOT / 'run_train.py'), *arguments]), \
                 patch.object(r, 'create_context', fixture_context), \
                 patch.object(r, 'verify_unchanged', verify_fixture_unchanged), \
                 patch.object(torch.Tensor, 'cuda', cpu_cuda):
                exit_code = r.main()
        paths = list((out / name).glob('*_mx_*'))
        assert len(paths) == 1, paths
        path = paths[0]
        report = json.loads((path / 'run_report.json').read_text(encoding='utf-8'))
        assert report['status'] == expected_status, (name, report.get('error'), report.get('traceback'))
        assert exit_code == (1 if expected_status == 'FAIL' else 0), (name, exit_code)
        assert report['synthetic_fixture_only'] and not report['dataset_downloaded']
        assert report['schema'] == 'mx-e4m3-dedicated-reference100-report-v1'
        cases[name].update(status=report['status'], exit_code=exit_code, output_directory=str(path))
        return path, report

    continuous_path, continuous_report = run_case('continuous100')
    trace = json.loads((continuous_path / 'history.json').read_text(encoding='utf-8'))
    records = trace['epochs']
    assert [row['epoch'] for row in records] == list(range(1, 101))
    assert continuous_report['optimizer_steps'] == 100
    assert cases['continuous100']['optimizer_updates_in_process'] == 100
    assert len(list((continuous_path / 'checkpoints').glob('epoch_*.pt'))) == 100
    assert continuous_report['full_training_completed'] is True
    assert continuous_report['training_summary']['full_100_epoch_training_completed'] is True
    assert 'pilot_completed' not in continuous_report['training_summary']
    elapsed = sum(row['total_epoch_seconds'] for row in records)
    assert continuous_report['timing_estimate']['measured_100_epoch_seconds'] == elapsed
    assert continuous_report['timing_estimate']['measured_completed_epoch_seconds'] == elapsed
    assert continuous_report['timing_estimate']['full_100_epoch_training_completed'] is True
    with (continuous_path / 'epoch_metrics.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 200

    # Independent phase specification, including both sides of every transition.
    phase_rates = [.001, .001 * .1, .001 * .1 * .1, .001 * .1 * .1 * .1]
    phase_ranges = [(1, 50), (51, 75), (76, 90), (91, 100)]
    for (first, last), rate in zip(phase_ranges, phase_rates):
        assert all(row['lr_used'] == rate for row in records[first - 1:last])
    for milestone, next_rate in zip([50, 75, 90], phase_rates[1:]):
        assert records[milestone - 1]['lr_next'] == next_rate
        assert records[milestone]['lr_used'] == next_rate
    boundary_lrs = {str(epoch): {'lr_used': records[epoch - 1]['lr_used'],
                                 'lr_next': records[epoch - 1]['lr_next']}
                    for epoch in [1, 50, 51, 75, 76, 90, 91, 99, 100]}
    assert all(row['optimizer_steps'] == row['epoch'] for row in records)
    assert all(row['test_optimizer_steps'] == 0 for row in records)
    assert all(row['batchnorm_counters_correct'] for row in records)

    scores = [row['test'][0]['top1_percent'] for row in records]
    expected_best = scores.index(max(scores)) + 1
    selection = continuous_report['checkpoint_selection']
    assert selection['best']['epoch'] == expected_best and selection['last']['epoch'] == 100
    final_checkpoint = continuous_path / 'checkpoints/epoch_00100.pt'
    final_state = core.load_verified(final_checkpoint)
    assert core.file_sha256(continuous_path / 'last_checkpoint.pt') == core.file_sha256(final_checkpoint)
    assert final_state['scheduler']['last_epoch'] == 100 and final_state['optimizer_steps'] == 100
    assert all(detail['decoded_weights_exact'] and detail['six_fp32_masters_absent']
               for detail in continuous_report['mx_exports'].values())
    with ZipFile(continuous_path / 'feedback_bundle.zip') as archive:
        assert {NOTEBOOK, 'notebook_support.py', 'training_summary.json', 'timing_estimate.json',
                'run_report.json', 'checkpoint_selection.json'}.issubset(archive.namelist())
    notebook = json.loads((continuous_path / NOTEBOOK).read_text(encoding='utf-8'))
    parameter_cells = [cell for cell in notebook['cells']
                       if 'parameters' in cell.get('metadata', {}).get('tags', [])]
    assert len(parameter_cells) == 1
    assert str(continuous_path.resolve()) in ''.join(parameter_cells[0]['source'])

    checkpoint99 = continuous_path / 'checkpoints/epoch_00099.pt'
    hash99 = core.file_sha256(checkpoint99)
    resumed_path, resumed_report = run_case('resume99_to100', checkpoint99)
    resumed_state = core.load_verified(resumed_path / 'checkpoints/epoch_00100.pt')
    assert cases['resume99_to100']['optimizer_updates_in_process'] == 1
    assert len(list((resumed_path / 'checkpoints').glob('epoch_*.pt'))) == 1
    assert all(resumed_report['restore_checks'].values())
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        differences = core.differences(final_state[key], resumed_state[key])
        assert not differences, (key, differences[:4])
    for key in ('train', 'test', 'model_sha256', 'lr_used', 'lr_next', 'best_epoch_so_far',
                'weight_updates', 'gradient_checks'):
        assert final_state['trace']['epochs'][-1][key] == resumed_state['trace']['epochs'][-1][key], key
    for phase in ('train', 'test'):
        assert final_state['trace']['epochs'][-1]['coverage'][phase]['order_sha256'] \
            == resumed_state['trace']['epochs'][-1]['coverage'][phase]['order_sha256']
    assert core.file_sha256(checkpoint99) == hash99
    assert resumed_report['full_training_completed'] is True
    assert resumed_report['selected_last_epoch'] == 100
    assert resumed_report['initial_model_sha256'] == continuous_report['initial_model_sha256']

    stopped_path, stopped_report = run_case('early_stop', stop_marker=True,
                                           expected_status='STOPPED_AT_EPOCH_BOUNDARY')
    assert stopped_report['completed_epoch'] == 1
    assert cases['early_stop']['optimizer_updates_in_process'] == 1
    assert stopped_report['full_training_completed'] is False
    assert stopped_report['training_summary']['full_100_epoch_training_completed'] is False
    assert stopped_report['timing_estimate']['measured_100_epoch_seconds'] is None
    assert stopped_report['timing_estimate']['full_100_epoch_training_completed'] is False
    assert (stopped_path / 'checkpoints/epoch_00001.pt').is_file()
    assert stopped_report['initial_model_sha256'] == continuous_report['initial_model_sha256']

    # Adversarial fixture, not a user checkpoint: a valid epoch5 payload carrying
    # the old pilot protocol must be refused before model restoration or updates.
    old_pilot = core.load_verified(continuous_path / 'checkpoints/epoch_00005.pt')
    old_pilot['contract'] = copy.deepcopy(old_pilot['contract'])
    old_config = old_pilot['contract']['config']
    old_config['schema'] = 'mx-e4m3-dedicated-pilot5-protocol-v1'
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
    completed_path, completed_report = run_case('reject_completed100', final_checkpoint,
                                                expected_status='FAIL')
    assert 'geen nieuwe referentie-epochs' in completed_report['error']
    assert cases['reject_completed100']['optimizer_updates_in_process'] == 0
    assert not list((completed_path / 'checkpoints').glob('*.pt'))

    # Independent best-score tie check: earliest maximum, never last-tie-wins.
    ties = [{'epoch': i + 1, 'test': [{'top1_percent': score}]}
            for i, score in enumerate([10, 80, 80, 20, 50])]
    assert t.best_record(ties)['epoch'] == 2

    cli_results = {}
    rejected_arguments = [['run', '--stop-after-epoch', value] for value in ['-1', '0', '5', '99', '101', '100.0']]
    rejected_arguments += [['run', '--device', 'cpu'], ['run', '--mode', 'control'],
                           ['run', '--synthetic'], ['resume'],
                           ['run', '--checkpoint', str(checkpoint99)]]
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

    report = {
        'status': 'PASS', 'synthetic_fixture_only': True,
        'scope': 'Real reference100 main, model, optimizer, loop and export; synthetic context on CPU only',
        'device': 'CPU; test-only Tensor.cuda redirection',
        'no_SVHN_data_loaded_or_downloaded': True,
        'not_a_SVHN_accuracy_or_GPU_timing_result': True,
        'environment': contexts['continuous100'].rt.environment,
        'production_config': production_config, 'package_sha256': package_hashes,
        'source_sha256': source_info['workcopy_files_sha256'],
        'test_duration_seconds': time.perf_counter() - started,
        'total_synthetic_optimizer_updates': sum(case['optimizer_updates_in_process'] for case in cases.values()),
        'cases': cases, 'actual_LR_boundaries': boundary_lrs,
        'tests': {
            'real_main_default100_and_PASS_only_at100': True,
            '100_synthetic_epochs_100_updates_200_metric_rows': True,
            'unchanged_50_75_90_LR_schedule_all_four_phases': True,
            'best_first_strict_maximum_and_last100_retained': True,
            'MX_best_last_exports_decoded_weights_exact': True,
            'measured100_time_only_after100': True,
            'new_notebook_and_feedback_bundle_present': True,
            'resume99_to100_model_optimizer_scheduler_rng_exact_CPU': True,
            'early_stop_complete_epoch_without_false_PASS': True,
            'old_pilot_contract_rejected_before_restore_and_updates': True,
            'completed100_resume_rejected_before_updates': True,
            'fresh_seed_model_identical_before_all_optional_restores': True,
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
