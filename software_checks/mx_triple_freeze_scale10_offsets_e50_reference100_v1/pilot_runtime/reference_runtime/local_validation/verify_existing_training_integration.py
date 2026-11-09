"""Read back existing synthetic full100 integration outputs without training.

The first execution completed every functional assertion, then its final
bookkeeping assertion mistakenly checked an unfinished package manifest. This
script records that harness error explicitly, verifies the stable runtime and
saved results, and writes the final validation report. No SVHN/CUDA training is
performed, and no checkpoints or raw result files are changed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_training_integration as h


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def case(directory, name):
    paths = list((directory / name).glob('*_joint_*'))
    assert len(paths) == 1, (name, paths)
    path = paths[0]
    return path, read_json(path / 'run_report.json')


def main(directory):
    started = time.perf_counter()
    directory = Path(directory).resolve()
    assert directory.is_dir()
    report_path = directory / 'training_integration_report.json'
    assert not report_path.exists(), 'Do not overwrite an earlier validation report.'
    production_config = h.r.config()
    config_review = h.compare_prior_config(production_config)
    runtime_hashes = h.runtime_hashes()
    continuous_path, continuous = case(directory, 'continuous100')
    resumed_path, resumed = case(directory, 'resume50_to100')
    partial_path, partial = case(directory, 'partial1')
    assert continuous['status'] == resumed['status'] == 'PASS'
    assert partial['status'] == 'STOPPED_AT_EPOCH_BOUNDARY'
    assert continuous['package_sha256'] == resumed['package_sha256'] == runtime_hashes
    assert continuous['synthetic_fixture_only'] and resumed['synthetic_fixture_only']
    assert continuous['dataset_downloaded'] is resumed['dataset_downloaded'] is False
    fixture_config = continuous['config']
    assert fixture_config['train_examples'] == fixture_config['test_examples'] == 3
    assert fixture_config['batch_size'] == 2 and fixture_config['synthetic_initial_theta_mixed']
    fixture_keys = {'train_examples', 'test_examples', 'batch_size', 'purpose',
                    'synthetic_initial_theta_mixed'}
    assert {k: v for k, v in fixture_config.items() if k not in fixture_keys} == {
        k: v for k, v in production_config.items() if k not in fixture_keys}
    trace = read_json(continuous_path / 'history.json')
    records = trace['epochs']
    assert [record['epoch'] for record in records] == list(range(1, 101))
    assert continuous['optimizer_steps'] == 200 and continuous['training_forward_passes'] == 600
    assert continuous['full_training_completed'] and continuous['full_100_epoch_training_completed']
    assert continuous['pilot_completed'] is False
    assert continuous['training_summary']['reference_end_epoch'] == 100
    assert continuous['training_summary']['expected_optimizer_steps_at_reference_completion'] == 200
    assert continuous['training_summary']['expected_training_forward_passes_at_reference_completion'] == 600
    assert continuous['timing_estimate']['measured_100_epoch_seconds'] == sum(
        record['total_epoch_seconds'] for record in records)
    assert len(list((continuous_path / 'checkpoints').glob('epoch_*.pt'))) == 100
    assert len(h.read_csv(continuous_path / 'epoch_metrics.csv')) == 600
    assert len(h.read_csv(continuous_path / 'weight_updates.csv')) == 1800
    assert len(h.read_csv(continuous_path / 'test_prediction_counts.csv')) == 300
    lr_boundaries = {}
    for record in records:
        epoch = record['epoch']
        assert record['lr_used'] == record['scale_lr_used'] == h.expected_lr(epoch)
        assert record['lr_next'] == record['scale_lr_next'] == h.expected_lr(epoch, after=True)
        assert record['optimizer_steps'] == epoch * 2
        assert record['test_optimizer_steps'] == 0 and record['batchnorm_counters_correct']
        checks = record['gradient_checks']
        assert checks['optimizer_steps_in_phase'] == checks['scale_optimizer_steps'] == 2
        assert checks['finite_gradient_checks'] == 2 and checks['detached_teacher_checks'] == 4
        assert checks['scale_projection_after_every_update']
        assert all(checks['forward_calls_' + name] == 2 for name in ('E4M3', 'E2M3', 'E2M1'))
        assert record['adam_checks']['all_active_adam_steps_correct']
        assert record['adam_checks']['moments_fp32_finite']
        for phase, checks in [('train', checks['shared_source_checks']),
                              ('test', record['test_source_checks'])]:
            assert checks['source_triplets'] == 2
            assert checks['high_encodings'] == checks['exponent_derivations'] == checks['mantissa_derivations'] == 12
            assert checks['offset_choices'] == list(range(10))
            assert checks['scale_factors'] == [2**k for k in range(10)]
            assert checks['learned_shared_scale'] and checks['no_direct_lower_quantizer']
            assert checks['same_hard_offsets_across_lower_modes'] and checks['theta_unchanged_within_triplet']
            assert checks['source_cache_cleared'] and checks['cache_cleared_before_adam']
            assert record['coverage'][phase]['complete_once'] and record['coverage'][phase]['n'] == 3
            for metric in record[phase]:
                assert metric['n'] == 3 and metric['activation_bits'] == 32
                assert abs(metric['top1_percent'] - 100 * metric['correct1'] / 3) < 1e-4
        if epoch in (49, 50, 51, 74, 75, 76, 89, 90, 91, 100):
            lr_boundaries[str(epoch)] = {key: record[key] for key in (
                'lr_used', 'lr_next', 'scale_lr_used', 'scale_lr_next')}
    scale_tables = h.verify_scale_tables(continuous_path, trace)
    assert scale_tables['block_state_rows'] == 412080
    assert scale_tables['epoch_metric_rows'] == 600
    final_path = continuous_path / 'checkpoints/epoch_00100.pt'
    final_state = h.core.load_verified(final_path)
    resumed_final = h.core.load_verified(resumed_path / 'checkpoints/epoch_00100.pt')
    equality = {}
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        differences = h.core.differences(final_state[key], resumed_final[key])
        assert not differences, (key, differences[:5])
        equality[key] = 'exact CPU tensor/container equality'
    assert final_state['optimizer_steps'] == resumed_final['optimizer_steps'] == 200
    assert len(list((resumed_path / 'checkpoints').glob('epoch_*.pt'))) == 50
    assert all(resumed['restore_checks'].values())
    assert resumed['training_summary']['resumed'] and resumed['full_training_completed']
    assert final_state['scheduler']['last_epoch'] == 100
    for before, after in zip(final_state['trace']['epochs'], resumed_final['trace']['epochs']):
        for key in ('train', 'test', 'model_sha256', 'lr_used', 'lr_next',
                    'scale_lr_used', 'scale_lr_next', 'best_epoch_so_far',
                    'weight_updates', 'gradient_checks', 'test_source_checks',
                    'test_prediction_counts', 'test_label_counts',
                    'scale_state', 'scale_epoch_metrics'):
            assert before[key] == after[key], (before['epoch'], key)
        for phase in ('train', 'test'):
            assert before['coverage'][phase]['order_sha256'] == after['coverage'][phase]['order_sha256']
    for name in ('scale_block_state.csv', 'scale_epoch_metrics.csv'):
        assert (continuous_path / name).read_bytes() == (resumed_path / name).read_bytes()
        equality[name] = 'byte-identical'
    checkpoint50 = continuous_path / 'checkpoints/epoch_00050.pt'
    state50 = h.core.load_verified(checkpoint50)
    sha50 = h.core.file_sha256(checkpoint50)
    metadata50 = read_json(continuous_path / 'epoch50_checkpoint.json')
    assert metadata50 == continuous['epoch50_checkpoint'] == continuous['training_summary']['epoch50_checkpoint']
    assert metadata50['available'] and metadata50['full_training_state']
    assert metadata50['sha256'] == sha50
    assert Path(metadata50['original_checkpoint']).resolve() == checkpoint50.resolve()
    assert resumed['epoch50_checkpoint']['sha256'] == sha50
    assert state50['completed_epoch'] == 50 and state50['next_epoch'] == 51
    assert state50['optimizer_steps'] == 100 and state50['scheduler']['last_epoch'] == 50
    assert state50['contract']['config']['scale_learning_end_epoch'] == 100
    assert state50['contract']['config']['scale_freeze_epoch'] is None
    assert set(state50['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'loader_generators'}
    assert state50['rng']['torch_cuda'] == []
    groups50 = state50['optimizer']['param_groups']
    assert [group['name'] for group in groups50] == ['weights', 'scale_proxy']
    assert all(group['lr'] == h.expected_lr(51) > 0 for group in groups50)
    theta_keys = [key for key in state50['model'] if key.endswith('.quantize_fn.theta')]
    assert len(theta_keys) == 6
    assert sum(state50['model'][key].numel() for key in theta_keys) == 4080
    assert len(groups50[1]['params']) == 6
    for parameter_id in groups50[1]['params']:
        item = state50['optimizer']['state'][parameter_id]
        assert item['step'].item() == 100
        assert item['exp_avg'].dtype == item['exp_avg_sq'].dtype == h.torch.float32
        assert h.torch.isfinite(item['exp_avg']).all() and h.torch.isfinite(item['exp_avg_sq']).all()
    changed_theta_after50 = sum(int((state50['model'][key] != final_state['model'][key]).sum())
                               for key in theta_keys)
    assert changed_theta_after50 > 0
    assert len(state50['trace']['epochs']) == 50
    best = max(records, key=lambda record: record['test'][-1]['top1_percent'])['epoch']
    assert continuous['selected_best_epoch'] == best and continuous['selected_last_epoch'] == 100
    selected = h.read_csv(continuous_path / 'selected_metrics.csv')
    assert len(selected) == 6
    export_hashes = {}
    for role, epoch in [('best_E4M3_test', best), ('last_epoch', 100)]:
        assert {int(row['epoch']) for row in selected if row['role'] == role} == {epoch}
        loaded = {}
        for bit in h.MODES:
            entry = continuous['mx_exports'][role][str(bit)]
            export_path = continuous_path / entry['file']
            assert entry['epoch'] == epoch and entry['decoded_weights_exact']
            assert entry['six_fp32_masters_absent'] and entry['fp32_scale_proxies_absent']
            assert h.core.file_sha256(export_path) == entry['sha256']
            loaded[bit] = h.a.load_verified_payload(export_path)
            export_hashes[entry['file']] = entry['sha256']
        assert sum(x['packed_choices'].numel() for x in loaded[8]['scale_choices'].values()) == 2040
        for layer in h.a.TARGET_LAYERS:
            high = h.a.codec(8).MXTensor.from_payload(loaded[8]['mx_layers'][layer])
            middle = h.a.codec(6).MXTensor.from_payload(loaded[6]['mx_layers'][layer])
            low = h.a.codec(4).MXTensor.from_payload(loaded[4]['mx_layers'][layer])
            offset = h.independent_offsets(loaded[8]['scale_choices'][layer], tuple(high.scale_codes.shape))
            codes = high.codes.long()
            expected6 = ((codes >> 7) << 5) | (((codes >> 5) & 3) << 3) | (codes & 7)
            assert h.torch.equal(middle.codes.long(), expected6)
            assert h.torch.equal(low.codes.long(), expected6 >> 2)
            assert h.torch.equal(middle.scale_codes.int(), high.scale_codes.int() + offset)
            assert h.torch.equal(low.scale_codes, middle.scale_codes)
    assert partial['completed_epoch'] == 1 and partial['optimizer_steps'] == 2
    assert not partial['full_training_completed'] and not partial['pilot_completed']
    assert partial['timing_estimate']['measured_100_epoch_seconds'] is None
    assert not partial['epoch50_checkpoint']['available']
    cases = {}
    negative_cases = ['wrong_learned_pilot', 'wrong_fixed32_full100', 'wrong_scale_gradient',
                      'completed100', 'endpoint_precedes_checkpoint']
    for name in ['continuous100', 'resume50_to100', 'partial1', *negative_cases]:
        path, result = case(directory, name)
        assert result['synthetic_fixture_only'] and result['dataset_downloaded'] is False
        assert result['package_sha256'] == runtime_hashes
        if name in negative_cases:
            assert result['status'] == 'FAIL'
            message = ('geen nieuwe referentie-epochs' if name in ('completed100', 'endpoint_precedes_checkpoint')
                       else 'Hervatprotocol/omgeving/bron verschilt')
            assert message in result['error']
            assert not list((path / 'checkpoints').glob('*.pt'))
        cases[name] = {'status': result['status'], 'output_directory': str(path),
                       'completed_epoch': result.get('completed_epoch'), 'error': result.get('error')}
        with ZipFile(path / 'feedback_bundle.zip') as archive:
            assert h.NOTEBOOK in archive.namelist() and 'notebook_support.py' in archive.namelist()
            assert any(name.startswith('pilot_reference/') for name in archive.namelist())
            assert any(name.startswith('fixed_references/') for name in archive.namelist())
            assert not any(name.endswith(('.pt', '.mat', '.npy')) for name in archive.namelist())
        assert (directory / (name + '_stdout.log')).is_file()
    assert runtime_hashes == h.runtime_hashes()
    assert h.r.config() == production_config
    assert h.core.file_sha256(final_path) == h.core.file_sha256(continuous_path / 'last_checkpoint.pt')
    harness_path = Path(h.__file__)
    corrected_text = harness_path.read_text()
    original_text = corrected_text.replace(
        'assert r.config()==production_config and runtime_hashes()==package_hashes',
        'assert r.config()==production_config and s.verify_package()==package_hashes')
    assert original_text != corrected_text
    original_harness_sha = hashlib.sha256(original_text.encode()).hexdigest()
    report = {
        'status': 'PASS', 'synthetic_fixture_only': True, 'CUDA_tested': False,
        'scope': 'Actual full100 runner cases plus independent readback of saved outputs; no actual SVHN',
        'no_SVHN_data_loaded_or_downloaded': True,
        'not_a_SVHN_accuracy_or_GPU_timing_result': True,
        'environment': continuous['environment'], 'production_config': production_config,
        'fixture_changes': {'train_examples': 3, 'test_examples': 3, 'batch_size': 2,
                            'theta_mixed_test_only': True},
        'validation_harness_correction': {
            'original_harness_exit_code': 1,
            'failure_stage': 'Final bookkeeping assertion after all functional runner/CLI assertions',
            'failure_message': 'RuntimeError: Pakketbestand gewijzigd: EXPERIMENT_MANIFEST.json',
            'cause': 'One stale s.verify_package call remained while package assembly was still in progress.',
            'runner_failure': False,
            'correction': 'Use the same recorded stable runtime_hashes contract at final bookkeeping.',
            'corrected_harness_file': str(harness_path),
            'corrected_harness_sha256': h.core.file_sha256(harness_path),
            'original_harness_sha256_reconstructed_from_single_line_change': original_harness_sha,
            'corrected_harness_fully_rerun': False,
            'actual_training_not_repeated_for_bookkeeping': True,
            'verification': 'Read-only independent checks of original checkpoints, histories, exports and runtime bindings.',
        },
        'comparison_to_learned_scale10_pilot': config_review,
        'tested_training_runtime_sha256': runtime_hashes,
        'final_package_manifest_verified_here': False,
        'fixture_integrity_scope': 'Stable training runtime/protocol only; final package manifest and notebook verified separately.',
        'source_sha256': continuous['source']['workcopy_files_sha256'],
        'initial_base_model_sha256': continuous['initial_base_model_sha256'],
        'LR_boundary_values_both_groups': lr_boundaries,
        'scale_tables_checked': scale_tables,
        'resume_equality': equality,
        'epoch50_checkpoint': {'sha256': sha50, 'saved_theta_parameters': 4080,
                               'saved_Adam_parameter_groups': 2, 'scale_lr_after_epoch50': h.expected_lr(51),
                               'changed_theta_values_epoch50_to100': changed_theta_after50},
        'export_sha256': export_hashes,
        'total_actual_synthetic_optimizer_updates': 302,
        'cases': cases,
        'tests': {
            'actual_default100_main_passes_only_at100': True,
            'all100_epochs_both_Adam_LRs_50_75_90_correct': True,
            'epoch50_theta_Adam_scheduler_RNG_saved_and_unfrozen': True,
            'theta_changes_after50_until_full100': True,
            'resume50_to100_model_theta_Adam_scheduler_RNG_exact_CPU': True,
            'all100_scale_history_and_CSVs_resume_exact': True,
            'partial1_reports_incomplete_and_epoch50_unavailable': True,
            'foreign_pilot_and_gradient_contracts_refused': True,
            'completed100_and_noprogress_resume_refused': True,
            'CLI_boundaries_CPU_and_warmstart_refusals_passed_in_original_execution': True,
            'all_batches_three_passes_one_update_A32_FP32_BN_source_audits': True,
            'independent2040byte_export_choices_and_bit_mappings': True,
            'source_only_reconstruction_logits_checks_passed_in_original_execution': True,
            'common_E4_best_and_epoch100_selection': True,
            'blank_notebook_and_references_in_all_feedback_bundles': True,
            'training_runtime_and_input_checkpoints_unchanged': True,
        },
        'readback_script_sha256': h.core.file_sha256(Path(__file__)),
        'readback_duration_seconds': time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'report': str(report_path),
                      'duration_seconds': report['readback_duration_seconds'],
                      'training_repeated': False}, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: verify_existing_training_integration.py EXISTING_OUTPUT_DIRECTORY')
    main(Path(sys.argv[1]))
