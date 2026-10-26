"""Focused runner integration on three synthetic CPU images, never real SVHN.

Only this test replaces create_context/final data verification and redirects the
authors' Tensor.cuda calls to CPU. The actual main, three-mode loop, optimizer,
checkpoint/restore, exports and feedback generation remain unchanged. The
production CLI has no CPU/synthetic bypass.

The fixture executes 100 tiny epochs with unchanged LR milestones 50/75/90.
Resume after epoch99 verifies the final optimizer/scheduler/RNG state exactly.
Three one-epoch resumes cross the actual LR boundaries at 50, 75 and 90.
Only dataset size, batch size and CPU routing differ from production.
No notebook is executed and no synthetic research-result notebook is produced.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import ast
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch
import joint_support as direct_support
import trunc_adapter as a
import trunc_support as t
import resume_core as core
import run_train as r
import svhn_support as s

NOTEBOOK = 'MX_triple_group16_scale64_reference100_resultaten.ipynb'
MODES = (4, 6, 8)
FORMATS = {4: 'MXFP4_E2M1', 6: 'MXFP6_E2M3', 8: 'MXFP8_E4M3'}
TEACHERS = {4: 'MXFP6_E2M3', 6: 'MXFP8_E4M3', 8: None}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def verify_exports(directory, exports, epoch, modules):
    """Read the actual produced files; derive expected bits from saved E4 only."""
    assert set(exports) == {'4', '6', '8'}
    loaded = {}
    for bit in MODES:
        detail = exports[str(bit)]
        assert detail['epoch'] == epoch and detail['weight_bits'] == bit
        assert detail['decoded_weights_exact'] and detail['six_fp32_masters_absent']
        assert detail['truncation'] is (bit != 8)
        assert detail['source_scale_bytes_copied'] is (bit == 8)
        assert detail['scale_exponent_offset'] == (0 if bit == 8 else 6)
        assert detail['scale_factor'] == (1 if bit == 8 else 64)
        assert detail['shared_scale_compensation'] is (bit != 8)
        assert detail['learned_shared_scale'] is False
        assert detail['full_test_forward_rerun'] is False
        assert detail['mx_block_size'] == 16
        assert detail['group_size_cost_audit']['status'] == 'PASS'
        export_path = directory / detail['file']
        assert core.file_sha256(export_path) == detail['sha256']
        assert export_path.with_suffix(export_path.suffix + '.sha256').read_text().strip() == detail['sha256']
        loaded[bit] = a.load_verified_payload(export_path)
        assert all(name + '.weight' not in loaded[bit]['auxiliary_fp32_state'] for name in a.TARGET_LAYERS)
        assert loaded[bit]['weight_bits'] == bit
        assert loaded[bit]['bn_bank_key'] == a.BANK[bit]
        assert loaded[bit]['provenance']['source_epoch'] == epoch
    for name in a.TARGET_LAYERS:
        h = a.codec(8).MXTensor.from_payload(loaded[8]['mx_layers'][name])
        mid = a.codec(6).MXTensor.from_payload(loaded[6]['mx_layers'][name])
        low = a.codec(4).MXTensor.from_payload(loaded[4]['mx_layers'][name])
        # Independent unpacked bit field expression; no training helper.
        u8 = h.codes.long()
        expected6 = ((u8 >> 7) << 5) | (((u8 >> 5) & 3) << 3) | (u8 & 7)
        assert torch.equal(mid.codes.long(), expected6)
        assert torch.equal(low.codes.long(), expected6 >> 2)
        shifted = h.scale_codes.to(torch.int32) + 6
        assert bool((shifted <= 254).all())
        assert torch.equal(mid.scale_codes.to(torch.int32), shifted)
        assert torch.equal(low.scale_codes, mid.scale_codes)
        assert h.shape == mid.shape == low.shape
        assert h.codes.shape[-1] == mid.codes.shape[-1] == low.codes.shape[-1] == 16
    # Reconstruct all three modes from the saved high source while quantizers
    # and G(W) are forbidden. Derived files must yield the same frozen logits.
    def forbidden(*args, **kwargs):
        raise AssertionError('Source-only export reconstruction attempted a weight encoder or G(W).')
    images = torch.linspace(-1, 1, 2 * 3 * 40 * 40).reshape(2, 3, 40, 40)
    with ExitStack() as stack:
        stack.enter_context(patch.object(a, 'effective_reference_weight', forbidden))
        for bit in MODES:
            stack.enter_context(patch.object(a.codec(bit), 'encode_tensor', forbidden))
            stack.enter_context(patch.object(a.codec(bit), 'encode_elements', forbidden))
        for bit in MODES:
            source_model = a.load_exported_model(modules.models, loaded[8], torch.device('cpu'), bit)
            own_model = a.load_exported_model(modules.models, loaded[bit], torch.device('cpu'), bit)
            before = s.legacy.state_digest(source_model.state_dict())
            with torch.no_grad():
                source_logits, own_logits = source_model(images), own_model(images)
            assert source_logits.shape == (2, 10) and torch.isfinite(source_logits).all()
            assert torch.equal(source_logits, own_logits)
            assert all(not module.training for module in source_model.modules())
            assert all(not p.requires_grad and p.grad is None for p in source_model.parameters())
            assert s.legacy.state_digest(source_model.state_dict()) == before


def compare_prior_config(current):
    """Only the duration identity and reporting may differ from the successful group16 fixed64 pilot."""
    source = ROOT / 'provenance/group16_pilot5_v2_run_train.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'config')
    namespace = {'s': s, 't': t, 'BLOCK_SIZE': 16}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    previous = namespace['config']()
    changed = {key: {'previous': previous.get(key), 'current': current.get(key)}
               for key in sorted(set(previous) | set(current)) if previous.get(key) != current.get(key)}
    allowed = {'schema', 'purpose', 'pilot_stop_epoch', 'reference_end_epoch', 'reporting_policy'}
    assert set(changed) == allowed, changed
    assert previous['planned_epochs'] == current['planned_epochs'] == 100
    assert previous['pilot_stop_epoch'] == 5 and current['reference_end_epoch'] == 100
    return {'same_numerical_route_training_hyperparameters_and_mechanics': True,
            'only_duration_identity_and_reporting_changed': True, 'differences': changed,
            'archived_runner_sha256': core.file_sha256(source)}


def verify_reused_core():
    """Bind prior successful B16 pilot evidence to identical numerical files."""
    source = ROOT / 'provenance/group16_pilot5_v2_PACKAGE_SHA256.json'
    manifest = json.loads(source.read_text(encoding='utf-8'))
    names = ['group_config.py', 'group_costs.py', 'mx_format.py', 'mx_e4m3.py', 'mx_e2m1.py',
             'posthoc_truncation.py', 'trunc_adapter.py', 'trunc_support.py', 'joint_adapter.py',
             'joint_support.py', 'mx_adapter.py', 'resume_core.py', 'smoke_support.py',
             'svhn_support.py', 'upstream/Any-Precision-DNNs-master.zip']
    for name in names:
        assert core.file_sha256(ROOT / name) == manifest[name], name
    return {'status': 'PASS', 'reused_files_verified': len(names),
            'reused_files_sha256': {name:manifest[name] for name in names},
            'successful_pilot_manifest': str(source.relative_to(ROOT)),
            'all_numerical_files_identical_to_successful_group16_pilot': True,
            'standalone_exhaustive_codec_suite_repeated': False}


def verify_independent_group16_scale64_and_STE():
    """Check arithmetic and declared STE independently before the runner fixture."""
    upstream = torch.linspace(-.71, 1.13, 64).reshape(2, 32)
    for bit in (8, 6, 4):
        effective = torch.linspace(-.39, .49, 64).reshape(2, 32).requires_grad_(True)
        chain = a.encode_chain(effective)
        source = chain[8]
        bits = source.codes.to(torch.int64)
        expected6 = ((bits >> 7) << 5) | (((bits >> 5) & 3) << 3) | (bits & 7)
        expected_scales = (source.scale_codes.to(torch.int32) + 6).to(torch.uint8)
        if bit == 8:
            expected = source.decode()
        else:
            codes = (expected6 if bit == 6 else expected6 >> 2).to(torch.uint8)
            expected = a.codec(bit).MXTensor(codes, expected_scales, source.shape).decode()
            assert torch.equal(chain[bit].codes, codes)
            assert torch.equal(chain[bit].scale_codes, expected_scales)
        result = a._CachedCodeSTE.apply(effective, chain[bit].codes, chain[bit].scale_codes, bit)
        assert torch.equal(result, expected)
        actual_gradient, = torch.autograd.grad((result * upstream).sum(), effective)
        assert torch.equal(actual_gradient, upstream), 'STE must not multiply gradient by64'

        weight = torch.linspace(-.39, .49, 64).reshape(2, 32).requires_grad_(True)
        quantizer = a.JointMXQuantizer()
        quantizer.weight_mode = bit
        actual_gradient, = torch.autograd.grad((quantizer(weight) * upstream).sum(), weight)
        expected_gradient, = torch.autograd.grad((a.effective_reference_weight(weight) * upstream).sum(), weight)
        assert torch.equal(actual_gradient, expected_gradient), 'Original G(W) derivative was changed'
    return {'independent_bit_and_plus6_scale_forward': True,
            'identity_backward_all_three_modes_exact': True,
            'no_factor64_in_backward': True,
            'original_G_derivatives_preserved': True}

def main(out: Path) -> dict:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    package_hashes = s.verify_package()
    production_config = r.config()
    assert production_config['schema'] == 'mx-triple-group16-scale64-reference100-protocol-v1'
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
    for flag in ('truncation', 'mantissa_truncation', 'exponent_truncation'):
        assert production_config[flag] is True, flag
    assert production_config['shared_scale_compensation'] is True
    assert production_config['scale_exponent_offset'] == 6 and production_config['scale_factor'] == 64
    assert production_config['mx_block_size'] == 16
    for flag in ('learned_shared_scale', 'extra_FP32_teacher_forward', 'direct_lower_quantization', 'exponent_bias_compensation'):
        assert production_config[flag] is False, flag
    numerical_checks = verify_independent_group16_scale64_and_STE()
    config_comparison = compare_prior_config(production_config)
    core_reuse = verify_reused_core()

    prepared = out / 'source'
    prepared.mkdir()
    source, source_info = s.legacy.prepare_source(prepared)
    modules = s.import_source(source)
    # Compare the unchanged direct-MX builder in the same CPU runtime and RNG
    # context. The user's actual GPU/data-loading run has a different context;
    # its stored initial hash is not the synthetic-fixture reference.
    reference_runtime = s.runtime(production_config['seed'], 'cpu', strict=False)
    direct_reference = direct_support.build_model(modules, 'joint', reference_runtime.device)
    direct_initial_sha256 = s.legacy.state_digest(direct_reference.state_dict())
    del direct_reference
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(3, 3, 40, 40, generator=generator) * 2 - 1
    labels = torch.tensor([0, 1, 2])
    datasets = {phase: torch.utils.data.TensorDataset(images.clone(), labels.clone())
                for phase in ('train', 'test')}
    cases, contexts = {}, {}

    def fixture_context(args, path, report, log, check_only=False):
        assert not check_only, 'This integration fixture tests run/resume only.'
        assert args.stop_after_epoch == 100 and not args.download
        rt = s.runtime(production_config['seed'], 'cpu', strict=False)
        c = r.config()
        c.update(train_examples=3, test_examples=3, batch_size=2,
                 purpose='SYNTHETIC_CPU_TRIPLE_GROUP16_SCALE64_REFERENCE100_INTEGRATION_ONLY')
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
        assert all(len(loader) == 2 for loader in loaders.values())
        contract = {'config': c, 'environment': rt.environment, 'settings': rt.settings,
                    'source_files': source_info['workcopy_files_sha256'],
                    'package_sha256': package_hashes, 'synthetic_fixture_only': True}
        report.update(config=c, environment=rt.environment, execution_settings=rt.settings,
                      source=source_info, package_sha256=package_hashes,
                      contract_sha256=core.json_digest(contract), synthetic_fixture_only=True,
                      dataset_downloaded=False)
        report['group_size_costs'] = r.group_costs.write_cost_report(path, active_group_size=16)
        report['limitations'].append(
            'SYNTHETIC CPU FIXTURE: three fake images, batch2, unchanged LR milestones50/75/90; no SVHN accuracy or GPU timing.')
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

    def forbid_lower_encoder(*args, **kwargs):
        raise AssertionError('Group16 fixed64 training invoked a forbidden direct lower encoder.')

    def run_case(name, checkpoint=None, stop_marker=False, expected_status='PASS'):
        cases[name] = {'optimizer_updates_in_process': 0, 'stop_marker': stop_marker}
        arguments = ['resume' if checkpoint else 'run', '--output-root', str(out / name)]
        if checkpoint is not None:
            arguments += ['--checkpoint', str(checkpoint)]
        print('SYNTHETIC triple-group16-scale64 reference100 runner case: ' + name, flush=True)
        with (out / (name + '_stdout.log')).open('x', encoding='utf-8') as stdout:
            with redirect_stdout(stdout), patch.object(sys, 'argv', [str(ROOT / 'run_train.py'), *arguments]), \
                 patch.object(r, 'create_context', fixture_context), \
                 patch.object(r, 'verify_unchanged', verify_fixture_unchanged), \
                 patch.object(torch.Tensor, 'cuda', cpu_cuda), \
                 patch.object(a.codec(6), 'encode_tensor', forbid_lower_encoder), \
                 patch.object(a.codec(6), 'encode_elements', forbid_lower_encoder), \
                 patch.object(a.codec(4), 'encode_tensor', forbid_lower_encoder), \
                 patch.object(a.codec(4), 'encode_elements', forbid_lower_encoder):
                exit_code = r.main()
        paths = list((out / name).glob('*_joint_*'))
        assert len(paths) == 1, paths
        path = paths[0]
        report = json.loads((path / 'run_report.json').read_text(encoding='utf-8'))
        assert report['status'] == expected_status, (name, report.get('error'), report.get('traceback'))
        assert exit_code == (1 if expected_status == 'FAIL' else 0), (name, exit_code)
        assert report['synthetic_fixture_only'] and not report['dataset_downloaded']
        assert report['schema'] == 'mx-triple-group16-scale64-reference100-report-v1'
        cases[name].update(status=report['status'], exit_code=exit_code, output_directory=str(path))
        return path, report

    continuous_path, continuous_report = run_case('continuous100')
    assert continuous_report['initial_model_sha256'] == direct_initial_sha256
    records = json.loads((continuous_path / 'history.json').read_text(encoding='utf-8'))['epochs']
    assert [record['epoch'] for record in records] == list(range(1, 101))
    assert continuous_report['optimizer_steps'] == 200
    assert continuous_report['training_forward_passes'] == 600
    assert cases['continuous100']['optimizer_updates_in_process'] == 200
    assert continuous_report['full_training_completed'] is True
    assert continuous_report['training_summary']['full_100_epoch_training_completed'] is True
    assert 'pilot_completed' not in continuous_report['training_summary']
    assert continuous_report['training_summary']['expected_optimizer_steps_at_reference_completion'] == 200
    assert continuous_report['training_summary']['expected_training_forward_passes_at_reference_completion'] == 600
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
    prediction_rows = read_csv(continuous_path / 'test_prediction_counts.csv')
    assert len(prediction_rows) == 100 * 3
    for row in prediction_rows:
        counts = records[int(row['epoch']) - 1]['test_prediction_counts'][row['weight_bits']]
        assert counts == [int(row['predicted_class_' + str(k)]) for k in range(10)]
        assert int(row['classes_predicted']) == sum(value > 0 for value in counts)
        assert int(row['dominant_predicted_class']) == counts.index(max(counts))
        assert float(row['dominant_prediction_fraction']) == max(counts) / sum(counts)
    assert len(list((continuous_path / 'sample_order').glob('*.npy'))) == 200
    assert continuous_report['timing_estimate']['measured_completed_epoch_seconds'] \
        == sum(record['total_epoch_seconds'] for record in records)

    phase_rates = [.001, .001 * .1, .001 * .1 * .1, .001 * .1 * .1 * .1]
    for (first, last), rate in zip([(1, 50), (51, 75), (76, 90), (91, 100)], phase_rates):
        assert all(record['lr_used'] == rate for record in records[first - 1:last])
    for record in records:
        epoch = record['epoch']
        assert record['optimizer_steps'] == 2 * epoch
        assert record['test_optimizer_steps'] == 0 and record['batchnorm_counters_correct']
        audit = record['gradient_checks']
        assert audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == 2
        assert audit['detached_teacher_checks'] == 4
        assert all(audit[key] == 2 for key in ('forward_calls_E4M3', 'forward_calls_E2M3', 'forward_calls_E2M1'))
        for phase, source_audit in [('train', audit['shared_source_checks']),
                                    ('test', record['test_source_checks'])]:
            assert source_audit['source_triplets'] == 2
            assert source_audit['high_encodings'] == 12
            assert source_audit['exponent_derivations'] == 12
            assert source_audit['mantissa_derivations'] == 12
            assert 'scale_copies_unchanged' not in source_audit
            assert source_audit['scale_offsets_plus6'] == 24
            assert source_audit['mantissa_scale_copies_unchanged'] == 12
            assert source_audit['scale_exponent_offset'] == 6 and source_audit['scale_factor'] == 64
            assert source_audit['source_cache_cleared']
            assert source_audit['no_direct_lower_quantizer']
            assert source_audit['cache_cleared_before_adam']
            assert source_audit['training_order'] == ([8, 6, 4] if phase == 'train' else [4, 6, 8])
        assert set(record['test_prediction_counts']) == {'8', '6', '4'}
        assert record['test_label_counts'] == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        for counts in record['test_prediction_counts'].values():
            assert len(counts) == 10 and sum(counts) == 3
            assert all(type(value) is int and value >= 0 for value in counts)
        assert record['adam_checks']['all_active_adam_steps_correct']
        assert record['adam_checks']['moments_fp32_finite']
        assert record['adam_checks']['unused_bn_has_no_optimizer_state']
        for phase in ('train', 'test'):
            assert [row['weight_bits'] for row in record[phase]] == list(MODES)
            assert record['coverage'][phase]['n'] == 3 and record['coverage'][phase]['complete_once']
            for row in record[phase]:
                bit = row['weight_bits']
                assert row['n'] == 3 and row['activation_bits'] == 32
                assert row['weight_format'] == FORMATS[bit]
                assert row['bn_bank_key'] == a.BANK[bit]
                assert row['teacher_weight_format'] == (TEACHERS[bit] if phase == 'train' else None)
                assert row['objective_type'] == (
                    'recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy')
                assert abs(row['top1_percent'] - 100 * row['correct1'] / 3) < 1e-4
        assert {row['weight_bits'] for row in record['weight_updates']} == set(MODES)
        assert all(row['master_dtype'] == 'torch.float32' for row in record['weight_updates'])
    for milestone, next_rate in zip([50, 75, 90], phase_rates[1:]):
        assert records[milestone - 1]['lr_next'] == next_rate
        assert records[milestone]['lr_used'] == next_rate
    boundary_lrs = {str(epoch): {'lr_used': records[epoch - 1]['lr_used'],
                                 'lr_next': records[epoch - 1]['lr_next']}
                    for epoch in [1, 50, 51, 75, 76, 90, 91, 99, 100]}

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
        verify_exports(continuous_path, continuous_report['mx_exports'][role], epoch, modules)
    assert len(list(continuous_path.glob('mx_export_layers_*.csv'))) == 6
    final_checkpoint = continuous_path / 'checkpoints/epoch_00100.pt'
    final_state = core.load_verified(final_checkpoint)
    assert core.file_sha256(continuous_path / 'last_checkpoint.pt') == core.file_sha256(final_checkpoint)
    assert final_state['scheduler']['last_epoch'] == 100 and final_state['optimizer_steps'] == 200

    with ZipFile(continuous_path / 'feedback_bundle.zip') as archive:
        expected = {NOTEBOOK, 'notebook_support.py', 'training_summary.json', 'timing_estimate.json',
                    'run_report.json', 'checkpoint_selection.json', 'selected_metrics.csv',
                    'test_prediction_counts.csv'}
        expected.update(f'mx_export_layers_{bit}_{role}.csv' for bit in MODES for role in ('best', 'last'))
        for reference in ['scale64_reference', 'pilot_reference']:
            expected.update(reference + '/' + p.name for p in (ROOT / reference).iterdir() if p.is_file())
            for original in (ROOT / reference).iterdir():
                if original.is_file():
                    relative = reference + '/' + original.name
                    assert (continuous_path / relative).read_bytes() == original.read_bytes()
                    assert archive.read(relative) == original.read_bytes()
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
    assert cases['resume99_to100']['optimizer_updates_in_process'] == 2
    assert len(list((resumed_path / 'checkpoints').glob('epoch_*.pt'))) == 1
    assert all(resumed_report['restore_checks'].values())
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        differences = core.differences(final_state[key], resumed_state[key])
        assert not differences, (key, differences[:4])
    for before, after in zip(final_state['trace']['epochs'], resumed_state['trace']['epochs']):
        for key in ('train', 'test', 'model_sha256', 'lr_used', 'lr_next', 'best_epoch_so_far',
                    'weight_updates', 'gradient_checks', 'test_source_checks',
                    'test_prediction_counts', 'test_label_counts'):
            assert before[key] == after[key], (before['epoch'], key)
        for phase in ('train', 'test'):
            assert before['coverage'][phase]['order_sha256'] == after['coverage'][phase]['order_sha256']
    assert core.file_sha256(checkpoint99) == checkpoint99_hash
    assert resumed_report['full_training_completed'] is True
    assert resumed_report['training_summary']['resumed'] is True
    assert resumed_report['initial_model_sha256'] == continuous_report['initial_model_sha256']

    # Restore immediately before each real scheduler milestone. The marker asks
    # the unchanged loop to finish exactly one epoch, including scheduler.step,
    # checkpoint and exports, then stop. No shortened LR fixture is used.
    lr_resume_checks = {}
    for milestone in (50, 75, 90):
        previous_epoch = milestone - 1
        checkpoint = continuous_path / f'checkpoints/epoch_{previous_epoch:05d}.pt'
        checkpoint_hash = core.file_sha256(checkpoint)
        name = f'resume{previous_epoch}_to{milestone}_stop'
        boundary_path, boundary_report = run_case(name, checkpoint, stop_marker=True,
                                                   expected_status='STOPPED_AT_EPOCH_BOUNDARY')
        assert cases[name]['optimizer_updates_in_process'] == 2
        assert boundary_report['completed_epoch'] == milestone
        assert boundary_report['full_training_completed'] is False
        assert boundary_report['training_summary']['full_100_epoch_training_completed'] is False
        assert boundary_report['timing_estimate']['measured_100_epoch_seconds'] is None
        assert all(boundary_report['restore_checks'].values())
        expected_state = core.load_verified(continuous_path / f'checkpoints/epoch_{milestone:05d}.pt')
        actual_state = core.load_verified(boundary_path / f'checkpoints/epoch_{milestone:05d}.pt')
        for key in ('model', 'optimizer', 'scheduler', 'rng'):
            differences = core.differences(expected_state[key], actual_state[key])
            assert not differences, (milestone, key, differences[:4])
        for key in ('train', 'test', 'model_sha256', 'lr_used', 'lr_next', 'best_epoch_so_far',
                    'weight_updates', 'gradient_checks', 'test_source_checks',
                    'test_prediction_counts', 'test_label_counts'):
            assert expected_state['trace']['epochs'][-1][key] == actual_state['trace']['epochs'][-1][key], (milestone, key)
        assert actual_state['scheduler']['last_epoch'] == milestone
        assert actual_state['trace']['epochs'][-1]['lr_next'] == records[milestone - 1]['lr_next']
        assert core.file_sha256(checkpoint) == checkpoint_hash
        lr_resume_checks[str(milestone)] = {
            'restored_epoch': previous_epoch, 'stopped_at_epoch': milestone,
            'optimizer_updates_in_process': 2,
            'model_optimizer_scheduler_rng_exact': True,
            'lr_used': records[milestone - 1]['lr_used'],
            'lr_next': records[milestone - 1]['lr_next'],
            'input_checkpoint_unchanged': True,
        }

    stopped_path, stopped_report = run_case('early_stop', stop_marker=True,
                                          expected_status='STOPPED_AT_EPOCH_BOUNDARY')
    assert stopped_report['completed_epoch'] == 1
    assert cases['early_stop']['optimizer_updates_in_process'] == 2
    assert stopped_report['full_training_completed'] is False
    assert 'pilot_completed' not in stopped_report['training_summary']
    assert stopped_report['training_summary']['full_100_epoch_training_completed'] is False
    assert stopped_report['timing_estimate']['measured_100_epoch_seconds'] is None
    assert stopped_report['timing_estimate']['full_100_epoch_training_completed'] is False
    partial_limits = ' '.join(stopped_report['timing_estimate']['limitations'])
    assert 'Only completed epochs are observed' in partial_limits and 'comparable' in partial_limits
    assert 'not measured or remaining time' in stopped_report['timing_estimate']['interpretation']
    assert (stopped_path / 'checkpoints/epoch_00001.pt').is_file()
    assert len(read_csv(stopped_path / 'selected_metrics.csv')) == 6

    # An adversarial synthetic epoch5 checkpoint carries the former pilot contract.
    # It must be refused before restoring model state or performing any update.
    old_contract = core.load_verified(continuous_path / 'checkpoints/epoch_00005.pt')
    old_contract['contract'] = copy.deepcopy(old_contract['contract'])
    old_contract['contract']['config']['schema'] = 'mx-triple-group16-scale64-pilot5-protocol-v1'
    old_contract['contract']['config']['pilot_stop_epoch'] = 5
    del old_contract['contract']['config']['reference_end_epoch']
    old_contract_file = out / 'old_pilot_contract_fixture.pt'
    core.atomic_save(old_contract_file, old_contract)
    old_contract_hash = core.file_sha256(old_contract_file)
    rejected_path, rejected_report = run_case('reject_old_pilot', old_contract_file, expected_status='FAIL')
    assert 'Hervatprotocol/omgeving/bron verschilt' in rejected_report['error']
    assert cases['reject_old_pilot']['optimizer_updates_in_process'] == 0
    assert not list((rejected_path / 'checkpoints').glob('*.pt'))
    assert core.file_sha256(old_contract_file) == old_contract_hash
    assert s.legacy.state_digest(contexts['reject_old_pilot'].model.state_dict()) \
        == cases['reject_old_pilot']['fresh_model_sha256_before_optional_restore']

    wrong_scale64 = core.load_verified(continuous_path / 'checkpoints/epoch_00005.pt')
    wrong_scale64['contract'] = copy.deepcopy(wrong_scale64['contract'])
    wrong_scale64['contract']['config']['mx_block_size'] = 32
    wrong_file = out / 'wrong_group32_reference100_contract_fixture.pt'
    core.atomic_save(wrong_file, wrong_scale64)
    wrong_hash = core.file_sha256(wrong_file)
    wrong_path, wrong_report = run_case('reject_group32_reference100', wrong_file, expected_status='FAIL')
    assert 'Hervatprotocol/omgeving/bron verschilt' in wrong_report['error']
    assert cases['reject_group32_reference100']['optimizer_updates_in_process'] == 0
    assert not list((wrong_path / 'checkpoints').glob('*.pt'))
    assert core.file_sha256(wrong_file) == wrong_hash
    assert s.legacy.state_digest(contexts['reject_group32_reference100'].model.state_dict()) \
        == cases['reject_group32_reference100']['fresh_model_sha256_before_optional_restore']

    completed_path, completed_report = run_case('reject_completed100', final_checkpoint, expected_status='FAIL')
    assert 'geen nieuwe referentie-epochs' in completed_report['error']
    assert cases['reject_completed100']['optimizer_updates_in_process'] == 0
    assert not list((completed_path / 'checkpoints').glob('*.pt'))

    # Independent selection fixture: lower modes peak elsewhere, and the high
    # mode ties. All modes must still inherit the first E4M3 maximum at epoch2.
    competing_scores = [(10, 20, 30), (40, 50, 80), (99, 60, 80), (70, 100, 70), (95, 90, 79)]
    selection_fixture = [
        {'epoch': epoch, 'test': [{'weight_bits': bit, 'top1_percent': score}
                                 for bit, score in zip(MODES, scores)]}
        for epoch, scores in enumerate(competing_scores, 1)]
    assert t.best_record(selection_fixture)['epoch'] == 2

    cli_results = {}
    rejected_arguments = [['run', '--stop-after-epoch', '5']]
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
        'scope': 'Real triple-group16-scale64 reference100 main/model/loop/exports on three synthetic CPU images only',
        'device': 'CPU; test-only Tensor.cuda redirection',
        'no_SVHN_data_loaded_or_downloaded': True,
        'not_a_SVHN_accuracy_or_GPU_timing_result': True,
        'notebook_executed': False,
        'environment': contexts['continuous100'].rt.environment,
        'production_config': production_config,
        'comparison_to_group16_scale64_pilot': config_comparison,
        'unchanged_core_validation': core_reuse,
        'fixture_changes': {'train_examples': 3, 'test_examples': 3, 'batch_size': 2},
        'initialization_reference': {
            'builder': 'unchanged joint_support.build_model',
            'same_runtime_seed_and_CPU_context': True,
            'initial_model_sha256': direct_initial_sha256,
        },
        'package_sha256': package_hashes, 'source_sha256': source_info['workcopy_files_sha256'],
        'test_file_sha256': core.file_sha256(Path(__file__)),
        'test_duration_seconds': time.perf_counter() - started,
        'total_synthetic_optimizer_updates': sum(case['optimizer_updates_in_process'] for case in cases.values()),
        'cases': cases, 'actual_LR_boundaries': boundary_lrs,
        'actual_LR_boundary_resume_checks': lr_resume_checks,
        'tests': {
            'real_main_default100_and_PASS_only_at100': True,
            '100_synthetic_epochs_200_updates_600_forwards_600_metrics': True,
            'short_final_batch_included_in_every_mode': True,
            'three_forwards_two_teacher_checks_one_update_per_batch': True,
            'one_E4_source_per_MX_layer_per_batch_no_E2_encoder': True,
            'lower_scale_codes_exact_source_plus6_then_mantissa_scale_copied': True,
            'all_lower_tensor_and_element_encoders_forbidden_during_real_run_and_resume': True,
            'independent_forward_and_identity_STE_no_extra64_gradient': numerical_checks,
            'source_caches_empty_before_Adam_and_after_every_phase': True,
            'test_prediction_histograms_all_modes_and_labels_complete': True,
            'A32_FP32_master_Adam_and_three_BN_audits': True,
            'train_recursive_teacher_labels_test_all_hard_CE': True,
            'highest_E4M3_first_strict_maximum_not_lower_mode_peak': True,
            'six_selected_metric_rows_from_common_best_or_epoch100_checkpoint': True,
            'two_high_sources_and_four_bit_derived_exports_without_masters': True,
            'all_export_modes_reconstruct_from_E4_source_with_encoders_and_G_forbidden': True,
            'feedback_and_unexecuted_notebook_with_unchanged_scale64_reference_copied': True,
            'resume99_to100_model_optimizer_scheduler_rng_exact_CPU': True,
            'unchanged_50_75_90_LR_schedule_all_four_phases': True,
            'resume49_to50_74_to75_89_to90_cross_actual_LR_boundaries_exact_CPU': True,
            'early_stop_complete_epoch_without_false_PASS_or100_result': True,
            'old_pilot_contract_rejected_before_restore_and_updates': True,
            'group32_reference100_contract_rejected_before_restore_and_updates': True,
            'completed100_resume_rejected_before_updates': True,
            'fresh_seed_state_identical_before_every_optional_restore': True,
            'initial_state_matches_unchanged_direct_builder_same_CPU_runtime': True,
            'production_config_package_source_and_fixture_data_unchanged': True,
            'production_CLI_refusals': cli_results,
            'rejected_CLI_created_no_production_output': True,
        },
    }
    report_path = out / 'training_integration_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(report_path),
                      'test_duration_seconds': report['test_duration_seconds'],
                      'total_synthetic_optimizer_updates': report['total_synthetic_optimizer_updates'],
                      'not_a_SVHN_accuracy_or_GPU_timing_result': True}, indent=2), flush=True)
    return report


def check_entry(destination: Path):
    """Actual main/check; adapt only device selection in this separate test process."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    production_runtime = s.runtime
    before = s.verify_package()
    def cpu_runtime(seed, requested_device):
        assert requested_device in ('cuda', 'cuda:0')
        return production_runtime(seed, 'cpu', strict=False)
    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor
    with (destination / 'check_stdout.log').open('x', encoding='utf-8') as stdout:
        with redirect_stdout(stdout), patch.object(s, 'runtime', cpu_runtime), \
             patch.object(torch.Tensor, 'cuda', cpu_cuda), \
             patch.object(sys, 'argv', [str(ROOT / 'run_train.py'), 'check',
                                        '--output-root', str(destination / 'output')]):
            exit_code = r.main()
    paths = list((destination / 'output').glob('*_joint_check'))
    assert len(paths) == 1
    report = json.loads((paths[0] / 'run_report.json').read_text(encoding='utf-8'))
    assert exit_code == 0 and report['status'] == 'PASS', report.get('traceback')
    assert report['synthetic_checks_only'] and report['dataset_downloaded'] is False
    assert report['codec_tests']['status'] == 'PASS'
    assert report['synthetic_training_checks']['status'] == 'PASS'
    assert (paths[0] / 'codec_report.json').is_file()
    assert list(paths[0].glob('*check_report.json'))
    assert (paths[0] / 'feedback_bundle.zip').is_file()
    assert s.verify_package() == before
    result = {'status': 'PASS', 'synthetic_fixture_only': True,
              'scope': 'Actual main/check entry with CPU-only runtime adaptation',
              'no_real_SVHN_data_or_training': True, 'synthetic_training_checks_executed': True, 'CUDA_tested_here': False,
              'environment': report['environment'], 'package_sha256': before,
              'codec_status': report['codec_tests']['status'],
              'truncation_training_status': report['synthetic_training_checks']['status'],
              'production_files_unchanged': True,
              'test_file_sha256': core.file_sha256(Path(__file__)),
              'check_report_directory': str(paths[0])}
    (destination / 'check_entry_validation_report.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'],
                      'report': str(destination / 'check_entry_validation_report.json')}))
    return result


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--check-entry':
        check_entry(Path(sys.argv[2]).resolve())
    elif len(sys.argv) == 2:
        main(Path(sys.argv[1]).resolve())
    else:
        raise SystemExit('Usage: test_training_integration.py [--check-entry] NEW_OUTPUT_DIRECTORY')
