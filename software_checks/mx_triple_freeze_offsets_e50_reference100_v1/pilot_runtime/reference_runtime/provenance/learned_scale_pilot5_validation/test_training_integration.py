"""Learned binary-scale pilot runner integration using three synthetic CPU images.

The production CLI remains CUDA/Python3.13 only. This harness substitutes only
context/data construction, final fixture-data verification and Tensor.cuda CPU
routing. Real main, loop, loss, optimizer, scheduler, checkpoint/restore, exports
and feedback assembly are exercised. No actual SVHN training is performed.
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

NOTEBOOK = 'MX_triple_learned_scale32_64_pilot5_resultaten.ipynb'
MODES = (4, 6, 8)
FORMATS = {4: 'MXFP4_E2M1', 6: 'MXFP6_E2M3', 8: 'MXFP8_E4M3'}
TEACHERS = {4: 'MXFP6_E2M3', 6: 'MXFP8_E4M3', 8: None}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def independent_offsets(choice, scale_shape):
    """Unpack exported bit choices without the production unpacking helper."""
    assert choice['schema'] == 'mx-binary-scale-choice-v1'
    assert choice['shape'] == list(scale_shape)
    assert choice['num_blocks'] == scale_shape[0] * scale_shape[1]
    assert choice['bit_order'] == 'little' and choice['mapping'] == {'0': 5, '1': 6}
    packed = choice['packed_choices']
    n = choice['num_blocks']
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == ((n + 7) // 8,)
    bits = ((packed.to(torch.int64)[:, None] >> torch.arange(8)) & 1).reshape(-1)
    assert not bool(bits[n:].any())
    return (bits[:n] + 5).reshape(scale_shape).to(torch.int32)


def verify_exports(directory, exports, epoch, modules):
    """Read actual files and derive expected fields from stored E4+choice bits."""
    assert set(exports) == {'4', '6', '8'}
    loaded = {}
    for bit in MODES:
        detail = exports[str(bit)]
        assert detail['epoch'] == epoch and detail['weight_bits'] == bit
        assert detail['decoded_weights_exact'] and detail['six_fp32_masters_absent']
        assert detail['truncation'] is (bit != 8)
        assert detail['source_scale_bytes_copied'] is (bit == 8)
        assert detail['full_test_forward_rerun'] is False
        export_path = directory / detail['file']
        assert core.file_sha256(export_path) == detail['sha256']
        assert export_path.with_suffix(export_path.suffix + '.sha256').read_text().strip() == detail['sha256']
        loaded[bit] = a.load_verified_payload(export_path)
        aux = loaded[bit]['auxiliary_fp32_state']
        assert all(name + '.weight' not in aux for name in a.TARGET_LAYERS)
        assert not any('.quantize_fn.theta' in name for name in aux)
        assert loaded[bit]['weight_bits'] == bit
        assert loaded[bit]['bn_bank_key'] == a.BANK[bit]
        assert loaded[bit]['provenance']['source_epoch'] == epoch
    choices = loaded[8]['scale_choices']
    assert set(choices) == set(a.TARGET_LAYERS)
    assert sum(v['num_blocks'] for v in choices.values()) == 4080
    assert sum(v['packed_choices'].numel() for v in choices.values()) == 510
    seen = set()
    for name in a.TARGET_LAYERS:
        h = a.codec(8).MXTensor.from_payload(loaded[8]['mx_layers'][name])
        mid = a.codec(6).MXTensor.from_payload(loaded[6]['mx_layers'][name])
        low = a.codec(4).MXTensor.from_payload(loaded[4]['mx_layers'][name])
        u8 = h.codes.long()
        expected6 = ((u8 >> 7) << 5) | (((u8 >> 5) & 3) << 3) | (u8 & 7)
        assert torch.equal(mid.codes.long(), expected6)
        assert torch.equal(low.codes.long(), expected6 >> 2)
        offsets = independent_offsets(choices[name], tuple(h.scale_codes.shape))
        seen.update(offsets.reshape(-1).tolist())
        shifted = h.scale_codes.to(torch.int32) + offsets
        assert bool((shifted <= 254).all())
        assert torch.equal(mid.scale_codes.to(torch.int32), shifted)
        assert torch.equal(low.scale_codes, mid.scale_codes)
        assert h.shape == mid.shape == low.shape
    assert seen == {5, 6}, 'Mixed-scale fixture must exercise both packed choices.'
    def forbidden(*args, **kwargs):
        raise AssertionError('Source-only reconstruction attempted encoder or G(W).')
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
            assert not any('.quantize_fn.theta' in name for name in source_model.state_dict())
            with torch.no_grad():
                source_logits, own_logits = source_model(images), own_model(images)
            assert source_logits.shape == (2, 10) and torch.isfinite(source_logits).all()
            assert torch.equal(source_logits, own_logits)
            assert all(not module.training for module in source_model.modules())
            assert all(not p.requires_grad and p.grad is None for p in source_model.parameters())
            assert s.legacy.state_digest(source_model.state_dict()) == before


def compare_prior_config(current):
    """Backbone, losses and schedule remain the same as the actual fixed pilot."""
    source = ROOT / 'provenance/scale32_pilot5_run_train.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'config')
    namespace = {'s': s, 't': t}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    previous = namespace['config']()
    unchanged = ['mode', 'weight_format', 'weight_bits', 'weight_formats', 'codec_revision',
                 'weight_preprocess', 'activation_bits', 'activation_policy', 'master_dtype',
                 'gradient_dtype', 'adam_moments_dtype', 'mx_block_size', 'mx_scale_format',
                 'mx_group_axis', 'mx_rounding', 'numeric_kernel', 'quantization_source',
                 'source_weight_bits', 'exponent_mapping', 'direct_lower_quantization',
                 'exponent_bias_compensation', 'distillation', 'loss', 'passes_per_training_batch',
                 'optimizer_steps_per_training_batch', 'training_mode_order', 'test_mode_order',
                 'bn_bank_mapping', 'batchnorm_policy', 'extra_FP32_teacher_forward',
                 'dataset', 'train_split', 'test_split', 'extra_used', 'train_examples',
                 'test_examples', 'validation_examples', 'model', 'expand', 'planned_epochs',
                 'pilot_stop_epoch', 'batch_size', 'optimizer', 'lr', 'weight_decay',
                 'scheduler_milestones', 'scheduler_gamma', 'workers', 'pin_memory',
                 'drop_last', 'train_shuffle', 'test_shuffle', 'seed', 'evaluation_policy',
                 'selection_policy', 'backend_policy', 'rng_policy']
    for key in unchanged:
        assert current[key] == previous[key], (key, previous[key], current[key])
    changed = {key: {'previous': previous.get(key), 'current': current.get(key)}
               for key in sorted(set(previous) | set(current)) if previous.get(key) != current.get(key)}
    return {'unchanged_core_training_keys': unchanged, 'differences': changed,
            'archived_runner_sha256': core.file_sha256(source)}


def validate_production_config(c):
    expected = {
        'schema': 'mx-triple-learned-scale32-64-pilot5-protocol-v1',
        'planned_epochs': 100, 'pilot_stop_epoch': 5, 'seed': 20261009,
        'scheduler_milestones': [50, 75, 90], 'scheduler_gamma': .1, 'lr': .001,
        'scale_lr': .001, 'weight_bits': [4, 6, 8], 'training_mode_order': [8, 6, 4],
        'test_mode_order': [4, 6, 8], 'passes_per_training_batch': 3,
        'optimizer_steps_per_training_batch': 1, 'activation_bits': 32,
        'master_dtype': 'float32', 'gradient_dtype': 'float32', 'adam_moments_dtype': 'float32',
        'train_examples': 73257, 'test_examples': 26032, 'batch_size': 128,
        'extra_used': False, 'distillation': True, 'truncation': True,
        'mantissa_truncation': True, 'exponent_truncation': True,
        'shared_scale_compensation': True, 'scale_exponent_offset_choices': [5, 6],
        'scale_factor_choices': [32, 64], 'learned_shared_scale': True,
        'parameters': 137034, 'base_parameters': 132954, 'learned_scale_parameters': 4080,
        'extra_FP32_teacher_forward': False, 'direct_lower_quantization': False,
        'exponent_bias_compensation': False,
    }
    for key, value in expected.items():
        assert c[key] == value, (key, c[key], value)
    assert tuple(a.MODES) == MODES
    assert 'reference_end_epoch' not in c


def initialization_and_projection_checks(modules, config):
    rt = s.runtime(config['seed'], 'cpu', strict=False)
    direct = direct_support.build_model(modules, 'joint', rt.device)
    direct_state = copy.deepcopy(direct.state_dict())
    direct_hash = s.legacy.state_digest(direct_state)
    direct_rng = core.capture_rng({}, rt.device)
    rt = s.runtime(config['seed'], 'cpu', strict=False)
    learned = t.build_model(modules, 'joint', rt.device)
    assert not core.differences(a.base_state_dict(learned), direct_state)
    assert not core.differences(core.capture_rng({}, rt.device), direct_rng)
    scales = a.scale_parameters(learned)
    assert len(scales) == 6 and sum(p.numel() for p in scales) == 4080
    assert sum(p.numel() for p in learned.parameters()) == 137034
    assert all(p.dtype == torch.float32 and bool((p == 6).all()) for p in scales)
    with torch.no_grad():
        for name in a.TARGET_LAYERS:
            layer = learned.get_submodule(name)
            effective = a.effective_reference_weight(layer.weight)
            high = a.codec(8).encode_tensor(effective)
            u = high.codes.long()
            c6 = (((u >> 7) << 5) | (((u >> 5) & 3) << 3) | (u & 7)).to(torch.uint8)
            lower_scales = (high.scale_codes.to(torch.int32) + 6).to(torch.uint8)
            expected = {8: high.decode(),
                        6: a.codec(6).MXTensor(c6, lower_scales, high.shape).decode(),
                        4: a.codec(4).MXTensor(c6 >> 2, lower_scales, high.shape).decode()}
            for bit in MODES:
                a.select_mode(learned, bit)
                assert torch.equal(layer.quantize_fn(layer.weight), expected[bit])
    optimizer = t.build_optimizer(modules, learned, config)
    assert [p['name'] for p in optimizer.param_groups] == ['weights', 'scale_proxy']
    assert all(p['lr'] == .001 for p in optimizer.param_groups)
    ids = [{id(p) for p in group['params']} for group in optimizer.param_groups]
    assert not ids[0] & ids[1] and ids[1] == {id(p) for p in scales}
    theta = scales[0]
    with torch.no_grad():
        theta.reshape(-1)[0] = 5.000001
        theta.reshape(-1)[1] = 5.999999
    theta.grad = torch.zeros_like(theta)
    theta.grad.reshape(-1)[0] = 1
    theta.grad.reshape(-1)[1] = -1
    optimizer.step()
    assert float(theta.detach().reshape(-1)[0]) == 5 and float(theta.detach().reshape(-1)[1]) == 6
    assert learned._last_scale_projection['projected_values'] == 2
    assert all(bool(((p >= 5) & (p <= 6)).all()) for p in scales)
    return {'base_initial_model_sha256': direct_hash, 'no_theta_RNG_draw': True,
            'all_theta_init6_and_fixed64_exact_forward': True,
            'two_disjoint_optimizer_groups': True,
            'real_Adam_post_hook_projects_both_bounds': True}


def set_mixed_fixture_theta(model):
    """Explicit test-only setup; real research starts entirely at theta=6."""
    pattern = torch.tensor([5.000001, 5.4999, 5.5001, 5.999999], dtype=torch.float32)
    with torch.no_grad():
        for p in a.scale_parameters(model):
            p.copy_(pattern.repeat((p.numel() + 3) // 4)[:p.numel()].reshape_as(p))


def verify_scale_tables(directory, trace):
    """Crosscheck human-readable scale data against actual checkpoint history."""
    records = trace['epochs']
    states = {0: trace['initial_scale_state']}
    states.update({rec['epoch']: rec['scale_state'] for rec in records})
    rows = read_csv(directory / 'scale_block_state.csv')
    assert len(rows) == 4080 * len(states)
    seen = set()
    for row in rows:
        epoch, layer = int(row['epoch']), row['layer']
        channel, block = int(row['output_channel']), int(row['block_index'])
        key = (epoch, layer, channel, block)
        assert key not in seen
        seen.add(key)
        value = states[epoch][layer][channel][block]
        assert float(row['theta']) == value
        offset = 5 if value < 5.5 else 6
        assert int(row['offset']) == offset and int(row['factor']) == 2 ** offset
        assert 5 <= value <= 6
    epoch_rows = read_csv(directory / 'scale_epoch_metrics.csv')
    assert len(epoch_rows) == len(records) * 6
    for row in epoch_rows:
        epoch, layer = int(row['epoch']), row['layer']
        current = torch.tensor(states[epoch][layer], dtype=torch.float64)
        prior = torch.tensor(states[epoch - 1][layer], dtype=torch.float64)
        count32 = int((current < 5.5).sum())
        net_changes = int(((current < 5.5) != (prior < 5.5)).sum())
        assert int(row['blocks']) == current.numel()
        assert int(row['count_x32']) == count32
        assert int(row['count_x64']) == current.numel() - count32
        assert abs(float(row['fraction_x32']) - count32 / current.numel()) < 1e-14
        assert float(row['theta_min']) == float(current.min())
        assert float(row['theta_max']) == float(current.max())
        assert int(row['net_choice_changes']) == net_changes
        assert int(row['choice_flips_during_epoch']) >= int(row['unique_blocks_switched_during_epoch']) >= net_changes
        assert int(row['unique_blocks_switched_during_epoch']) <= current.numel()
        assert int(row['projected_values_during_epoch']) >= 0
        assert float(row['scale_grad_l2_max']) >= float(row['scale_grad_l2_min']) >= 0
        assert float(row['scale_lr']) == .001
    return {'block_state_rows': len(rows), 'epoch_metric_rows': len(epoch_rows),
            'all_rows_match_trace_and_hard_threshold': True}


def main(out: Path) -> dict:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    package_hashes = s.verify_package()
    production_config = r.config()
    validate_production_config(production_config)
    config_comparison = compare_prior_config(production_config)
    reference_files = {path.relative_to(ROOT / 'fixed_references').as_posix(): core.file_sha256(path)
                       for path in (ROOT / 'fixed_references').rglob('*') if path.is_file()}
    assert len(reference_files) == 14
    assert all(name + '/REFERENCE.json' in reference_files for name in ['scale32', 'scale64'])

    prepared = out / 'source'
    prepared.mkdir()
    source, source_info = s.legacy.prepare_source(prepared)
    modules = s.import_source(source)
    initial_checks = initialization_and_projection_checks(modules, production_config)
    direct_initial_sha256 = initial_checks['base_initial_model_sha256']
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(3, 3, 40, 40, generator=generator) * 2 - 1
    labels = torch.tensor([0, 1, 2])
    datasets = {phase: torch.utils.data.TensorDataset(images.clone(), labels.clone())
                for phase in ('train', 'test')}
    cases, contexts = {}, {}

    def fixture_context(args, path, report, log, check_only=False):
        assert not check_only, 'This fixture tests run/resume only.'
        assert args.stop_after_epoch == 5 and not args.download
        rt = s.runtime(production_config['seed'], 'cpu', strict=False)
        c = r.config()
        c.update(train_examples=3, test_examples=3, batch_size=2,
                 purpose='SYNTHETIC_CPU_LEARNED_SCALE32_64_PILOT5_INTEGRATION_ONLY',
                 synthetic_initial_theta_mixed=True)
        model = t.build_model(modules, 'joint', rt.device)
        assert s.legacy.state_digest(a.base_state_dict(model)) == direct_initial_sha256
        assert all(bool((v == 6).all()) for v in a.scale_parameters(model))
        set_mixed_fixture_theta(model)
        optimizer = t.build_optimizer(modules, model, c)
        scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
        case_name = path.parent.name
        case = cases[case_name]

        def count_update(opt, hook_args, hook_kwargs):
            case['optimizer_updates_in_process'] += 1
            assert all(bool(torch.isfinite(p).all()) and bool(((p >= 5) & (p <= 6)).all())
                       for p in a.scale_parameters(model))
            case['projected_values_observed'] += model._last_scale_projection['projected_values']

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
        report['limitations'].append(
            'SYNTHETIC CPU FIXTURE: three fake images, batch2, deliberately mixed theta around threshold; no SVHN result.')
        ctx = SimpleNamespace(rt=rt, c=c, source=source, source_info=source_info,
                              modules=modules, model=model, optimizer=optimizer,
                              scheduler=scheduler, loaders=loaders, contract=contract, report=report)
        contexts[case_name] = ctx
        case['fresh_model_sha256_before_optional_restore'] = s.legacy.state_digest(model.state_dict())
        if case.get('stop_at_epoch') == 1:
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
        raise AssertionError('Learned binary-scale truncation training invoked a direct lower encoder.')

    def run_case(name, checkpoint=None, stop_at_epoch=None, expected_status='PASS'):
        cases[name] = {'optimizer_updates_in_process': 0, 'stop_at_epoch': stop_at_epoch,
                       'projected_values_observed': 0}
        arguments = ['resume' if checkpoint else 'run', '--output-root', str(out / name)]
        if checkpoint is not None:
            arguments += ['--checkpoint', str(checkpoint)]
        real_write_json = s.write_json

        def write_with_stop_marker(path, value):
            result = real_write_json(path, value)
            if stop_at_epoch and path.parent.name == 'epoch_summaries' \
                    and path.name == f'epoch_{stop_at_epoch:05d}.json':
                (path.parent.parent / 'STOP_AFTER_EPOCH').touch(exist_ok=True)
            return result

        print('SYNTHETIC learned-scale32/64 pilot5 runner case: ' + name, flush=True)
        with (out / (name + '_stdout.log')).open('x', encoding='utf-8') as stdout:
            with redirect_stdout(stdout), patch.object(sys, 'argv', [str(ROOT / 'run_train.py'), *arguments]), \
                 patch.object(r, 'create_context', fixture_context), \
                 patch.object(r, 'verify_unchanged', verify_fixture_unchanged), \
                 patch.object(s, 'write_json', write_with_stop_marker), \
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
        assert report['schema'] == 'mx-triple-learned-scale32-64-pilot5-report-v1'
        assert (path / 'feedback_bundle.zip').is_file()
        cases[name].update(status=report['status'], exit_code=exit_code, output_directory=str(path))
        return path, report

    continuous_path, continuous_report = run_case('continuous5')
    assert continuous_report['initial_base_model_sha256'] == direct_initial_sha256
    trace = json.loads((continuous_path / 'history.json').read_text(encoding='utf-8'))
    records = trace['epochs']
    scale_table_check = verify_scale_tables(continuous_path, trace)
    assert [record['epoch'] for record in records] == [1, 2, 3, 4, 5]
    assert continuous_report['optimizer_steps'] == 10
    assert continuous_report['training_forward_passes'] == 30
    assert cases['continuous5']['optimizer_updates_in_process'] == 10
    assert continuous_report['pilot_completed'] is True
    assert continuous_report['full_training_completed'] is False
    summary = continuous_report['training_summary']
    assert summary['pilot_completed'] and not summary['full_100_epoch_training_completed']
    assert summary['expected_optimizer_steps_at_pilot_completion'] == 10
    assert summary['expected_training_forward_passes_at_pilot_completion'] == 30
    elapsed = sum(record['total_epoch_seconds'] for record in records)
    assert continuous_report['timing_estimate']['measured_100_epoch_seconds'] is None
    assert continuous_report['timing_estimate']['measured_completed_epoch_seconds'] == elapsed
    assert continuous_report['timing_estimate']['full_100_epoch_training_completed'] is False
    assert 'not measured or remaining time' in continuous_report['timing_estimate']['interpretation']
    assert len(list((continuous_path / 'checkpoints').glob('epoch_*.pt'))) == 5
    assert len(read_csv(continuous_path / 'epoch_metrics.csv')) == 30
    assert len(read_csv(continuous_path / 'weight_updates.csv')) == 5 * 6 * 3
    assert len(list((continuous_path / 'sample_order').glob('*.npy'))) == 10
    prediction_rows = read_csv(continuous_path / 'test_prediction_counts.csv')
    assert len(prediction_rows) == 15
    for row in prediction_rows:
        counts = records[int(row['epoch']) - 1]['test_prediction_counts'][row['weight_bits']]
        assert counts == [int(row['predicted_class_' + str(k)]) for k in range(10)]
        assert int(row['classes_predicted']) == sum(value > 0 for value in counts)
        assert int(row['dominant_predicted_class']) == counts.index(max(counts))
        assert float(row['dominant_prediction_fraction']) == max(counts) / sum(counts)

    for record in records:
        epoch = record['epoch']
        assert record['lr_used'] == record['lr_next'] == .001
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
            assert source_audit['scale_offsets_learned'] == 24
            assert source_audit['mantissa_scale_copies_unchanged'] == 12
            assert source_audit['offset_choices'] == [5, 6] and source_audit['scale_factors'] == [32, 64]
            assert source_audit['learned_shared_scale'] is True
            for flag in ('source_cache_cleared', 'no_direct_lower_quantizer', 'cache_cleared_before_adam'):
                assert source_audit[flag]
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
                assert row['weight_format'] == FORMATS[bit] and row['bn_bank_key'] == a.BANK[bit]
                assert row['teacher_weight_format'] == (TEACHERS[bit] if phase == 'train' else None)
                assert row['objective_type'] == (
                    'recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy')
                assert abs(row['top1_percent'] - 100 * row['correct1'] / 3) < 1e-4
        assert {row['weight_bits'] for row in record['weight_updates']} == set(MODES)
        assert all(row['master_dtype'] == 'torch.float32' for row in record['weight_updates'])

    scores = [record['test'][-1]['top1_percent'] for record in records]
    expected_best = scores.index(max(scores)) + 1
    selection = continuous_report['checkpoint_selection']
    assert selection['best']['epoch'] == expected_best and selection['last']['epoch'] == 5
    assert selection['selection_mode'] == 'MXFP8_E4M3/A32'
    assert selection['all_modes_from_same_selected_checkpoint'] is True
    selected = read_csv(continuous_path / 'selected_metrics.csv')
    assert len(selected) == 6
    for role, epoch in [('best_E4M3_test', expected_best), ('last_epoch', 5)]:
        rows = [row for row in selected if row['role'] == role]
        assert sorted(int(row['weight_bits']) for row in rows) == list(MODES)
        assert {int(row['epoch']) for row in rows} == {epoch}
        for row in rows:
            metric = next(item for item in records[epoch - 1]['test'] if item['weight_bits'] == int(row['weight_bits']))
            assert float(row['top1_percent']) == metric['top1_percent']
        verify_exports(continuous_path, continuous_report['mx_exports'][role], epoch, modules)
    assert len(list(continuous_path.glob('mx_export_layers_*.csv'))) == 6
    final_checkpoint = continuous_path / 'checkpoints/epoch_00005.pt'
    final_state = core.load_verified(final_checkpoint)
    assert core.file_sha256(continuous_path / 'last_checkpoint.pt') == core.file_sha256(final_checkpoint)
    assert final_state['scheduler']['last_epoch'] == 5 and final_state['optimizer_steps'] == 10

    with ZipFile(continuous_path / 'feedback_bundle.zip') as archive:
        expected = {NOTEBOOK, 'notebook_support.py', 'training_summary.json', 'timing_estimate.json',
                    'run_report.json', 'checkpoint_selection.json', 'selected_metrics.csv',
                    'test_prediction_counts.csv'}
        expected.update(f'mx_export_layers_{bit}_{role}.csv' for bit in MODES for role in ('best', 'last'))
        expected.update('fixed_references/' + name for name in reference_files)
        expected.update(['scale_epoch_metrics.csv', 'scale_block_state.csv'])
        assert expected.issubset(archive.namelist()), sorted(expected - set(archive.namelist()))
        for name, digest in reference_files.items():
            assert core.file_sha256(continuous_path / 'fixed_references' / name) == digest
            import hashlib
            assert hashlib.sha256(archive.read('fixed_references/' + name)).hexdigest() == digest
    notebook = json.loads((continuous_path / NOTEBOOK).read_text(encoding='utf-8'))
    parameter_cells = [cell for cell in notebook['cells'] if 'parameters' in cell.get('metadata', {}).get('tags', [])]
    assert len(parameter_cells) == 1 and str(continuous_path.resolve()) in ''.join(parameter_cells[0]['source'])
    assert all(not cell.get('outputs') and cell.get('execution_count') is None
               for cell in notebook['cells'] if cell['cell_type'] == 'code')
    assert core.file_sha256(continuous_path / 'notebook_support.py') == core.file_sha256(ROOT / 'notebook_support.py')

    stopped_path, stopped_report = run_case('stop2', stop_at_epoch=2,
                                          expected_status='STOPPED_AT_EPOCH_BOUNDARY')
    assert stopped_report['completed_epoch'] == 2 and cases['stop2']['optimizer_updates_in_process'] == 4
    assert stopped_report['pilot_completed'] is False and stopped_report['full_training_completed'] is False
    assert stopped_report['timing_estimate']['measured_100_epoch_seconds'] is None
    checkpoint2 = stopped_path / 'checkpoints/epoch_00002.pt'
    checkpoint2_hash = core.file_sha256(checkpoint2)
    stopped_state = core.load_verified(checkpoint2)
    continuous2_state = core.load_verified(continuous_path / 'checkpoints/epoch_00002.pt')
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        assert not core.differences(continuous2_state[key], stopped_state[key]), key
    resumed_path, resumed_report = run_case('resume2_to5', checkpoint2)
    resumed_state = core.load_verified(resumed_path / 'checkpoints/epoch_00005.pt')
    assert cases['resume2_to5']['optimizer_updates_in_process'] == 6
    assert len(list((resumed_path / 'checkpoints').glob('epoch_*.pt'))) == 3
    assert all(resumed_report['restore_checks'].values())
    for key in ('model', 'optimizer', 'scheduler', 'rng'):
        differences = core.differences(final_state[key], resumed_state[key])
        assert not differences, (key, differences[:4])
    for before, after in zip(final_state['trace']['epochs'], resumed_state['trace']['epochs']):
        for key in ('train', 'test', 'model_sha256', 'lr_used', 'lr_next', 'best_epoch_so_far',
                    'weight_updates', 'gradient_checks', 'test_source_checks',
                    'test_prediction_counts', 'test_label_counts', 'scale_state', 'scale_epoch_metrics'):
            assert before[key] == after[key], (before['epoch'], key)
        for phase in ('train', 'test'):
            assert before['coverage'][phase]['order_sha256'] == after['coverage'][phase]['order_sha256']
    assert core.file_sha256(checkpoint2) == checkpoint2_hash
    assert resumed_report['pilot_completed'] is True
    assert resumed_report['training_summary']['resumed'] is True
    assert resumed_report['initial_model_sha256'] == continuous_report['initial_model_sha256']
    assert resumed_report['initial_base_model_sha256'] == continuous_report['initial_base_model_sha256']
    for name in ['scale_block_state.csv', 'scale_epoch_metrics.csv']:
        assert (resumed_path / name).read_bytes() == (continuous_path / name).read_bytes(), name
    verify_scale_tables(resumed_path, json.loads((resumed_path / 'history.json').read_text()))

    stopped1_path, stopped1_report = run_case('early_stop1', stop_at_epoch=1,
                                             expected_status='STOPPED_AT_EPOCH_BOUNDARY')
    assert stopped1_report['completed_epoch'] == 1
    assert cases['early_stop1']['optimizer_updates_in_process'] == 2
    assert stopped1_report['pilot_completed'] is False and stopped1_report['full_training_completed'] is False
    assert stopped1_report['timing_estimate']['measured_100_epoch_seconds'] is None
    assert (stopped1_path / 'checkpoints/epoch_00001.pt').is_file()
    assert len(read_csv(stopped1_path / 'selected_metrics.csv')) == 6

    rejection_checks = {}
    for name, changed in [
        ('wrong_fixed32_pilot', {'schema': 'mx-triple-scale32-pilot5-protocol-v1',
                                'learned_shared_scale': False, 'scale_exponent_offset': 5, 'scale_factor': 32}),
        ('wrong_fixed64_pilot', {'schema': 'mx-triple-scale64-pilot5-protocol-v1',
                                'learned_shared_scale': False, 'scale_exponent_offset': 6, 'scale_factor': 64}),
        ('wrong_scale_gradient_contract', {'scale_gradient': 'straight_line_between_candidates'}),
    ]:
        state = core.load_verified(checkpoint2)
        state['contract'] = copy.deepcopy(state['contract'])
        state['contract']['config'].update(changed)
        checkpoint = out / (name + '_fixture.pt')
        core.atomic_save(checkpoint, state)
        digest = core.file_sha256(checkpoint)
        rejected_path, rejected_report = run_case(name, checkpoint, expected_status='FAIL')
        assert 'Hervatprotocol/omgeving/bron verschilt' in rejected_report['error']
        assert cases[name]['optimizer_updates_in_process'] == 0
        assert not list((rejected_path / 'checkpoints').glob('*.pt'))
        assert core.file_sha256(checkpoint) == digest
        assert s.legacy.state_digest(contexts[name].model.state_dict()) \
            == cases[name]['fresh_model_sha256_before_optional_restore']
        rejection_checks[name] = {'rejected_before_restore_and_updates': True, 'source_unchanged': True}

    for name, completed_epoch in [('completed5', 5), ('completed100', 100)]:
        if completed_epoch == 5:
            checkpoint = final_checkpoint
        else:
            state = core.load_verified(final_checkpoint)
            state['completed_epoch'] = 100
            checkpoint = out / 'completed100_fixture.pt'
            core.atomic_save(checkpoint, state)
        digest = core.file_sha256(checkpoint)
        rejected_path, rejected_report = run_case('reject_' + name, checkpoint, expected_status='FAIL')
        assert 'geen nieuwe pilot-epochs' in rejected_report['error']
        assert cases['reject_' + name]['optimizer_updates_in_process'] == 0
        assert not list((rejected_path / 'checkpoints').glob('*.pt'))
        assert core.file_sha256(checkpoint) == digest
        rejection_checks[name] = {'rejected_before_updates': True, 'source_unchanged': True}

    # A first strict high-precision maximum controls every mode, including ties.
    competing_scores = [(10, 20, 30), (40, 50, 80), (99, 60, 80), (70, 100, 70), (95, 90, 79)]
    selection_fixture = [
        {'epoch': epoch, 'test': [{'weight_bits': bit, 'top1_percent': score}
                                 for bit, score in zip(MODES, scores)]}
        for epoch, scores in enumerate(competing_scores, 1)]
    assert t.best_record(selection_fixture)['epoch'] == 2

    cli_results = {}
    rejected_arguments = [['run', '--stop-after-epoch', value] for value in ['-1', '0', '4', '6', '100', '5.0']]
    rejected_arguments += [['run', '--device', 'cpu'], ['run', '--synthetic'],
                           ['run', '--mode', 'control'], ['resume'],
                           ['run', '--checkpoint', str(checkpoint2)],
                           ['run', '--warmstart', str(checkpoint2)]]
    for arguments in rejected_arguments:
        result = subprocess.run([sys.executable, str(ROOT / 'run_train.py'), *arguments],
                                cwd=out, text=True, capture_output=True)
        assert result.returncode == 2, (arguments, result.stdout, result.stderr)
        cli_results[' '.join(arguments)] = {'exit_code': result.returncode,
                                           'reason': result.stderr.splitlines()[-1]}
    assert not (out / 'mx_runs').exists()
    assert r.config() == production_config and s.verify_package() == package_hashes
    assert len({case['fresh_model_sha256_before_optional_restore'] for case in cases.values()}) == 1
    assert core.file_sha256(final_checkpoint) == core.file_sha256(continuous_path / 'last_checkpoint.pt')

    result = {
        'status': 'PASS', 'synthetic_fixture_only': True,
        'scope': 'Actual learned-scale32/64 pilot main/model/loop/exports on three synthetic CPU images only',
        'device': 'CPU; test-only Tensor.cuda redirection',
        'no_SVHN_data_loaded_or_downloaded': True, 'not_a_SVHN_accuracy_or_GPU_timing_result': True,
        'notebook_executed': False, 'environment': contexts['continuous5'].rt.environment,
        'production_config': production_config, 'comparison_to_fixed_scale32_pilot': config_comparison,
        'fixture_changes': {'train_examples': 3, 'test_examples': 3, 'batch_size': 2},
        'initialization_reference': initial_checks,
        'scale_tables_checked': scale_table_check,
        'package_sha256': package_hashes, 'source_sha256': source_info['workcopy_files_sha256'],
        'test_file_sha256': core.file_sha256(Path(__file__)),
        'test_duration_seconds': time.perf_counter() - started,
        'total_synthetic_optimizer_updates': sum(case['optimizer_updates_in_process'] for case in cases.values()),
        'cases': cases,
        'tests': {
            'real_main_default5_PASS_only_at5': True,
            'five_synthetic_epochs_10_updates_30_forwards_30_metrics': True,
            'short_final_batch_included_in_every_mode': True,
            'three_forwards_two_detached_teachers_one_update_per_batch': True,
            'one_E4_source_per_layer_per_batch_lower_encoders_forbidden': True,
            'independent_export_bits_and_packed_binary_offsets': True,
            'mixed32_and64_source_only_exports_without_theta_or_masters': True,
            'mantissa_scale_identical_to_E2M3': True,
            'source_caches_empty_before_Adam_and_after_phases': True,
            'all_modes_prediction_histograms_complete': True,
            'A32_FP32_master_Adam_three_BN_audits': True,
            'loss_semantics_match_original_recursive_distillation': True,
            'highest_E4M3_first_strict_maximum_shared_by_all_modes': True,
            'six_selected_metric_rows_common_best_and_epoch5': True,
            'source_only_export_reconstructs_same_logits_without_G_or_encoders': True,
            'feedback_and_blank_notebook_with_both_unchanged_fixed_references': True,
            'fresh_stop2_then_resume5_model_optimizer_scheduler_rng_exact_CPU': True,
            'fresh_stop2_equals_continuous_epoch2_exact_CPU': True,
            'unchanged50_75_90_schedule_no_LR_drop_during_pilot': True,
            'early_stop1_reports_partial_not_PASS_or100': True,
            'checkpoint_rejections': rejection_checks,
            'fresh_base_initial_state_matches_unchanged_direct_builder_no_extra_RNG': True,
            'all6_initial_forward_matches_fixed64_independent_bits': True,
            'theta_adam_state_and_scale_CSV_resume_exact_CPU': True,
            'theta_projected_after_every_real_Adam_update': True,
            'production_config_package_source_and_synthetic_inputs_unchanged': True,
            'production_CLI_refusals': cli_results,
            'rejected_CLI_created_no_production_output': True,
        },
    }
    report_path = out / 'training_integration_report.json'
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'report': str(report_path),
                      'test_duration_seconds': result['test_duration_seconds'],
                      'total_synthetic_optimizer_updates': result['total_synthetic_optimizer_updates'],
                      'not_a_SVHN_accuracy_or_GPU_timing_result': True}, indent=2), flush=True)
    return result


def check_entry(destination: Path):
    """Actual main/check with CPU selection adapted only in this test process."""
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
              'no_real_SVHN_data_or_training': True, 'synthetic_training_checks_executed': True,
              'CUDA_tested_here': False, 'environment': report['environment'], 'package_sha256': before,
              'codec_status': report['codec_tests']['status'],
              'truncation_training_status': report['synthetic_training_checks']['status'],
              'production_files_unchanged': True, 'test_file_sha256': core.file_sha256(Path(__file__)),
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

