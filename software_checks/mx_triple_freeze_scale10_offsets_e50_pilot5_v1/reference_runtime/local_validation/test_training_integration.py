"""Learned ten-scale full100 runner integration using three synthetic CPU images.

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
import math
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

NOTEBOOK = 'MX_triple_learned_scale10_reference100_resultaten.ipynb'
MODES = (4, 6, 8)
FORMATS = {4: 'MXFP4_E2M1', 6: 'MXFP6_E2M3', 8: 'MXFP8_E4M3'}
TEACHERS = {4: 'MXFP6_E2M3', 6: 'MXFP8_E4M3', 8: None}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def independent_offsets(choice, scale_shape):
    """Decode the documented nibble layout without production helper calls."""
    assert choice['schema'] == 'mx-ten-scale-choices-four-bit-v1'
    assert choice['shape'] == list(scale_shape)
    assert choice['num_blocks'] == scale_shape[0] * scale_shape[1]
    assert choice['bits_per_choice'] == 4
    assert choice['bit_order'] == 'little'
    assert choice['mapping'] == {str(i): i for i in range(10)}
    packed = choice['packed_choices']
    n = choice['num_blocks']
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == ((n + 1) // 2,)
    unpacked = ((packed.to(torch.int64)[:, None] >> torch.tensor([0, 4])) & 15).reshape(-1)
    assert not bool(unpacked[n:].any())
    assert bool(((unpacked[:n] >= 0) & (unpacked[:n] <= 9)).all())
    return unpacked[:n].reshape(scale_shape).to(torch.int32)


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
    assert sum(v['packed_choices'].numel() for v in choices.values()) == 2040
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
    assert seen == set(range(10)), 'Mixed-scale fixture must exercise all ten packed choices.'
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
        theta.reshape(-1)[0] = 0.000001
        theta.reshape(-1)[1] = 8.999999
    theta.grad = torch.zeros_like(theta)
    theta.grad.reshape(-1)[0] = 1
    theta.grad.reshape(-1)[1] = -1
    optimizer.step()
    assert float(theta.detach().reshape(-1)[0]) == 0 and float(theta.detach().reshape(-1)[1]) == 9
    assert learned._last_scale_projection['projected_values'] == 2
    assert all(bool(((p >= 0) & (p <= 9)).all()) for p in scales)
    return {'base_initial_model_sha256': direct_hash, 'no_theta_RNG_draw': True,
            'all_theta_init6_and_fixed64_exact_forward': True,
            'two_disjoint_optimizer_groups': True,
            'real_Adam_post_hook_projects_both_bounds': True}

def set_mixed_fixture_theta(model):
    """Test-only ten-choice fixture; production remains theta=6 everywhere."""
    pattern = torch.tensor([0.000001, 0.4999, 0.5001, 1.4999, 1.5001,
        2.4999, 2.5001, 3.4999, 3.5001, 4.4999, 4.5001, 5.4999,
        5.5001, 6.4999, 6.5001, 7.4999, 7.5001, 8.4999, 8.5001, 8.999999],
        dtype=torch.float32)
    with torch.no_grad():
        for p in a.scale_parameters(model):
            p.copy_(pattern.repeat((p.numel() + len(pattern)-1) // len(pattern))[:p.numel()].reshape_as(p))

def verify_scale_tables(directory, trace):
    """Independently crosscheck all ten histograms and per-group offsets."""
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
        offset = math.floor(value + 0.5)
        assert int(row['offset']) == offset and int(row['factor']) == 2 ** offset
        assert 0 <= value <= 9
    epoch_rows = read_csv(directory / 'scale_epoch_metrics.csv')
    assert len(epoch_rows) == len(records) * 6
    for row in epoch_rows:
        epoch, layer = int(row['epoch']), row['layer']
        current = torch.tensor(states[epoch][layer], dtype=torch.float64)
        prior = torch.tensor(states[epoch - 1][layer], dtype=torch.float64)
        offsets = torch.floor(current + 0.5).to(torch.int64)
        prior_offsets = torch.floor(prior + 0.5).to(torch.int64)
        net_changes = int((offsets != prior_offsets).sum())
        assert int(row['blocks']) == current.numel()
        for k in range(10):
            count = int((offsets == k).sum())
            assert int(row['count_x'+str(2**k)]) == count
            assert abs(float(row['fraction_x'+str(2**k)])-count/current.numel()) < 1e-14
        assert sum(int(row['count_x'+str(2**k)]) for k in range(10)) == current.numel()
        assert float(row['theta_min']) == float(current.min())
        assert float(row['theta_max']) == float(current.max())
        assert int(row['net_choice_changes']) == net_changes
        assert int(row['choice_flips_during_epoch']) >= int(row['unique_blocks_switched_during_epoch']) >= net_changes
        assert int(row['unique_blocks_switched_during_epoch']) <= current.numel()
        assert int(row['choice_jump_distance_during_epoch']) >= int(row['choice_flips_during_epoch'])
        assert 0 <= int(row['max_choice_jump']) <= 9
        assert int(row['choice_min_blocks']) == int((offsets == 0).sum())
        assert int(row['choice_max_blocks']) == int((offsets == 9).sum())
        assert int(row['theta_at_min_blocks']) == int((current == 0).sum())
        assert int(row['theta_at_max_blocks']) == int((current == 9).sum())
        assert int(row['projected_values_during_epoch']) >= 0
        assert float(row['scale_grad_l2_max']) >= float(row['scale_grad_l2_min']) >= 0
        assert float(row['scale_lr']) == expected_lr(epoch)
    summary = json.loads((directory/'scale_learning_summary.json').read_text())
    assert summary['packed_scale_choice_bytes'] == 2040
    return {'block_state_rows': len(rows), 'epoch_metric_rows': len(epoch_rows),
            'all_ten_choice_histograms_match_trace': True, 'packed_choice_bytes': 2040}

def expected_lr(epoch, after=False):
    value = .001
    for milestone in (50, 75, 90):
        if milestone <= epoch if after else milestone < epoch:
            value *= .1
    return value


def compare_prior_config(current):
    path = ROOT / 'provenance/scale10_pilot5_PROTOCOL.json'
    prior = json.loads(path.read_text(encoding='utf-8'))
    changed_keys = (set(prior) ^ set(current)) | {key for key in set(prior) & set(current)
                                              if prior[key] != current[key]}
    changes = {key: {'pilot_present': key in prior, 'full100_present': key in current,
                     'pilot': prior.get(key), 'full100': current.get(key)}
               for key in sorted(changed_keys)}
    assert set(changes) == {'schema', 'purpose', 'pilot_stop_epoch', 'reference_end_epoch',
                           'scale_learning_end_epoch', 'scale_freeze_epoch',
                           'reporting_policy'}, changes
    assert current['schema'] == 'mx-triple-learned-scale10-reference100-protocol-v1'
    assert current['reference_end_epoch'] == current['planned_epochs'] == 100
    assert current['scale_learning_end_epoch'] == 100 and current['scale_freeze_epoch'] is None
    assert current['scale_lr'] == current['lr'] == .001
    assert current['scheduler_milestones'] == [50, 75, 90]
    assert current['train_examples'] == 73257 and current['test_examples'] == 26032
    assert current['batch_size'] == 128 and current['parameters'] == 137034
    assert current['activation_bits'] == 32 and current['learned_scale_parameters'] == 4080
    assert current['scale_exponent_offset_choices'] == list(range(10))
    assert current['scale_factor_choices'] == [2**k for k in range(10)]
    assert current['scale_proxy_bounds'] == [0.0, 9.0] and current['scale_proxy_init'] == 6.0
    assert 'pilot_stop_epoch' not in current
    prior_hashes = json.loads((ROOT / 'provenance/scale10_pilot5_PACKAGE_SHA256.json').read_text())
    if 'files' in prior_hashes:
        prior_hashes = prior_hashes['files']
    unchanged_files = ['trunc_adapter.py', 'trunc_support.py', 'resume_core.py',
                       'svhn_support.py', 'smoke_support.py', 'joint_adapter.py',
                       'joint_support.py', 'mx_adapter.py', 'mx_format.py',
                       'mx_e2m1.py', 'mx_e4m3.py', 'posthoc_truncation.py',
                       'test_learned_scale.py', 'test_joint.py', 'test_e4m3.py', 'test_mx.py']
    for name in unchanged_files:
        assert core.file_sha256(ROOT / name) == prior_hashes[name], name
    return {'status': 'PASS', 'only_endpoint_and_reporting_metadata_change': True,
            'changes': changes, 'unchanged_numerical_core_files': unchanged_files,
            'archive_protocol_sha256': core.file_sha256(path)}


def runtime_hashes():
    """Bind tests to training files independently of an assembly-stage manifest.

    The fixture replaces production context construction. Its contract hashes
    cover the stable training runtime and protocol, but exclude notebook helpers,
    historical-reference presentation metadata and the final packaging manifest.
    Those are validated separately during package assembly/notebook validation.
    """
    files = [path for path in ROOT.glob('*.py') if path.name != 'notebook_support.py']
    files += [ROOT / 'PROTOCOL.json', ROOT / 'SOURCE_MANIFEST.json',
              ROOT / 'upstream/Any-Precision-DNNs-master.zip']
    return {path.relative_to(ROOT).as_posix(): core.file_sha256(path)
            for path in sorted(files)}



def main(out: Path) -> dict:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    package_hashes = runtime_hashes()
    production_config = r.config()
    config_comparison = compare_prior_config(production_config)
    reference_files = {p.relative_to(ROOT).as_posix(): core.file_sha256(p)
                       for d in ('fixed_references', 'pilot_reference')
                       for p in (ROOT / d).rglob('*') if p.is_file()}
    assert len([n for n in reference_files if n.startswith('fixed_references/')]) >= 25
    prepared = out / 'source'
    prepared.mkdir()
    source, source_info = s.legacy.prepare_source(prepared)
    modules = s.import_source(source)
    initial_checks = initialization_and_projection_checks(modules, production_config)
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(3, 3, 40, 40, generator=generator) * 2 - 1
    labels = torch.tensor([0, 1, 2])
    datasets = {phase: torch.utils.data.TensorDataset(images.clone(), labels.clone())
                for phase in ('train', 'test')}
    cases, contexts = {}, {}

    def fixture_context(args, path, report, log, check_only=False):
        assert not check_only and not args.download
        assert 1 <= args.stop_after_epoch <= 100
        rt = s.runtime(production_config['seed'], 'cpu', strict=False)
        c = r.config()
        c.update(train_examples=3, test_examples=3, batch_size=2,
                 purpose='SYNTHETIC_CPU_LEARNED_SCALE10_REFERENCE100_INTEGRATION_ONLY',
                 synthetic_initial_theta_mixed=True)
        model = t.build_model(modules, 'joint', rt.device)
        assert s.legacy.state_digest(a.base_state_dict(model)) == initial_checks['base_initial_model_sha256']
        assert all(bool((v == 6).all()) for v in a.scale_parameters(model))
        set_mixed_fixture_theta(model)
        optimizer = t.build_optimizer(modules, model, c)
        scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
        case_name = path.parent.name
        case = cases[case_name]

        def count_update(opt, hook_args, hook_kwargs):
            case['optimizer_updates_in_process'] += 1
            assert all(bool(torch.isfinite(p).all()) and bool(((p >= 0) & (p <= 9)).all())
                       for p in a.scale_parameters(model))
            case['projected_values_observed'] += model._last_scale_projection['projected_values']
            assert [g['name'] for g in opt.param_groups] == ['weights', 'scale_proxy']
            assert opt.param_groups[0]['lr'] == opt.param_groups[1]['lr'] > 0
            assert all(p.requires_grad for p in a.scale_parameters(model))

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
            'SYNTHETIC CPU FIXTURE: three fake images, batch2, mixed theta spanning all ten choices; no SVHN result.')
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
        assert runtime_hashes() == package_hashes
        assert all(torch.equal(d.tensors[0], images) and torch.equal(d.tensors[1], labels)
                   for d in datasets.values())
        ctx.report.update(source_unchanged=True, package_unchanged=True,
                          synthetic_input_tensors_unchanged=True)

    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor

    def forbid_lower_encoder(*args, **kwargs):
        raise AssertionError('Training invoked a direct lower encoder.')

    def run_case(name, checkpoint=None, stop_after=100, stop_marker=False, expected_status='PASS'):
        cases[name] = {'optimizer_updates_in_process': 0, 'stop_marker': stop_marker,
                       'projected_values_observed': 0}
        arguments = ['resume' if checkpoint else 'run', '--output-root', str(out / name)]
        if stop_after != 100:
            arguments += ['--stop-after-epoch', str(stop_after)]
        if checkpoint is not None:
            arguments += ['--checkpoint', str(checkpoint)]
        print('SYNTHETIC learned-scale10 full100 runner case: ' + name, flush=True)
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
        assert exit_code == (1 if expected_status == 'FAIL' else 0)
        assert report['synthetic_fixture_only'] and not report['dataset_downloaded']
        assert report['schema'] == 'mx-triple-learned-scale10-reference100-report-v1'
        assert (path / 'feedback_bundle.zip').is_file()
        cases[name].update(status=report['status'], exit_code=exit_code, output_directory=str(path))
        return path, report

    continuous_path, report = run_case('continuous100')
    assert report['initial_base_model_sha256'] == initial_checks['base_initial_model_sha256']
    trace = json.loads((continuous_path / 'history.json').read_text())
    records = trace['epochs']
    assert [rec['epoch'] for rec in records] == list(range(1,101))
    assert report['optimizer_steps'] == 200 and report['training_forward_passes'] == 600
    assert cases['continuous100']['optimizer_updates_in_process'] == 200
    assert report['full_training_completed'] is True and report.get('pilot_completed') is False
    summary = report['training_summary']
    assert summary['full_100_epoch_training_completed'] and summary['reference_end_epoch'] == 100
    assert summary['expected_optimizer_steps_at_reference_completion'] == 200
    assert summary['expected_training_forward_passes_at_reference_completion'] == 600
    elapsed = sum(rec['total_epoch_seconds'] for rec in records)
    assert report['timing_estimate']['measured_100_epoch_seconds'] == elapsed
    assert report['timing_estimate']['full_100_epoch_training_completed'] is True
    assert len(list((continuous_path / 'checkpoints').glob('epoch_*.pt'))) == 100
    assert len(read_csv(continuous_path / 'epoch_metrics.csv')) == 600
    assert len(read_csv(continuous_path / 'weight_updates.csv')) == 1800
    assert len(list((continuous_path / 'sample_order').glob('*.npy'))) == 200
    assert len(read_csv(continuous_path / 'test_prediction_counts.csv')) == 300
    scale_table_check = verify_scale_tables(continuous_path, trace)
    assert scale_table_check['block_state_rows'] == 4080 * 101
    assert scale_table_check['epoch_metric_rows'] == 600
    lr_boundaries = {}
    for rec in records:
        epoch = rec['epoch']
        assert rec['lr_used'] == rec['scale_lr_used'] == expected_lr(epoch)
        assert rec['lr_next'] == rec['scale_lr_next'] == expected_lr(epoch, after=True)
        assert rec['optimizer_steps'] == 2 * epoch
        assert rec['test_optimizer_steps'] == 0 and rec['batchnorm_counters_correct']
        audit = rec['gradient_checks']
        assert audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == audit['scale_optimizer_steps'] == 2
        assert audit['scale_projection_after_every_update'] is True
        assert audit['detached_teacher_checks'] == 4
        assert all(audit[key] == 2 for key in ('forward_calls_E4M3', 'forward_calls_E2M3', 'forward_calls_E2M1'))
        for phase, source_audit in [('train',audit['shared_source_checks']),('test',rec['test_source_checks'])]:
            assert source_audit['source_triplets'] == 2
            assert source_audit['high_encodings'] == source_audit['exponent_derivations'] == source_audit['mantissa_derivations'] == 12
            assert source_audit['scale_offsets_learned'] == 24 and source_audit['mantissa_scale_copies_unchanged'] == 12
            assert source_audit['offset_choices'] == list(range(10)) and source_audit['scale_factors'] == [2**k for k in range(10)]
            assert source_audit['learned_shared_scale']
            assert all(source_audit[k] for k in ('source_cache_cleared','no_direct_lower_quantizer','cache_cleared_before_adam'))
            assert source_audit['training_order'] == ([8,6,4] if phase == 'train' else [4,6,8])
        assert rec['adam_checks']['all_active_adam_steps_correct']
        assert rec['adam_checks']['moments_fp32_finite']
        assert rec['adam_checks']['unused_bn_has_no_optimizer_state']
        assert rec['test_label_counts'] == [1,1,1,0,0,0,0,0,0,0]
        for phase in ('train','test'):
            assert [m['weight_bits'] for m in rec[phase]] == list(MODES)
            assert rec['coverage'][phase]['complete_once'] and rec['coverage'][phase]['n'] == 3
            for metric in rec[phase]:
                bit = metric['weight_bits']
                assert metric['n'] == 3 and metric['activation_bits'] == 32
                assert metric['weight_format'] == FORMATS[bit] and metric['bn_bank_key'] == a.BANK[bit]
                assert metric['teacher_weight_format'] == (TEACHERS[bit] if phase == 'train' else None)
                assert metric['objective_type'] == ('recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy')
                assert abs(metric['top1_percent'] - 100 * metric['correct1'] / 3) < 1e-4
        if epoch in (49,50,51,74,75,76,89,90,91,100):
            lr_boundaries[str(epoch)] = {k:rec[k] for k in ('lr_used','lr_next','scale_lr_used','scale_lr_next')}
    scores = [rec['test'][-1]['top1_percent'] for rec in records]
    best = scores.index(max(scores)) + 1
    assert report['selected_best_epoch'] == best and report['selected_last_epoch'] == 100
    assert report['checkpoint_selection']['all_modes_from_same_selected_checkpoint']
    selected = read_csv(continuous_path / 'selected_metrics.csv')
    assert len(selected) == 6
    for role, epoch in [('best_E4M3_test',best),('last_epoch',100)]:
        assert {int(x['epoch']) for x in selected if x['role']==role} == {epoch}
        verify_exports(continuous_path, report['mx_exports'][role], epoch, modules)
    final_checkpoint = continuous_path / 'checkpoints/epoch_00100.pt'
    final_state = core.load_verified(final_checkpoint)
    assert final_state['scheduler']['last_epoch'] == 100 and final_state['optimizer_steps'] == 200
    assert core.file_sha256(continuous_path / 'last_checkpoint.pt') == core.file_sha256(final_checkpoint)
    with ZipFile(continuous_path / 'feedback_bundle.zip') as z:
        expected = {NOTEBOOK,'notebook_support.py','scale_epoch_metrics.csv','scale_block_state.csv',
                    'scale_learning_summary.json','training_summary.json','timing_estimate.json',
                    'run_report.json','selected_metrics.csv','epoch50_checkpoint.json'} | set(reference_files)
        assert expected.issubset(z.namelist())
        import hashlib
        for name,digest in reference_files.items():
            assert core.file_sha256(continuous_path/name)==digest
            assert hashlib.sha256(z.read(name)).hexdigest()==digest
    notebook = json.loads((continuous_path / NOTEBOOK).read_text())
    parameters = [c for c in notebook['cells'] if 'parameters' in c.get('metadata',{}).get('tags',[])]
    assert len(parameters)==1 and str(continuous_path) in ''.join(parameters[0]['source'])
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells'] if c['cell_type']=='code')

    # Resume the preserved epoch50 checkpoint; test restored first LR and both later transitions.
    checkpoint50 = continuous_path / 'checkpoints/epoch_00050.pt'
    checkpoint50_hash = core.file_sha256(checkpoint50)
    checkpoint50_state = core.load_verified(checkpoint50)
    meta50 = json.loads((continuous_path / 'epoch50_checkpoint.json').read_text())
    assert meta50 == report['epoch50_checkpoint'] == summary['epoch50_checkpoint']
    assert meta50['available'] and meta50['epoch'] == 50 and meta50['full_training_state']
    assert Path(meta50['original_checkpoint']).resolve() == checkpoint50.resolve()
    assert meta50['sha256'] == checkpoint50_hash
    assert not meta50['checkpoint_in_feedback_bundle']
    assert checkpoint50_state['completed_epoch'] == 50 and checkpoint50_state['next_epoch'] == 51
    assert checkpoint50_state['optimizer_steps'] == 100
    assert checkpoint50_state['scheduler']['last_epoch'] == 50
    groups50 = checkpoint50_state['optimizer']['param_groups']
    assert [group['name'] for group in groups50] == ['weights', 'scale_proxy']
    assert all(group['lr'] == expected_lr(51) > 0 for group in groups50)
    assert set(checkpoint50_state['rng']) == {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'loader_generators'}
    assert checkpoint50_state['rng']['torch_cuda'] == []  # This fixture is CPU only.
    theta_names = [name for name in checkpoint50_state['model'] if name.endswith('.quantize_fn.theta')]
    assert len(theta_names) == 6
    assert sum(checkpoint50_state['model'][name].numel() for name in theta_names) == 4080
    assert len(groups50[1]['params']) == 6
    for parameter_id in groups50[1]['params']:
        adam_state = checkpoint50_state['optimizer']['state'][parameter_id]
        assert adam_state['step'].item() == 100
        assert adam_state['exp_avg'].dtype == adam_state['exp_avg_sq'].dtype == torch.float32
        assert torch.isfinite(adam_state['exp_avg']).all() and torch.isfinite(adam_state['exp_avg_sq']).all()
    assert len(checkpoint50_state['trace']['epochs']) == 50
    assert checkpoint50_state['contract']['config']['scale_freeze_epoch'] is None
    assert checkpoint50_state['contract']['config']['scale_learning_end_epoch'] == 100
    assert any(not torch.equal(checkpoint50_state['model'][name], final_state['model'][name])
               for name in theta_names), 'Theta must remain trainable after epoch50 in this fixture.'
    assert all(a.scale_parameters(contexts['continuous100'].model)[i].requires_grad for i in range(6))
    resumed_path, resumed_report = run_case('resume50_to100', checkpoint50)
    resumed_state = core.load_verified(resumed_path / 'checkpoints/epoch_00100.pt')
    assert cases['resume50_to100']['optimizer_updates_in_process'] == 100
    assert len(list((resumed_path/'checkpoints').glob('epoch_*.pt'))) == 50
    assert all(resumed_report['restore_checks'].values())
    for key in ('model','optimizer','scheduler','rng'):
        differences = core.differences(final_state[key],resumed_state[key])
        assert not differences,(key,differences[:4])
    for before,after in zip(final_state['trace']['epochs'],resumed_state['trace']['epochs']):
        for key in ('train','test','model_sha256','lr_used','lr_next','scale_lr_used','scale_lr_next',
                    'best_epoch_so_far','weight_updates','gradient_checks','test_source_checks',
                    'test_prediction_counts','test_label_counts','scale_state','scale_epoch_metrics'):
            assert before[key] == after[key],(before['epoch'],key)
        for phase in ('train','test'):
            assert before['coverage'][phase]['order_sha256'] == after['coverage'][phase]['order_sha256']
    assert core.file_sha256(checkpoint50)==checkpoint50_hash
    assert resumed_report['full_training_completed'] and resumed_report['training_summary']['resumed']
    assert resumed_report['initial_model_sha256']==report['initial_model_sha256']
    assert resumed_report['initial_base_model_sha256']==report['initial_base_model_sha256']
    for name in ('scale_block_state.csv','scale_epoch_metrics.csv'):
        assert (resumed_path/name).read_bytes()==(continuous_path/name).read_bytes()

    partial_path, partial = run_case('partial1',stop_after=1,expected_status='STOPPED_AT_EPOCH_BOUNDARY')
    assert partial['completed_epoch']==1 and cases['partial1']['optimizer_updates_in_process']==2
    assert not partial['full_training_completed'] and partial['timing_estimate']['measured_100_epoch_seconds'] is None
    assert len(read_csv(partial_path/'selected_metrics.csv'))==6
    rejections={}
    for name,changes in [
        ('wrong_learned_pilot',{'schema':'mx-triple-learned-scale10-pilot5-protocol-v1','pilot_stop_epoch':5}),
        ('wrong_fixed32_full100',{'schema':'mx-triple-scale32-reference100-protocol-v1','learned_shared_scale':False}),
        ('wrong_scale_gradient',{'scale_gradient':'straight_line_between_candidates'}),
    ]:
        state=core.load_verified(checkpoint50)
        state['contract']=copy.deepcopy(state['contract'])
        state['contract']['config'].update(changes)
        checkpoint=out/(name+'_fixture.pt')
        core.atomic_save(checkpoint,state)
        digest=core.file_sha256(checkpoint)
        path,rejected=run_case(name,checkpoint,expected_status='FAIL')
        assert 'Hervatprotocol/omgeving/bron verschilt' in rejected['error']
        assert cases[name]['optimizer_updates_in_process']==0
        assert not list((path/'checkpoints').glob('*.pt'))
        assert core.file_sha256(checkpoint)==digest
        assert s.legacy.state_digest(contexts[name].model.state_dict())==cases[name]['fresh_model_sha256_before_optional_restore']
        rejections[name]={'before_restore_and_updates':True,'input_unchanged':True}
    path,rejected=run_case('completed100',final_checkpoint,expected_status='FAIL')
    assert 'geen nieuwe referentie-epochs' in rejected['error']
    assert cases['completed100']['optimizer_updates_in_process']==0
    rejections['completed100']={'before_restore_and_updates':True,'input_unchanged':True}
    path,rejected=run_case('endpoint_precedes_checkpoint',checkpoint50,stop_after=50,expected_status='FAIL')
    assert 'geen nieuwe referentie-epochs' in rejected['error']
    assert cases['endpoint_precedes_checkpoint']['optimizer_updates_in_process']==0
    rejections['endpoint_precedes_checkpoint']={'before_restore_and_updates':True,'input_unchanged':True}

    cli_results={}
    invalid=[['run','--stop-after-epoch',x] for x in ['0','101']]
    invalid += [['run','--device','cpu'],['resume'],
                ['run','--checkpoint',str(checkpoint50)],['run','--warmstart',str(checkpoint50)]]
    for args in invalid:
        result=subprocess.run([sys.executable,str(ROOT/'run_train.py'),*args],cwd=out,text=True,capture_output=True)
        assert result.returncode==2,(args,result.stdout,result.stderr)
        cli_results[' '.join(args)]={'exit_code':2,'reason':result.stderr.splitlines()[-1]}
    assert not (out/'mx_runs').exists()
    assert r.config()==production_config and runtime_hashes()==package_hashes
    assert core.file_sha256(final_checkpoint)==core.file_sha256(continuous_path/'last_checkpoint.pt')
    result={
        'status':'PASS','synthetic_fixture_only':True,'CUDA_tested':False,
        'scope':'Actual scale10 reference100 main/loop/export/resume on three synthetic CPU images; no actual SVHN',
        'final_package_manifest_verified_here':False,
        'fixture_integrity_scope':'Stable training runtime and protocol hashes; notebook/reference packaging checked separately',
        'no_SVHN_data_loaded_or_downloaded':True,'not_a_SVHN_accuracy_or_GPU_timing_result':True,
        'notebook_executed':False,'environment':contexts['continuous100'].rt.environment,
        'production_config':production_config,'comparison_to_learned_scale10_pilot':config_comparison,
        'fixture_changes':{'train_examples':3,'test_examples':3,'batch_size':2,'theta_mixed_test_only':True},
        'initialization_reference':initial_checks,'scale_tables_checked':scale_table_check,
        'LR_boundary_values_both_groups':lr_boundaries,'tested_training_runtime_sha256':package_hashes,
        'source_sha256':source_info['workcopy_files_sha256'],'test_file_sha256':core.file_sha256(Path(__file__)),
        'test_duration_seconds':time.perf_counter()-started,
        'total_synthetic_optimizer_updates':sum(c['optimizer_updates_in_process'] for c in cases.values()),
        'cases':cases,
        'tests':{
            'actual_default100_only_complete_PASS_at100':True,
            'all100_epochs_both_Adam_LRs_50_75_90_correct':True,
            'epoch50_checkpoint_theta_Adam_scheduler_RNG_complete_unfrozen_and_unchanged':True,
            'theta_remains_trainable_and_changes_after_epoch50':True,
            'resume50_to100_after_first_LR_drop_exact_model_theta_Adam_scheduler_RNG':True,
            'all100_scale_history_and_both_CSVs_resume_exact':True,
            'partial_CLI_endpoint_preserves_incomplete_status':True,
            'all_batches_three_passes_one_update_A32_FP32_BN_and_source_audits':True,
            'scale_projection_after_each_real_update':True,
            'independent_compact_export_choices2040bytes_no_theta_or_master':True,
            'source_only_reconstruction_all_three_logits_exact_without_encoders':True,
            'common_E4_best_and_epoch100_selected':True,
            'blank_notebook_and_unchanged_reference_feedback':True,
            'old_or_incompatible_checkpoint_contracts_rejected':rejections,
            'invalid_CLI_refused_before_outputs':cli_results,
            'production_inputs_unchanged':True,
        },
    }
    (out/'training_integration_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'PASS','report':str(out/'training_integration_report.json'),
                      'duration_seconds':result['test_duration_seconds'],
                      'synthetic_updates':result['total_synthetic_optimizer_updates']},indent=2),flush=True)
    return result


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: test_training_integration.py NEW_OUTPUT_DIRECTORY')
    main(Path(sys.argv[1]))
