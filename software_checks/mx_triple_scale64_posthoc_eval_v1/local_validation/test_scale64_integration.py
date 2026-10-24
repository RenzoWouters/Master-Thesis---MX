"""Bounded independent integration of the real scale64 CLI on synthetic CPU data.

Usage: python test_scale64_integration.py NEW_OUTPUT DIRECT_FEEDBACK TRUNC_FEEDBACK

The feedback ZIPs supply archived JSON/CSV protocol metadata only. All model
exports and three images are generated privately. Production has no synthetic,
CPU, dependency, or provenance bypass. Test-only substitutions are enumerated in
the report. No original checkpoint, SVHN data, training, or BN update is used.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import csv
import hashlib
import io
import json
import math
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

import numpy as np
import torch
import torch.nn.functional as F
import evaluation as ev
import joint_adapter as adapter
import mx_e2m1 as low
import mx_e4m3 as high
import mx_format as middle
import resume_core as core
import run_eval as runner
import scale64 as tr
import svhn_support as support
import trunc_adapter as naive

TRAININGS = ('direct', 'truncation')
ROLES = ('best_E4M3_test', 'last_epoch')
METADATA = ('run_report.json', 'checkpoint_selection.json', 'selected_metrics.csv',
            'epoch_metrics.csv', 'training_summary.json')
ORIGINAL_IMPORT = support.import_source
ORIGINAL_RUNTIME = support.runtime
ORIGINAL_READ_EXPORT = ev.read_export
ORIGINAL_ASSEMBLE = ev.assemble_paths
ORIGINAL_TRAIN = torch.nn.Module.train


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tree_hashes(path):
    return {p.relative_to(path).as_posix(): core.file_sha256(p)
            for p in sorted(Path(path).rglob('*')) if p.is_file()}


def fresh_source_import(source):
    # Production is invoked in a new process. Reproduce that module isolation
    # for repeated real main() calls inside this bounded test process.
    for name in tuple(sys.modules):
        if name.split('.')[0] in ('models', 'datasets', 'optimizer', 'utils'):
            del sys.modules[name]
    return ORIGINAL_IMPORT(source)


def no_operation(*args, **kwargs):
    raise AssertionError('Forbidden operation in post-hoc inference')


def eval_only(self, mode=True):
    assert mode is False, 'Evaluation entered training mode'
    return ORIGINAL_TRAIN(self, False)


def cpu_cuda_identity(self, *args, **kwargs):
    # Archived upstream initialization unconditionally calls Tensor.cuda().
    # This alias exists solely within the isolated CPU software test.
    return self


def independent_C6_lut():
    """Scalar IEEE-like decoding + explicit nearest/tie-even search; no codec."""
    out = []
    target = [(c & 7) / 8 if c >> 3 == 0 else
              (1 + (c & 7) / 8) * 2 ** ((c >> 3) - 1) for c in range(32)]
    for c in range(256):
        if c in (127, 255):
            out.append(0)
            continue
        e, m = (c >> 3) & 15, c & 7
        magnitude = m / 512 if e == 0 else (8 + m) * 2 ** (e - 1) / 512
        rounded = min(range(32), key=lambda q: (abs(target[q] - magnitude / 64), q % 2, q))
        out.append(((c >> 7) << 5) | rounded)
    return torch.tensor(out, dtype=torch.uint8)


def predict(model, images):
    with torch.no_grad():
        return torch.cat([model(images[:2]), model(images[2:])]).detach()


def make_exports(folder, report, payloads, synthetic):
    """Generate or clone truthful private files and update exact source hashes."""
    folder.mkdir(parents=True)
    (folder / 'SYNTHETIC_FIXTURE_ONLY.txt').write_text(
        'SOFTWARE VALIDATION ONLY. Random model exports; no real SVHN result.\n', encoding='utf-8')
    kind = report['fixture_training_kind']
    result = copy.deepcopy(report)
    saved_payloads = {}
    for role in ROLES:
        saved_payloads[role] = {}
        for bit in (8, 6, 4):
            item = result['mx_exports'][role][str(bit)]
            if kind == 'truncation' and bit != 8:
                payload = naive.derive_payload(saved_payloads[role][8], bit,
                                              result['mx_exports'][role]['8']['sha256'])
            else:
                payload = copy.deepcopy(payloads[role][bit])
                payload['provenance']['synthetic_fixture_only'] = synthetic
            item.update(core.atomic_save(folder / item['file'], payload))
            item['encoding_and_comparison_device'] = 'cpu; SYNTHETIC FIXTURE'
            if kind == 'truncation':
                item['source_file_sha256'] = result['mx_exports'][role]['8']['sha256']
            saved_payloads[role][bit] = payload
    return result, saved_payloads


def main(out: Path, direct_feedback: Path, truncation_feedback: Path):
    started = time.perf_counter()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    feedbacks = dict(zip(TRAININGS, map(lambda p: Path(p).resolve(), (direct_feedback, truncation_feedback))))
    feedback_hashes = {k: core.file_sha256(p) for k, p in feedbacks.items()}
    package_hashes = support.verify_package()
    source_dir = out / 'fixture_source'
    source_dir.mkdir()
    source, source_info = support.legacy.prepare_source(source_dir)
    rt = ORIGINAL_RUNTIME(20261009, 'cpu', strict=False)
    modules = fresh_source_import(source)
    originals, metadata = {}, {}
    for kind, feedback in feedbacks.items():
        with ZipFile(feedback) as archive:
            metadata[kind] = {name: archive.read(name) for name in METADATA}
        originals[kind] = json.loads(metadata[kind]['run_report.json'])
        assert originals[kind]['status'] == 'PASS' and originals[kind]['completed_epoch'] == 100
        assert originals[kind]['synthetic_fixture_only'] is False
        assert originals[kind]['config'] == json.loads((ROOT / (kind.upper() + '_REFERENCE_PROTOCOL.json')).read_text())
    assert tuple(tr.PATHS) == ('source8', 'trained6', 'trained4', 'A6', 'A4', 'B6', 'B4', 'C6', 'C4')
    assert len(tr.COMPARISONS) == 12 and len(tr.WEIGHT_COMPARISONS) == 18

    images = torch.rand((3, 3, 40, 40), generator=torch.Generator().manual_seed(8951)) * 2 - 1
    raw, full_folders, full_reports, eval_folders, eval_reports, eval_payloads = {}, {}, {}, {}, {}, {}
    expected_logits, expected_metrics = {}, {}
    # Create distinct per-bank affine/running state, so using the wrong BN bank
    # cannot accidentally pass because every bank has the same defaults.
    with patch.object(torch.Tensor, 'cuda', cpu_cuda_identity):
        for kind in TRAININGS:
            raw[kind], expected_logits[kind] = {}, {}
            for i, role in enumerate(ROLES):
                torch.manual_seed(9991 + i)
                original_model = modules.models.svhnq([1, 2, 4, 8, 32], expand=2)
                model = (adapter if kind == 'direct' else naive).adapt_copy(original_model).eval()
                with torch.no_grad():
                    for layer in adapter.BN_LAYERS:
                        for j, bank_key in enumerate(('8', '32', '4')):
                            bank = model.get_submodule(layer).bn_dict[bank_key]
                            bank.weight.fill_(0.9 + j * .07)
                            bank.bias.fill_(j * .013)
                            bank.running_mean.fill_(j * .011)
                            bank.running_var.fill_(1.1 + j * .1)
                selection = originals[kind]['checkpoint_selection']['best' if i == 0 else 'last']
                provenance = {'source_epoch': selection['epoch'], 'checkpoint_sha256': selection['sha256'],
                              'synthetic_fixture_only': True, 'encoding_device': 'cpu',
                              'fixture_notice': 'Random synthetic model for software tests only'}
                if kind == 'direct':
                    raw[kind][role] = {b: adapter.export_payload(model, b, provenance)[0] for b in (8, 6, 4)}
                else:
                    src = naive.source_payload(model, provenance)[0]
                    raw[kind][role] = {8: src, 6: None, 4: None}
            base = copy.deepcopy(originals[kind])
            base.update(fixture_training_kind=kind, fixture_synthetic_not_a_research_result=True)
            full_folders[kind] = out / 'SYNTHETIC_guard_contract' / kind
            full_reports[kind], _ = make_exports(full_folders[kind], base, raw[kind], False)
            for name, content in metadata[kind].items():
                (full_folders[kind] / name).write_bytes(content)
            json_write(full_folders[kind] / 'run_report.json', full_reports[kind])
            eval_folders[kind] = out / 'SYNTHETIC_evaluation_inputs' / kind
            eval_reports[kind], eval_payloads[kind] = make_exports(eval_folders[kind], base, raw[kind], True)
            for role in ROLES:
                expected_logits[kind][role] = {}
                for bit in (8, 6, 4):
                    payload = eval_payloads[kind][role][bit]
                    frozen = (adapter if kind == 'direct' else naive).load_exported_model(modules.models, payload, 'cpu')
                    expected_logits[kind][role][bit] = predict(frozen, images)
    # A known high-precision best fixture allows the notebook's archived-best
    # selection checks to remain coherent even with only three examples.
    labels = expected_logits['direct'][ROLES[0]][8].topk(5, dim=1, largest=True, sorted=True).indices[:, 0].clone()
    assert torch.equal(expected_logits['direct'][ROLES[0]][8], expected_logits['truncation'][ROLES[0]][8])
    label_hash = hashlib.sha256(labels.numpy().astype(np.int64).tobytes()).hexdigest()
    data_file = out / 'SYNTHETIC_three_images.pt'
    torch.save({'images': images, 'labels': labels}, data_file)
    data_hashes = support.hashes(data_file)
    loader = torch.utils.data.DataLoader(support.IndexedDataset(torch.utils.data.TensorDataset(images, labels)),
        batch_size=2, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    for kind in TRAININGS:
        expected_metrics[kind] = {}
        for role in ROLES:
            expected_metrics[kind][role] = {}
            for bit, logits in expected_logits[kind][role].items():
                top5 = logits.topk(5, dim=1, largest=True, sorted=True).indices
                c1 = int((top5[:, 0] == labels).sum()); c5 = int((top5 == labels[:, None]).any(dim=1).sum())
                expected_metrics[kind][role][bit] = dict(n=3, correct1=c1, correct5=c5,
                    top1_percent=100*c1/3, count_top1_percent=100*c1/3, top5_percent=100*c5/3,
                    objective_loss=float(F.cross_entropy(logits, labels, reduction='none').double().mean()))
        report = eval_reports[kind]
        report['synthetic_fixture_only'] = True
        report['config']['test_examples'] = 3
        report['training_summary'].update(test_examples=3, synthetic_fixture_only=True)
        report['dataset_selection']['test'].update(n=3, label_sha256=label_hash,
            labels_histogram=np.bincount(labels.numpy(), minlength=10).tolist())
        report['dataset_files']['files'] = {'SYNTHETIC_three_images.pt': data_hashes}
        report['transforms']['test'] = 'SYNTHETIC already transformed float32 images; no SVHN read'
        history = list(csv.DictReader(io.StringIO(metadata[kind]['epoch_metrics.csv'].decode('utf-8-sig'))))
        for row in history:
            if row['phase'] != 'test': continue
            row.update(n=3, correct1=0, correct5=0, top1_percent=0., count_top1_percent=0., top5_percent=0., objective_loss=2.3)
            for role in ROLES:
                selection = report['checkpoint_selection']['best' if role == ROLES[0] else 'last']
                if int(row['epoch']) == selection['epoch']:
                    row.update(expected_metrics[kind][role][int(row['weight_bits'])])
        selected = []
        for role in ROLES:
            choice = report['checkpoint_selection']['best' if role == ROLES[0] else 'last']
            choice['selection_score_E4M3'] = expected_metrics[kind][role][8]['top1_percent']
            for row in history:
                if row['phase'] == 'test' and int(row['epoch']) == choice['epoch']:
                    selected.append({'role': role, **row})
        support.write_csv(eval_folders[kind] / 'epoch_metrics.csv', history)
        support.write_csv(eval_folders[kind] / 'selected_metrics.csv', selected)
        json_write(eval_folders[kind] / 'checkpoint_selection.json', report['checkpoint_selection'])
        json_write(eval_folders[kind] / 'training_summary.json', report['training_summary'])
        json_write(eval_folders[kind] / 'run_report.json', report)

    guard_tests = {}
    for kind in TRAININGS:
        folder, fixture = full_folders[kind], full_reports[kind]
        guard_rt = SimpleNamespace(environment=originals[kind]['environment'], settings=originals[kind]['execution_settings'])
        checked, fingerprints, selected = runner.inspect_reference(folder, guard_rt, kind)
        assert checked['fixture_synthetic_not_a_research_result'] and len(selected) == 6
        assert all(core.file_sha256(folder / name) == digest for name, digest in fingerprints.items())
        guard_tests[kind + '/valid_archived_contract'] = {'accepted': True, 'selected_rows': len(selected)}

        def rejected_metadata(label, mutate):
            altered = copy.deepcopy(fixture)
            mutate(altered)
            json_write(folder / 'run_report.json', altered)
            try:
                runner.inspect_reference(folder, guard_rt, kind)
            except (AssertionError, ValueError, RuntimeError, KeyError) as exc:
                guard_tests[kind + '/' + label] = {'rejected': True, 'message': str(exc)}
            else:
                raise AssertionError('Invalid reference accepted: ' + kind + '/' + label)
            finally:
                json_write(folder / 'run_report.json', fixture)

        rejected_metadata('wrong_training_schema', lambda r: r.update(schema=originals[TRAININGS[1-TRAININGS.index(kind)]]['schema']))
        rejected_metadata('unfinished_reference', lambda r: r.update(completed_epoch=99, full_training_completed=False))
        rejected_metadata('synthetic_reference', lambda r: r.update(synthetic_fixture_only=True))
        rejected_metadata('protocol_A16', lambda r: r['config'].update(activation_bits=16))
        rejected_metadata('changed_export_hash', lambda r: r['mx_exports'][ROLES[0]]['8'].update(sha256='0'*64))
        rejected_metadata('unsafe_export_path', lambda r: r['mx_exports'][ROLES[0]]['8'].update(file='../outside.pt'))
        if kind == 'truncation':
            rejected_metadata('wrong_export_source_hash', lambda r: r['mx_exports'][ROLES[0]]['6'].update(source_file_sha256='0'*64))
        # Resign both JSON copies: the fixed selection must still be checked
        # against all 600 archived metric rows, not merely against itself.
        for which, epoch in [('best', 53), ('last', 99)]:
            altered = copy.deepcopy(fixture)
            altered['checkpoint_selection'][which]['epoch'] = epoch
            json_write(folder / 'run_report.json', altered)
            json_write(folder / 'checkpoint_selection.json', altered['checkpoint_selection'])
            try:
                runner.inspect_reference(folder, guard_rt, kind)
            except (AssertionError, ValueError, RuntimeError, KeyError) as exc:
                guard_tests[kind + '/resigned_wrong_' + which + '_epoch'] = {'rejected': True, 'message': str(exc)}
            else: raise AssertionError('Changed archived selection was accepted')
            finally:
                json_write(folder / 'run_report.json', fixture)
                (folder / 'checkpoint_selection.json').write_bytes(metadata[kind]['checkpoint_selection.json'])
        # Valid-count and LR invariants are checked independently of selection.
        history_path = folder / 'epoch_metrics.csv'; before = history_path.read_bytes()
        for field, value in [('correct1', '999999'), ('lr', '0.9')]:
            rows = csv_rows(history_path); rows[0][field] = value
            history_path.unlink(); support.write_csv(history_path, rows)
            try:
                runner.inspect_reference(folder, guard_rt, kind)
            except (AssertionError, ValueError, RuntimeError, KeyError) as exc:
                guard_tests[kind + '/invalid_history_' + field] = {'rejected': True, 'message': str(exc)}
            else: raise AssertionError('Invalid archived history was accepted')
            finally: history_path.write_bytes(before)
    before_full = {k: tree_hashes(p) for k, p in full_folders.items()}
    before_eval = {k: tree_hashes(p) for k, p in eval_folders.items()}
    source_paths = {eval_folders[k] / item['file'] for k in TRAININGS for items in eval_reports[k]['mx_exports'].values() for item in items.values()}

    def fixture_inspection(ref, runtime, training_kind):
        assert Path(ref) == eval_folders[training_kind]
        report = eval_reports[training_kind]
        names = list(METADATA)
        for exports in report['mx_exports'].values():
            for item in exports.values(): names += [item['file'], item['file'] + '.sha256']
        return copy.deepcopy(report), {name: core.file_sha256(ref / name) for name in names}, csv_rows(ref / 'selected_metrics.csv')

    def fixture_data(args, current_modules, reference, path, log):
        assert reference['config']['test_examples'] == 3
        support.write_json(path / 'dataset_manifest.json', {
            'synthetic_fixture_only': True, 'test_file': {'path': str(data_file), **data_hashes},
            'selection': reference['dataset_selection']['test'], 'train_read': False, 'extra_read': False})
        return loader, data_file, data_hashes

    def fixture_dependencies(path, log):
        (path / 'pip_freeze.txt').write_text('Temporary isolated CPU validation environment; dependencies separately checked.\n')
        (path / 'pip_check.txt').write_text('Dependency subprocess replaced only inside test harness.\n')

    def fixture_runtime(seed, requested_device, strict=True):
        assert requested_device in ('cuda', 'cuda:0')
        return ORIGINAL_RUNTIME(seed, 'cpu', strict=False)

    def fixture_read(path, digest):
        payload = ORIGINAL_READ_EXPORT(path, digest)
        if Path(path) in source_paths:
            assert payload['provenance']['synthetic_fixture_only'] is True
            payload['provenance']['synthetic_fixture_only'] = False
        return payload

    def fixture_assemble(models, payloads, hashes, path, role, device, training_kind):
        for payload in payloads.values():
            assert payload['provenance']['synthetic_fixture_only'] is False
            payload['provenance']['synthetic_fixture_only'] = True
        return ORIGINAL_ASSEMBLE(models, payloads, hashes, path, role, device, training_kind)

    original_evaluate = runner.evaluate

    def marked_evaluate(args, path, report, runtime, current_modules, log):
        report['limitations'].append('SYNTHETIC CPU SOFTWARE TEST ONLY; three artificial images and random exports. No SVHN result.')
        try: return original_evaluate(args, path, report, runtime, current_modules, log)
        finally: report['synthetic_fixture_only'] = True

    cases = {}

    def invoke(command, case_name=None, expected_status='PASS'):
        case_name = case_name or command
        arguments = [str(ROOT / 'run_eval.py'), command, '--output-root', str(out / case_name)]
        if command == 'evaluate':
            arguments += ['--direct-run', str(eval_folders['direct']), '--truncation-run', str(eval_folders['truncation'])]
        with ExitStack() as stack:
            for target, name, replacement in [
                (sys, 'argv', arguments), (runner, 'dependencies', fixture_dependencies),
                (support, 'runtime', fixture_runtime), (support, 'import_source', fresh_source_import),
                (torch.Tensor, 'cuda', cpu_cuda_identity), (torch.optim.Optimizer, '__init__', no_operation),
                (torch.Tensor, 'backward', no_operation), (torch.autograd, 'backward', no_operation)]:
                stack.enter_context(patch.object(target, name, replacement))
            if command == 'evaluate':
                for target, name, replacement in [
                    (runner, 'inspect_reference', fixture_inspection), (runner, 'prepare_test_only', fixture_data),
                    (runner, 'evaluate', marked_evaluate), (ev, 'read_export', fixture_read),
                    (ev, 'assemble_paths', fixture_assemble), (torch.nn.Module, 'train', eval_only),
                    (adapter, 'effective_reference_weight', no_operation),
                    (adapter.original_adapter, 'effective_reference_weight', no_operation),
                    (naive, 'effective_reference_weight', no_operation),
                    (high, 'encode_tensor', no_operation), (middle, 'encode_tensor', no_operation),
                    (low, 'encode_tensor', no_operation), (high, 'encode_elements', no_operation),
                    (low, 'encode_elements', no_operation)]:
                    stack.enter_context(patch.object(target, name, replacement))
            with (out / (case_name + '_stdout.log')).open('x', encoding='utf-8') as log:
                stack.enter_context(redirect_stdout(log))
                exit_code = runner.main()
        paths = list((out / case_name).glob('*_' + command))
        assert len(paths) == 1
        path = paths[0]; report = json.loads((path / 'run_report.json').read_text())
        assert exit_code == (0 if expected_status == 'PASS' else 1) and report['status'] == expected_status, (case_name, report.get('error'), report.get('traceback'))
        assert report['optimizer_steps'] == 0 and report['batchnorm_recalibration'] is False and report['new_checkpoint_selection'] is False
        assert report['synthetic_fixture_only'] is True
        with ZipFile(path / 'feedback_bundle.zip') as archive:
            assert archive.testzip() is None and 'run_report.json' in archive.namelist()
        cases[case_name] = {'status': report['status'], 'expected_status': expected_status,
                            'output_directory': str(path), 'error': report.get('error')}
        return path, report

    check_path, check_report = invoke('check')
    eval_path, eval_report = invoke('evaluate')
    assert set(eval_report['roles']) == set(TRAININGS)
    for flag in ('input_files_unchanged', 'test_file_unchanged', 'original_checkpoint_selection_unchanged',
                 'historical_correct_counts_all_match', 'derived_models_all_evaluated_from_saved_and_reloaded_codes',
                 'same_initial_model_and_dataset_across_training_kinds'):
        assert eval_report[flag] is True, flag
    assert eval_report['train_and_extra_read'] is False
    metrics = csv_rows(eval_path / 'metrics.csv'); pairs = csv_rows(eval_path / 'paired_metrics.csv')
    assert len(metrics) == 36 and len(pairs) == 48
    assert len(csv_rows(eval_path / 'weight_pair_metrics.csv')) == 432
    assert len(csv_rows(eval_path / 'layer_code_metrics.csv')) == 216
    assert len(csv_rows(eval_path / 'historical_reproduction.csv')) == 12
    assert {tuple((r[k] for k in ('training', 'role', 'path'))) for r in metrics} == {
        (k, r, p) for k in TRAININGS for r in ROLES for p in tr.PATHS}
    lut = independent_C6_lut(); independently_verified_layers = 0
    for kind in TRAININGS:
        assert eval_report['references'][kind]['checkpoint_files_opened'] is False
        assert eval_report['references'][kind]['FP32_master_weights_opened'] is False
        for role in ROLES:
            rr = eval_report['roles'][kind][role]
            for flag in ('all_model_and_bn_states_unchanged', 'all_models_no_grad',
                         'all_paths_use_frozen_target_precision_bn', 'six_FP32_master_parameters_absent_all_paths',
                         'one_E4_source_for_all_derived', 'A_source_scales_unchanged', 'B_element_codes_equal_A',
                         'mantissa_shift_after_each_E2M3_path', 'original_auxiliary_state_preserved', 'no_tensor_scale_reselection'):
                assert rr[flag] is True, (kind, role, flag)
            assert rr['optimizer_steps'] == 0 and rr['batchnorm_recalibrated'] is False
            assert rr['n_test'] == 3 and rr['label_sha256'] == label_hash
            assert rr['BC_source_scale_exponent_offset'] == 6
            assert set(rr['derived_exports']) == set(tr.DERIVED_PATHS)
            loaded = {}
            for key, details in rr['derived_exports'].items():
                assert details['saved_and_reloaded'] is True and details['master_weights_read_to_derive'] is False
                assert details['source_scale_bytes_copied'] is (key[0] == 'A')
                assert details['scale_exponent_offset'] == (0 if key[0] == 'A' else 6)
                loaded[key] = ORIGINAL_READ_EXPORT(eval_path / kind / details['file'], details['sha256'])
                tr.validate_derived(loaded[key])
                assert loaded[key]['provenance']['synthetic_fixture_only'] is True
                assert loaded[key]['derivation']['path_id'] == key
                assert loaded[key]['derivation']['source_file_sha256'] == eval_reports[kind]['mx_exports'][role]['8']['sha256']
                assert not core.differences(loaded[key]['auxiliary_fp32_state'], eval_payloads[kind][role][8]['auxiliary_fp32_state'])
            for layer in adapter.TARGET_LAYERS:
                source_mx = high.MXTensor.from_payload(eval_payloads[kind][role][8]['mx_layers'][layer])
                mx = {key: adapter.codec(tr.PATH_SPEC[key]['target_bits']).MXTensor.from_payload(payload['mx_layers'][layer])
                      for key, payload in loaded.items()}
                raw_codes = source_mx.codes.long()
                oracle_A6 = (((raw_codes >> 7) << 5) | (((raw_codes >> 5) & 3) << 3) | (raw_codes & 7)).byte()
                assert torch.equal(mx['A6'].codes, oracle_A6)
                assert torch.equal(mx['C6'].codes, lut[raw_codes])
                for family in ('A', 'B', 'C'):
                    assert torch.equal(mx[family + '4'].codes, mx[family + '6'].codes >> 2)
                    for bit in (6, 4):
                        expected_scales = source_mx.scale_codes.int() + (0 if family == 'A' else 6)
                        assert torch.equal(mx[family + str(bit)].scale_codes.int(), expected_scales)
                for bit in (6, 4):
                    assert torch.equal(mx['A' + str(bit)].codes, mx['B' + str(bit)].codes)
                    assert torch.equal(mx['B' + str(bit)].decode(), mx['A' + str(bit)].decode() * 64)
                independently_verified_layers += 1
            with np.load(eval_path / rr['per_image_file'], allow_pickle=False) as archive:
                assert np.array_equal(archive['indices'], np.arange(3)) and np.array_equal(archive['labels'], labels.numpy())
                for row in (x for x in metrics if x['training'] == kind and x['role'] == role):
                    key = row['path']; pred = archive[key + '__predictions']; logits = archive[key + '__logits']; ce = archive[key + '__per_image_ce']
                    correct = int((pred == labels.numpy()).sum())
                    assert int(row['n']) == 3 and int(row['correct1']) == correct
                    assert float(row['top1_percent']) == 100 * correct / 3
                    assert math.isclose(float(row['hard_cross_entropy']), ce.astype(np.float64).mean())
                    assert logits.shape == (3, 10) and np.isfinite(logits).all()
                    assert row['bn_bank_key'] == tr.PATH_SPEC[key]['bn_bank_key']
                    if key in ('source8', 'trained6', 'trained4'):
                        bit = tr.PATH_SPEC[key]['target_bits']
                        assert np.array_equal(logits, expected_logits[kind][role][bit].numpy())
                if kind == 'truncation':
                    for bit in (6, 4):
                        assert np.array_equal(archive[f'A{bit}__logits'], archive[f'trained{bit}__logits'])
                for row in (x for x in pairs if x['training'] == kind and x['role'] == role):
                    r = archive[row['reference_path'] + '__predictions']; o = archive[row['other_path'] + '__predictions']
                    rc, oc = r == labels.numpy(), o == labels.numpy()
                    values = {'n': 3, 'reference_correct1': int(rc.sum()), 'other_correct1': int(oc.sum()),
                        'wrong_to_correct': int((~rc & oc).sum()), 'correct_to_wrong': int((rc & ~oc).sum()),
                        'both_correct': int((rc & oc).sum()), 'both_wrong': int((~rc & ~oc).sum()),
                        'prediction_changes': int((r != o).sum()), 'wrong_to_other_wrong': int((~rc & ~oc & (r != o)).sum()),
                        'accuracy_difference_pp': 100 * (int(oc.sum())-int(rc.sum())) / 3,
                        'net_correct_change': int(oc.sum())-int(rc.sum())}
                    error = archive[row['other_path'] + '__logits'].astype(np.float64) - archive[row['reference_path'] + '__logits'].astype(np.float64)
                    values.update(mean_abs_logit_difference=float(np.abs(error).mean()), max_abs_logit_difference=float(np.abs(error).max()))
                    assert all(math.isclose(float(row[k]), v) for k, v in values.items())
    notebooks = list(eval_path.glob('*.ipynb')); assert len(notebooks) == 1
    notebook = json.loads(notebooks[0].read_text())
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in notebook['cells'] if c['cell_type'] == 'code')
    with ZipFile(eval_path / 'feedback_bundle.zip') as archive:
        assert all(eval_report['roles'][k][r]['per_image_file'] in archive.namelist() for k in TRAININGS for r in ROLES)
        assert all('reference_inputs/' + k + '/' + n in archive.namelist() for k in TRAININGS for n in METADATA)

    # File hashes are deliberately updated after each private mutation, so these
    # exercise the real payload/scale checks rather than just a stale checksum.
    def payload_case(label, kind, bit, mutation, message_fragment):
        item = eval_reports[kind]['mx_exports'][ROLES[0]][str(bit)]
        path = eval_folders[kind] / item['file']; marker = path.with_suffix('.pt.sha256')
        saved_bytes, saved_marker, saved_item = path.read_bytes(), marker.read_bytes(), copy.deepcopy(item)
        payload = torch.load(path, map_location='cpu', weights_only=True); mutation(payload)
        path.unlink(); marker.unlink(); item.update(core.atomic_save(path, payload))
        try:
            failed_path, failed_report = invoke('evaluate', label, 'FAIL')
            assert message_fragment.lower() in failed_report['error'].lower(), (label, failed_report['error'])
            assert not (failed_path / 'metrics.csv').exists()
        finally:
            path.write_bytes(saved_bytes); marker.write_bytes(saved_marker)
            item.clear(); item.update(saved_item)

    payload_case('reject_checkpoint_provenance', 'direct', 8,
        lambda p: p['provenance'].update(checkpoint_sha256='0'*64), 'checkpointprovenance')
    payload_case('reject_source_format', 'direct', 8,
        lambda p: p.update(weight_format='MXFP6_E2M3'), 'formaat')
    payload_case('reject_hidden_master', 'direct', 8,
        lambda p: p['auxiliary_fp32_state'].update({adapter.TARGET_LAYERS[0] + '.weight': torch.zeros(1)}), 'master')
    payload_case('reject_naive_source_hash', 'truncation', 6,
        lambda p: p['derivation'].update(source_sha256='0'*64), 'bronhash')

    def mutate_naive_code(payload):
        layer = adapter.TARGET_LAYERS[0]
        mx = middle.MXTensor.from_payload(payload['mx_layers'][layer])
        mx.codes.reshape(-1)[0] ^= 1
        payload['mx_layers'][layer] = mx.payload()
    payload_case('reject_naive_lower_codes', 'truncation', 6, mutate_naive_code, 'exacte naive')

    def overflow_scale(payload):
        # A is still finite for this block. B/C must reject +6 >254 before
        # converting a code to uint8, hence no wrapping or clipping is possible.
        layer = adapter.TARGET_LAYERS[0]
        mx = high.MXTensor.from_payload(payload['mx_layers'][layer])
        mx.scale_codes.reshape(-1)[0] = 249
        mx.codes.reshape(-1, 32)[0].fill_(0)
        payload['mx_layers'][layer] = mx.payload()
    payload_case('reject_scale_plus6_overflow', 'direct', 8, overflow_scale, 'E8M0')

    # Cross-training invariants are real runner gates, independent of each
    # separate reference inspection. Test before any model is evaluated.
    for key, replacement in [('initial_model_sha256', '0'*64), ('transforms', {'test': 'changed'})]:
        original = copy.deepcopy(eval_reports['truncation'][key])
        eval_reports['truncation'][key] = replacement
        try:
            failed_path, _ = invoke('evaluate', 'reject_cross_training_' + key, 'FAIL')
            assert not list(failed_path.rglob('*_derived.pt')) and not (failed_path / 'metrics.csv').exists()
        finally: eval_reports['truncation'][key] = original

    bad_cli = [['run'], ['check', '--device', 'cpu'], ['check', '--download'],
        ['evaluate'], ['evaluate', '--direct-run', str(eval_folders['direct'])],
        ['evaluate', '--direct-run', str(eval_folders['direct']), '--truncation-run', str(eval_folders['direct'])],
        ['evaluate', '--direct-run', str(eval_folders['direct']), '--truncation-run', str(eval_folders['truncation']),
         '--output-root', str(eval_folders['direct'] / 'inside')],
        ['check', '--direct-run', str(eval_folders['direct'])], ['check', '--synthetic'], ['check', '--epochs', '5']]
    cli_results = []
    for arguments in bad_cli:
        completed = subprocess.run([sys.executable, str(ROOT / 'run_eval.py'), *arguments], cwd=out,
            capture_output=True, text=True, timeout=45)
        assert completed.returncode == 2, (arguments, completed.stdout, completed.stderr)
        cli_results.append({'arguments': arguments, 'exit_code': completed.returncode})
    assert {k: tree_hashes(p) for k, p in full_folders.items()} == before_full
    assert {k: tree_hashes(p) for k, p in eval_folders.items()} == before_eval
    assert support.hashes(data_file) == data_hashes
    assert {k: core.file_sha256(p) for k, p in feedbacks.items()} == feedback_hashes
    assert support.verify_package() == package_hashes
    modifications = [
        'Temporary isolated torch 2.10.0+cpu / torchvision 0.25.0+cpu venv; primary and laptop environments untouched.',
        'runtime CUDA request redirected to CPU; archived upstream Tensor.cuda initializer aliased to identity inside harness.',
        'Dependency subprocess writes fixture notices; CPU package imports checked separately.',
        'prepare_test_only supplies three artificial 40x40 images in batches 2+1; no SVHN data opened.',
        'inspect_reference uses truthful three-image fixture for evaluate; actual strict inspector tested separately against both full archived JSON/CSV contracts.',
        'At-rest source exports are synthetic=True; read_export transiently presents False solely to production real-run provenance gate, assemble restores True before all core checks, derivation, and inference.',
        'Output report explicitly synthetic=True and contains a synthetic limitation on success and expected failure.',
        'Fresh upstream module isolation reproduces a new production CLI process for each real main invocation.',
        'No production source edited or CLI bypass flag added.'
    ]
    result = {'status': 'PASS', 'schema': 'mx-triple-scale64-integration-validation-v1',
        'synthetic_fixture_only': True, 'scope': 'Real check/evaluate main on two random CPU model populations, both roles, three images in batches 2+1',
        'production_CUDA_or_SVHN_tested': False, 'no_real_SVHN_data_or_user_checkpoint_loaded': True,
        'optimizer_backward_G_tensor_encoders_training_mode_forbidden_during_evaluate': True,
        'no_training_or_BN_recalibration': True, 'production_bypass_added': False,
        'fixture_only_adaptations': modifications, 'runtime_environment': rt.environment,
        'package_sha256': package_hashes, 'test_harness_sha256': core.file_sha256(Path(__file__)),
        'feedback_metadata_inputs_sha256': feedback_hashes, 'cases': cases, 'reference_guard_tests': guard_tests,
        'CLI_guards': cli_results, 'metric_rows': 36, 'paired_rows': 48, 'weight_metric_rows': 432,
        'code_metric_rows': 216, 'historical_reproduction_rows': 12, 'per_image_NPZ_files': 4,
        'all_24_derived_models_saved_reloaded_and_evaluated': True,
        'independent_code_scale_oracle_layer_groups': independently_verified_layers,
        'all_12_baseline_logits_exactly_match_archived_loaders': True,
        'naive_trained_A_routes_exactly_match_stored_trained_logits': True,
        'all_input_trees_feedback_and_package_unchanged': True,
        'notebook_blank_only': True, 'successful_evaluation_directory': str(eval_path),
        'seconds': time.perf_counter()-started}
    json_write(out / 'integration_report.json', result)
    json_write(ROOT / 'local_validation/integration_report.json', result)
    check_result = {'status': 'PASS', 'schema': 'mx-triple-scale64-check-entry-validation-v1',
        'synthetic_fixture_only': True, 'actual_cli_main_invoked': True, 'runtime_environment': rt.environment,
        'production_CUDA_or_SVHN_tested': False, 'fixture_only_adaptations': modifications[:3] + modifications[-2:],
        'package_sha256': package_hashes, 'test_harness_sha256': core.file_sha256(Path(__file__)),
        'check_output_directory': str(check_path), 'check_tests': check_report['tests'],
        'source_unchanged': check_report['source_unchanged'], 'zero_optimizer_steps': check_report['optimizer_steps'] == 0}
    json_write(ROOT / 'local_validation/check_entry_validation_report.json', check_result)
    print(json.dumps({'status':'PASS', 'report':str(out / 'integration_report.json'),
        'evaluation_directory':str(eval_path), 'seconds':result['seconds']}, indent=2))
    return result


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
