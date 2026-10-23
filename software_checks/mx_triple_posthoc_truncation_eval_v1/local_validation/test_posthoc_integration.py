"""Bounded offline integration of the actual post-hoc CLI with synthetic CPU data.

Usage: python local_validation/test_posthoc_integration.py NEW_OUTPUT FEEDBACK_ZIP

The production CLI is not given a CPU/data/provenance bypass. This test patches
runtime/dependency/data setup in memory and constructs private synthetic exports.
No original checkpoint, SVHN tensor or user's run file is opened or changed.
The supplied feedback ZIP provides protocol metadata for guard tests only.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import csv
import hashlib
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
import resume_core as core
import run_eval as runner
import svhn_support as support
import truncation as tr

ROLES = ('best_E4M3_test', 'last_epoch')
NOTEBOOK = 'MX_triple_posthoc_truncatie_resultaten.ipynb'


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tree_hashes(path):
    return {str(p.relative_to(path)): core.file_sha256(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}


def fresh_source_import(source):
    # Every production command is a fresh process. Recreate only that module
    # isolation while invoking main repeatedly in this bounded test process.
    for name in tuple(sys.modules):
        if name.split('.')[0] in ('models', 'datasets', 'optimizer', 'utils'):
            del sys.modules[name]
    return ORIGINAL_IMPORT_SOURCE(source)


ORIGINAL_IMPORT_SOURCE = support.import_source
ORIGINAL_RUNTIME = support.runtime


def main(out: Path, feedback: Path):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    feedback = Path(feedback).resolve()
    feedback_hash = core.file_sha256(feedback)
    started = time.perf_counter()
    torch.set_num_threads(2)
    package_hashes = support.verify_package()
    source_dir = out / 'fixture_source'
    source_dir.mkdir()
    source, source_info = support.legacy.prepare_source(source_dir)
    rt = ORIGINAL_RUNTIME(20261009, 'cpu', strict=False)
    modules = fresh_source_import(source)
    with ZipFile(feedback) as archive:
        metadata = {name: archive.read(name) for name in ('run_report.json', 'checkpoint_selection.json',
                                                        'selected_metrics.csv', 'epoch_metrics.csv',
                                                        'training_summary.json')}
    original_report = json.loads(metadata['run_report.json'])
    assert original_report['status'] == 'PASS' and original_report['completed_epoch'] == 100
    assert original_report['synthetic_fixture_only'] is False
    assert original_report['config'] == json.loads((ROOT / 'REFERENCE_PROTOCOL.json').read_text())
    assert tuple(tr.PATHS) == ('direct8', 'direct6', 'direct4', 'round64', 'mantissa64',
                              'round86', 'exponent86', 'combined864')

    generator = torch.Generator().manual_seed(8951)
    images = torch.rand((3, 3, 40, 40), generator=generator) * 2 - 1
    labels = torch.tensor([2, 5, 9], dtype=torch.long)
    data_file = out / 'SYNTHETIC_three_images.pt'
    torch.save({'images': images, 'labels': labels}, data_file)
    data_hashes = support.hashes(data_file)
    dataset = torch.utils.data.TensorDataset(images, labels)
    loader = torch.utils.data.DataLoader(support.IndexedDataset(dataset), batch_size=2,
                                       shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    label_hash = hashlib.sha256(labels.numpy().astype(np.int64).tobytes()).hexdigest()

    # Test-only fixture exports deliberately carry the protocol's real-run
    # provenance flags so the production guard can be exercised. The fixture
    # directory and test reports unambiguously identify synthetic random data.
    # No such payload is included in the distribution or accepted via a CLI flag.
    full_fixture = out / 'SYNTHETIC_EXPORTS_WITH_FULL_PROTOCOL_METADATA'
    full_fixture.mkdir()
    (full_fixture / 'FIXTURE_ONLY.txt').write_text(
        'SOFTWARE TEST ONLY. Random model exports plus copied protocol metadata. '
        'These are not user checkpoints or SVHN results.\n', encoding='utf-8')
    for name, content in metadata.items():
        (full_fixture / name).write_bytes(content)
    fixture_report = copy.deepcopy(original_report)
    fixture_report['fixture_synthetic_not_a_research_result'] = True
    expected_metrics = {}
    direct_payloads = {}
    for role_index, role in enumerate(ROLES):
        torch.manual_seed(9991 + role_index)
        model = adapter.adapt_copy(modules.models.svhnq([1, 2, 4, 8, 32], expand=2)).eval()
        selection = fixture_report['checkpoint_selection']['best' if role == ROLES[0] else 'last']
        suffix = 'best' if role == ROLES[0] else 'last'
        direct_payloads[role] = {}
        expected_metrics[role] = {}
        for bit in (8, 6, 4):
            provenance = {'source_epoch': selection['epoch'], 'checkpoint_sha256': selection['sha256'],
                          'synthetic_fixture_only': False, 'encoding_device': 'cpu',
                          'fixture_notice': 'SYNTHETIC random model used only for software validation'}
            payload, unused = adapter.export_payload(model, bit, provenance)
            name = f'mx_direct_{adapter.FORMATS[bit]}_{suffix}.pt'
            saved = core.atomic_save(full_fixture / name, payload)
            fixture_report['mx_exports'][role][str(bit)] = {
                **fixture_report['mx_exports'][role][str(bit)], **saved,
                'epoch': selection['epoch'], 'encoding_and_comparison_device': 'cpu'}
            direct_payloads[role][bit] = payload
            frozen = adapter.load_exported_model(modules.models, payload, 'cpu')
            logits = []
            with torch.no_grad():
                for x, unused_y, unused_i in loader:
                    logits.append(frozen(x))
            logits = torch.cat(logits)
            top5 = logits.topk(5, dim=1, largest=True, sorted=True).indices
            expected_metrics[role][bit] = {
                'correct1': int((top5[:, 0] == labels).sum()),
                'correct5': int((top5 == labels[:, None]).any(dim=1).sum()),
                'objective_loss': float(F.cross_entropy(logits, labels, reduction='none').double().mean()),
            }
    json_write(full_fixture / 'run_report.json', fixture_report)

    guard_rt = SimpleNamespace(environment=original_report['environment'], settings=original_report['execution_settings'])
    checked_report, checked_hashes, checked_selected = runner.inspect_reference(full_fixture, guard_rt)
    assert checked_report['fixture_synthetic_not_a_research_result']
    assert len(checked_selected) == 6
    assert all(core.file_sha256(full_fixture / name) == digest for name, digest in checked_hashes.items())
    guard_tests = {}

    def rejected_metadata(label, mutate):
        altered = copy.deepcopy(fixture_report)
        mutate(altered)
        json_write(full_fixture / 'run_report.json', altered)
        try:
            runner.inspect_reference(full_fixture, guard_rt)
        except (AssertionError, ValueError, RuntimeError, KeyError) as exc:
            guard_tests[label] = {'rejected': True, 'message': str(exc)}
        else:
            raise AssertionError('Invalid reference accepted: ' + label)
        finally:
            json_write(full_fixture / 'run_report.json', fixture_report)

    rejected_metadata('wrong_reference_schema', lambda value: value.update(schema='mx-weights-triple-direct-pilot5-report-v1'))
    rejected_metadata('unfinished_reference', lambda value: value.update(completed_epoch=99, full_training_completed=False))
    rejected_metadata('synthetic_reference_flag', lambda value: value.update(synthetic_fixture_only=True))
    rejected_metadata('different_checkpoint_selection', lambda value: value['checkpoint_selection']['best'].update(epoch=53))
    rejected_metadata('changed_direct_export_hash', lambda value: value['mx_exports'][ROLES[0]]['8'].update(sha256='0' * 64))
    rejected_metadata('truncated_source_export', lambda value: value['mx_exports'][ROLES[0]]['8'].update(truncation=True))
    rejected_metadata('unsafe_source_export_path', lambda value: value['mx_exports'][ROLES[0]]['8'].update(file='../outside.pt'))
    before_full_fixture = tree_hashes(full_fixture)

    # The evaluate fixture uses the exact same frozen export files, with three
    # artificial test examples and self-consistent direct historical counts.
    evaluate_fixture = out / 'SYNTHETIC_evaluation_reference'
    evaluate_fixture.mkdir()
    for path in full_fixture.iterdir():
        if path.is_file():
            (evaluate_fixture / path.name).write_bytes(path.read_bytes())
    evaluate_report = copy.deepcopy(fixture_report)
    evaluate_report['synthetic_fixture_only'] = True
    evaluate_report['config']['test_examples'] = len(labels)
    evaluate_report['dataset_selection']['test'].update(n=len(labels), label_sha256=label_hash)
    # Actual inference fixtures keep truthful synthetic provenance. The full
    # production-contract guard fixture above is never used for inference.
    for role in ROLES:
        for bit in (8, 6, 4):
            item = evaluate_report['mx_exports'][role][str(bit)]
            payload = copy.deepcopy(direct_payloads[role][bit])
            payload['provenance']['synthetic_fixture_only'] = True
            path = evaluate_fixture / item['file']
            path.unlink()
            path.with_suffix('.pt.sha256').unlink()
            saved = core.atomic_save(path, payload)
            item.update(saved)
    evaluate_selected = csv_rows(evaluate_fixture / 'selected_metrics.csv')
    for row in evaluate_selected:
        values = expected_metrics[row['role']][int(row['weight_bits'])]
        row.update(n=len(labels), **values)
        row['top1_percent'] = row['count_top1_percent'] = 100 * values['correct1'] / len(labels)
        row['top5_percent'] = 100 * values['correct5'] / len(labels)
    (evaluate_fixture / 'selected_metrics.csv').unlink()
    support.write_csv(evaluate_fixture / 'selected_metrics.csv', evaluate_selected)
    json_write(evaluate_fixture / 'run_report.json', evaluate_report)
    before_evaluate_fixture = tree_hashes(evaluate_fixture)

    def fixture_inspection(ref, runtime):
        assert Path(ref) == evaluate_fixture and runtime.device.type == 'cpu'
        return copy.deepcopy(evaluate_report), tree_hashes(ref), copy.deepcopy(evaluate_selected)

    def fixture_data(args, current_modules, reference, path, log):
        assert reference['config']['test_examples'] == len(labels)
        support.write_json(path / 'dataset_manifest.json', {
            'synthetic_fixture_only': True, 'test_file': {'path': str(data_file), **data_hashes},
            'selection': {'n': len(labels), 'label_sha256': label_hash}, 'train_read': False, 'extra_read': False})
        return loader, data_file, data_hashes

    def fixture_dependencies(path, log):
        (path / 'pip_freeze.txt').write_text('SYNTHETIC CPU integration: dependencies inspected separately.\n')
        (path / 'pip_check.txt').write_text('Dependency operation replaced only by test harness.\n')

    def fixture_runtime(seed, requested_device, strict=True):
        assert requested_device in ('cuda', 'cuda:0')
        return ORIGINAL_RUNTIME(seed, 'cpu', strict=False)

    real_evaluate = runner.evaluate

    def marked_evaluate(args, path, report, runtime, current_modules, log):
        report['synthetic_fixture_only'] = True
        report['limitations'].append('SYNTHETIC CPU SOFTWARE TEST; three artificial images and random exports, no SVHN result.')
        result = real_evaluate(args, path, report, runtime, current_modules, log)
        report['synthetic_fixture_only'] = True
        return result

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError('Post-hoc evaluation attempted to construct an optimizer')

    def forbidden_backward(*args, **kwargs):
        raise AssertionError('Post-hoc evaluation attempted backward')

    cases = {}

    def invoke(command, case_name=None, expected_status='PASS'):
        case_name = case_name or command
        arguments = [str(ROOT / 'run_eval.py'), command, '--output-root', str(out / case_name)]
        if command == 'evaluate':
            arguments += ['--reference-run', str(evaluate_fixture)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', arguments))
            stack.enter_context(patch.object(runner, 'dependencies', fixture_dependencies))
            stack.enter_context(patch.object(support, 'runtime', fixture_runtime))
            stack.enter_context(patch.object(support, 'import_source', fresh_source_import))
            stack.enter_context(patch.object(torch.optim.Optimizer, '__init__', forbidden_optimizer))
            stack.enter_context(patch.object(torch.Tensor, 'backward', forbidden_backward))
            stack.enter_context(patch.object(torch.autograd, 'backward', forbidden_backward))
            if command == 'evaluate':
                stack.enter_context(patch.object(runner, 'inspect_reference', fixture_inspection))
                stack.enter_context(patch.object(runner, 'prepare_test_only', fixture_data))
                stack.enter_context(patch.object(runner, 'evaluate', marked_evaluate))
            with (out / (case_name + '_stdout.log')).open('x', encoding='utf-8') as output:
                stack.enter_context(redirect_stdout(output))
                exit_code = runner.main()
        paths = list((out / case_name).glob('*_' + command))
        assert len(paths) == 1
        path = paths[0]
        report = json.loads((path / 'run_report.json').read_text())
        assert exit_code == (0 if expected_status == 'PASS' else 1) and report['status'] == expected_status, (case_name, report.get('error'), report.get('traceback'))
        assert report['optimizer_steps'] == 0 and report['batchnorm_recalibration'] is False
        assert report['new_checkpoint_selection'] is False
        with ZipFile(path / 'feedback_bundle.zip') as archive:
            assert archive.testzip() is None and 'run_report.json' in archive.namelist()
        cases[case_name] = {'status': report['status'], 'expected_status': expected_status,
                            'output_directory': str(path)}
        return path, report

    check_path, check_report = invoke('check')
    eval_path, eval_report = invoke('evaluate')
    assert eval_report['synthetic_fixture_only'] is True
    assert set(eval_report['roles']) == set(ROLES)
    assert eval_report['input_files_unchanged'] is True and eval_report['test_file_unchanged'] is True
    assert eval_report['train_and_extra_read'] is False
    assert eval_report['original_checkpoint_selection_unchanged'] is True
    assert eval_report['historical_correct_counts_all_match'] is True
    assert eval_report['derived_models_all_evaluated_from_saved_and_reloaded_codes'] is True
    assert eval_report['reference']['checkpoint_files_opened'] is False
    assert eval_report['reference']['FP32_master_weights_opened'] is False
    metrics = csv_rows(eval_path / 'metrics.csv')
    pairs = csv_rows(eval_path / 'paired_metrics.csv')
    assert len(metrics) == 2 * len(tr.PATHS)
    assert len(pairs) == 2 * len(ev.COMPARISONS)
    assert len(csv_rows(eval_path / 'weight_pair_metrics.csv')) == 2 * 6 * len(tr.WEIGHT_COMPARISONS)
    assert len(csv_rows(eval_path / 'layer_code_metrics.csv')) == 2 * 6 * len(tr.PATHS)
    assert len(csv_rows(eval_path / 'historical_reproduction.csv')) == 6
    for role in ROLES:
        role_report = eval_report['roles'][role]
        assert role_report['all_model_and_bn_states_unchanged']
        assert role_report['all_models_no_grad'] and role_report['all_paths_use_frozen_target_precision_bn']
        assert role_report['six_FP32_master_parameters_absent_all_paths']
        assert role_report['optimizer_steps'] == 0 and role_report['batchnorm_recalibrated'] is False
        assert role_report['n_test'] == 3 and role_report['label_sha256'] == label_hash
        assert set(role_report['derived_exports']) == set(tr.DERIVED_PATHS)
        for key, details in role_report['derived_exports'].items():
            assert details['saved_and_reloaded'] and details['source_scale_bytes_copied']
            assert details['master_weights_read_to_derive'] is False
            payload = ev.read_export(eval_path / details['file'], details['sha256'])
            tr.validate_derived(payload)
            assert payload['derivation']['path_id'] == key
        with np.load(eval_path / role_report['per_image_file'], allow_pickle=False) as archive:
            assert np.array_equal(archive['indices'], np.arange(3))
            assert np.array_equal(archive['labels'], labels.numpy())
            for row in (x for x in metrics if x['role'] == role):
                key = row['path']
                prediction = archive[key + '__predictions']
                logits = archive[key + '__logits']
                ce = archive[key + '__per_image_ce']
                correct = int((prediction == labels.numpy()).sum())
                assert int(row['n']) == 3 and int(row['correct1']) == correct
                assert float(row['top1_percent']) == 100 * correct / 3
                assert math.isclose(float(row['hard_cross_entropy']), ce.astype(np.float64).mean())
                assert logits.shape == (3, 10) and np.isfinite(logits).all()
                assert row['bn_bank_key'] == tr.PATH_SPEC[key]['bn_bank_key']
            for row in (x for x in pairs if x['role'] == role):
                ref, other = row['reference_path'], row['other_path']
                calculated = ev.paired_outputs(
                    {field: archive[ref + '__' + field] for field in ('predictions', 'logits')},
                    {field: archive[other + '__' + field] for field in ('predictions', 'logits')},
                    archive['labels'])
                for name, value in calculated.items():
                    assert math.isclose(float(row[name]), value), (name, row[name], value)
                assert int(row['both_correct']) + int(row['both_wrong']) + int(row['wrong_to_correct']) + int(row['correct_to_wrong']) == 3
    notebook_files = list(eval_path.glob('*.ipynb'))
    assert len(notebook_files) == 1
    notebook = json.loads(notebook_files[0].read_text())
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in notebook['cells'] if c['cell_type'] == 'code')
    with ZipFile(eval_path / 'feedback_bundle.zip') as archive:
        assert all(eval_report['roles'][role]['per_image_file'] in archive.namelist() for role in ROLES)

    # Confirm that the real evaluate path rejects bad source tensors/provenance
    # even after file hashes are valid. These mutations touch fixture copies only.
    source_item = evaluate_report['mx_exports'][ROLES[0]]['8']
    source_file = evaluate_fixture / source_item['file']
    source_marker = source_file.with_suffix('.pt.sha256')
    original_source_bytes = source_file.read_bytes()
    original_marker_bytes = source_marker.read_bytes()
    original_item = copy.deepcopy(source_item)
    for case_name, mutation in (
        ('reject_payload_checkpoint', lambda value: value['provenance'].update(checkpoint_sha256='0' * 64)),
        ('reject_payload_format', lambda value: value.update(weight_format='MXFP6_E2M3')),
        ('reject_payload_truncated', lambda value: value.update(truncation=True)),
    ):
        payload = torch.load(source_file, map_location='cpu', weights_only=True)
        mutation(payload)
        source_file.unlink()
        source_marker.unlink()
        source_item.update(core.atomic_save(source_file, payload))
        failed_path, failed_report = invoke('evaluate', case_name=case_name, expected_status='FAIL')
        assert failed_report['optimizer_steps'] == 0
        assert not (failed_path / 'metrics.csv').exists()
        assert not list(failed_path.glob('*_derived.pt'))
        source_file.write_bytes(original_source_bytes)
        source_marker.write_bytes(original_marker_bytes)
        source_item.clear()
        source_item.update(copy.deepcopy(original_item))

    bad_cli = [['run'], ['check', '--device', 'cpu'], ['check', '--download'],
               ['evaluate', '--reference-run', str(evaluate_fixture), '--output-root', str(evaluate_fixture / 'inside')],
               ['check', '--reference-run', str(evaluate_fixture)], ['check', '--synthetic'], ['check', '--epochs', '5']]
    cli_results = []
    for arguments in bad_cli:
        completed = subprocess.run([sys.executable, str(ROOT / 'run_eval.py'), *arguments], cwd=out,
                                   capture_output=True, text=True, timeout=45)
        assert completed.returncode == 2, (arguments, completed.stdout, completed.stderr)
        cli_results.append({'arguments': arguments, 'exit_code': completed.returncode})

    assert tree_hashes(full_fixture) == before_full_fixture
    assert tree_hashes(evaluate_fixture) == before_evaluate_fixture
    assert support.hashes(data_file) == data_hashes
    assert core.file_sha256(feedback) == feedback_hash
    assert support.verify_package() == package_hashes
    result = {'status': 'PASS', 'schema': 'mx-triple-posthoc-integration-validation-v1',
              'synthetic_fixture_only': True,
              'scope': 'Actual check/evaluate CLI with three synthetic CPU images, random exports and test-only dependency/runtime/data patches',
              'no_real_SVHN_data_or_user_checkpoint_loaded': True, 'no_training_or_BN_recalibration': True,
              'production_bypass_added': False, 'executed_notebook_delivered': False,
              'protocol_guard_fixture_note': 'Private synthetic exports plus copied real protocol metadata; not research results.',
              'runtime_environment': rt.environment, 'package_sha256': package_hashes,
              'feedback_metadata_input_sha256': feedback_hash, 'cases': cases, 'reference_guard_tests': guard_tests,
              'CLI_guards': cli_results, 'metric_rows': len(metrics), 'paired_rows': len(pairs),
              'paths_per_checkpoint': len(tr.PATHS), 'checkpoint_roles': list(ROLES),
              'per_image_logits_counts_CE_and_paired_tables_consistent': True,
              'all_10_derived_models_saved_reloaded_and_evaluated': True,
              'historical_direct_correct_counts_match_synthetic_baselines': True,
              'all_inputs_and_package_unchanged': True,
              'seconds': time.perf_counter() - started}
    json_write(out / 'posthoc_integration_report.json', result)
    print(json.dumps({'status': 'PASS', 'report': str(out / 'posthoc_integration_report.json'),
                      'seconds': result['seconds'], 'no_SVHN_or_training': True}, indent=2))
    return result


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
