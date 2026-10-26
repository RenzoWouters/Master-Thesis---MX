"""Execute the actual check command with a genuine synthetic CPU parent55.

Only the device, synthetic evidence pins and unpublished outer package manifest
are test substitutions. Inherited package/codec/gradient checks, exact parent
restore, selected-input preflight, reporting and archive paths run unchanged.
No SVHN data is loaded and no CUDA numerical assertion is made.
"""
from __future__ import annotations
from contextlib import ExitStack, redirect_stdout
import importlib.util
import json
from pathlib import Path
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


def main(package, output, reference_root):
    package, output, reference_root = map(lambda p:Path(p).resolve(), (package, output, reference_root))
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sys.path.insert(0, str(package))
    r = load_module('freeze100_check_entry_runner', package / 'run_train.py')
    import continuation_core as cc
    import torch
    torch.set_num_threads(2)
    paths = json.loads((reference_root / 'reference_paths.json').read_text(encoding='utf-8'))
    evidence = cc.load_evidence(Path(paths['dynamic_run']), Path(paths['pilot_dir']),
        Path(paths['dynamic_dir']), paths['expected_pilot_sha'], paths['expected_dynamic_sha'])
    parent55 = Path(paths['parent_checkpoint55'])
    parent_sha = r.core.file_sha256(parent55)
    code_binding = {name:r.core.file_sha256(package / name) for name in
        ('run_train.py', 'continuation_core.py', 'continuation_bridge.py')}
    package_id = {'synthetic_fixture_only_check_sha256':r.core.file_sha256(Path(__file__))}
    original_runtime = r.s.runtime

    def cpu_runtime(seed, device='cpu', strict=True):
        return original_runtime(seed, 'cpu', strict=False)

    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor

    argv = [str(package / 'run_train.py'), 'check', '--parent-checkpoint', str(parent55),
        '--dynamic-run', str(evidence['dynamic_run']), '--output-root', str(output / 'check_results')]
    with (output / 'stdout.log').open('x', encoding='utf-8') as stdout:
        with redirect_stdout(stdout), ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(r.s, 'runtime', cpu_runtime))
            stack.enter_context(patch.object(torch.Tensor, 'cuda', cpu_cuda))
            stack.enter_context(patch.object(r, 'verify_package', lambda:package_id))
            stack.enter_context(patch.object(r, 'load_evidence', lambda *args, **kwargs:evidence))
            exit_code = r.main()
    runs = list((output / 'check_results').iterdir())
    assert len(runs) == 1
    out = runs[0]
    report = json.loads((out / 'run_report.json').read_text(encoding='utf-8'))
    assert exit_code == 0 and report['status'] == 'PASS', report.get('traceback')
    verification = report['parent55_verification']
    assert verification['parent_model_matches_reference']
    assert verification['parent_contract_matches_reference']
    assert verification['parent_trace_matches_reference']
    assert all(verification['strict_restore_checks'].values())
    assert verification['parent_epoch'] == 55
    inherited = report['inherited_technical_checks']
    assert inherited['codec_tests']['status'] == 'PASS'
    assert inherited['synthetic_training_checks']['status'] == 'PASS'
    assert report['freeze_tests']['status'] == 'PASS'
    assert all(report['freeze_tests']['checks'].values())
    assert report['dataset_downloaded'] is False
    assert report['full_dataset_verified'] is False
    assert report['continuation_completed'] is False
    assert report['synthetic_fixture_only'] is True
    assert report['check_only'] is True
    assert not (out / 'checkpoints').exists()
    assert [item['epoch'] for item in report['selected_input_preflight']] == [16, 51]
    assert r.core.file_sha256(parent55) == parent_sha
    assert r.s.verify_package() == evidence['dynamic_contract']['package_sha256']
    assert r.core.file_sha256(evidence['dynamic_run'] / 'history.json') == evidence['dynamic_meta']['source_history_sha256']
    assert (out / 'parent_training_change_manifest.json').is_file()
    assert (out / 'training_change_manifest.json').is_file()
    with ZipFile(out / 'feedback_bundle.zip') as archive:
        names = set(archive.namelist())
        assert {'run_report.json', 'freeze_check_report.json', 'parent_training_change_manifest.json',
            'training_change_manifest.json', 'pilot_reference/REFERENCE.json',
            'dynamic_reference/history.json'}.issubset(names)
        assert not any(name.endswith('.pt') for name in names)
    for name, digest in code_binding.items():
        assert r.core.file_sha256(package / name) == digest
    result = dict(status='PASS', synthetic_fixture_only=True, CUDA_tested=False,
        no_SVHN_data_loaded_or_downloaded=True, not_a_research_accuracy_or_GPU_timing_result=True,
        actual_check_main_exit_code=exit_code, original_codec_and_gradient_checks_unchanged=True,
        strict_parent55_model_optimizer_scheduler_rng_restored=True,
        original_pre16_and_pilot51_checkpoints_preflighted=True,
        no_new_training_epochs_run=True, nested_manifests_archived_without_overwrite=True,
        feedback_bundle_verified=True, parent_checkpoint_sha256=parent_sha,
        code_binding=code_binding, test_file_sha256=r.core.file_sha256(Path(__file__)),
        duration_seconds=time.perf_counter() - started,
        actual_check_report=str(out / 'run_report.json'),
        parent55_verification=verification, freeze_tests=report['freeze_tests'])
    r.s.write_json(output / 'check_entry_validation_report.json', result)
    print(json.dumps(dict(status='PASS', report=str(output / 'check_entry_validation_report.json'),
        seconds=result['duration_seconds']), indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit('Usage: test_check_entry.py PACKAGE OUTPUT REFERENCES')
    main(*sys.argv[1:])
