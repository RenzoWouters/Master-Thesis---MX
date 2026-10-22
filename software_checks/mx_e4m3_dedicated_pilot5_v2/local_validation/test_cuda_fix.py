"""Focused v2 regression tests on CPU; no SVHN, optimizer updates or GPU claim.

The production runner remains CUDA-only. A subprocess-local runtime redirect
exercises its real ``check`` entry point on CPU. Its separately tested CNN smoke
is explicitly stubbed here so these tests concern codec gates and report
preservation only. Neither package hashes nor source checks are bypassed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch
import mx_e4m3 as mx
import test_e4m3 as codec_tests


FIRST_CHECK = 'all_254_finite_codes_match_independent_decode'


def _write(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def _broken_decoder(original):
    """Inject two distinct exactness failures, without changing source files."""
    def decode(codes):
        result = original(codes)
        advanced = torch.nextafter(result, torch.full_like(result, math.inf))
        result = torch.where(codes == 56, advanced, result)  # 1.0 -> +1 FP64 ULP.
        return torch.where(codes == 128, torch.zeros_like(result), result)  # -0 -> +0.
    return decode


def _check_injected_report(report: dict) -> dict:
    assert report['status'] == 'FAIL' and report['failed_check'] == FIRST_CHECK
    assert report['checks'] == {FIRST_CHECK: False}
    detail = report['exact_comparisons'][FIRST_CHECK]
    assert detail['exact'] is False
    assert detail['numeric_mismatch_count'] == 1 and detail['signbit_mismatch_count'] == 1
    differences = {row['label']: row for row in detail['first_mismatches']}
    assert set(differences) == {56, 128}
    assert differences[56]['same_sign_ulp_distance'] == 1
    assert differences[56]['expected_hex'] == float(1.0).hex()
    assert differences[128]['actual_fp64_bits'] == '0x0000000000000000'
    assert differences[128]['expected_fp64_bits'] == '0x8000000000000000'
    assert differences[128]['actual_signbit'] is False and differences[128]['expected_signbit'] is True
    assert report['training_performed'] is False and report['dataset_used'] is False
    json.dumps(report, allow_nan=False)
    return detail


def _runner_case(case: str, out: Path) -> dict:
    """Fresh interpreter per case: no cross-run upstream-module cache leakage."""
    import run_train as runner
    import smoke_single

    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    original_runtime = runner.s.runtime
    original_decode = mx.decode_elements

    def cpu_test_runtime(seed, requested_device, strict=True):
        assert requested_device in ('cuda', 'cuda:0') and strict is True
        return original_runtime(seed, 'cpu', strict=False)

    smoke_result = {'status': 'TEST_STUB_NO_OPTIMIZER_UPDATES',
                    'purpose': 'report-flow test only; CNN smoke tested separately',
                    'synthetic_fixture_only': True, 'optimizer_updates': 0}
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner.s, 'runtime', cpu_test_runtime))
        smoke = stack.enter_context(patch.object(smoke_single, 'check', return_value=smoke_result))
        data = stack.enter_context(patch.object(runner.s, 'ensure_dataset',
            side_effect=AssertionError('No dataset access is permitted in this check-only regression.')))
        epochs = stack.enter_context(patch.object(runner, 'run_epochs',
            side_effect=AssertionError('No training is permitted in this check-only regression.')))
        exports = stack.enter_context(patch.object(runner, 'export_results',
            side_effect=AssertionError('No training export is permitted in this check-only regression.')))
        stack.enter_context(patch.object(sys, 'argv', [str(ROOT / 'run_train.py'), 'check',
                                                      '--output-root', str(out / 'runs')]))
        if case == 'injected_failure':
            stack.enter_context(patch.object(mx, 'decode_elements', _broken_decoder(original_decode)))
        return_code = runner.main()
        assert not data.called and not epochs.called and not exports.called
        assert smoke.call_count == (1 if case == 'success' else 0)

    folders = list((out / 'runs').iterdir())
    assert len(folders) == 1
    run_dir = folders[0]
    report = json.loads((run_dir / 'run_report.json').read_text(encoding='utf-8'))
    codec_report = json.loads((run_dir / 'codec_report.json').read_text(encoding='utf-8'))
    legacy = json.loads((run_dir / 'legacy_codec_diagnostics.json').read_text(encoding='utf-8'))
    assert report['codec_tests'] == codec_report
    assert report['legacy_codec_diagnostics'] == legacy
    assert report['command'] == 'check'
    assert report['environment']['device'] == 'cpu'
    assert legacy['finite_code_decode']['exact'] and legacy['e8m0_powers']['exact']
    assert 'completed_epoch' not in report and 'optimizer_steps' not in report
    for absent in ('checkpoints', 'dataset_manifest.json', 'epoch_metrics.csv', 'training_summary.json'):
        assert not (run_dir / absent).exists(), absent
    required = {'run_report.json', 'codec_report.json', 'legacy_codec_diagnostics.json'}
    with ZipFile(run_dir / 'feedback_bundle.zip') as feedback:
        assert required.issubset(feedback.namelist())
        for name in required:
            assert feedback.read(name) == (run_dir / name).read_bytes()

    if case == 'success':
        assert return_code == 0 and report['status'] == 'PASS' and codec_report['status'] == 'PASS'
        assert report['synthetic_training_checks'] == smoke_result
        assert report['dataset_downloaded'] is False and report['synthetic_checks_only'] is True
        detail = None
    else:
        assert return_code == 1 and report['status'] == 'FAIL'
        detail = _check_injected_report(codec_report)
        assert FIRST_CHECK in report['error']
        assert 'synthetic_training_checks' not in report
    result = {'status': 'PASS', 'case': case, 'runner_exit_code': return_code,
              'run_directory': str(run_dir), 'feedback_contains_all_three_reports': True,
              'package_and_source_hash_checks_executed': True,
              'cpu_redirect_is_test_only': True, 'CNN_smoke_is_explicit_test_stub': True,
              'dataset_access': False, 'optimizer_updates': 0,
              'injected_failure_details': detail}
    _write(out / 'runner_case_result.json', result)
    return result


def main(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    strict = codec_tests.run_tests('cpu')
    assert strict['status'] == 'PASS'
    assert strict['exact_comparisons']['all_255_scale_powers_are_exact']['exact']
    assert strict['exact_comparisons']['all_64770_code_scale_products_are_exact_fp64']['exact']
    _write(out / 'strict_codec_cpu_report.json', strict)

    legacy = codec_tests.legacy_decode_diagnostics('cpu')
    assert legacy['finite_code_decode']['exact'] and legacy['e8m0_powers']['exact']
    _write(out / 'legacy_cpu_report.json', legacy)

    injected = {}
    with patch.object(mx, 'decode_elements', _broken_decoder(mx.decode_elements)):
        try:
            codec_tests.run_tests('cpu', report=injected)
        except AssertionError as exc:
            assert FIRST_CHECK in str(exc)
        else:
            raise AssertionError('A one-ULP and signed-zero error must not pass the codec gate.')
    _check_injected_report(injected)
    _write(out / 'injected_failure_cpu_report.json', injected)

    nonfinite = codec_tests._comparison_details(
        torch.tensor([float('nan'), float('inf'), -float('inf'), -0.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0, -3.0, 0.0], dtype=torch.float64),
        labels=['nan', 'positive_inf', 'negative_inf', 'negative_zero'])
    assert nonfinite['numeric_mismatch_count'] == 3 and nonfinite['signbit_mismatch_count'] == 1
    assert nonfinite['max_abs_error'] == 'nan'
    assert all(row['same_sign_ulp_distance'] is None for row in nonfinite['first_mismatches'])
    _write(out / 'nonfinite_diagnostics_json_safe.json', nonfinite)

    runner_results = {}
    for case in ('success', 'injected_failure'):
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--runner-case', case,
                   str(out / case)]
        process = subprocess.run(command, cwd=out, text=True, encoding='utf-8',
                                 errors='replace', capture_output=True, timeout=120)
        with (out / (case + '_process.log')).open('x', encoding='utf-8') as handle:
            handle.write(process.stdout + process.stderr)
        assert process.returncode == 0, (case, process.returncode, process.stdout[-2000:], process.stderr[-2000:])
        runner_results[case] = json.loads((out / case / 'runner_case_result.json').read_text(encoding='utf-8'))

    report = {'status': 'PASS', 'scope': 'Focused v2 codec and real check-entry reporting regression on CPU',
              'torch_version': torch.__version__, 'device': 'cpu',
              'strict_codec_checks': len(strict['checks']),
              'strict_expected_rejections': len(strict['expected_rejection_cases']),
              'finite_codes_checked': 254, 'scale_powers_checked': 255,
              'code_scale_products_checked': 64770,
              'one_ulp_and_signed_zero_fault_rejected_with_exact_details': True,
              'nan_inf_diagnostics_serialize_with_allow_nan_false': True,
              'legacy_CPU_probe_exact': True, 'runner_cases': runner_results,
              'training_performed': False, 'optimizer_updates': 0, 'dataset_used': False,
              'not_a_CUDA_runtime_validation': True}
    _write(out / 'cuda_fix_validation_report.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner-case', choices=['success', 'injected_failure'])
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    if args.runner_case:
        _runner_case(args.runner_case, args.output_directory.resolve())
    else:
        main(args.output_directory.resolve())
