"""Run the actual check entry on CPU in this local test process only.

Production CLI/runtime requirements stay unchanged. No data is loaded and
no SVHN training is run. The authors' explicit Tensor.cuda calls are redirected
only for this CPU integration test. Production/laptop check uses actual CUDA.
"""
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch
import run_train as runner
import svhn_support as support


def main(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    production_runtime = support.runtime
    before = support.verify_package()

    def cpu_runtime(seed, requested_device):
        assert requested_device in ('cuda', 'cuda:0')
        return production_runtime(seed, 'cpu', strict=False)

    def cpu_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu'
        return tensor

    with (destination / 'check_stdout.log').open('x', encoding='utf-8') as stdout:
        with redirect_stdout(stdout), patch.object(support, 'runtime', cpu_runtime), \
             patch.object(torch.Tensor, 'cuda', cpu_cuda), \
             patch.object(sys, 'argv', [str(ROOT / 'run_train.py'), 'check',
                                        '--output-root', str(destination / 'output')]):
            exit_code = runner.main()
    paths = list((destination / 'output').glob('*_joint_check'))
    assert len(paths) == 1
    report = json.loads((paths[0] / 'run_report.json').read_text(encoding='utf-8'))
    assert exit_code == 0 and report['status'] == 'PASS', report.get('traceback')
    assert report['synthetic_checks_only'] and report['dataset_downloaded'] is False
    assert report['codec_tests']['status'] == 'PASS'
    assert report['synthetic_training_checks']['status'] == 'PASS'
    assert (paths[0] / 'codec_report.json').is_file()
    assert (paths[0] / 'joint_check_report.json').is_file()
    assert (paths[0] / 'feedback_bundle.zip').is_file()
    assert support.verify_package() == before
    result = {'status': 'PASS', 'synthetic_fixture_only': True,
              'scope': 'actual main/check entry with CPU-only runtime adaptation',
              'no_SVHN_data_or_training': True, 'CUDA_tested_here': False,
              'environment': report['environment'], 'package_sha256': before,
              'codec_status': report['codec_tests']['status'],
              'joint_status': report['synthetic_training_checks']['status'],
              'production_files_unchanged': True,
              'check_report_directory': str(paths[0])}
    (destination / 'check_entry_validation_report.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'report': str(destination / 'check_entry_validation_report.json')}))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: test_check_entry.py NEW_OUTPUT_DIRECTORY')
    main(sys.argv[1])
