r"""Lees de Python/PyTorch-omgeving zonder pakketten te installeren of code te wijzigen.

Gebruik in PowerShell, vanuit de projectmap:
  .\.venv\Scripts\python.exe -B .\check_training_environment.py --project-root .

Het script schrijft uitsluitend nieuwe rapporten in environment_checks/<tijdstip>/.
Een GPU-rekentest gebeurt alleen als PyTorch al geinstalleerd is. Er worden geen
modellen of datasets gedownload en er wordt geen onderzoekstraining uitgevoerd.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import traceback
from typing import Any

sys.dont_write_bytecode = True


def command(args: list[str], timeout: int = 60) -> dict[str, Any]:
    """Voer een lokale lees-/diagnoseopdracht uit en bewaar ook fouten."""
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, errors='replace', timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1'),
        )
        return {'command': args, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': args, 'returncode': None, 'stdout': '',
                'stderr': f'{type(exc).__name__}: {exc}'}


def torch_probe() -> dict[str, Any]:
    """Aparte procescontrole: een importfout verliest niet het hoofdrapport."""
    report: dict[str, Any] = {}
    try:
        import torch
    except Exception:
        report['torch_import'] = {'status': 'FAILED', 'traceback': traceback.format_exc()}
        return report
    report['torch_import'] = {'status': 'OK', 'version': torch.__version__}
    report['torch_cuda_build'] = torch.version.cuda
    report['cudnn_version'] = torch.backends.cudnn.version()
    try:
        import torchvision
        report['torchvision_import'] = {'status': 'OK', 'version': torchvision.__version__}
    except Exception:
        report['torchvision_import'] = {'status': 'FAILED', 'traceback': traceback.format_exc()}
    torch.set_num_threads(1)
    torch.manual_seed(42)

    def compute_test(device: str) -> dict[str, Any]:
        try:
            # GEMM plus backward; this tests execution, not just GPU detection.
            a = torch.randn(64, 64, device=device, requires_grad=True)
            b = torch.randn(64, 64, device=device, requires_grad=True)
            product = a @ b
            matrix_loss = product.square().mean()
            matrix_loss.backward()
            # Convolution plus backward exercises a second relevant backend.
            layer = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
            image = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
            output = layer(image)
            conv_loss = output.square().mean()
            conv_loss.backward()
            # Operations used by the supplied integer quantizers.
            values = torch.tensor([0.0, 0.21, 0.5, 1.0], device=device)
            codes = torch.round(values * 255).to(torch.uint8) >> 6
            expected = torch.tensor([0, 0, 2, 3], dtype=torch.uint8, device=device)
            gradients = [a.grad, b.grad, image.grad, layer.weight.grad]
            finite = all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
            finite = finite and bool(torch.isfinite(matrix_loss)) and bool(torch.isfinite(conv_loss))
            quant_ops_ok = bool(torch.equal(codes, expected))
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            if not finite or not quant_ops_ok:
                raise RuntimeError(f'finite={finite}, quantizer_operations={quant_ops_ok}')
            return {'status': 'OK', 'matrix_forward_backward': True,
                    'conv_forward_backward': True, 'finite_values_and_gradients': finite,
                    'uint8_shift': quant_ops_ok}
        except Exception:
            return {'status': 'FAILED', 'traceback': traceback.format_exc()}

    report['cpu_test'] = compute_test('cpu')
    try:
        available = bool(torch.cuda.is_available())
        report['cuda_available'] = available
        report['cuda_device_count'] = torch.cuda.device_count()
        report['compiled_cuda_architectures'] = torch.cuda.get_arch_list() if available else []
        report['cuda_devices'] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report['cuda_devices'].append({'index': index, 'name': props.name,
                'compute_capability': list(torch.cuda.get_device_capability(index)),
                'total_memory_GiB': props.total_memory / 2**30})
        report['cuda_test'] = compute_test('cuda:0') if available else {
            'status': 'SKIPPED', 'reason': 'torch.cuda.is_available() is False'}
    except Exception:
        report['cuda_test'] = {'status': 'FAILED', 'traceback': traceback.format_exc()}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path.cwd())
    parser.add_argument('--probe-torch', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_torch:
        print(json.dumps(torch_probe(), indent=2))
        return 0

    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Projectmap bestaat niet: {root}')
    timestamp = datetime.now().astimezone()
    directory = root / 'environment_checks' / timestamp.strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        'timestamp': timestamp.isoformat(),
        'purpose': 'Omgevingsdiagnose; geen dataset-, referentie- of MX-training',
        'project_root': str(root), 'python_executable': sys.executable,
        'python_version': sys.version, 'python_implementation': platform.python_implementation(),
        'python_pointer_bits': struct.calcsize('P') * 8,
        'python_prefix': sys.prefix, 'python_base_prefix': sys.base_prefix,
        'in_virtual_environment': sys.prefix != sys.base_prefix,
        'platform': platform.platform(), 'machine': platform.machine(),
        'logical_cpu_count': os.cpu_count(),
        'free_disk_GiB': shutil.disk_usage(root).free / 2**30,
    }
    distributions = ['pip', 'torch', 'torchvision', 'numpy', 'Pillow', 'tensorboardX', 'gpustat', 'scipy']
    report['package_versions'] = {}
    for name in distributions:
        try:
            report['package_versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report['package_versions'][name] = None
    report['pip_version'] = command([sys.executable, '-B', '-m', 'pip', '--version'])
    report['pip_check'] = command([sys.executable, '-B', '-m', 'pip', 'check'])
    freeze = command([sys.executable, '-B', '-m', 'pip', 'freeze', '--all'])
    report['pip_freeze'] = freeze
    with (directory / 'pip_freeze.txt').open('x', encoding='utf-8') as stream:
        stream.write(freeze['stdout'])
        if freeze['stderr']:
            stream.write('\n# STDERR\n' + freeze['stderr'])
    report['nvidia_smi'] = command(['nvidia-smi'], timeout=20)
    if sys.platform == 'win32':
        report['python_launcher'] = command(['py', '-0p'], timeout=20)
        report['physical_memory_bytes'] = command([
            'powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
            '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'], timeout=30)
    if report['package_versions']['torch'] is None:
        report['torch_probe'] = {'status': 'SKIPPED', 'reason': 'PyTorch is niet geinstalleerd.'}
    else:
        probe = command([sys.executable, '-B', str(Path(__file__).resolve()), '--probe-torch'], timeout=120)
        report['torch_probe_process'] = probe
        try:
            report['torch_probe'] = json.loads(probe['stdout'])
        except (json.JSONDecodeError, TypeError):
            report['torch_probe'] = {'status': 'FAILED', 'reason': 'Geen geldig JSON-resultaat; zie procesuitvoer.'}
    with (directory / 'environment_report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    lines = ['OMGEVINGSCONTROLE - geen onderzoekstraining',
             f"Python: {platform.python_version()} ({report['python_pointer_bits']} bit)",
             f'Interpreter: {sys.executable}',
             f"Virtual environment: {report['in_virtual_environment']}",
             f"Vrije schijfruimte: {report['free_disk_GiB']:.1f} GiB", '', 'Pakketten:']
    lines += [f"  {key}: {value if value is not None else 'NIET GEINSTALLEERD'}"
              for key, value in report['package_versions'].items()]
    lines += ['', 'pip check:', (report['pip_check']['stdout'] or report['pip_check']['stderr']).strip(),
              '', 'nvidia-smi:', (report['nvidia_smi']['stdout'] or report['nvidia_smi']['stderr']).strip(),
              '', 'PyTorch-tests:', json.dumps(report['torch_probe'], ensure_ascii=False, indent=2),
              '', f'Rapportmap: {directory}',
              'Deel deze uitvoer of environment_report.json. Er zijn geen pakketten geinstalleerd.']
    summary = '\n'.join(lines)
    with (directory / 'summary.txt').open('x', encoding='utf-8') as stream:
        stream.write(summary + '\n')
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
