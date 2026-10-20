"""Reproducible OFFLINE synthetic tests. Does not invoke production bypasses.
Usage: python local_validation/test_all.py <NEW_output_directory> [--notebook]
The --notebook option additionally requires nbformat/nbclient in the test environment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import svhn_support as s
import resume_core as core
import evaluation as ev
import truncation as tr
import test_truncation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--notebook', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source, info = s.legacy.prepare_source(out)
    rt = s.runtime(20261009, 'cpu', strict=False)
    modules = s.import_source(source)
    result = test_truncation.run_tests(modules, rt.device, out)
    # Standalone child sees only high export and new synthetic input images.
    isolated = out / 'isolated_high_source'; isolated.mkdir()
    hp = out / 'synthetic_tests/synthetic_high_only.pt'
    high_file = isolated / hp.name
    shutil.copy2(hp, high_file); shutil.copy2(hp.with_suffix('.pt.sha256'), high_file.with_suffix('.pt.sha256'))
    torch.manual_seed(889)
    inputs = torch.randn(5, 3, 40, 40)
    images = isolated / 'synthetic_images.pt'
    torch.save(inputs, images)
    childout = out / 'child'
    child = subprocess.run([sys.executable, str(ROOT / 'local_validation/fresh_source_check.py'),
                            str(high_file), str(images), str(childout)],
                           capture_output=True, text=True, timeout=120)
    (out / 'fresh_process.log').write_text(child.stdout + child.stderr, encoding='utf-8')
    assert child.returncode == 0, child.stderr
    high_payload = ev.read_export(high_file, core.file_sha256(high_file))
    payload = tr.derive_payload(high_payload, 'mantissa_truncation', core.file_sha256(high_file))
    model = tr.load_derived_model(modules.models, payload, 'cpu')
    with torch.no_grad(): expected = model(inputs)
    actual = torch.load(childout / 'logits.pt', map_location='cpu', weights_only=True)['logits']
    assert torch.equal(actual, expected)
    result['fresh_process_C_logits_exact'] = True
    result['fresh_process_no_master_no_direct_low'] = True
    # CLI must reject training, CPU fallback, missing ref, output in source, and downloads.
    bad_args = [['run'], ['check', '--device', 'cpu'], ['evaluate'], ['check', '--download'],
                ['evaluate', '--reference-run', str(out), '--output-root', str(out / 'inside')]]
    rejected = []
    for arguments in bad_args:
        p = subprocess.run([sys.executable, str(ROOT / 'run_eval.py'), *arguments],
                            capture_output=True, text=True, timeout=30)
        assert p.returncode == 2, (arguments, p.stdout, p.stderr)
        rejected.append(arguments)
    result['cli_guards_rejected'] = rejected
    if args.notebook:
        import nbformat
        from nbclient import NotebookClient
        f = out / 'synthetic_tests'
        fake = {'status': 'PASS', 'synthetic_fixture_only': True, 'optimizer_steps': 0,
                'batchnorm_recalibration': False, 'warnings': [], 'session_wall_seconds': 0,
                'roles': {'synthetic': {**result['synthetic_full_evaluator'], 'all_source_scales_equal': True,
                                      'same_auxiliary_state': True}},
                'limitations': ['ONLY SYNTHETIC CPU SOFTWARE TEST; no real SVHN result.']}
        s.write_json(f / 'run_report.json', fake)
        notebook = nbformat.read(ROOT / 'MX_mantissa_truncatie_resultaten.ipynb', as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.source = cell.source.replace('RUN_DIR = Path(".")  # RUN_DIRECTORY',
                                                  'RUN_DIR = Path(' + json.dumps(str(f)) + ')  # RUN_DIRECTORY')
        NotebookClient(notebook, timeout=120, kernel_name='python3',
                       resources={'metadata': {'path': str(f)}}).execute()
        nbformat.write(notebook, out / 'synthetic_notebook_executed.ipynb')
        result['notebook_executed_only_synthetic_data'] = True
    result['environment'] = rt.environment
    result['production_GPU_training_or_evaluation_performed'] = False
    s.write_json(out / 'validation_report.json', result)
    print('All synthetic CPU tests PASS. No user checkpoint or real SVHN data evaluated.')


if __name__ == '__main__':
    main()
