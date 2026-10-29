"""Local CPU-only synthetic validation; never part of laptop production path."""
from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
import svhn_support as s
import test_e4m3
import test_mx
import test_joint
import test_group_size
import test_trunc_training

torch.set_num_threads(2)
report = {'scope': 'CPU synthetic codec and model tests; no dataset; no real training; no CUDA test'}
with tempfile.TemporaryDirectory(prefix='mx_group64_codec_runtime_') as td:
    source, source_info = s.legacy.prepare_source(Path(td))
    rt = s.runtime(20261009, 'cpu', strict=False)
    modules = s.import_source(source)
    report['environment'] = rt.environment
    report['source'] = source_info
    original_cuda = torch.Tensor.cuda
    # Author loop uses .cuda() unconditionally. The production files are
    # unchanged; this temporary local substitution permits a CPU oracle test.
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    try:
        for key, fn in [
            ('e4m3', lambda: test_e4m3.run_tests('cpu')),
            ('group_size', lambda: test_group_size.run_tests('cpu')),
            ('e2m3_model', lambda: test_mx.run_tests(modules, 'cpu')),
            ('joint_direct_oracles', lambda: test_joint.run_tests(modules, rt, source, print)),
            ('truncation_runtime', lambda: test_trunc_training.run_tests(modules, rt, source, print)),
        ]:
            print('TEST', key, flush=True)
            report[key] = fn()
            (OUT / 'codec_runtime_checks.json').write_text(json.dumps(report, indent=2) + '\n')
            print('PASS', key, flush=True)
    finally:
        torch.Tensor.cuda = original_cuda
report['status'] = 'PASS'
(OUT / 'codec_runtime_checks.json').write_text(json.dumps(report, indent=2) + '\n')
print('ALL CPU CHECKS PASS')
