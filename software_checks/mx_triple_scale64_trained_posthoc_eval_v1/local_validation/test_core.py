"""Bounded synthetic CPU core validation. Never trains or opens SVHN data."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import svhn_support as s
import test_scale64 as test


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    rt = s.runtime(20261009, 'cpu', strict=False)
    original_cuda = torch.Tensor.cuda
    with tempfile.TemporaryDirectory(prefix='mx_scale64_trained_posthoc_core_') as folder:
        out = Path(folder)
        source, _ = s.legacy.prepare_source(out)
        modules = s.import_source(source)
        try:
            # Archived upstream initialization calls Tensor.cuda unconditionally.
            # This fixture-only adaptation is never installed by the production CLI.
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
            result = test.run_tests(modules, rt.device, out)
        finally:
            torch.Tensor.cuda = original_cuda
    result.update(environment=rt.environment, seconds=time.perf_counter() - start,
        production_CUDA_or_SVHN_tested=False,
        tested_file_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
            for f in ['scale64.py','evaluation.py','test_scale64.py','posthoc_truncation.py',
                      'trunc_adapter.py','joint_adapter.py','mx_e4m3.py','mx_format.py','mx_e2m1.py']})
    target = ROOT / 'local_validation/test_core_validation_report.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],
        'report':str(target),'negative_controls':len(result['negative_controls']),
        'model_comparisons':len(result['model_comparisons'])}))


if __name__ == '__main__': main()
