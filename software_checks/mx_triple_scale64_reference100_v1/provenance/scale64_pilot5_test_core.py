"""Bounded CPU-only harness for the real synthetic checks, never a train CLI mode."""
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
import test_trunc_training as test

def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    rt = s.runtime(20261009, 'cpu', strict=False)
    original_cuda = torch.Tensor.cuda
    checks = {}
    with tempfile.TemporaryDirectory(prefix='mx_triple_scale64_core_') as folder:
        source, _ = s.legacy.prepare_source(Path(folder))
        modules = s.import_source(source)
        try:
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
            result = test.run_tests(modules, rt, source, lambda value: None, checks)
        finally:
            torch.Tensor.cuda = original_cuda
    result.update(synthetic_fixture_only=True, SVHN_used=False,
                  environment=rt.environment, seconds=time.perf_counter() - start,
                  tested_file_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                     for f in ['trunc_adapter.py','trunc_support.py','test_trunc_training.py',
                                               'posthoc_truncation.py','joint_adapter.py','joint_support.py',
                                               'mx_e4m3.py','mx_format.py','mx_e2m1.py',
                                               'provenance/posthoc_scale64.py']})
    target = ROOT / 'local_validation/core_training_validation.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status':result['status'],'seconds':result['seconds'],
                      'report':str(target), 'checks':list(result)}))
if __name__ == '__main__': main()
