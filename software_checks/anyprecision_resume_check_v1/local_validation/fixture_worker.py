"""Developer-only fixture wrapper. NOT called by the user-facing command.

Changes CIFAR checksum metadata IN MEMORY for artificial, locally generated files;
never changes source files. No claim about real CIFAR-10 accuracy or GPU execution.
"""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import resume_worker as worker
real_ensure = worker.pilot.ensure_dataset


def fixture_ensure(cache, cls, download, log):
    metadata = json.loads((cache / 'FIXTURE_ONLY.json').read_text())
    cls.train_list = metadata['entries'][:5]
    cls.test_list = metadata['entries'][5:]
    log('LOCAL SYNTHETIC FIXTURE ONLY: geen echte CIFAR-10-data.')
    result = real_ensure(cache, cls, False, log)
    result['synthetic_test_fixture_only'] = True
    return result

worker.pilot.ensure_dataset = fixture_ensure
if __name__ == '__main__':
    raise SystemExit(worker.main())
