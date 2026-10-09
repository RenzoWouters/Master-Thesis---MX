"""Developer-only validation, NOT a CIFAR-10 experiment.

Creates synthetic image/label batches with the CIFAR file layout. Changes checksum
metadata IN MEMORY only to permit the unchanged upstream loader to read fixtures.
Production run_pilot.py does not contain or call this fixture bypass.
No real CIFAR-10 image, pretrained model or research-accuracy result is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_pilot as pilot
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--fixture-root', type=Path, required=True)
parser.add_argument('--output-root', type=Path, required=True)
args = parser.parse_args()
folder = args.fixture_root / 'cifar-10-batches-py'
folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(77)
entries = []
for name in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']:
    f = folder / name
    if not f.exists():
        with f.open('xb') as h:
            pickle.dump({'data': rng.integers(0,256,(10000,3072),dtype=np.uint8),
                'labels': np.tile(np.arange(10),1000).tolist()}, h, protocol=2)
    entries.append([name, pilot.hash_file(f, 'md5')])
real_ensure = pilot.ensure_dataset
real_run = pilot.run

def offline_data(cache, cls, download, log):
    log('OFFLINE FIXTURE: kunstmatige beelden/labels; GEEN echte CIFAR-10-accuracy.')
    cls.train_list = entries[:5]
    cls.test_list = entries[5:]
    result = real_ensure(cache, cls, False, log)
    result['synthetic_test_fixture_only'] = True
    return result


def offline_run(cli, out, report, log):
    report['purpose'] = 'OFFLINE SYNTHETIC FIXTURE VALIDATION ONLY; not CIFAR-10 accuracy'
    report['synthetic_test_fixture_only'] = True
    report['fixture_harness_sha256'] = pilot.sha256(Path(__file__))
    real_run(cli, out, report, log)

pilot.ensure_dataset = offline_data
pilot.run = offline_run
sys.argv = ['run_pilot.py', '--profile', 'local-check', '--device', 'cpu',
    '--data-root', str(args.fixture_root), '--output-root', str(args.output_root)]
raise SystemExit(pilot.main())
