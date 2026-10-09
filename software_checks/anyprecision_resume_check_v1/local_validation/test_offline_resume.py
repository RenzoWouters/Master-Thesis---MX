"""Local CPU validation via three real subprocesses and fake CIFAR-format files.

Not a user command. Writes only to the explicitly specified new work directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_resume_check as check

p=argparse.ArgumentParser()
p.add_argument('--work-root', type=Path, required=True)
args=p.parse_args()
args.work_root.mkdir(parents=True, exist_ok=False)
cache=args.work_root / 'fixture_data'
folder=cache / 'cifar-10-batches-py'
folder.mkdir(parents=True)
rng=np.random.default_rng(9182)
entries=[]
for name in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']:
    path=folder / name
    with path.open('xb') as f:
        pickle.dump({'data':rng.integers(0,256,(10000,3072),dtype=np.uint8),
                     'labels':np.tile(np.arange(10),1000).tolist()}, f, protocol=4)
    entries.append([name,hashlib.md5(path.read_bytes()).hexdigest()])
(cache / 'FIXTURE_ONLY.json').write_text(json.dumps({'entries':entries}))
check.WORKER_PATH=ROOT / 'local_validation/fixture_worker.py'
sys.argv=['run_resume_check.py','--device','cpu','--data-root',str(cache),
          '--output-root',str(args.work_root/'output')]
raise SystemExit(check.main())
