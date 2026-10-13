"""LOCAL synthetic-file/tiny-protocol shim. Never used by the public CLI."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_reference as ref
real_config, real_prepare, real_ensure = ref.load_config, ref.prepare_training_data, ref.pilot.ensure_dataset

def config():
    c = real_config()
    c.update(planned_epochs=3, scheduler_milestones=[1, 2], batch_size=8,
             train_examples=20, test_examples=20)
    c['purpose'] = 'LOCAL_SYNTHETIC_FIXTURE_ONLY_not_reference_training'
    return c

def ensure(cache, cls, download, log):
    m = json.loads((cache / 'FIXTURE_ONLY.json').read_text())
    cls.train_list, cls.test_list = m['entries'][:5], m['entries'][5:]
    log('LOCAL SYNTHETIC FIXTURE: geen echte CIFAR-10-data, geen paperbudget.')
    r = real_ensure(cache, cls, False, log)
    r['synthetic_test_fixture_only'] = True
    return r

def prepare(root, source, cfg, log):
    td, selection, info, tf = real_prepare(root, source, real_config(), log)
    td.indices = list(range(20))
    selection = {'split': 'synthetic_train', 'n': 20, 'indices': list(range(20)),
                 'validation_examples': 0, 'official_test_examples_used': 0}
    dm = ref.importlib.import_module('datasets.data')
    get = dm.get_dataset
    def small_test(name, split, transform, **kwargs):
        ds = get(name, split, transform, **kwargs)
        if split == 'val':
            ds.test_data = ds.test_data[:20]
            ds.test_labels = ds.test_labels[:20]
        return ds
    dm.get_dataset = small_test
    return td, selection, info, tf

ref.load_config, ref.pilot.ensure_dataset, ref.prepare_training_data = config, ensure, prepare
if __name__ == '__main__':
    raise SystemExit(ref.main())
