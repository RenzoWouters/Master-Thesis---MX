"""Local synthetic CPU integration test. Creates a NEW workdir only."""
from pathlib import Path
import argparse, hashlib, json, pickle, subprocess, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import resume_core as core
p = argparse.ArgumentParser()
p.add_argument('--work-root', type=Path, required=True)
a = p.parse_args()
w = a.work_root.resolve()
w.mkdir(parents=True, exist_ok=False)
data = w / 'fixture_data'
folder = data / 'cifar-10-batches-py'
folder.mkdir(parents=True)
rng = np.random.default_rng(1234)
entries = []
for name in [f'data_batch_{i}' for i in range(1, 6)] + ['test_batch']:
    path = folder / name
    with path.open('xb') as f:
        pickle.dump({'data': rng.integers(0, 256, (10000, 3072), dtype=np.uint8),
                     'labels': np.tile(np.arange(10), 1000).tolist()}, f, protocol=4)
    entries.append([name, hashlib.md5(path.read_bytes()).hexdigest()])
(data / 'FIXTURE_ONLY.json').write_text(json.dumps({'entries': entries}))
initial_data_hashes = {p.name: core.file_sha256(p) for p in folder.iterdir()}

def launch(label, argv, expected=0, shim=True):
    dest = w / label
    target = ROOT / 'local_validation/fixture_reference.py' if shim else ROOT / 'run_reference.py'
    cmd = [sys.executable, '-B', str(target), *argv, '--device', 'cpu',
           '--data-root', str(data), '--output-root', str(dest)]
    with (w / (label + '.log')).open('x') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if r.returncode != expected:
        print((w / (label + '.log')).read_text(), flush=True)
        raise RuntimeError(f'{label}: exit {r.returncode}, expected {expected}')
    reports = list(dest.glob('*/reference_report.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    print(label, report['status'], flush=True)
    return report, reports[0].parent

ra, da = launch('uninterrupted', ['train', '--session-epochs', '3'])
rb, db = launch('before_stop', ['train', '--session-epochs', '1'])
b1 = Path(rb['last_checkpoint'])
rc, dc = launch('after_resume', ['train', '--session-epochs', '2', '--resume', str(b1)])
sa, sc = core.load_verified(Path(ra['last_checkpoint'])), core.load_verified(Path(rc['last_checkpoint']))
checks = {
 'same_checkpoint_epoch1': not core.differences(core.load_verified(da/'checkpoints/epoch_00001.pt'),core.load_verified(b1)),
 'same_entire_final_checkpoint': not core.differences(sa, sc),
 'same_model': not core.differences(sa['model'], sc['model']),
 'same_optimizer': not core.differences(sa['optimizer'], sc['optimizer']),
 'same_scheduler': not core.differences(sa['scheduler'], sc['scheduler']),
 'same_rng': not core.differences(sa['rng'], sc['rng']),
 'same_training_history': not core.differences(sa['trace'], sc['trace']),
 'same_exported_metrics_csv': (da/'metrics.csv').read_bytes() == (dc/'metrics.csv').read_bytes(),
 'lr_sequence': [e['lr_used'] for e in sa['trace']['epochs']] == [0.001, 0.0001, 0.00001],
 'cumulative_steps9': sa['optimizer_steps'] == sc['optimizer_steps'] == 9,
 'train_only': all(r['test_not_evaluated'] for r in [ra, rb, rc]) and sa['validation_logits'] == {},
 'bounded_session': rb['completed_epoch'] == 1 and rb['status'] == 'PAUSED',
 'unchanged_planning': rb['config']['planned_epochs'] == rc['config']['planned_epochs'] == 3,
 'readback_all': all(e['full_readback_identical'] for r in [ra, rb, rc] for e in r['checkpoints']),
 'no_warnings': not any(r['warnings'] for r in [ra, rb, rc]),
}
r0, d0 = launch('resume_initialization', ['train', '--session-epochs', '3', '--resume',str(da/'checkpoints/epoch_00000.pt')])
checks['same_from_epoch0'] = not core.differences(sa, core.load_verified(Path(r0['last_checkpoint'])))
re, de = launch('endpoint_evaluation', ['evaluate', '--checkpoint', str(ra['last_checkpoint']),
                 '--endpoint-epoch', '3', '--endpoint-reason', 'LOCAL_SYNTHETIC_TEST_ONLY'])
checks['endpoint_evaluation'] = (re['status'] == 'PASS' and re['official_test_examples_used'] == 20 and re['model_unchanged_by_evaluation'])
checks['checkpoint_unchanged_by_evaluation'] = not core.differences(sa, core.load_verified(Path(ra['last_checkpoint'])))
rg, dg = launch('production_data_guard', ['train', '--session-epochs', '1'], expected=1, shim=False)
checks['production_rejects_fake_data'] = rg['status'] == 'FAIL' and 'Checksum' in rg['error']
checks['fixture_files_unchanged'] = initial_data_hashes == {p.name: core.file_sha256(p) for p in folder.iterdir()}
result = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
          'environment': ra['environment'], 'fixture_only': True,
          'full_reference_config_tested': False, 'CUDA_tested_here': False,
          'checkpoint_bytes': ra['checkpoints'][-1]['bytes'],
          'notes': 'New runner in separate processes; unchanged source, artificial files and tiny test-only protocol.'}
(w/'integration_result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
assert all(checks.values()), result
