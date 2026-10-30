"""CPU checks of the bounded real-data runner; mocked GPU values are not measurements."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _reject(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Expected invalid evidence to be rejected.')


def _metrics(n=256, training=True):
    result = {'samples': n, 'optimizer_updates': int(training), 'forwards': 20 if training else 5,
              'by_bit': {str(bit): {'samples': n, 'loss': 6.9, 'top1_correct': 1, 'top5_correct': 2}
                         for bit in (1, 2, 4, 8, 32)}}
    if training:
        result.update(mode='serial_dp4', chunk_sizes=[64] * 4, replicas_used=4)
    return result


def _bn(updates):
    result = {f'shared_{i}': {'running_mean': 0, 'running_var': 1, 'num_batches_tracked': 5 * updates}
              for i in range(4)}
    result.update({f'block_{layer}.bn_dict.{bit}':
                   {'running_mean': 0, 'running_var': 1, 'num_batches_tracked': updates}
                   for layer in range(16) for bit in (1, 2, 4, 8, 32)})
    return result


def _contract_tests():
    import torch
    import run_pilot
    import runtime_checks as c
    import support as s
    s.verify_fixed_contract()
    s.verify_sources()
    proof = run_pilot.verify_approved_check()
    assert proof['status'] == 'PASS' and proof['synthetic_probe_repeated'] is False
    for command in ('resume', 'train', 'check'):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                run_pilot.parser().parse_args([command])
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError('Unexpected public command: ' + command)
    for option in ('--epochs', '--batches', '--checkpoint'):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                run_pilot.parser().parse_args(['run', option, '201'])
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError('The bounded pilot accepted an out-of-scope option.')
    c.check_metrics(_metrics(), training=True, expected_samples=256)
    for key, value in [('samples', 128), ('optimizer_updates', 2), ('chunk_sizes', [128, 128])]:
        broken = _metrics()
        broken[key] = value
        _reject(lambda: c.check_metrics(broken, training=True, expected_samples=256))
    total = c.fresh_totals()
    first, last = _metrics(64, False), _metrics(8, False)
    for value in first['by_bit'].values():
        value.update(loss=2., top1_correct=32, top5_correct=48)
    for value in last['by_bit'].values():
        value.update(loss=4., top1_correct=2, top5_correct=6)
    for item in (first, last):
        c.check_metrics(item, training=False, expected_samples=item['samples'])
        c.add_totals(total, item)
    summary = c.summarize_totals(total)
    assert all(v['samples'] == 72 and abs(v['loss'] - 160/72) < 1e-12 and
               abs(v['top1'] - 100*34/72) < 1e-12 for v in summary.values())
    bn_proof = c.check_bn_training(_bn(0), _bn(200), 200)
    broken = _bn(200)
    broken['shared_0']['num_batches_tracked'] = 200
    _reject(lambda: c.check_bn_training(_bn(0), broken, 200))
    before = {name: {key: torch.tensor(value) for key, value in data.items()} for name, data in _bn(200).items()}
    after = copy.deepcopy(before)
    c.check_bn_unchanged(before, after)
    after['shared_0']['running_mean'].add_(1)
    _reject(lambda: c.check_bn_unchanged(before, after))
    return {'status': 'PASS', 'prior_synthetic_PASS_reused': True,
            'bounded_cli': True, 'weighted_partial_validation_batch': True,
            'BN_training_retention': bn_proof, 'changed_validation_BN_rejected': True}


def _checkpoint_roundtrip():
    import torch
    import pilot_worker
    state = torch.get_rng_state()
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.3, momentum=.9, weight_decay=1e-5)
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix='r18-checkpoint-test-') as temporary:
            out = Path(temporary)
            with patch.object(torch.cuda, 'get_rng_state_all', return_value=[]):
                metadata = pilot_worker.save_diagnostic_checkpoint(out, model, optimizer, {'fixture': True})
                saved = torch.load(metadata['path'], map_location='cpu', weights_only=True)
                assert saved['global_step'] == 200 and saved['completed_epochs'] == 0
                assert saved['eligible_for_reference_resume'] is False
                assert saved['kind'] == 'diagnostic_partial_epoch'
                assert saved['full_training_must_start_from_zero'] is True
                assert all(torch.equal(saved['model'][name], value) for name, value in expected.items())
                assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())
                assert len(saved['optimizer']['state']) == 2
                assert all('momentum_buffer' in value for value in saved['optimizer']['state'].values())
                assert saved['rng']['torch_cpu'].dtype == torch.uint8
                assert saved['rng']['numpy']['keys'] and saved['rng']['python']
                _reject(lambda: pilot_worker.save_diagnostic_checkpoint(out, model, optimizer, {}))
        return {'status': 'PASS', 'weights_only_load_succeeded': True, 'FP32_model_and_SGD_preserved': True,
                'not_full_epoch_or_reference_resume': True, 'existing_checkpoint_not_overwritten': True}
    finally:
        torch.set_rng_state(state)


class _FakeOOM(RuntimeError):
    pass


class _FakeCUDA:
    OutOfMemoryError = _FakeOOM

    def __init__(self, trace):
        self.trace = trace

    def synchronize(self):
        self.trace.append('sync')

    def is_initialized(self):
        return True

    def reset_peak_memory_stats(self):
        self.trace.append('reset_peak')

    def mem_get_info(self):
        self.trace.append('memory')
        return 4000, 8000

    def memory_allocated(self):
        self.trace.append('memory')
        return 2000

    memory_reserved = memory_allocated
    max_memory_allocated = memory_allocated
    max_memory_reserved = memory_allocated


class _FakeTensor:
    def __init__(self, n, trace, label=False):
        self.shape = (n,) if label else (n, 3, 224, 224)
        self.trace = trace

    def to(self, device, non_blocking=False):
        assert device == 'cuda:0' and non_blocking
        self.trace.append('transfer')
        return self


class _SizedDataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _FakeLoader:
    def __init__(self, n, batch, trace, phase):
        self.dataset = _SizedDataset(n)
        self.batch_size = batch
        self.trace = trace
        self.phase = phase
        self.returned = 0

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for start in range(0, len(self.dataset), self.batch_size):
            n = min(self.batch_size, len(self.dataset) - start)
            self.returned += 1
            self.trace.append('fetch_' + self.phase)
            yield _FakeTensor(n, self.trace), _FakeTensor(n, self.trace, label=True)


class _FileLog:
    def __init__(self, path):
        self.stream = Path(path).open('x', encoding='utf-8')

    def __call__(self, value):
        self.stream.write(str(value) + '\n')

    def close(self):
        self.stream.close()


def _mock_worker_case(fail_at=None):
    import data_support
    import engine
    import pilot_worker
    import resnet18_model
    import runtime_checks as c
    import support as s
    trace = []
    model = SimpleNamespace(updates=0, evals=0)
    fake_torch = ModuleType('torch')
    fake_torch.cuda = _FakeCUDA(trace)
    fake_torch.equal = lambda a, b: a == b
    train_loader = _FakeLoader(51200, 256, trace, 'train')
    val_loader = _FakeLoader(5000, 64, trace, 'val')
    clock = [0.]

    def now():
        clock[0] += .001
        trace.append('timer')
        return clock[0]

    def train_step(candidate, optimizer, x, y, soft, **kwargs):
        assert candidate is model and x.shape[0] == 256
        assert kwargs == {'mode': 'serial_dp4', 'validate': model.updates == 0}
        trace.append('engine_train')
        if fail_at is not None and model.updates + 1 == fail_at:
            raise _FakeOOM('Fixture OOM before completion')
        model.updates += 1
        return _metrics()

    def evaluate(candidate, x, y):
        assert candidate is model and model.updates == 200
        model.evals += 1
        trace.append('engine_eval')
        return _metrics(x.shape[0], False)

    def checkpoint(out, *args):
        assert model.updates == 200 and model.evals == 0
        trace.append('checkpoint')
        path = Path(out) / 'checkpoint_step0200.pt'
        path.write_bytes(b'fixture-only')
        return {'filename': path.name, 'path': str(path), 'size_bytes': path.stat().st_size,
                'seconds': .001, 'kind': 'diagnostic_partial_epoch', 'eligible_for_reference_resume': False}

    with tempfile.TemporaryDirectory(prefix='r18-worker-test-') as temporary:
        root = Path(temporary)
        dataset = root / 'imagenet'
        audit = {'strict': True, 'root': str(dataset), 'train_count': 1281167, 'val_count': 50000,
                 'class_count': 1000, 'class_to_idx_sha256': 'fixture'}
        s.json_write(root / 'dataset_manifest.json', audit)
        s.json_write(root / 'source_manifest.json', {'fixture': True})
        args = SimpleNamespace(run_directory=root, imagenet_root=dataset, workers=4, cpu_threads=4)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch}))
            stack.enter_context(patch.object(s, 'Log', _FileLog))
            stack.enter_context(patch.object(s, 'verify_source_copy', return_value={'anyprecision': root, 'truncquant': root}))
            stack.enter_context(patch.object(s, 'configure', return_value={'device': {'name': 'MOCK_GPU'}}))
            stack.enter_context(patch.object(c, 'model_and_optimizer', return_value=(model, object(), object())))
            stack.enter_context(patch.object(c, 'bn_snapshot', side_effect=lambda _: _bn(model.updates)))
            stack.enter_context(patch.object(c, 'assert_finite_state', side_effect=lambda *args: trace.append('finite_check')))
            stack.enter_context(patch.object(resnet18_model, 'model_report', return_value={'fixture': True}))
            stack.enter_context(patch.object(data_support, 'build_loaders', return_value=(train_loader, val_loader, {'fixture': True})))
            stack.enter_context(patch.object(engine, 'train_step', side_effect=train_step))
            stack.enter_context(patch.object(engine, 'eval_batch', side_effect=evaluate))
            stack.enter_context(patch.object(pilot_worker, 'save_diagnostic_checkpoint', side_effect=checkpoint))
            stack.enter_context(patch.object(pilot_worker.time, 'perf_counter', side_effect=now))
            code = pilot_worker.run_worker(args)
        result = json.loads((root / 'pilot/result.json').read_text(encoding='utf-8'))
        assert len(result['train_steps']) == model.updates
        if fail_at is None:
            assert code == 0 and result['status'] == 'PASS'
            assert model.updates == 200 and model.evals == 79
            assert train_loader.returned == 200 and val_loader.returned == 79
            assert result['validation']['samples'] == 5000
            assert result['bn_training_check']['updates_per_shared_bn'] == 1000
            assert result['bn_validation_unchanged']['status'] == 'PASS'
            import run_pilot
            run_pilot.validate_completed_result(result, root)
            for mutation in ('missing_step', 'val_short', 'no_bn', 'bad_timing'):
                invalid = copy.deepcopy(result)
                if mutation == 'missing_step':
                    invalid['train_steps'].pop()
                elif mutation == 'val_short':
                    invalid['validation']['samples'] = 4999
                elif mutation == 'no_bn':
                    invalid.pop('bn_validation_unchanged')
                else:
                    invalid['train_steps'][1]['total_seconds'] = float('nan')
                _reject(lambda: run_pilot.validate_completed_result(invalid, root))
            Path(result['checkpoint']['path']).unlink()
            _reject(lambda: run_pilot.validate_completed_result(result, root))
        else:
            assert code == 3 and result['status'] == 'OOM'
            assert model.updates == fail_at - 1 and model.evals == 0
            assert 'checkpoint' not in result and result['validation']['status'] == 'NOT_STARTED'
        for index, event in enumerate(trace):
            if event == 'engine_train' and (index + 2 < len(trace)) and trace[index + 1] == 'sync':
                assert trace[index - 1] == 'timer' and trace[index + 2] == 'timer'
        assert (root / 'pilot/step_times.csv').is_file()
        assert (root / 'pilot/train_metrics.csv').is_file()
    return {'status': 'PASS', 'mock_only': True, 'observed_worker_status': result['status'],
            'completed_training_updates': model.updates, 'validation_batches': model.evals,
            'total_time_components_verified': True, 'partial_results_preserved': True,
            'no_accuracy_or_GPU_speed_claim': True}


def run_tests():
    import torch
    state = torch.get_rng_state()
    try:
        result = {'status': 'PASS', 'scope': 'CPU contract/serialization tests plus mocked GPU worker orchestration.',
                  'contract': _contract_tests(), 'checkpoint': _checkpoint_roundtrip(),
                  'worker_success': _mock_worker_case(), 'worker_partial_OOM': _mock_worker_case(51)}
        return result
    finally:
        torch.set_rng_state(state)
