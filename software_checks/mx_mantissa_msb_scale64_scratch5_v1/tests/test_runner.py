"""Small runner-boundary tests, with no dataset or numerical training.

The model/phase layer is deliberately mocked: the engine suite owns the actual
forward/backward and continuous-versus-resumed training checks. All checkpoint
bytes written here are explicitly fake and live only in temporary directories.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


runner = _module('msb64_boundary_runner', ROOT / 'run_train.py')
reporter = _module('msb64_boundary_reporter', ROOT / 'training_report.py')


class Loader:
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return self.batches

    def validate(self, path):
        return {'complete_once': True, 'n': 3, 'order_file': str(path),
                'order_sha256': 'software-fixture-only'}


class Scheduler:
    def __init__(self):
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1


def context(out, report=None):
    report = report if report is not None else {
        'status': 'RUNNING', 'completed_epochs': 0, 'last_checkpoint': None,
        'training_performed': False, 'network_evaluated': False,
        'synthetic_only': True, 'synthetic_fixture_only': True,
    }
    core = SimpleNamespace()
    core.capture_checkpoint = mock.Mock(side_effect=lambda model, optimizer, scheduler, generators,
            device, epoch, steps, contract, trace, logits: {
                'completed_epoch': epoch, 'optimizer_steps': steps,
                'contract': copy.deepcopy(contract), 'trace': copy.deepcopy(trace)})

    def save(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'SOFTWARE TEST FIXTURE; NOT A PYTORCH CHECKPOINT\n')
        digest = runner.sha256(path)
        path.with_suffix('.pt.sha256').write_text(digest + '\n')
        return {'sha256': digest}

    core.atomic_save = mock.Mock(side_effect=save)
    core.load_verified = mock.Mock()
    core.restore_checkpoint = mock.Mock()
    core.differences = lambda a, b: [] if a == b else ['contract']
    rows = [{'weight_bits': bit, 'weight_format': f'MXFP{bit}_' + {4: 'E2M1', 6: 'E2M3', 8: 'E4M3'}[bit],
             'n': 3, 'correct1': 2, 'correct5': 3, 'objective_loss': .5, 'lr': .001}
            for bit in (4, 6, 8)]
    engine = SimpleNamespace(
        run_phase=mock.Mock(side_effect=lambda *args, **kwargs: {
            'rows': copy.deepcopy(rows), 'audit': {'software_fixture_only': True},
            'predictions': [], 'seconds': 0.}),
        verify_state=mock.Mock(return_value={'software_fixture_only': True}))
    return SimpleNamespace(
        core=core, rt=SimpleNamespace(device='cpu'), engine=engine,
        state=SimpleNamespace(model=object(), optimizer=SimpleNamespace(param_groups=[{'lr': .001}]),
                              scheduler=Scheduler()),
        trace={'schema': runner.SCHEMA + '-history', 'method': runner.METHOD,
               'train_from_scratch': True, 'initial_model_sha256': 'software-fixture-only',
               'initial_rng_sha256': 'software-fixture-only', 'epochs': []},
        completed=0, report=report, loaders={'train': Loader(2), 'test': Loader(1)},
        contract={'schema': runner.SCHEMA + '-contract', 'method': runner.METHOD,
                  'synthetic_fixture_only': True},
        source_info={'workcopy_files_sha256': {}}, source=out, data_info=None)


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='msb64_runner_boundary_fixture_')
        self.out = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_checkpoint_checksum_failure_does_not_commit_epoch(self):
        ctx = context(self.out)
        before = copy.deepcopy(ctx.trace)

        def fail_after_checkpoint_bytes(path, payload):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'INCOMPLETE SOFTWARE FIXTURE WITHOUT CHECKSUM')
            raise OSError('injected checksum publication failure')

        ctx.core.atomic_save.side_effect = fail_after_checkpoint_bytes
        with self.assertRaisesRegex(OSError, 'checksum publication'):
            runner.run_epochs(ctx, self.out, lambda message: None, stop_after=1)
        self.assertEqual(ctx.engine.run_phase.call_count, 2)
        self.assertEqual(ctx.completed, 0)
        self.assertEqual(ctx.trace, before)
        self.assertIsNone(ctx.report['last_checkpoint'])
        self.assertFalse((self.out / 'checkpoints/epoch_00001.pt.sha256').exists())
        runner.export_results(ctx, self.out)
        self.assertEqual(json.loads((self.out / 'history.json').read_text())['epochs'], [])
        with (self.out / 'accuracy.csv').open(newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])
        self.assertFalse(ctx.report['training_completed'])

    def test_summary_failure_after_saved_checkpoint_preserves_commit(self):
        ctx = context(self.out)
        write_json = runner.write_json

        def fail_summary(path, value):
            if Path(path).name == 'epoch_summary.json':
                raise OSError('injected epoch summary failure')
            return write_json(path, value)

        with mock.patch.object(runner, 'write_json', side_effect=fail_summary):
            with self.assertRaisesRegex(OSError, 'epoch summary'):
                runner.run_epochs(ctx, self.out, lambda message: None, stop_after=1)
        self.assertEqual(ctx.completed, 1)
        self.assertEqual([record['epoch'] for record in ctx.trace['epochs']], [1])
        checkpoint = Path(ctx.report['last_checkpoint']['path'])
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(checkpoint.with_suffix('.pt.sha256').read_text().strip(), runner.sha256(checkpoint))
        runner.export_results(ctx, self.out)
        history = json.loads((self.out / 'history.json').read_text())
        self.assertEqual([record['epoch'] for record in history['epochs']], [1])
        with (self.out / 'accuracy.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({(row['epoch'], row['weight_bits']) for row in rows},
                         {('1', str(bit)) for bit in (4, 6, 8)})
        self.assertEqual(ctx.report['optimizer_steps_in_completed_epochs'], 2)

    def test_wrong_resume_method_is_rejected_before_restore_or_update(self):
        for mismatch in ('contract', 'trace'):
            with self.subTest(mismatch=mismatch):
                ctx = context(self.out)
                payload = {'contract': copy.deepcopy(ctx.contract), 'completed_epoch': 1,
                           'optimizer_steps': 2, 'trace': copy.deepcopy(ctx.trace)}
                payload['trace']['epochs'] = [{'epoch': 1, 'method': runner.METHOD, 'optimizer_steps': 2}]
                if mismatch == 'contract':
                    payload['contract']['method'] = 'lsb_trunc'
                else:
                    payload['trace']['method'] = 'lsb_trunc'
                ctx.core.load_verified.return_value = payload
                with self.assertRaises(RuntimeError):
                    runner.run_epochs(ctx, self.out, lambda message: None, stop_after=5,
                                      checkpoint=self.out / 'wrong_method.pt')
                ctx.core.restore_checkpoint.assert_not_called()
                ctx.engine.run_phase.assert_not_called()
                ctx.core.atomic_save.assert_not_called()
                self.assertEqual(ctx.completed, 0)

    def test_public_check_route_exports_no_research_scores(self):
        package = self.out / 'package'
        package.mkdir()
        shutil.copyfile(ROOT / 'training_report.py', package / 'training_report.py')
        (package / 'REFERENCE.json').write_text('{}\n')
        minimal_protocol = {'planned_epochs': 5, 'reference_end_epoch': 100,
                            'lr': .001, 'limitations': ['software-only public-route test']}

        def prepare(args, out, report, log):
            return context(out, report)

        def synthetic(ctx, out, log):
            ctx.report.update(synthetic_only=True, synthetic_training_steps=2,
                              training_performed=False, network_evaluated=False)

        with mock.patch.object(runner, 'ROOT', package), \
             mock.patch.object(runner, 'project_root', return_value=self.out), \
             mock.patch.object(runner, 'protocol', return_value=minimal_protocol), \
             mock.patch.object(runner, 'prepare_context', side_effect=prepare), \
             mock.patch.object(runner, 'prepare_state'), \
             mock.patch.object(runner, 'synthetic_check', side_effect=synthetic) as check, \
             mock.patch.object(runner, 'run_epochs') as run, \
             mock.patch.dict(sys.modules, {'training_report': reporter}):
            self.assertEqual(runner.main(['check']), 0)
        check.assert_called_once()
        run.assert_not_called()
        output = next((self.out / 'mx_runs' / runner.NAME).iterdir())
        report = json.loads((output / 'run_report.json').read_text())
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['synthetic_only'])
        self.assertFalse(report['training_performed'])
        self.assertFalse(report['network_evaluated'])
        self.assertEqual(report['completed_epochs'], 0)
        with (output / 'accuracy.csv').open(newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])
        analysis = reporter.load_results(output)
        self.assertFalse(analysis['display_new_scores'])
        self.assertFalse(analysis['endpoint_ready'])
        self.assertTrue((output / 'feedback_bundle.zip').is_file())


if __name__ == '__main__':
    unittest.main()
