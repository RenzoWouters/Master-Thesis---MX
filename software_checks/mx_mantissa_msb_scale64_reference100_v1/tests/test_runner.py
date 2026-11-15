"""Small runner-boundary tests, with no dataset or numerical training.

The model/phase layer is deliberately mocked: the engine suite owns the actual
forward/backward and continuous-versus-resumed training checks. All checkpoint
bytes written here are explicitly fake and live only in temporary directories.
"""
from __future__ import annotations

import copy
import csv
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings

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
    def __init__(self, optimizer, events):
        self.last_epoch = 0
        self.optimizer = optimizer
        self.events = events

    def step(self):
        self.last_epoch += 1
        self.events.append(('scheduler', self.last_epoch))
        self.optimizer.param_groups[0]['lr'] = expected_lr_after(self.last_epoch)


def expected_lr_after(completed_epoch):
    value = .001
    for milestone in (50, 75, 90):
        if milestone <= completed_epoch:
            value *= .1
    return value


def fixture_protocol():
    return {'planned_epochs': 100, 'reference_end_epoch': 100, 'lr': .001,
            'scheduler_milestones': [50, 75, 90], 'scheduler_gamma': .1,
            'method': 'msb_bestexp', 'train_from_scratch': True,
            'scale_factor': 64, 'weight_decay': 0.,
            'limitations': ['software-only runner test; no research measurements']}


def context(out, report=None):
    report = report if report is not None else {
        'status': 'RUNNING', 'completed_epochs': 0, 'last_checkpoint': None,
        'training_performed': False, 'network_evaluated': False,
        'synthetic_only': True, 'synthetic_fixture_only': True,
    }
    report['synthetic_fixture_only'] = True
    events = []
    optimizer = SimpleNamespace(param_groups=[{'lr': .001}])
    scheduler = Scheduler(optimizer, events)
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
    def phase(ctx, state, loader, epoch, train, phase_out, log):
        lr = state.optimizer.param_groups[0]['lr']
        events.append(('train' if train else 'test', epoch, lr, state.scheduler.last_epoch))
        current = copy.deepcopy(rows)
        for row in current:
            row['lr'] = lr
        if train:
            ctx.report['training_performed'] = True
            ctx.report['actual_new_optimizer_steps'] = ctx.report.get('actual_new_optimizer_steps', 0) + len(loader)
        return {'rows': current, 'audit': {'software_fixture_only': True},
                'predictions': [], 'seconds': 0.}

    engine = SimpleNamespace(
        run_phase=mock.Mock(side_effect=phase),
        verify_state=mock.Mock(return_value={'software_fixture_only': True}))
    return SimpleNamespace(
        core=core, rt=SimpleNamespace(device='cpu'), engine=engine,
        state=SimpleNamespace(model=object(), optimizer=optimizer, scheduler=scheduler),
        trace={'schema': runner.SCHEMA + '-history', 'method': runner.METHOD,
               'train_from_scratch': True, 'initial_model_sha256': 'software-fixture-only',
               'initial_rng_sha256': 'software-fixture-only', 'epochs': []},
        completed=0, report=report, loaders={'train': Loader(2), 'test': Loader(1)},
        contract={'schema': runner.SCHEMA + '-contract', 'method': runner.METHOD,
                  'synthetic_fixture_only': True},
        protocol=fixture_protocol(), events=events,
        source_info={'workcopy_files_sha256': {}}, source=out, data_info=None)


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='msb64_runner_boundary_fixture_')
        self.out = Path(self.temp.name)
        self.public_case_count = 0

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
                    runner.run_epochs(ctx, self.out, lambda message: None, stop_after=100,
                                      checkpoint=self.out / 'wrong_method.pt')
                ctx.core.restore_checkpoint.assert_not_called()
                ctx.engine.run_phase.assert_not_called()
                ctx.core.atomic_save.assert_not_called()
                self.assertEqual(ctx.completed, 0)

    def test_original_five_epoch_pilot_checkpoint_is_rejected_before_updates(self):
        for mismatch in ('contract', 'history'):
            with self.subTest(mismatch=mismatch):
                ctx = context(self.out)
                payload = {'contract': copy.deepcopy(ctx.contract), 'completed_epoch': 5,
                           'optimizer_steps': 10, 'trace': copy.deepcopy(ctx.trace)}
                payload['trace']['epochs'] = [
                    {'epoch': epoch, 'method': runner.METHOD, 'optimizer_steps': epoch * 2}
                    for epoch in range(1, 6)]
                if mismatch == 'contract':
                    payload['contract']['schema'] = 'mx-msb-scale64-scratch5-v1-contract'
                else:
                    payload['trace']['schema'] = 'mx-msb-scale64-scratch5-v1-history'
                ctx.core.load_verified.return_value = payload
                with self.assertRaises(RuntimeError):
                    runner.run_epochs(ctx, self.out, lambda message: None,
                                      checkpoint=self.out / 'old_pilot_epoch5.pt')
                ctx.core.restore_checkpoint.assert_not_called()
                ctx.engine.run_phase.assert_not_called()
                ctx.core.atomic_save.assert_not_called()
                self.assertEqual(ctx.completed, 0)

    def test_protocol_and_stop_boundaries_are_100_epochs_from_scratch(self):
        protocol = runner.protocol()
        self.assertEqual(protocol['planned_epochs'], 100)
        self.assertEqual(protocol['reference_end_epoch'], 100)
        self.assertEqual(protocol['scheduler_milestones'], [50, 75, 90])
        self.assertEqual(protocol['scheduler_gamma'], .1)
        self.assertEqual(protocol['lr'], .001)
        self.assertTrue(protocol['train_from_scratch'])
        self.assertFalse(protocol['trained_parent_used'])
        self.assertFalse(protocol['pilot_checkpoint_used'])
        self.assertEqual(protocol['method'], 'msb_bestexp')
        self.assertEqual(protocol['scale_factor'], 64)
        self.assertFalse(protocol['learned_shared_scale'])
        self.assertNotIn('scratch5', runner.SCHEMA)
        for stop in (0, 101):
            with self.subTest(stop=stop):
                ctx = context(self.out)
                with self.assertRaises(RuntimeError):
                    runner.run_epochs(ctx, self.out, lambda message: None, stop_after=stop)
                ctx.engine.run_phase.assert_not_called()
                ctx.core.load_verified.assert_not_called()
                ctx.core.atomic_save.assert_not_called()

    def test_fresh_100_epoch_flow_uses_original_lr_and_steps_after_train_test(self):
        ctx = context(self.out)
        runner.run_epochs(ctx, self.out, lambda message: None)
        self.assertEqual(ctx.completed, 100)
        self.assertEqual(ctx.engine.run_phase.call_count, 200)
        self.assertEqual(ctx.core.atomic_save.call_count, 100)
        ctx.core.load_verified.assert_not_called()
        ctx.core.restore_checkpoint.assert_not_called()
        self.assertEqual([record['epoch'] for record in ctx.trace['epochs']], list(range(1, 101)))
        self.assertEqual(len(ctx.events), 300)
        for epoch in range(1, 101):
            used = expected_lr_after(epoch - 1)
            self.assertEqual(ctx.events[(epoch - 1) * 3:epoch * 3], [
                ('train', epoch, used, epoch - 1),
                ('test', epoch, used, epoch - 1), ('scheduler', epoch)])
            record = ctx.trace['epochs'][epoch - 1]
            self.assertEqual(record['optimizer_steps'], epoch * 2)
            self.assertEqual(record['learning_rate_used'], used)
            self.assertEqual(record['learning_rate_next_epoch'], expected_lr_after(epoch))
            self.assertTrue(all(row['lr'] == used for row in record['train'] + record['test']))
        self.assertEqual(ctx.state.optimizer.param_groups[0]['lr'], expected_lr_after(100))
        runner.export_results(ctx, self.out)
        self.assertTrue(ctx.report['training_completed'])
        self.assertEqual(ctx.report['completed_epochs'], 100)
        self.assertEqual(ctx.report['new_completed_epochs'], 100)
        self.assertEqual(ctx.report['optimizer_steps_in_completed_epochs'], 200)

    def test_saved_epoch99_is_partial_and_not_completed_training(self):
        ctx = context(self.out)
        # This boundary check needs no fake computation of the preceding epochs.
        ctx.completed = 99
        ctx.trace['epochs'] = [{'epoch': epoch, 'optimizer_steps': epoch * 2,
                                'train': [], 'test': [], 'method': runner.METHOD}
                               for epoch in range(1, 100)]
        runner.export_results(ctx, self.out)
        self.assertEqual(ctx.report['completed_epochs'], 99)
        self.assertFalse(ctx.report['training_completed'])

    def public_case(self, command='check', *, stop=None, failure=None, warn=False, render_warning=False):
        """Exercise the public finalization with mocked numerics, real files/ZIP."""
        self.public_case_count += 1
        case = self.out / ('public_case_' + str(self.public_case_count))
        case.mkdir()
        package = case / 'package'
        package.mkdir()
        shutil.copyfile(ROOT / 'training_report.py', package / 'training_report.py')
        (package / 'REFERENCE.json').write_text('{}\n')
        contexts = []

        def prepare(args, out, report, log):
            value = context(out, report)
            contexts.append(value)
            return value

        def synthetic(ctx, out, log):
            if warn:
                for index in range(7):
                    warnings.warn_explicit('repeated software fixture warning', DeprecationWarning,
                                           filename='warning_fixture.py', lineno=10 + index)
                for index in range(3):
                    warnings.warn_explicit('other software fixture warning', UserWarning,
                                           filename='warning_fixture.py', lineno=30 + index)
            if failure == 'check':
                raise RuntimeError('injected numerical check failure')
            if failure == 'interrupt':
                raise KeyboardInterrupt('injected check interruption')
            ctx.report.update(synthetic_only=True, synthetic_training_steps=2,
                              training_performed=False, network_evaluated=False)

        def train(ctx, out, log, stop_after, checkpoint):
            self.assertIsNone(checkpoint)
            ctx.completed = stop_after
            ctx.trace['epochs'] = [{'epoch': epoch, 'optimizer_steps': 2 * epoch,
                                   'method': runner.METHOD, 'train': [], 'test': []}
                                  for epoch in range(1, stop_after + 1)]
            ctx.report.update(training_performed=True, network_evaluated=True,
                              actual_new_optimizer_steps=2 * stop_after)

        original_renderer = reporter.write_outputs

        def optional_renderer_warning(out):
            rendered = original_renderer(out)
            rendered.setdefault('warnings', []).append('Optional Matplotlib unavailable (software fixture).')
            return rendered

        captured = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(captured))
            stack.enter_context(redirect_stderr(captured))
            stack.enter_context(mock.patch.object(runner, 'ROOT', package))
            stack.enter_context(mock.patch.object(runner, 'project_root', return_value=case))
            stack.enter_context(mock.patch.object(runner, 'protocol', return_value=fixture_protocol()))
            prepared = stack.enter_context(mock.patch.object(runner, 'prepare_context', side_effect=prepare))
            data = stack.enter_context(mock.patch.object(runner, 'prepare_data'))
            state = stack.enter_context(mock.patch.object(runner, 'prepare_state'))
            checked = stack.enter_context(mock.patch.object(runner, 'synthetic_check', side_effect=synthetic))
            trained = stack.enter_context(mock.patch.object(runner, 'run_epochs', side_effect=train))
            stack.enter_context(mock.patch.dict(sys.modules, {'training_report': reporter}))
            if render_warning:
                stack.enter_context(mock.patch.object(reporter, 'write_outputs', side_effect=optional_renderer_warning))
            if failure == 'report':
                stack.enter_context(mock.patch.object(runner, 'write_run_report',
                                      side_effect=OSError('injected run_report publication failure')))
            if failure == 'bundle':
                stack.enter_context(mock.patch.object(runner, 'feedback_bundle',
                                      side_effect=OSError('injected feedback publication failure')))
            argv = [command] + ([] if stop is None else ['--stop-after-epoch', str(stop)])
            exit_code = runner.main(argv)
        output = next((case / 'mx_runs' / runner.NAME).iterdir())
        report_path = output / 'run_report.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        return SimpleNamespace(exit_code=exit_code, text=captured.getvalue(), output=output,
                               report=report, prepare=prepared, data=data, state=state,
                               check=checked, run=trained, contexts=contexts)

    def assert_last_global_line(self, case, result, exit_code):
        self.assertEqual(case.exit_code, exit_code)
        lines = case.text.rstrip().splitlines()
        global_lines = [line for line in lines if line.startswith('GLOBAAL RESULTAAT:')]
        self.assertEqual(global_lines, ['GLOBAAL RESULTAAT: ' + result])
        self.assertEqual(lines[-1], 'GLOBAAL RESULTAAT: ' + result)

    def test_public_check_final_line_pass_after_real_bundle_no_research_scores(self):
        case = self.public_case()
        self.assert_last_global_line(case, 'PASS', 0)
        case.check.assert_called_once()
        case.run.assert_not_called()
        case.data.assert_not_called()
        report = case.report
        self.assertEqual(report['status'], 'PASS')
        self.assertTrue(report['synthetic_only'])
        self.assertFalse(report['training_performed'])
        self.assertFalse(report['network_evaluated'])
        self.assertEqual(report['completed_epochs'], 0)
        with (case.output / 'accuracy.csv').open(newline='') as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])
        analysis = reporter.load_results(case.output)
        self.assertFalse(analysis['display_new_scores'])
        self.assertFalse(analysis['endpoint_ready'])
        self.assertTrue((case.output / 'feedback_bundle.zip').is_file())

    def test_public_check_failure_is_last_line_fail_and_exit1(self):
        case = self.public_case(failure='check')
        self.assert_last_global_line(case, 'FAIL', 1)
        self.assertEqual(case.report['status'], 'FAIL')
        case.run.assert_not_called()

    def test_public_check_interruption_is_last_line_fail_and_exit130(self):
        case = self.public_case(failure='interrupt')
        self.assert_last_global_line(case, 'FAIL', 130)
        self.assertEqual(case.report['status'], 'FAIL')

    def test_report_publication_failure_cannot_finish_with_pass(self):
        case = self.public_case(failure='report')
        self.assert_last_global_line(case, 'FAIL', 1)
        self.assertIn('injected run_report publication failure', case.text)
        if case.report is not None:
            self.assertEqual(case.report['status'], 'FAIL')

    def test_feedback_publication_failure_cannot_finish_with_pass(self):
        case = self.public_case(failure='bundle')
        self.assert_last_global_line(case, 'FAIL', 1)
        self.assertEqual(case.report['status'], 'FAIL')
        self.assertIn('injected feedback publication failure', case.text)

    def test_optional_render_warning_keeps_last_line_pass_and_exit0(self):
        case = self.public_case(render_warning=True)
        self.assert_last_global_line(case, 'PASS', 0)
        self.assertEqual(case.report['status'], 'PASS')
        self.assertIn('Optional Matplotlib unavailable (software fixture).', case.report['warnings'])
        self.assertIn('RAPPORTWAARSCHUWING:', case.text)
        self.assertFalse(case.report.get('finalization_errors'))
        self.assertTrue((case.output / reporter.NOTEBOOK_NAME).is_file())
        self.assertTrue((case.output / 'feedback_bundle.zip').is_file())

    def test_public_fresh_run_has_no_checkpoint_and_100_epoch_default(self):
        case = self.public_case(command='run')
        self.assert_last_global_line(case, 'PASS', 0)
        case.check.assert_not_called()
        case.state.assert_called_once()
        case.run.assert_called_once()
        self.assertEqual(case.run.call_args.args[-2:], (100, None))
        self.assertEqual(case.report['completed_epochs'], 100)
        self.assertEqual(case.report['inherited_epochs'], 0)
        self.assertTrue(case.report['training_completed'])

    def test_public_partial_run_is_stopped_not_pass(self):
        case = self.public_case(command='run', stop=2)
        self.assert_last_global_line(case, 'GESTOPT', 0)
        self.assertEqual(case.report['status'], 'STOPPED')
        self.assertEqual(case.report['completed_epochs'], 2)
        self.assertFalse(case.report['training_completed'])

    def test_public_warning_aggregation_preserves_all_frequencies(self):
        case = self.public_case(warn=True)
        self.assert_last_global_line(case, 'PASS', 0)
        records = case.report['warning_counts']
        counts = {(row['message'], row['category']): row['count'] for row in records}
        self.assertEqual(counts, {('repeated software fixture warning', 'DeprecationWarning'): 7,
                                  ('other software fixture warning', 'UserWarning'): 3})
        self.assertEqual(len(case.report['warnings']), 2)
        self.assertEqual(sum(row['count'] for row in records), 10)

    def test_warning_collector_dedup_is_by_message_category_and_uses_no_rng(self):
        # Installed numerical libraries are used only to inspect RNG states.
        # No model, data, random draw or training is performed in this test.
        import numpy as np
        import torch
        collector = runner.WarningCollector()
        python_before, numpy_before = random.getstate(), np.random.get_state()
        torch_before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        captured = io.StringIO()
        with redirect_stdout(captured), redirect_stderr(captured), warnings.catch_warnings():
            warnings.simplefilter('always')
            with mock.patch.object(warnings, 'showwarning', collector.capture):
                for index in range(7):
                    warnings.warn_explicit('same text', DeprecationWarning,
                                           filename='fixture_' + str(index) + '.py', lineno=index + 1)
                for index in range(3):
                    warnings.warn_explicit('same text', RuntimeWarning, filename='fixture.py', lineno=index + 1)
                for index in range(2):
                    warnings.warn_explicit('different text', UserWarning, filename='fixture.py', lineno=index + 1)
        records = collector.as_records()
        self.assertEqual({(r['message'], r['category']): r['count'] for r in records},
                         {('same text', 'DeprecationWarning'): 7,
                          ('same text', 'RuntimeWarning'): 3, ('different text', 'UserWarning'): 2})
        self.assertTrue(all(set(('message', 'category', 'count', 'filename', 'lineno')) <= set(row)
                            for row in records))
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        self.assertTrue(np.array_equal(numpy_after[1], numpy_before[1]))
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        if cuda_before is not None:
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all())))


if __name__ == '__main__':
    unittest.main()
