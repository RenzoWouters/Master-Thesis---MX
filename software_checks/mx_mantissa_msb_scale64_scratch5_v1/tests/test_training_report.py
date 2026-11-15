"""Report guards only; temporary metric fixtures are never research results."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('msb64_report_guard_module', ROOT / 'training_report.py')
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class TrainingReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='msb64_report_fixture_only_')
        self.out = Path(self.temp.name)
        self.historical = []
        with (ROOT / 'reference_lsb_scale64/epoch_metrics.csv').open(newline='') as handle:
            for raw in csv.DictReader(handle):
                if raw['phase'] == 'test' and 1 <= int(raw['epoch']) <= 5:
                    self.historical.append({
                        'method': 'lsb_trunc', 'epoch': int(raw['epoch']), 'phase': 'test',
                        'weight_bits': int(raw['weight_bits']), 'weight_format': raw['weight_format'],
                        'n': int(raw['n']), 'correct1': int(raw['correct1']), 'correct5': int(raw['correct5']),
                        'top1_percent': float(raw['top1_percent']), 'top5_percent': float(raw['top5_percent']),
                        'loss': float(raw['objective_loss']), 'lr': float(raw['lr'])})
        self.new_fixture = []
        for old in self.historical:
            row = dict(old, method='msb_bestexp')
            row['correct1'] -= 10 * row['epoch']
            row['top1_percent'] = 100 * row['correct1'] / row['n']
            self.new_fixture.append(row)
        self.report_fixture = {
            'schema': reporter.SCHEMA, 'status': 'PASS', 'command': 'run',
            'protocol': {'planned_epochs': 5, 'reference_end_epoch': 100, 'lr': .001},
            'completed_epochs': 5, 'training_completed': True, 'network_evaluated': True,
            'training_performed': True, 'synthetic_only': False, 'synthetic_fixture_only': False,
            'warnings': [], 'reference_comparison': dict.fromkeys(reporter.COMPARISON_KEYS, True),
            'test_fixture_note': 'temporary report-branch test; no numerical training or research evidence'}
        self.csv('reference_lsb.csv', self.historical)
        self.csv('accuracy.csv', self.new_fixture)
        self.report(self.report_fixture)

    def tearDown(self):
        self.temp.cleanup()

    def csv(self, name, rows):
        with (self.out / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.historical[0]))
            writer.writeheader()
            writer.writerows(rows)

    def report(self, value):
        (self.out / 'run_report.json').write_text(json.dumps(value))

    def test_full_endpoint_requires_all_new_and_historical_rows(self):
        result = reporter.load_results(self.out)
        self.assertTrue(result['endpoint_ready'])
        self.assertEqual(len(result['new_rows']), 15)
        self.assertEqual(len(result['reference_rows']), 15)
        self.assertFalse(result['errors'])
        self.assertIn('Nieuw − eerder', reporter.endpoint_markdown(result))
        self.csv('accuracy.csv', self.new_fixture[:-1])
        result = reporter.load_results(self.out)
        self.assertFalse(result['endpoint_ready'])
        self.assertFalse(result['display_new_scores'])
        self.assertTrue(result['errors'])

    def test_partial_run_has_no_epoch5_comparison(self):
        self.report({**self.report_fixture, 'status': 'STOPPED', 'completed_epochs': 2,
                     'training_completed': False})
        self.csv('accuracy.csv', [row for row in self.new_fixture if row['epoch'] <= 2])
        result = reporter.load_results(self.out)
        self.assertTrue(result['display_new_scores'])
        self.assertEqual(len(result['new_rows']), 6)
        self.assertFalse(result['endpoint_ready'])
        self.assertIn('nog niet beschikbaar', reporter.summary_markdown(result))

    def test_every_comparison_guard_is_required(self):
        for key in reporter.COMPARISON_KEYS:
            for value in (False, None):
                with self.subTest(key=key, value=value):
                    self.report({**self.report_fixture, 'reference_comparison': {
                        **self.report_fixture['reference_comparison'], key: value}})
                    result = reporter.load_results(self.out)
                    self.assertFalse(result['endpoint_ready'])
                    self.assertTrue(result['display_new_scores'])

    def test_check_failure_and_synthetic_flags_never_show_new_scores(self):
        for change in ({'command': 'check'}, {'status': 'FAIL'},
                       {'synthetic_only': True}, {'synthetic_fixture_only': True}):
            with self.subTest(change=change):
                self.report({**self.report_fixture, **change})
                result = reporter.load_results(self.out)
                self.assertFalse(result['display_new_scores'])
                self.assertFalse(result['endpoint_ready'])
                self.assertEqual(result['new_rows'], [])
                self.assertEqual(len(result['reference_rows']), 15)
                self.assertIn('historische metingen', reporter.summary_markdown(result))

    def test_wrong_method_duplicate_and_inconsistent_counts_are_rejected(self):
        wrong_method = [dict(row) for row in self.new_fixture]
        wrong_method[0]['method'] = 'lsb_trunc'
        wrong_percent = [dict(row) for row in self.new_fixture]
        wrong_percent[0]['top1_percent'] = 0.
        for rows in (wrong_method, wrong_percent, self.new_fixture + self.new_fixture[:1]):
            with self.subTest(kind=rows[0]['method'], length=len(rows)):
                self.csv('accuracy.csv', rows)
                result = reporter.load_results(self.out)
                self.assertTrue(result['errors'])
                self.assertFalse(result['display_new_scores'])
                self.assertFalse(result['endpoint_ready'])

    def test_historical_counts_are_pinned_beyond_percentage_consistency(self):
        changed = [dict(row) for row in self.historical]
        changed[0]['correct1'] -= 1
        changed[0]['top1_percent'] = 100 * changed[0]['correct1'] / changed[0]['n']
        self.csv('reference_lsb.csv', changed)
        result = reporter.load_results(self.out)
        self.assertEqual(result['reference_rows'], [])
        self.assertFalse(result['endpoint_ready'])
        self.assertTrue(result['errors'])

    def test_pending_notebook_has_visible_status_and_valid_code_cells(self):
        for name in ('run_report.json', 'accuracy.csv', 'reference_lsb.csv'):
            (self.out / name).unlink()
        output = reporter.write_outputs(self.out)
        self.assertFalse(output['figures'])
        self.assertFalse(output['errors'])
        notebook = json.loads(Path(output['notebook_path']).read_text())
        self.assertIn('NOG NIET UITGEVOERD', ''.join(notebook['cells'][1]['source']))
        for cell in notebook['cells']:
            self.assertIn('id', cell)
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'generated-notebook-cell', 'exec')


if __name__ == '__main__':
    unittest.main()
