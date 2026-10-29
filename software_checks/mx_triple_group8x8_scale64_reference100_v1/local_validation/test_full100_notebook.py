"""Full100/partial notebook controls. All artificial metrics are software fixtures."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import group_costs as costs
import notebook_support as n
import pandas as pd
import matplotlib.pyplot as plt
import nbformat


class Full100NotebookTests(unittest.TestCase):
    @staticmethod
    def fixture_run_and_references(through=100, completed=True):
        """Formatting objects only; fabricated values never join research outputs."""
        def metrics(stop, e4_peak, future_peak=None):
            rows = []
            for epoch in range(1, stop + 1):
                for bit in (8, 6, 4):
                    value = 70.0 + epoch * .02 + (bit - 4) * .2
                    # Separate per-mode maxima test that the lower modes never
                    # choose their own checkpoint. Tied E4 maxima choose first.
                    if bit == 8 and epoch in (e4_peak, e4_peak + 1):
                        value = 89.0
                    if bit == 8 and future_peak is not None and epoch == future_peak:
                        value = 95.0
                    if bit == 6 and epoch == 79:
                        value = 97.0
                    if bit == 4 and epoch == 91:
                        value = 98.0
                    rows.append({'phase': 'test', 'epoch': epoch, 'weight_bits': bit,
                                 'weight_format': n.FORMATS[bit], 'top1_percent': value,
                                 'correct1': round(value * 100), 'n': 10000,
                                 'objective_loss': .5})
            return pd.DataFrame(rows)
        config = json.loads((ROOT / 'PROTOCOL.json').read_text(encoding='utf-8'))
        run = n.RunAnalysis(ROOT,
                            report={'status': 'PASS', 'synthetic_fixture_only': True,
                                    'schema': 'mx-triple-group8x8-scale64-reference100-report-v1',
                                    'config': config},
                            summary={'full_training_completed': completed,
                                     'full_100_epoch_training_completed': completed},
                            metrics=metrics(through, 37))
        # In-memory compatible=True is for rendering/selection tests only.
        # Actual loader provenance and reference hashes have separate tests.
        refs = [n.GroupReference(ROOT, size, metrics=metrics(100, 12, peak),
                                 safe=True, compatible=True, synthetic=True)
                for size, peak in ((32, 60), (64, 70))]
        return run, refs

    def tearDown(self):
        plt.close('all')

    def test_exact_six_layer_costs_unchanged(self):
        report = costs.build_cost_report()
        actual = {r['layout_id']: (r['n_weights'], r['blocks'], r['padded_slots'],
                  r['packed_e4m3_bytes'], r['packed_e2m3_bytes'], r['packed_e2m1_bytes'])
                  for r in report['summary']}
        self.assertEqual(actual, {
            'row32': (129472, 4080, 1088, 134640, 102000, 69360),
            'row64': (129472, 2160, 8768, 140400, 105840, 71280),
            'tile8x8': (129472, 2024, 64, 131560, 99176, 66792)})
        self.assertEqual([r['layout_label'] for r in report['summary']], ['1×32', '1×64', '8×8'])

    def test_two_axis_padding_cost(self):
        row = costs.layer_cost('fixture', (9, 9), 64, 8, 'tile8x8')
        self.assertEqual((row['blocks'], row['stored_weight_slots'], row['padded_slots'],
                          row['packed_weight_and_scale_bytes']), (4, 256, 175, 260))
        self.assertEqual((row['padded_output_rows'], row['padded_weights_per_row']), (16, 16))

    def test_tampered_costs_rejected(self):
        report = costs.build_cost_report()
        report['summary'][-1]['padded_slots'] += 1
        with self.assertRaises(ValueError):
            costs.validate_cost_report(report)

    def test_row64_export_cannot_pass_tile8x8_audit(self):
        rows = [costs.layer_cost(k, v, 64, 8, 'row') for k, v in costs.LAYER_SHAPES.items()]
        with self.assertRaises(ValueError):
            costs.validate_export_layers(rows, 8)
        tiles = [costs.layer_cost(k, v, 64, 8, 'tile8x8') for k, v in costs.LAYER_SHAPES.items()]
        self.assertEqual(costs.validate_export_layers(tiles, 8)['packed_weight_and_scale_bytes'], 131560)

    def test_wrong_group_shape_rejected_with_correct_counts(self):
        rows = [costs.layer_cost(k, v, 64, 8, 'tile8x8') for k, v in costs.LAYER_SHAPES.items()]
        for row in rows:
            row['group_shape'] = json.dumps(row['group_shape'])
        self.assertEqual(costs.validate_export_layers(rows, 8)['status'], 'PASS')
        rows[0]['group_shape'] = '[1, 64]'
        with self.assertRaisesRegex(ValueError, 'group_shape'):
            costs.validate_export_layers(rows, 8)

    def test_wrong_group_axis_rejected_with_correct_counts(self):
        rows = [costs.layer_cost(k, v, 64, 8, 'tile8x8') for k, v in costs.LAYER_SHAPES.items()]
        rows[0]['group_axis'] = costs.ROW_AXIS
        with self.assertRaisesRegex(ValueError, 'group_axis'):
            costs.validate_export_layers(rows, 8)

    def test_actual_reference_hashes_and_all_hundred_epochs(self):
        refs = n.load_group_references(n.RunAnalysis(ROOT))
        self.assertEqual([r.group_size for r in refs], [32, 64])
        for ref, best in zip(refs, (42, 53)):
            self.assertTrue(ref.safe, ref.issues)
            self.assertFalse(ref.compatible)
            self.assertEqual(len(ref.metrics), 600)
            self.assertEqual(len(ref.test), 300)
            self.assertEqual(sorted(ref.test.epoch.unique()), list(range(1, 101)))
            self.assertEqual(ref.best_epoch(), best)
            self.assertEqual(ref.best_epoch(5), 5)

    def test_corrupt_reference_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'group64_reference'
            shutil.copytree(ROOT / 'group64_reference', target)
            with (target / 'epoch_metrics.csv').open('a') as handle:
                handle.write('\ncorruption\n')
            ref = n.load_group64_reference(n.RunAnalysis(ROOT), target)
            self.assertFalse(ref.safe)
            self.assertTrue(any('hash' in message for message in ref.issues))

    def test_pilot_reference_metadata_cannot_pass_full100_loader(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'scale64_reference'
            shutil.copytree(ROOT / 'scale64_reference', target)
            path = target / 'REFERENCE.json'
            obj = json.loads(path.read_text())
            obj['comparison_epochs'] = list(range(1, 6))
            path.write_text(json.dumps(obj))
            ref = n.load_scale64_reference(n.RunAnalysis(ROOT), target)
            self.assertFalse(ref.safe)

    def test_synthetic_run_never_compared_to_real_research(self):
        run, _ = self.fixture_run_and_references()
        refs = n.load_group_references(run)
        self.assertTrue(all(ref.safe for ref in refs))
        self.assertTrue(all(not ref.compatible for ref in refs))
        self.assertTrue(n.comparison_table(run, refs).empty)

    def test_full100_endpoint_and_common_e4_checkpoint(self):
        run, refs = self.fixture_run_and_references()
        self.assertTrue(run.reference_complete)
        table = n.comparison_table(run, refs)
        self.assertEqual(set(table.Groepsindeling), {'8×8', '1×32', '1×64'})
        self.assertEqual(len(table), 18)
        self.assertEqual(set(table.loc[table.Controlepunt == 'Epoch 100'].Epoch), {100})
        chosen = table.loc[table.Controlepunt.str.startswith('Beste')]
        self.assertEqual(chosen.groupby('Groepsindeling').Epoch.nunique().to_dict(),
                         {'1×32': 1, '1×64': 1, '8×8': 1})
        self.assertEqual(chosen.groupby('Groepsindeling').Epoch.first().to_dict(),
                         {'1×32': 60, '1×64': 70, '8×8': 37})
        self.assertEqual(run.observed_best_epoch, 37)  # first E4 tie
        for fig in (n.plot_checkpoint_accuracy(run),
                    n.plot_scale_comparison_checkpoints(run, refs)):
            fig.canvas.draw()

    def test_partial_run_uses_common_range_and_never_invents_epoch100(self):
        run, refs = self.fixture_run_and_references(through=45, completed=False)
        table = n.comparison_table(run, refs)
        self.assertFalse(run.reference_complete)
        self.assertTrue(table.loc[table.Controlepunt == 'Epoch 100', 'Testaccuracy [%]'].isna().all())
        chosen = table.loc[table.Controlepunt.str.startswith('Beste')]
        self.assertEqual(chosen.groupby('Groepsindeling').Epoch.first().to_dict(),
                         {'1×32': 12, '1×64': 12, '8×8': 37})
        fig = n.plot_scale_comparison_curves(run, refs)
        fig.canvas.draw()
        self.assertIn('voorlopig: epochs 1–45', fig._suptitle.get_text())
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 3)
            for line in ax.lines:
                self.assertEqual(list(line.get_xdata()), list(range(1, 46)))
        own = n.checkpoint_table(run)
        self.assertTrue(own.loc[own.Rol == 'epoch100', 'Testaccuracy [%]'].isna().all())
        n.plot_scale_comparison_checkpoints(run, refs).canvas.draw()

    def test_full_curves_have_all_hundred_points(self):
        run, refs = self.fixture_run_and_references()
        fig = n.plot_scale_comparison_curves(run, refs)
        fig.canvas.draw()
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 3)
            for line in ax.lines:
                self.assertEqual(list(line.get_xdata()), list(range(1, 101)))

    def test_false_completion_flag_cannot_turn_five_epochs_into_full100(self):
        run, refs = self.fixture_run_and_references(through=5, completed=True)
        self.assertFalse(run.reference_complete)
        self.assertTrue(n.comparison_table(run, refs).query('Controlepunt == "Epoch 100"')
                        ['Testaccuracy [%]'].isna().all())

    def test_missing_mode_epoch_blocks_full100(self):
        run, _ = self.fixture_run_and_references()
        run.metrics = run.metrics.loc[~((run.metrics.epoch == 30) & (run.metrics.weight_bits == 4))]
        self.assertFalse(run.reference_complete)

    def test_unconfirmed_status_or_selection_blocks_final_endpoint(self):
        run, _ = self.fixture_run_and_references()
        run.report['status'] = 'STOPPED_AT_EPOCH_BOUNDARY'
        self.assertFalse(run.reference_complete)
        run.report['status'] = 'PASS'
        run.selection_safe = False
        self.assertFalse(run.reference_complete)

    def test_measured_full100_duration_never_uses_old_projection(self):
        run, _ = self.fixture_run_and_references()
        run.times = pd.DataFrame({'epoch': range(1, 101), 'total_epoch_seconds': [2.0] * 100})
        run.timing = {'available': True, 'estimated_100_epoch_seconds': 999999.0}
        self.assertEqual(n.measured_completed_seconds(run), 200.0)
        text = n.timing_markdown(run)
        self.assertIn('200.00', text)
        self.assertIn('Alle honderd epochs', text)
        self.assertNotIn('999999', text)

    def test_contract_rejects_row64_with_same_group_size(self):
        config = json.loads((ROOT / 'PROTOCOL.json').read_text())
        report = {'schema': 'mx-triple-group8x8-scale64-reference100-report-v1', 'config': config}
        run = n.RunAnalysis(ROOT, report=copy.deepcopy(report))
        n._validate_contract(run)
        self.assertTrue(run.metrics_safe, run.issues)
        report['config']['mx_group_layout'] = 'row64'
        run = n.RunAnalysis(ROOT, report=report)
        n._validate_contract(run)
        self.assertFalse(run.metrics_safe)

    def test_check_only_or_empty_run_has_no_invented_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            run = n.load_run(td)
            self.assertTrue(run.test.empty)
            self.assertFalse(run.reference_complete)
            self.assertTrue(n.checkpoint_table(run)['Testaccuracy [%]'].isna().all())
            self.assertIn('Epoch100 niet gemeten', n.status_markdown(run))
            n.plot_checkpoint_accuracy(run).canvas.draw()

    def test_blank_notebook_schema_and_code_syntax(self):
        path = ROOT / 'MX_triple_group8x8_scale64_reference100_resultaten.ipynb'
        doc = nbformat.read(path, as_version=4)
        nbformat.validate(doc)
        cells = [cell for cell in doc.cells if cell.cell_type == 'code']
        self.assertEqual(len(cells), 14)
        for cell in cells:
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            compile(cell.source, 'notebook-cell', 'exec')
        self.assertNotIn('import torch', (ROOT / 'notebook_support.py').read_text())
        self.assertFalse((ROOT / 'MX_triple_group8x8_scale64_pilot5_resultaten.ipynb').exists())
        self.assertIn('Nieuwe initialisatie', doc.cells[0].source)
        self.assertIn('epoch 100', doc.cells[3].source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
