"""Read-only notebook and layout checks. Every artificial metric is a test fixture."""
from __future__ import annotations
import copy
import importlib.util
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


class NotebookAnalysisTests(unittest.TestCase):
    def test_exact_six_layer_costs(self):
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

    def test_actual_reference_hashes_and_first_five_epochs(self):
        run = n.RunAnalysis(ROOT)
        refs = n.load_group_references(run)
        self.assertEqual([r.group_size for r in refs], [32, 64])
        for ref in refs:
            self.assertTrue(ref.safe, ref.issues)
            self.assertFalse(ref.compatible)  # no current training result supplied
            self.assertEqual(len(ref.test), 15)
            self.assertEqual(sorted(ref.test.epoch.unique()), [1, 2, 3, 4, 5])
            self.assertLessEqual(ref.best_epoch(), 5)

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

    def test_corrupt_reference_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)/'group64_reference'
            shutil.copytree(ROOT/'group64_reference', target)
            with (target/'epoch_metrics.csv').open('a') as h:
                h.write('\ncorruption\n')
            ref = n.load_group64_reference(n.RunAnalysis(ROOT), target)
            self.assertFalse(ref.safe)
            self.assertTrue(any('hash' in message for message in ref.issues))

    @staticmethod
    def fixture_run_and_references(through=5):
        rows = []
        # Artificial values deliberately give different per-mode optima. All
        # selected modes must nevertheless follow E4M3's first maximum (epoch2).
        curves = {8: [80., 90., 90., 88., 89.], 6: [81., 82., 93., 84., 85.],
                  4: [79., 80., 81., 92., 83.]}
        for epoch in range(1, through+1):
            for bit in (8, 6, 4):
                rows.append({'phase':'test', 'epoch':epoch, 'weight_bits':bit,
                             'weight_format':n.FORMATS[bit], 'top1_percent':curves[bit][epoch-1],
                             'correct1':int(curves[bit][epoch-1]), 'n':100, 'objective_loss':.5})
        data = pd.DataFrame(rows)
        run = n.RunAnalysis(ROOT, report={'synthetic_fixture_only':True}, metrics=data)
        # The in-memory test objects exercise formatting only; provenance has
        # already been independently verified above. They are never exported.
        refs = [n.GroupReference(ROOT, size, metrics=data.copy(), safe=True, compatible=True)
                for size in (32, 64)]
        return run, refs

    def test_common_checkpoint_and_distinct_layout_labels(self):
        run, refs = self.fixture_run_and_references()
        table = n.comparison_table(run, refs)
        self.assertEqual(set(table.Groepsindeling), {'8×8', '1×32', '1×64'})
        self.assertEqual(len(table), 18)
        chosen = table.loc[table.Controlepunt.str.startswith('Beste')]
        self.assertEqual(set(chosen.Epoch), {2})
        self.assertEqual(set(table.loc[table.Controlepunt == 'Epoch 5'].Epoch), {5})
        for figure in (n.plot_scale_comparison_curves(run, refs), n.plot_scale_comparison_checkpoints(run, refs)):
            figure.canvas.draw()
            plt.close(figure)

    def test_partial_run_does_not_invent_epoch_five(self):
        run, refs = self.fixture_run_and_references(through=3)
        table = n.comparison_table(run, refs)
        self.assertTrue(table.loc[table.Controlepunt == 'Epoch 5', 'Testaccuracy [%]'].isna().all())
        self.assertEqual(set(table.loc[table.Controlepunt.str.startswith('Beste')].Epoch), {2})
        fig = n.plot_scale_comparison_checkpoints(run, refs)
        fig.canvas.draw(); plt.close(fig)

    def test_contract_rejects_row64_with_same_group_size(self):
        config = json.loads((ROOT/'group64_reference/run_report.json').read_text())['config']
        config.update(schema='mx-triple-group8x8-scale64-pilot5-protocol-v1', pilot_stop_epoch=5,
                      mx_group_layout='tile8x8', mx_group_shape=[8, 8], mx_group_geometry_revision='tile8x8-v1')
        report = {'schema':'mx-triple-group8x8-scale64-pilot5-report-v1', 'config':config}
        run = n.RunAnalysis(ROOT, report=copy.deepcopy(report))
        n._validate_contract(run)
        self.assertTrue(run.metrics_safe, run.issues)
        report['config']['mx_group_layout']='row64'
        run = n.RunAnalysis(ROOT, report=report)
        n._validate_contract(run)
        self.assertFalse(run.metrics_safe)

    def test_blank_notebook_and_code_syntax(self):
        doc = json.loads((ROOT/'MX_triple_group8x8_scale64_pilot5_resultaten.ipynb').read_text())
        count = 0
        for cell in doc['cells']:
            if cell['cell_type']=='code':
                count += 1
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                compile(''.join(cell['source']), 'notebook-cell', 'exec')
        self.assertEqual(count,14)
        self.assertNotIn('import torch', (ROOT/'notebook_support.py').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
