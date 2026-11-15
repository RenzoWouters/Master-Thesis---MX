"""Execute full100 notebook fixtures without training or inventing research data.

The historical reference is the supplied real source. All NEW run metrics in
this developer test are explicitly synthetic and stay in a fresh temp folder.
"""
from pathlib import Path
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

os.environ['MPLBACKEND'] = 'Agg'
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import notebook_support as ns
import matplotlib.pyplot as plt

FORMATS = ns.FORMATS
BITS = {'E4M3': 8, 'E5M2': 8, 'E2M3': 6, 'E3M2': 6, 'E2M1': 4}
FIELDS = ['epoch', 'phase', 'weight_format', 'weight_bits', 'activation_bits',
          'n', 'correct1', 'correct5', 'top1_percent', 'top5_percent',
          'objective_loss', 'hard_ce', 'objective_type', 'teacher_formats', 'lr']
REFERENCE = json.loads((PACKAGE / 'reference_data/triple_direct_reference.json').read_text())


def fixture(root, name, epochs, completed=None, mutate=None):
    out = root / name
    out.mkdir()
    configuration = copy.deepcopy(REFERENCE['config'])
    configuration.update(planned_epochs=100, synthetic_fixture_only=True)
    report = {'status': 'DEVELOPMENT_FIXTURE', 'synthetic_fixture_only': True,
              'completed_epochs': epochs if completed is None else completed,
              'config': configuration}
    (out / 'report.json').write_text(json.dumps(report))
    rows = []
    for epoch in range(1, epochs + 1):
        lr = 0.001 * (0.1 ** sum(epoch > milestone for milestone in (50, 75, 90)))
        for phase in ('train', 'test'):
            n = configuration[phase + '_examples']
            for index, fmt in enumerate(FORMATS):
                accuracy = (80 + 17 * (1 - math.exp(-epoch / 7)) if phase == 'train'
                            else 85 + 10 * (1 - math.exp(-epoch / 9))) - index * 0.07
                correct = round(n * accuracy / 100)
                row = dict(zip(FIELDS, [epoch, phase, fmt, BITS[fmt], 32, n, correct, n,
                    100.0 * correct / n, 100.0, 0.7, 0.6,
                    'hard_cross_entropy' if phase == 'test' or BITS[fmt] == 8 else 'soft_cross_entropy', '', lr]))
                rows.append(row)
    if mutate: rows = mutate(rows)
    if rows:
        with (out / 'metrics.csv').open('x', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
    ns.write_notebook(out)
    original_files = {f.relative_to(PACKAGE / 'reference_data').as_posix(): f.read_bytes()
                      for f in (PACKAGE / 'reference_data').rglob('*') if f.is_file()}
    copied_files = {f.relative_to(out / 'reference_data').as_posix(): f.read_bytes()
                    for f in (out / 'reference_data').rglob('*') if f.is_file()}
    assert copied_files == original_files, 'Reference snapshot is not self-contained'
    return out


def execute(out):
    notebook = json.loads((out / ns.NOTEBOOK_NAME).read_text())
    namespace = {}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                ast.parse(source)
                exec(compile(source, str(out / ns.NOTEBOOK_NAME), 'exec'), namespace)
    (out / 'execution_stdout.txt').write_text(capture.getvalue())
    return namespace


def main(root=None):
    root = Path(tempfile.mkdtemp(prefix='mx_reference100_notebook_')) if root is None else Path(root)
    root.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Synthetic NEW-run fixtures; real supplied historical reference; no new SVHN results',
              'status': 'RUNNING', 'checks': []}
    for name, epochs in [('complete100', 100), ('partial75', 75), ('check_only', 0)]:
        out = fixture(root, name, epochs)
        scope = execute(out)
        assert len(scope['endpoint']) == (5 if epochs == 100 else 0)
        assert len(scope['endpoint_comparison']) == (3 if epochs == 100 else 0)
        assert scope['reference_compatible']
        if epochs:
            assert set(scope['matched_curves']) == {'E4M3', 'E2M3', 'E2M1'}
            assert all(len(frame) == epochs for frame in scope['matched_curves'].values())
        if epochs == 100:
            assert set(scope['endpoint_comparison']['Formaat']) == {'E4M3', 'E2M3', 'E2M1'}
            assert len(scope['fig_comparison'].axes) == 3
            assert len(scope['fig_comparison_curves'].axes) == 3
            with (PACKAGE / 'reference_data/triple_direct_metrics.csv').open() as file:
                old_rows = [row for row in csv.DictReader(file) if row['phase'] == 'test' and int(row['epoch']) == 100]
            for row in old_rows:
                actual = scope['endpoint_comparison'].set_index('Formaat').loc[row['weight_format'], '3 formaten, direct (%)']
                assert math.isclose(actual, 100 * int(row['correct1']) / int(row['n']), abs_tol=1e-12)
            for fig_name in ['fig_comparison', 'fig_comparison_curves', 'fig_accuracy']:
                figure = scope[fig_name]
                figure.suptitle('SYNTHETISCHE LAYOUTTEST — GEEN NIEUWE TRAININGSRESULTATEN', fontsize=12)
                figure.savefig(root / (fig_name + '_layout.png'), dpi=110)
        plt.close('all')
        report['checks'].append({'case': name, 'status': 'PASS'})
    out = fixture(root, 'uncommitted100', 100, completed=99)
    scope = execute(out)
    assert scope['endpoint'].empty and scope['endpoint_comparison'].empty
    assert len(scope['complete_test'][scope['complete_test']['epoch'] == 100]) == 5
    plt.close('all')
    report['checks'].append({'case': 'uncommitted_epoch100_excluded_from_primary_and_comparison', 'status': 'PASS'})
    out = fixture(root, 'missing_new_mode100', 100,
                  mutate=lambda rows: [r for r in rows if not (r['epoch'] == 100 and r['phase'] == 'test' and r['weight_format'] == 'E5M2')])
    scope = execute(out)
    assert scope['endpoint'].empty and scope['endpoint_comparison'].empty
    plt.close('all')
    report['checks'].append({'case': 'incomplete_five_format_endpoint_excluded', 'status': 'PASS'})
    out = fixture(root, 'reference_missing', 100)
    shutil.rmtree(out / 'reference_data')
    scope = execute(out)
    assert len(scope['endpoint']) == 5 and scope['endpoint_comparison'].empty
    plt.close('all')
    report['checks'].append({'case': 'missing_reference_not_fabricated', 'status': 'PASS'})
    out = fixture(root, 'reference_mismatch', 100)
    metadata_path = out / 'reference_data/triple_direct_reference.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['config']['batch_size'] = 256
    metadata_path.write_text(json.dumps(metadata))
    scope = execute(out)
    assert not scope['reference_compatible'] and scope['endpoint_comparison'].empty
    plt.close('all')
    report['checks'].append({'case': 'incompatible_shared_protocol_blocks_comparison', 'status': 'PASS'})
    rejected = [('duplicate_new_rows', lambda rows: rows + [rows[0].copy()]),
                ('invalid_new_accuracy', lambda rows: [{**row, 'top1_percent': row['top1_percent'] + 1} if i == 0 else row for i, row in enumerate(rows)])]
    for name, mutate in rejected:
        out = fixture(root, name, 100, mutate=mutate)
        try: execute(out)
        except ValueError as exc:
            report['checks'].append({'case': name, 'status': 'PASS_expected_rejection', 'error': str(exc)})
        else: raise AssertionError('Invalid current metrics accepted: ' + name)
        plt.close('all')
    out = fixture(root, 'reference_hash_mismatch', 100)
    target = out / 'reference_data/triple_direct_metrics.csv'
    target.write_bytes(target.read_bytes() + b'\n')
    try: execute(out)
    except ValueError as exc:
        assert 'hash' in str(exc)
        report['checks'].append({'case': 'modified_reference_table_rejected', 'status': 'PASS_expected_rejection'})
    else: raise AssertionError('Changed reference file accepted')
    plt.close('all')
    report['status'] = 'PASS'
    report['reference_source_sha256'] = REFERENCE['source_sha256']
    report['reference_metrics_sha256'] = REFERENCE['metrics_sha256']
    (root / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Only developer fixtures were generated at:', root)
    return report


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
