"""Validate read-only analysis on explicit software fixtures, without training.

Usage: python test_notebook_analysis.py --package PATH --integration PATH --out PATH
The integration directory must contain frozen51_to55, stop_after52,
resume52_to55 and softstop51 from the training-state integration test.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path):
    return {p.relative_to(path).as_posix(): digest(p) for p in path.rglob('*')
            if p.is_file() and p.suffix in ('.json', '.csv', '.ipynb', '.py') and '__pycache__' not in p.parts}


def read_helper(package):
    name = '_scale10_freeze_analysis_validation'
    spec = importlib.util.spec_from_file_location(name, package / 'notebook_support.py')
    helper = importlib.util.module_from_spec(spec)
    sys.modules[name] = helper
    spec.loader.exec_module(helper)
    return helper


def view(source, destination, package):
    destination.mkdir(exist_ok=False)
    if source is not None:
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.json', '.csv'):
                shutil.copy2(path, destination / path.name)
        shutil.copytree(source / 'parent_reference', destination / 'parent_reference')
    else:
        shutil.copytree(package / 'parent_reference', destination / 'parent_reference')
    shutil.copy2(package / 'notebook_support.py', destination / 'notebook_support.py')
    notebook = package / 'MX_triple_freeze_scale10_offsets_e50_pilot5_resultaten.ipynb'
    shutil.copy2(notebook, destination / notebook.name)
    return destination


def execute_notebook(path):
    before = manifest(path)
    nb = nbformat.read(path / 'MX_triple_freeze_scale10_offsets_e50_pilot5_resultaten.ipynb', as_version=4)
    nbformat.validate(nb)
    env = {'__name__': '__main__'}
    code_cells = 0
    with patch('IPython.display.display', lambda *args, **kwargs: None), contextlib.redirect_stdout(io.StringIO()):
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue
            assert cell.execution_count is None and cell.outputs == []
            code = cell.source
            if 'parameters' in cell.metadata.get('tags', []):
                code = f'from pathlib import Path\nRUN_DIR = Path({str(path)!r})\nSAVE_FIGURES = False\n'
            exec(compile(code, f'<notebook_cell_{code_cells}>', 'exec'), env)
            code_cells += 1
    assert manifest(path) == before, 'Analysis changed source results or the unexecuted notebook.'
    assert len(env['figures']) == 10
    plt.close('all')
    return code_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--integration', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    package, integration, output = args.package.resolve(), args.integration.resolve(), args.out.resolve()
    output.mkdir(exist_ok=False, parents=True)
    ns = read_helper(package)
    results = []
    for name, epochs, complete in [('frozen51_to55', list(range(51,56)), True),
                                    ('stop_after52', [51,52], False),
                                    ('resume52_to55', list(range(51,56)), True),
                                    ('softstop51', [51], False)]:
        destination = view(integration / name, output / name, package)
        run = ns.load_run(destination)
        assert run.synthetic and run.observed_epochs == epochs and run.complete == complete
        assert not [issue for issue in run.issues if issue['Niveau'] in ('FOUT','RUNFOUT')], run.issues
        assert run.freeze_safe and run.reference_safe and run.metrics_safe
        for role in ('last', 'best'):
            table = ns.factor_distribution_table(run, role)
            assert len(table) == 10 and table.Groepen.sum() == 4080
            assert np.isclose(table['Aandeel [%]'].sum(), 100)
            assert table.Factor.tolist() == ['×'+str(factor) for factor in ns.SCALE_FACTORS]
        assert run.best_epoch() in epochs
        if not complete:
            assert ns.checkpoint_table(run, 'endpoint').empty
        code_cells = execute_notebook(destination)
        results.append(dict(fixture=name, status='PASS', complete=complete, epochs=epochs,
                            code_cells_executed=code_cells, input_files_unchanged=True,
                            issues=run.issues))
    for name, command, status in [('early_failure', 'run', 'FAIL'), ('check_only', 'check', 'PASS')]:
        destination = view(None, output / name, package)
        report = dict(schema='mx-triple-freeze-scale10-offsets-e50-pilot5-report-v1',
                      command=command, status=status, config=json.loads((package/'PROTOCOL.json').read_text()),
                      synthetic_fixture_only=False, continuation_completed=False)
        if status == 'FAIL': report['error'] = 'Explicit early-failure software fixture; no training.'
        (destination / 'run_report.json').write_text(json.dumps(report)+'\n')
        run = ns.load_run(destination)
        assert not run.complete and not run.observed_epochs and ns.factor_distribution_table(run).empty
        if command == 'check':
            assert not [issue for issue in run.issues if issue['Niveau'] in ('FOUT', 'RUNFOUT')], run.issues
            assert 'geen training uitgevoerd' in ns.status_markdown(run)
        else:
            assert any(issue['Niveau'] == 'RUNFOUT' for issue in run.issues)
        code_cells = execute_notebook(destination)
        results.append(dict(fixture=name, status='PASS', no_results_presented=True,
                            code_cells_executed=code_cells, input_files_unchanged=True, issues=run.issues))
    destination = view(integration/'frozen51_to55', output/'corrupt_factor', package)
    path = destination/'scale_block_state.csv'
    frame = pd.read_csv(path, converters={'theta':float})
    frame.loc[0, 'factor'] = int(frame.loc[0, 'factor']) * 2
    frame.to_csv(path,index=False)
    run = ns.load_run(destination)
    assert not run.freeze_safe and ns.factor_distribution_table(run).empty
    assert any('blok-CSV' in issue['Melding'] for issue in run.issues)
    results.append(dict(fixture='corrupt_factor', status='PASS', corrupt_data_rejected=True))
    # User-requested zero categories and percentages are verified against actual parent50 state,
    # without labelling it as a measurement from the still-unrun freeze experiment.
    parent = ns.Analysis(path=package)
    parent.report={'synthetic_fixture_only':True}
    parent.baseline=json.loads((package/'parent_reference/baseline_epoch50.json').read_text())
    table = ns.factor_distribution_table(parent, 'parent')
    assert table.Groepen.tolist() == [0,0,0,0,5,1191,2745,139,0,0]
    assert table.Groepen.eq(0).sum()==6 and table.Groepen.sum()==4080
    report = dict(status='PASS', scope='read-only notebook integration; explicit software fixtures only',
                  no_training_started=True, outputs_in_deliverable_notebook_empty=True,
                  all_ten_factors_including_zero_visible=True,
                  actual_parent_epoch50_counts_verified=True, fixtures=results,
                  helper_sha256=digest(package/'notebook_support.py'),
                  notebook_sha256=digest(package/'MX_triple_freeze_scale10_offsets_e50_pilot5_resultaten.ipynb'))
    (output/'notebook_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','fixtures':len(results),'report':str(output/'notebook_validation.json')}))


if __name__=='__main__':
    main()
