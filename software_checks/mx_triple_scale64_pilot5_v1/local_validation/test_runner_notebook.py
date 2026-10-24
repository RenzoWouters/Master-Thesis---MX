"""Read-only notebook execution on actual synthetic runner output.

Usage: python test_runner_notebook.py OUTPUT_REPORT RUN_DIR [RUN_DIR ...]
No model/checkpoint execution and no executed notebook saved.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_notebook as validation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(folder):
    return {p.relative_to(folder).as_posix(): digest(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def main(output, directories):
    name = 'MX_triple_scale64_pilot5_resultaten.ipynb'
    template = json.loads((ROOT / name).read_text(encoding='utf8'))
    cases = []
    for folder in map(lambda p: Path(p).resolve(), directories):
        before = tree_hashes(folder)
        notebook = json.loads((folder / name).read_text(encoding='utf8'))
        assert digest(folder / 'notebook_support.py') == digest(ROOT / 'notebook_support.py')
        comparable = copy.deepcopy(notebook)
        parameters = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                assert cell.get('execution_count') is None and cell.get('outputs') == []
            if 'parameters' in cell.get('metadata', {}).get('tags', []):
                source = ''.join(cell['source'])
                assignments = {node.targets[0].id: node.value for node in ast.parse(source).body
                               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
                assert assignments['SAVE_FIGURES'].value is False
                assert Path(assignments['RUN_DIR'].args[0].value).resolve() == folder
                parameters.append({'cell': i, 'RUN_DIR': str(folder), 'SAVE_FIGURES': False})
                comparable['cells'][i]['source'] = template['cells'][i]['source']
        assert len(parameters) == 1 and comparable == template
        context, _ = validation.execute_cells(notebook, folder, save=False)
        run, figures = context['run'], context['figures']
        assert run.synthetic and not run.reference_complete
        assert run.metrics_safe and run.selection_safe and run.prediction_safe
        assert not any(x['Niveau'] in ('FOUT', 'RUNFOUT', 'ONTBREEKT') for x in run.issues), run.issues
        assert len(figures) == 6
        assert run.pilot_complete == (run.report['status'] == 'PASS' and run.last_epoch == 5)
        assert all('SYNTHETISCHE CONTROLE' in fig._suptitle.get_text() for fig in figures.values())
        assert validation.ns.checkpoint_table(run).loc[
            validation.ns.checkpoint_table(run).Rol == 'epoch100', 'Testaccuracy [%]'].isna().all()
        assert before == tree_hashes(folder)
        cases.append({'directory': str(folder), 'source_report_sha256': digest(folder / 'run_report.json'),
                      'source_report_status': run.report['status'], 'synthetic_fixture_only': True,
                      'last_epoch': run.last_epoch, 'selected_E4_best_epoch': run.observed_best_epoch,
                      'pilot_complete': run.pilot_complete, 'metric_rows': len(run.metrics),
                      'selected_rows': len(run.selected), 'prediction_rows': len(run.predictions),
                      'figures': sorted(figures), 'blank_notebook': True,
                      'code_cells_executed_in_memory': sum(c['cell_type'] == 'code' for c in notebook['cells']),
                      'all_run_files_hashed_unchanged': len(before),
                      'copied_helper_matches_package': True, 'parameters': parameters,
                      'issues': run.issues, 'figures_saved': False, 'executed_notebook_saved': False,
                      'model_checkpoint_or_training_imported': False})
        validation.plt.close('all')
    report = {'schema': 'mx-triple-scale64-pilot5-runner-notebook-integration-v1', 'status': 'PASS',
              'scope': 'Actual check/run/resume runner products with test-only synthetic CPU data; no SVHN or research accuracy',
              'helper_sha256': digest(ROOT / 'notebook_support.py'),
              'notebook_sha256': digest(ROOT / name), 'runner_sha256': digest(ROOT / 'run_train.py'),
              'test_script_sha256': digest(__file__), 'cases': cases}
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'status': 'PASS', 'cases': len(cases), 'report': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit('Usage: test_runner_notebook.py OUTPUT_REPORT RUN_DIR [RUN_DIR ...]')
    main(sys.argv[1], sys.argv[2:])
