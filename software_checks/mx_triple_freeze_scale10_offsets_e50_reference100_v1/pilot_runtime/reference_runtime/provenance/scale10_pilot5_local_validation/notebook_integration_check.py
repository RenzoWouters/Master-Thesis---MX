"""Read-only notebook checks against actual CPU runner outputs and discovery fixtures."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scale10_notebook_fixture_checks', HERE / 'test_notebook.py')
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)
ns = checks.ns


def main(source, out):
    out.mkdir(parents=True, exist_ok=False)
    package = HERE.parent
    template = package / 'MX_triple_learned_scale10_pilot5_resultaten.ipynb'
    notebook = json.loads(template.read_text())
    initial_hash = hashlib.sha256(template.read_bytes()).hexdigest()
    results = {}
    for name, expected_epoch, completed in [('continuous5', 5, True), ('resume2_to5', 5, True),
                                            ('stop2', 2, False), ('wrong_binary_learned_contract', None, False)]:
        reports = list((source / name).glob('*/run_report.json'))
        assert len(reports) == 1
        original = reports[0].parent
        folder = out / name
        folder.mkdir()
        for p in original.iterdir():
            if p.is_file() and p.suffix in ('.json', '.csv', '.log', '.txt'):
                shutil.copy2(p, folder / p.name)
        if (original / 'fixed_references').is_dir():
            shutil.copytree(original / 'fixed_references', folder / 'fixed_references')
        shutil.copy2(package / 'notebook_support.py', folder / 'notebook_support.py')
        before = checks.raw_hashes(folder)
        context, stdout = checks.execute_cells(notebook, folder)
        run = context['run']
        assert run.synthetic and run.last_epoch == expected_epoch
        assert run.pilot_complete is completed
        assert run.scales_safe
        errors = [x for x in run.issues if x['Niveau'] in ('FOUT', 'RUNFOUT')]
        if expected_epoch is not None:
            assert not errors, errors
        else:
            assert run.report['status'] == 'FAIL' and run.test.empty
            assert any(x['Niveau'] == 'RUNFOUT' for x in run.issues)
        assert before == checks.raw_hashes(folder)
        results[name] = {'status': 'PASS', 'completed_epochs': expected_epoch,
                         'all_16_notebook_code_cells_executed': True,
                         'generated_figures': len(context['figures']),
                         'copied_runner_inputs_unchanged': True,
                         'no_SVHN_accuracy_claim': True}
        (out / (name + '_stdout.txt')).write_text(stdout)
        checks.plt.close('all')
    project = out / 'discovery_project'
    runs = project / 'mx_runs' / 'mx_triple_learned_scale10_pilot5'
    runs.mkdir(parents=True)
    schema = 'mx-triple-learned-scale10-pilot5-report-v1'
    for name, command, status in [('20261030_090000_joint_run', 'run', 'PASS'),
                                  ('20261030_100000_joint_run', 'run', 'FAIL'),
                                  ('20261030_110000_joint_check', 'check', 'PASS')]:
        folder = runs / name
        folder.mkdir()
        checks.write_json(folder / 'run_report.json', {'schema': schema, 'command': command, 'status': status})
    latest = ns.resolve_run_dir(None, start=project)
    assert latest == runs / '20261030_100000_joint_run'
    assert ns.resolve_run_dir(runs / '20261030_090000_joint_run', start=project) == runs / '20261030_090000_joint_run'
    assert ns.resolve_run_dir(Path('/missing_original_drive/result'), start=out / 'continuous5') == out / 'continuous5'
    empty = out / 'empty_template'
    empty.mkdir()
    empty_run = ns.load_run(empty)
    assert empty_run.test.empty and not empty_run.pilot_complete
    results['discovery'] = {'status': 'PASS', 'latest_failed_training_is_visible': True,
        'later_check_does_not_hide_training': True, 'explicit_older_run_supported': True,
        'relocated_feedback_supported': True, 'empty_template_reports_missing_data': True}
    # Compare recipe metadata only; the result below is not a new training observation.
    config = json.loads((package / 'PROTOCOL.json').read_text())
    ref_root = package / 'fixed_references'
    binary = json.loads((ref_root / 'learned_binary_pilot5' / 'run_report.json').read_text())
    differences = [k for k in ns.COMPARABLE_CONFIG_KEYS if config.get(k) != binary['config'].get(k)]
    assert not differences, differences
    assert hashlib.sha256(template.read_bytes()).hexdigest() == initial_hash
    report = {'status': 'PASS', 'synthetic_fixture_only': True, 'tests': results,
              'binary_non_scale_recipe_fields_identical': True,
              'template_sha256': initial_hash,
              'helper_sha256': hashlib.sha256((package / 'notebook_support.py').read_bytes()).hexdigest()}
    checks.write_json(out / 'notebook_integration_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.source.resolve(), args.out.resolve())
