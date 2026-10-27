from pathlib import Path
import argparse
import copy
import json
import shutil
import sys
sys.dont_write_bytecode = True
from validate_group64_full100_notebook import PKG, OUT, ROOT, ACTUAL, load_helper, execute_cells, digest_tree, sha, plt, pd

parser = argparse.ArgumentParser()
parser.add_argument('--complete', type=Path, required=True)
parser.add_argument('--partial', type=Path, required=True)
args = parser.parse_args()
m = load_helper(PKG / 'notebook_support.py', 'group64_full100_runner_notebook_validation')
report = {'status': 'PASS', 'synthetic_fixture_only': True, 'real_SVHN_training_performed': False, 'cases': {}}
for case, path in [('complete100', args.complete), ('partial1', args.partial)]:
    for name in ['notebook_support.py', 'group_costs.py']:
        assert sha(path / name) == sha(PKG / name), name + ': final helper differs from runner copy'
    before = digest_tree(path)
    run = m.load_run(path)
    refs = m.load_group_references(run)
    assert run.metrics_safe and run.selection_safe and run.prediction_safe
    assert not [x for x in run.issues if x['Niveau'] in ('FOUT', 'RUNFOUT')], run.issues
    assert run.synthetic
    assert len(run.layout_audits) == 6 and run.cost_report_verified
    assert all(r.safe and not r.compatible for r in refs)
    assert run.last_epoch == (100 if case == 'complete100' else 1)
    assert run.reference_complete == (case == 'complete100')
    summary, env = execute_cells(PKG / 'MX_triple_group64_scale64_reference100_resultaten.ipynb', path, 'fixture_' + case)
    assert before == digest_tree(path), 'Input mutation'
    for name in ['checkpoint_accuracy', 'quantization_diagnostics', 'timing']:
        env['figures'][name].savefig(OUT / (case + '_' + name + '.png'), dpi=130, bbox_inches='tight')
    plt.close('all')
    endpoint = m.checkpoint_table(run)
    if case == 'complete100':
        assert not endpoint.loc[endpoint.Rol == 'epoch100', 'Testaccuracy [%]'].isna().any()
    else:
        assert endpoint.loc[endpoint.Rol == 'epoch100', 'Testaccuracy [%]'].isna().all()
    report['cases'][case] = {**summary, 'path': str(path), 'metrics_safe': True, 'selection_safe': True, 'reference_complete': run.reference_complete, 'costs_verified': True, 'six_export_layouts_verified': True, 'comparison_to_real_references_blocked': True}

# Deliberately corrupted isolated copies; actual input bytes remain untouched.
for case in ['mixed_checkpoint_epochs', 'corrupt_cost_geometry', 'wrong_pilot_contract', 'tampered_reference', 'false_full100_claim']:
    target = OUT / 'negative' / case
    if target.exists():
        shutil.rmtree(target)
    source = args.partial if case == 'false_full100_claim' else args.complete
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('*.pt', '*.sha256', '__pycache__', 'checkpoints', 'epoch_summaries', 'batch_metrics', 'sample_order'))
    if case == 'mixed_checkpoint_epochs':
        p = target / 'selected_metrics.csv'
        d = pd.read_csv(p, float_precision='round_trip')
        current = int(d.loc[d.role == 'best_E4M3_test', 'epoch'].iloc[0])
        d.loc[(d.role == 'best_E4M3_test') & (d.weight_bits == 4), 'epoch'] = 99 if current == 100 else 100
        d.to_csv(p, index=False)
    elif case == 'corrupt_cost_geometry':
        p = target / 'group_size_costs.json'
        d = json.loads(p.read_text())
        next(x for x in d['summary'] if x['group_size'] == 64)['padded_slots'] += 1
        p.write_text(json.dumps(d))
    elif case == 'wrong_pilot_contract':
        p = target / 'run_report.json'
        d = json.loads(p.read_text())
        d['config']['schema'] = 'mx-triple-group64-scale64-pilot5-protocol-v1'
        d['config'].pop('reference_end_epoch')
        d['config']['pilot_stop_epoch'] = 5
        p.write_text(json.dumps(d))
    elif case == 'tampered_reference':
        p = target / 'group16_reference/epoch_metrics.csv'
        p.write_bytes(p.read_bytes() + b'\n')
    else:
        p = target / 'training_summary.json'
        d = json.loads(p.read_text())
        d['full_training_completed'] = True
        d['full_100_epoch_training_completed'] = True
        p.write_text(json.dumps(d))
    run = m.load_run(target)
    if case == 'mixed_checkpoint_epochs':
        assert not run.selection_safe and not run.reference_complete
    elif case == 'corrupt_cost_geometry':
        assert not run.cost_report_verified and not run.reference_complete
    elif case == 'wrong_pilot_contract':
        assert not run.metrics_safe and not run.reference_complete
    elif case == 'tampered_reference':
        assert not m.load_group16_reference(run).safe
    else:
        assert not run.reference_complete and any('honderd' in x['Melding'] and x['Niveau'] == 'FOUT' for x in run.issues)
    report['cases'][case] = {'rejected': True, 'synthetic_fixture_only': True}

run = m.load_run(args.complete)
run.report['status'] = 'FAIL'
assert not run.reference_complete and 'niet als PASS afgesloten' in m.status_markdown(run)
report['cases']['failed_completion_status'] = {'full100_not_claimed': True}
old = m.load_run(ACTUAL)
assert not old.metrics_safe and not old.reference_complete
report['cases']['actual_pilot_schema_not_full100'] = {'rejected': True}
report['final_source_sha256'] = {n: sha(PKG / n) for n in ['notebook_support.py', 'MX_triple_group64_scale64_reference100_resultaten.ipynb']}
(OUT / 'runner_notebook_validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'cases': {k: {key: val for key, val in v.items() if key != 'issues'} for k, v in report['cases'].items()}, 'final_source_sha256': report['final_source_sha256']}, ensure_ascii=False, indent=2))
