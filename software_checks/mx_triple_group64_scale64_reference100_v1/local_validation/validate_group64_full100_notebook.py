from pathlib import Path
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mx_group64_full100_notebook_mpl')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path('/workspace/scratch/4619498905be')
OUT = ROOT / 'work/group64_full100_notebook_validation_v1'
OUT.mkdir(exist_ok=True)
PKG = ROOT / 'output/mx_triple_group64_scale64_reference100_v1'
ACTUAL = ROOT / 'work/group64_pilot5_actual_20261027_161453'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest_tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def execute_cells(nb_path, run_path, name):
    nb = json.loads(nb_path.read_text())
    env = {'__name__': name}
    codes = 0
    before = digest_tree(run_path)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'code':
                continue
            source = ''.join(c['source'])
            if i == 1:
                source = 'from pathlib import Path\nRUN_DIR = Path(' + repr(str(run_path)) + ')\nSAVE_FIGURES = False\n'
            exec(compile(source, str(nb_path) + ':cell' + str(i), 'exec'), env)
            codes += 1
    assert before == digest_tree(run_path), 'Notebook changed its inputs'
    plt.close('all')
    return {'code_cells_executed': codes, 'figures_created': len(env.get('figures', {})), 'inputs_unchanged': True, 'run_last': env['run'].last_epoch, 'run_best': env['run'].observed_best_epoch, 'issues': env['run'].issues}, env


def main():
    report = {}
    actual, env = execute_cells(ACTUAL / 'MX_triple_group64_scale64_pilot5_resultaten.ipynb', ACTUAL, 'actual64pilot')
    assert env['run'].pilot_complete and env['run'].metrics_safe and env['run'].selection_safe
    assert not [x for x in env['run'].issues if x['Niveau'] in ('FOUT', 'RUNFOUT')]
    assert all(ref.safe and ref.compatible for ref in env['references'])
    report['actual_group64_pilot_notebook'] = actual

    m = load_helper(PKG / 'notebook_support.py', 'group64_full100_analysis_validation')
    blank = m.load_run(PKG)
    assert blank.test.empty and not blank.reference_complete
    refs = m.load_group_references(blank)
    assert len(refs) == 2 and all(ref.safe and not ref.compatible for ref in refs), [r.issues for r in refs]
    assert [r.best_epoch() for r in refs] == [42, 53]
    assert all(len(ref.metrics) == 600 and ref.metrics.epoch.max() == 100 for ref in refs)
    report['references'] = {str(ref.group_size): {'safe': ref.safe, 'global_best_epoch': ref.best_epoch(), 'rows': len(ref.metrics), 'epoch100_retained': ref.metric(100, 8) is not None} for ref in refs}
    report['empty_run'] = {'no_accuracy_invented': True, 'no_full100_claim': True, 'references_not_compared': True}

    # In-memory, explicitly synthetic display fixtures. They are never delivered
    # as SVHN results, and never rewrite the pinned historical references.
    base = refs[0].report['config'].copy()
    base.update(schema='mx-triple-group64-scale64-reference100-protocol-v1', mx_block_size=64, reference_end_epoch=100)
    metrics = refs[0].metrics.copy()
    special = {8: {54: 24900, 55: 24900}, 6: {77: 24700}, 4: {83: 24600}}
    for bit in (8, 6, 4):
        for e in range(1, 101):
            count = special[bit].get(e, 23800 + e + bit)
            mask = (metrics.phase == 'test') & (metrics.weight_bits == bit) & (metrics.epoch == e)
            metrics.loc[mask, 'correct1'] = count
            metrics.loc[mask, 'top1_percent'] = 100 * count / 26032
            metrics.loc[mask, 'count_top1_percent'] = 100 * count / 26032
    run = m.RunAnalysis(PKG, report={'schema': 'mx-triple-group64-scale64-reference100-report-v1', 'config': base, 'synthetic_fixture_only': True, 'status': 'PASS', 'initial_model_sha256': m.REFERENCE_INITIAL_SHA256}, summary={'full_training_completed': True, 'full_100_epoch_training_completed': True}, metrics=metrics)
    m._validate_contract(run)
    m._validate_metrics(run, [])
    assert run.metrics_safe, run.issues
    assert run.reference_complete and run.observed_best_epoch == 54
    synthetic_refs = []
    for ref, best_epoch in zip(refs, [42, 53]):
        data = ref.metrics.copy()
        for e in range(1, 101):
            for bit in (8, 6, 4):
                count = 23600 + e + bit
                if bit == 8 and e == best_epoch:
                    count = 24800
                mask = (data.phase == 'test') & (data.weight_bits == bit) & (data.epoch == e)
                data.loc[mask, 'correct1'] = count
                data.loc[mask, 'top1_percent'] = 100 * count / 26032
                data.loc[mask, 'count_top1_percent'] = 100 * count / 26032
        synthetic_refs.append(m.GroupReference(ref.path, ref.group_size, metrics=data, safe=True, compatible=True, synthetic=True))
    table = m.comparison_table(run, synthetic_refs)
    assert len(table) == 18
    best = table.loc[table.Controlepunt.str.startswith('Beste')]
    assert {int(b): set(rows.Epoch) for b, rows in best.groupby('Groepsgrootte')} == {16: {53}, 32: {42}, 64: {54}}
    assert set(table.loc[table.Controlepunt == 'Epoch 100', 'Epoch']) == {100}
    assert best.loc[(best.Groepsgrootte == 64) & (best.Gewichtsformaat == m.FORMATS[6]), 'Testaccuracy [%]'].iloc[0] == 100 * (23800 + 54 + 6) / 26032
    report['selection_fixture'] = {'synthetic_only': True, 'current_e4_first_tie_epoch': 54, 'reference32_epoch': 42, 'reference16_epoch': 53, 'lower_mode_own_maxima': {'6': 77, '4': 83}, 'lower_mode_maxima_not_mixed': True, 'all_modes_one_epoch_each_run': True, 'epoch100_all_same_endpoint': True}
    for name, fn in [('comparison_curves', m.plot_scale_comparison_curves), ('comparison_checkpoints', m.plot_scale_comparison_checkpoints)]:
        f = fn(run, synthetic_refs)
        f.savefig(OUT / (name + '.png'), dpi=130, bbox_inches='tight')
        plt.close(f)
    partial = copy.deepcopy(run)
    partial.metrics = partial.metrics.loc[partial.metrics.epoch <= 3].copy()
    partial.summary = {'full_training_completed': False, 'full_100_epoch_training_completed': False}
    partial.report['status'] = 'STOPPED_AT_EPOCH_BOUNDARY'
    table = m.comparison_table(partial, synthetic_refs)
    assert table.loc[table.Controlepunt == 'Epoch 100', 'Testaccuracy [%]'].isna().all()
    assert set(table.loc[table.Controlepunt.str.startswith('Beste'), 'Epoch']) <= {1, 2, 3}
    assert not partial.reference_complete
    assert 'tussentijdse' in m.comparison_markdown(partial, synthetic_refs)
    report['partial_fixture'] = {'synthetic_only': True, 'epoch100_unavailable': True, 'reference_selection_bounded_by_current_epoch': True, 'no_epoch100_claim': True}
    blocked = m.load_group_references(run)
    assert all(r.safe and not r.compatible for r in blocked)
    report['synthetic_real_comparison_blocked'] = True
    assert blank.costs['active_group_size'] == 64
    f = m.plot_group_costs(blank)
    f.savefig(OUT / 'costs.png', dpi=130, bbox_inches='tight')
    plt.close(f)
    report['costs'] = {'active_group_size': 64, 'counts': blank.costs['summary']}
    report['empty_notebook'], _ = execute_cells(PKG / 'MX_triple_group64_scale64_reference100_resultaten.ipynb', PKG, 'empty64fullnb')
    report['status'] = 'PASS'
    report['final_source_sha256'] = {n: sha(PKG / n) for n in ['notebook_support.py', 'MX_triple_group64_scale64_reference100_resultaten.ipynb']}
    report['unexecuted_deliverable'] = True
    (OUT / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'costs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
