"""Read-only notebook integration checks. Mutations use disposable fixture copies."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import shutil
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('package')
    parser.add_argument('--fixture')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    package=Path(args.package).resolve(); out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('notebook_fixture_test',package/'notebook_support.py')
    h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
    checks=[]
    def check(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    n=json.loads((package/'MX_triple_independent_e2_scale_pilot5_resultaten.ipynb').read_text())
    check(all(c.get('execution_count') is None and c.get('outputs',[])==[] for c in n['cells']),'deliverable_notebook_unexecuted')
    for c in n['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),str(package),'exec')
    checks.append('all_notebook_code_cells_compile')
    pending=h.load_run(package)
    check(pending.test.empty and pending.last_epoch is None and not pending.pilot_complete,'pending_not_turned_into_zeros_or_results')
    check(pending.reference_safe and len(pending.references)==2,'pinned_historical_references_verified')
    figures=['plot_learning','plot_checkpoints','plot_scale_distributions','plot_scale_evolution','plot_scale_by_layer','plot_scale_changes','plot_scale_gradients','plot_storage_cost','plot_timing']
    for name in figures:
        f=getattr(h,name)(pending);f.canvas.draw();h.plt.close(f)
    checks.append('all_pending_figures_render')
    if args.fixture:
        fixture=Path(args.fixture).resolve()
        r=h.load_run(fixture)
        print('fixtureissues',r.issues)
        check(r.metrics_safe and r.scales_safe and r.selection_safe and r.reference_safe,'synthetic_run_evidence_consistent')
        check(r.synthetic and set(h._variants(r))=={'independent'},'synthetic_results_never_mixed_with_research_baselines')
        check(r.last_epoch==5 and r.best_epoch is not None,'pilot5_last_and_best_checkpoint_available')
        # Tables must function for a synthetic run without research comparisons.
        for f in (h.status_markdown,h.protocol_table,h.comparison_markdown,h.checkpoint_table,h.difference_table,
                  h.scale_distribution_table,h.scale_layer_table,h.source_change_table,h.storage_cost_table,
                  h.timing_markdown,h.audit_table,h.reference_table,h.export_table,h.diagnostics_table):
            f(r)
        checks.append('all_notebook_tables_and_text_render')
        for name in figures:
            f=getattr(h,name)(r);f.canvas.draw();f.savefig(out/(name+'.png'),dpi=100,bbox_inches='tight');h.plt.close(f)
        checks.append('all_synthetic_figures_render')
        with tempfile.TemporaryDirectory(dir=out) as td:
            td=Path(td)
            def copied(name):
                dst=td/name;dst.mkdir()
                for path in fixture.iterdir():
                    if path.suffix in ('.json','.csv'):
                        shutil.copy2(path,dst/path.name)
                return dst
            bad=copied('stale_accuracy')
            f=h.pd.read_csv(bad/'epoch_metrics.csv');f.loc[f.index[0],'top1_percent']+=.01;f.to_csv(bad/'epoch_metrics.csv',index=False)
            rr=h.load_run(bad)
            check(not rr.metrics_safe and rr.test.empty,'stale_accuracy_csv_rejected')
            bad=copied('stale_scale')
            f=h.pd.read_csv(bad/'scale_block_state.csv');f.loc[f.index[0],'phi']+=.01;f.to_csv(bad/'scale_block_state.csv',index=False)
            rr=h.load_run(bad)
            check(not rr.scales_safe,'stale_absolute_scale_csv_rejected')
            bad=copied('wrong_contract')
            p=bad/'run_report.json';data=json.loads(p.read_text());data['schema']='mx-triple-learned-scale10-pilot5-report-v1';p.write_text(json.dumps(data))
            rr=h.load_run(bad)
            check(not rr.metrics_safe and not rr.scales_safe,'old_relative_contract_rejected')
            bad=copied('wrong_selection')
            p=bad/'checkpoint_selection.json';data=json.loads(p.read_text());data['best']['epoch']=99;p.write_text(json.dumps(data))
            rr=h.load_run(bad)
            check(not rr.selection_safe,'incorrect_best_checkpoint_rejected')
            bad=copied('changed_reference')
            d=bad/'relative_scale10_reference';shutil.copytree(package/'relative_scale10_reference',d)
            p=d/'REFERENCE.json';data=json.loads(p.read_text());data['completed_epochs']=4;p.write_text(json.dumps(data))
            rr=h.load_run(bad)
            check(not rr.reference_safe,'changed_historical_marker_rejected')
        # Check genuine line plots keep every visible accuracy in their zoomed axes.
        fig=h.plot_learning(r)
        for ax in fig.axes:
            lo,hi=ax.get_ylim()
            for line in ax.lines:
                check(all(lo<=float(y)<=hi for y in line.get_ydata()),'accuracy_zoom_contains_all_values')
        h.plt.close(fig)
    report={'passed':True,'checks':checks,'synthetic_fixture_only':True,
            'actual_SVHN_training_executed':False,'figures_are_synthetic_test_outputs_not_research_results':True}
    (out/'notebook_validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
