"""Validate the 100-epoch notebook against synthetic runner fixtures only."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('package');parser.add_argument('--fixture');parser.add_argument('--out',required=True)
    args=parser.parse_args()
    package=Path(args.package).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('nb100_fixture_test',package/'notebook_support.py')
    h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
    checks=[]
    def check(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    nbname='MX_triple_independent_e2_scale_reference100_resultaten.ipynb'
    n=json.loads((package/nbname).read_text())
    check(all(c.get('execution_count') is None and c.get('outputs',[])==[] for c in n['cells']),'deliverable_notebook_unexecuted')
    for c in n['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),str(package),'exec')
    checks.append('all_notebook_code_cells_compile')
    pending=h.load_run(package)
    check(pending.test.empty and pending.last_epoch is None and not pending.reference_complete,'pending_not_fabricated_as_results')
    check(pending.reference_safe and len(pending.references)==3,'all_three_historical_references_verified')
    pilot=h.pilot_evidence_table(pending)
    check(len(pilot)==8,'actual_pilot_evidence_kept_separate_from_new_run')
    figures=['plot_learning','plot_final_window','plot_checkpoints','plot_scale_distributions','plot_scale_evolution','plot_scale_by_layer','plot_scale_changes','plot_scale_gradients','plot_storage_cost','plot_timing']
    for name in figures:
        f=getattr(h,name)(pending);f.canvas.draw();h.plt.close(f)
    checks.append('all_pending_figures_render')
    if args.fixture:
        fixture=Path(args.fixture).resolve();r=h.load_run(fixture)
        print('fixtureissues',r.issues,flush=True)
        check(r.metrics_safe and r.scales_safe and r.selection_safe and r.reference_safe,'full100_runner_evidence_consistent')
        check(r.synthetic and set(h._variants(r))=={'independent'},'synthetic_run_never_mixed_with_historical_accuracy')
        check(r.last_epoch==100 and r.best_epoch is not None and r.reference_complete,'true_hundred_epochs_and_common_best_selection')
        check(r.times.shape[0]==100 and r.timing['measured_100_epoch_seconds'] is not None,'measured_full100_time_distinct_from_projection')
        for f in (h.status_markdown,h.protocol_table,h.comparison_markdown,h.checkpoint_table,h.difference_table,
                  h.scale_distribution_table,h.scale_layer_table,h.source_change_table,h.storage_cost_table,
                  h.timing_markdown,h.audit_table,h.reference_table,h.export_table,h.diagnostics_table,
                  h.pilot_evidence_table,h.pilot_comparison_table):f(r)
        checks.append('all_notebook_tables_text_render')
        for name in figures:
            f=getattr(h,name)(r);f.canvas.draw()
            if name in ('plot_learning','plot_final_window','plot_checkpoints','plot_scale_evolution','plot_storage_cost'):
                f.savefig(out/(name+'.png'),dpi=100,bbox_inches='tight')
            if name in ('plot_learning','plot_final_window'):
                for ax in f.axes:
                    lo,hi=ax.get_ylim()
                    for line in ax.lines:
                        if line.get_transform()==ax.transData:
                            check(all(lo<=float(y)<=hi for y in line.get_ydata()),name+'_zoom_preserves_all_accuracy_points')
            h.plt.close(f)
        checks.append('all_full100_figures_render')
        with tempfile.TemporaryDirectory(dir=out) as td:
            td=Path(td)
            def copied(name):
                dst=td/name;dst.mkdir()
                for path in fixture.iterdir():
                    if path.suffix in ('.json','.csv'):shutil.copy2(path,dst/path.name)
                return dst
            bad=copied('stale_accuracy')
            f=h.pd.read_csv(bad/'epoch_metrics.csv');f.loc[f.index[-1],'top1_percent']+=.01;f.to_csv(bad/'epoch_metrics.csv',index=False)
            rr=h.load_run(bad)
            check(not rr.metrics_safe and rr.test.empty,'stale_accuracy_csv_rejected')
            bad=copied('wrong_lr_boundary')
            p=bad/'history.json';data=json.loads(p.read_text());data['epochs'][50]['scale_lr_used']=.001;p.write_text(json.dumps(data))
            rr=h.load_run(bad)
            check(not rr.metrics_safe,'incorrect_lr_after_epoch50_rejected')
            bad=copied('pilot_contract')
            p=bad/'run_report.json';data=json.loads(p.read_text());data['schema']='mx-triple-independent-e2-scale-pilot5-report-v1';p.write_text(json.dumps(data))
            rr=h.load_run(bad)
            check(not rr.metrics_safe and not rr.scales_safe,'pilot_contract_rejected_by_full100_reader')
            bad=copied('wrong_selection')
            p=bad/'checkpoint_selection.json';data=json.loads(p.read_text());data['best']['epoch']=101;p.write_text(json.dumps(data))
            rr=h.load_run(bad)
            check(not rr.selection_safe,'incorrect_shared_best_selection_rejected')
        from IPython import display as d
        old=d.display;seen=[];d.display=lambda *values,**kw:seen.extend(type(v).__name__ for v in values)
        path=fixture/nbname;before=hashlib.sha256(path.read_bytes()).hexdigest();nb=json.loads(path.read_text());env={}
        capture=io.StringIO()
        try:
            with contextlib.redirect_stdout(capture):
                for i,c in enumerate(nb['cells']):
                    if c['cell_type']=='code':exec(compile(''.join(c['source']),f'notebook-cell-{i}','exec'),env)
        finally:d.display=old
        check(before==hashlib.sha256(path.read_bytes()).hexdigest(),'complete_notebook_execution_does_not_save_outputs')
        check(env['SAVE_FIGURES'] is False and seen.count('Figure')==10,'all_notebook_cells_execute_without_writing_figures')
    report={'passed':True,'checks':checks,'synthetic_fixture_only':True,'actual_SVHN_training_executed':False,
            'saved_figures_are_synthetic_validation_outputs_not_research_results':True}
    (out/'notebook_validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
