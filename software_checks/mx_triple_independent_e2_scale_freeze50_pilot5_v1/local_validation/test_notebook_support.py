"""Read-only validation of the freeze50 notebook; all fixture figures are synthetic."""
from pathlib import Path
import argparse,contextlib,copy,hashlib,importlib.util,io,json,shutil,sys,tempfile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('package',type=Path)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--partial-fixture',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();package=args.package.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('freeze50_notebook_validation',package/'notebook_support.py')
    h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
    checks=[]
    def check(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    notebook_name='MX_triple_independent_e2_scale_freeze50_pilot5_resultaten.ipynb'
    notebook=json.loads((package/notebook_name).read_text())
    check(all(c.get('execution_count') is None and not c.get('outputs') for c in notebook['cells']),'deliverable_notebook_unexecuted')
    for c in notebook['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),notebook_name,'exec')
    checks.append('all_notebook_code_cells_compile')
    pending=h.load_run(package)
    check(pending.last_epoch is None and pending.test.empty and not pending.reference_complete,'no_results_fabricated_before_run')
    check(pending.reference_safe and set(pending.references)=={'independent_full100_reference','fixed_scale64_reference','relative_scale10_full100_reference'},'all_three_pinned_historical_references_verified')
    figure_calls=[('learning',lambda r:h.plot_learning(r)),('branch',lambda r:h.plot_final_window(r)),
        ('last',lambda r:h.plot_checkpoints(r,'last')),('best',lambda r:h.plot_checkpoints(r,'best')),
        ('branch_best',lambda r:h.plot_checkpoints(r,'branch_best')),('freeze',lambda r:h.plot_freeze_comparison(r)),
        ('distributions',lambda r:h.plot_scale_distributions(r)),('evolution',lambda r:h.plot_scale_evolution(r)),
        ('layers',lambda r:h.plot_scale_by_layer(r)),('changes',lambda r:h.plot_scale_changes(r)),
        ('gradients',lambda r:h.plot_scale_gradients(r)),('storage',lambda r:h.plot_storage_cost(r)),('timing',lambda r:h.plot_timing(r))]
    for name,fn in figure_calls:
        f=fn(pending);f.canvas.draw();h.plt.close(f)
    checks.append('all13_pending_figures_render')

    # Explicit hypothetical scores verify categorical axes/annotations, not training outcomes.
    variants={}
    for j,name in enumerate(['independent_full100_reference','fixed_scale64_reference','relative_scale10_full100_reference','independent']):
        rows=[]
        for epoch in range(1,56):
            for i,bit in enumerate(h.BITS):
                correct=round((91.5+3.1*(1-h.np.exp(-epoch/9))+.12*i-.11*j+.04*h.np.sin(epoch*.7+j))*100)
                rows.append({'epoch':epoch,'phase':'test','weight_bits':bit,'top1_percent':correct/100,
                    'correct1':correct,'n':10000})
        variants[name]=h.pd.DataFrame(rows)
    hypothetical=h.RunAnalysis(out,report={'synthetic_fixture_only':True},metrics=variants['independent'])
    original=h._variants
    try:
        h._variants=lambda run,context=True:variants if context else {k:v for k,v in variants.items() if k in ('independent_full100_reference','independent')}
        for role in ('last','best','branch_best'):
            f=h.plot_checkpoints(hypothetical,role);f.canvas.draw()
            table=h.checkpoint_table(hypothetical,role)
            check(len(f.axes)==3,'checkpoint_'+role+'_one_panel_per_precision')
            for bit,ax in zip(h.BITS,f.axes):
                check(len(ax.collections)==4,'checkpoint_'+role+'_one_circle_per_variant_'+str(bit))
                expected=[f'{v:.3f}%'.replace('.',',') for v in table[h.LABELS[bit]+' testaccuracy [%]']]
                check([t.get_text() for t in ax.texts]==expected,'checkpoint_'+role+'_every_score_labelled_'+str(bit))
                check(all(any(label in t.get_text() for t in ax.get_yticklabels()) for label in table.Variant),
                    'checkpoint_'+role+'_variants_on_category_axis_'+str(bit))
                lo,hi=ax.get_xlim();check(all(lo<=v<=hi for v in table[h.LABELS[bit]+' testaccuracy [%]']),
                    'checkpoint_'+role+'_zoom_preserves_all_scores_'+str(bit))
                shapes=[c.get_paths()[0].vertices for c in ax.collections]
                check(all(h.np.array_equal(shapes[0],x) for x in shapes),'checkpoint_'+role+'_identical_marker_shape_'+str(bit))
            f.savefig(out/('synthetic_categorical_'+role+'.png'),dpi=100,bbox_inches='tight');h.plt.close(f)
    finally:h._variants=original

    if args.fixture:
        fixture=args.fixture.resolve();raw_before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.ipynb')}
        r=h.load_run(fixture)
        check(r.synthetic,'fixture_explicitly_synthetic')
        if not (r.metrics_safe and r.scales_safe and r.selection_safe):print(r.issues,flush=True)
        check(r.metrics_safe and r.scales_safe and r.selection_safe,'runner_fixture_evidence_consistent')
        check(r.last_epoch==55 and r.branch_epochs==5 and r.reference_complete,'five_new_epochs_distinguished_from_inherited50')
        check(list(h._variants(r))==['independent'],'synthetic_accuracy_never_mixed_with_research_baselines')
        check(h.checkpoint_table(r,'last').Epoch.tolist()==[55],'endpoint55_not100')
        check(h.checkpoint_table(r,'branch_best').Epoch.between(51,55).all(),'branch_best_selection_excludes_inherited_prefix')
        proof=h.freeze_evidence_table(r).set_index('Freezecontrole').Waarde
        check(proof['Groepen met gewijzigde continue φ sinds50']==0 and proof['Groepen met andere absolute E2-exponent sinds50']==0,'frozen_absolute_scale_demonstrated')
        for name,fn in figure_calls:
            f=fn(r);f.canvas.draw()
            if name in ('branch','last','freeze','distributions','gradients'):
                f.savefig(out/('runner_'+name+'.png'),dpi=100,bbox_inches='tight')
            h.plt.close(f)
        checks.append('all13_runner_figures_render')
        check('niet opnieuw uitgevoerd' in h.timing_markdown(r),'inherited_timing_not_counted_as_new_work')

        with tempfile.TemporaryDirectory(dir=out) as td:
            td=Path(td)
            for path in fixture.iterdir():
                if path.suffix in ('.json','.csv'):shutil.copy2(path,td/path.name)
            original_files={n:(td/n).read_bytes() for n in ('epoch_metrics.csv','history.json','run_report.json','checkpoint_selection.json')}
            p=td/'epoch_metrics.csv';f=h.pd.read_csv(p);f.loc[f.index[-1],'top1_percent']+=.1;f.to_csv(p,index=False)
            bad=h.load_run(td);check(not bad.metrics_safe and bad.test.empty,'stale_accuracy_csv_rejected');p.write_bytes(original_files[p.name])
            p=td/'history.json';data=json.loads(original_files[p.name]);data['epochs'][50]['scale_lr_used']=.0001;p.write_text(json.dumps(data))
            bad=h.load_run(td);check(not bad.metrics_safe,'nonzero_scale_lr_after_freeze_rejected');p.write_bytes(original_files[p.name])
            data=json.loads(original_files[p.name]);data['epochs'][-1]['scale_state']['layers.3'][0][0]+=.125;p.write_text(json.dumps(data))
            bad=h.load_run(td);check(not bad.scales_safe,'continuous_phi_drift_even_without_hard_switch_rejected');p.write_bytes(original_files[p.name])
            data=json.loads(original_files[p.name]);data['epochs'][-1]['freeze_checks']['scale_optimizer_state_unchanged']=False;p.write_text(json.dumps(data))
            bad=h.load_run(td);check(not bad.metrics_safe and not bad.scales_safe,'unconfirmed_frozen_adam_state_rejected');p.write_bytes(original_files[p.name])
            p=td/'checkpoint_selection.json';data=json.loads(original_files[p.name]);data['branch_best']['epoch']=50;p.write_text(json.dumps(data))
            bad=h.load_run(td);check(not bad.selection_safe,'inherited_epoch_cannot_be_branch_best');p.write_bytes(original_files[p.name])
            p=td/'run_report.json';data=json.loads(original_files[p.name]);data['schema']='mx-triple-independent-e2-scale-reference100-report-v1';p.write_text(json.dumps(data))
            bad=h.load_run(td);check(not bad.metrics_safe,'parent100_contract_not_mislabeled_as_freeze_pilot');p.write_bytes(original_files[p.name])

        from IPython import display as d
        old_display=d.display;seen=[];d.display=lambda *values,**kwargs:seen.extend(type(v).__name__ for v in values)
        nb=json.loads((package/notebook_name).read_text());env={}
        for cell in nb['cells']:
            if 'parameters' in cell['metadata'].get('tags',[]):
                cell['source']=['from pathlib import Path\n',f'RUN_DIR = Path({str(fixture)!r})\n','SAVE_FIGURES = False\n']
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for i,cell in enumerate(nb['cells']):
                    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),f'notebook-cell-{i}','exec'),env)
        finally:d.display=old_display
        check(seen.count('Figure')==13 and env['SAVE_FIGURES'] is False,'all31_cells_execute_and13_figures_display_without_saving')
        raw_after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.ipynb')}
        check(raw_before==raw_after,'runner_fixture_files_left_unchanged')

    if args.partial_fixture:
        partial=h.load_run(args.partial_fixture.resolve())
        check(partial.synthetic and partial.metrics_safe and partial.scales_safe and partial.selection_safe,
            'partial_runner_evidence_consistent')
        check(50 < partial.last_epoch < 55,'partial_epoch_is_between50and55')
        check(not partial.reference_complete and partial.branch_epochs==partial.last_epoch-50,
            'partial_run_does_not_claim_completed_pilot')
        check(h.checkpoint_table(partial,'last').Epoch.tolist()==[partial.last_epoch],
            'partial_plot_uses_actual_endpoint')

    report={'status':'PASS','checks':checks,'synthetic_fixture_only':True,
        'no_SVHN_training_executed':True,'all_saved_figures_are_synthetic_software_checks_not_research_results':True,
        'notebook_cells':len(notebook['cells']),'notebook_delivered_unexecuted':True}
    (out/'notebook_validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
