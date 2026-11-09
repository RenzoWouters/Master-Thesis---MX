"""Read-only notebook validation on actual synthetic runner exports.

This does not train a model or produce research results. Pass paths from the
CPU training integration test. All deliberate mutations remain in memory or
in a new temporary directory; original metrics/references are not changed.
"""
from pathlib import Path
import argparse,contextlib,copy,hashlib,importlib.util,io,json,os,sys,tempfile,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_helper(package):
 name='_scale10_reference100_notebook_test_helper'
 spec=importlib.util.spec_from_file_location(name,package/'notebook_support.py')
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def errors(run):return [v for v in run.issues if v['Niveau'] in ('FOUT','RUNFOUT')]

def execute_cells(package,run_dir):
 notebook=package/'MX_triple_learned_scale10_reference100_resultaten.ipynb'
 nb=json.loads(notebook.read_text())
 assert all(c.get('execution_count') is None and c.get('outputs')==[] for c in nb['cells'] if c['cell_type']=='code')
 namespace={'__name__':'__notebook_validation__'};done=[]
 with contextlib.redirect_stdout(io.StringIO()):
  for i,cell in enumerate(nb['cells']):
   if cell['cell_type']!='code':continue
   code=''.join(cell['source'])
   # Use the final package helper, even if a fixture copied an earlier wording.
   if 'helper_path = next(' in code:
    code=code.replace("helper_path = next((p for p in candidates if p.is_file()), None)",f"helper_path = Path({str(package/'notebook_support.py')!r})")
   exec(compile(code,f'cell_{i}','exec'),namespace)
   if 'RUN_DIR = None' in code:namespace['RUN_DIR']=run_dir
   done.append(i)
 return namespace,done


def main():
 a=argparse.ArgumentParser();a.add_argument('--package',type=Path,required=True);a.add_argument('--full-run',type=Path,required=True)
 a.add_argument('--partial-run',type=Path,required=True);a.add_argument('--failure-run',type=Path,required=True)
 a.add_argument('--resume-run',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 p=args.package.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();m=load_helper(p)
 report={'schema':'mx-scale10-reference100-notebook-validation-v1','synthetic_fixture_only':True,
  'no_SVHN_training':True,'no_dataset_or_checkpoint_loaded':True,'helper_sha256':digest(p/'notebook_support.py'),
  'notebook_sha256':digest(p/'MX_triple_learned_scale10_reference100_resultaten.ipynb'),'checks':{},'scenarios':{}}
 sources=[args.full_run,args.partial_run,args.failure_run]
 if args.resume_run:sources.append(args.resume_run)
 before={str(q):digest(q) for folder in sources for q in folder.iterdir() if q.is_file() and q.suffix in ('.json','.csv')}
 before.update({str(q):digest(q) for folder in [p/'fixed_references',p/'pilot_reference'] for q in folder.rglob('*') if q.is_file()})
 scenarios={'full100':args.full_run,'partial':args.partial_run,'early_fail':args.failure_run,'empty_template':p}
 if args.resume_run:scenarios['resumed100']=args.resume_run
 loaded={}
 for name,folder in scenarios.items():
  ns,cells=execute_cells(p,folder);run=ns['run'];loaded[name]=run
  report['scenarios'][name]={'code_cells_executed':len(cells),'figures':len(ns['figures']),
   'last_epoch':run.last_epoch,'reference_complete':run.reference_complete,
   'error_messages':errors(run),'synthetic':run.synthetic}
  assert len(cells)==17 and len(ns['figures'])==14
  if name in ('full100','resumed100'):
   assert run.synthetic and run.reference_complete and run.last_epoch==100 and not errors(run)
  elif name=='partial':assert run.synthetic and not run.reference_complete and run.last_epoch==1 and not errors(run)
  elif name=='early_fail':assert not run.reference_complete and run.test.empty
  else:assert run.test.empty and not run.reference_complete
  plt.close('all')
 report['checks']['all_delivered_cells_execute_in_each_scenario']=True
 run=loaded['full100'];partial=loaded['partial']
 refs=m.load_fixed_references(run,p/'fixed_references');pilot=m.load_pilot_reference(run,p/'pilot_reference')
 assert all(r.safe and not r.compatible and len(r.metrics)==600 for r in refs.values())
 assert pilot.safe and not pilot.compatible and len(pilot.metrics)==30
 report['checks']['three_real100_references_and_separate_real5_pilot_pinned']=True
 report['checks']['synthetic_results_do_not_enter_research_comparisons']=True
 # Only this explicit visualization QA enables mixed lines, retaining the large
 # SYNTHETISCHE CONTROLE title. It never edits metadata or produces research tables.
 for ref in refs.values():ref.compatible=True
 figs={'comparison_curves':m.plot_scale_comparison_curves(run,refs),
       'comparison_checkpoints':m.plot_scale_comparison_checkpoints(run,refs),
       'scale_choices':m.plot_scale_choices(run)}
 assert all(len(ax.lines)==4 for ax in figs['comparison_curves'].axes)
 assert all(len(ax.patches)==12 for ax in figs['comparison_checkpoints'].axes)
 assert all('SYNTHETISCHE' in f._suptitle.get_text() for f in figs.values())
 for name,figure in figs.items():
  figure.savefig(out/(name+'_SYNTHETIC.png'),dpi=110,bbox_inches='tight');plt.close(figure)
 table=m.comparison_table(run,refs)
 assert len(table)==18 and (table.loc[table.Controlepunt.eq('Epoch 100'),'Epoch tienkeuze']==100).all()
 best=table.loc[~table.Controlepunt.eq('Epoch 100')]
 assert best['Epoch tienkeuze'].nunique()==1 and best['Epoch tienkeuze'].iloc[0]==run.observed_best_epoch
 report['checks']['full100_curves_four_series_and_common_best_checkpoint']=True
 partial_refs=m.load_fixed_references(partial,p/'fixed_references')
 for ref in partial_refs.values():ref.compatible=True
 fig=m.plot_scale_comparison_curves(partial,partial_refs)
 assert all(max(line.get_xdata())<=1 for ax in fig.axes for line in ax.lines)
 plt.close(fig)
 t=m.comparison_table(partial,partial_refs)
 assert t.loc[t.Controlepunt.eq('Epoch 100'),'Tienkeuze accuracy [%]'].isna().all()
 assert t.loc[t.Controlepunt.ne('Epoch 100'),'Epoch referentie'].eq(1).all()
 report['checks']['partial_prefix_filters_reference_curves_and_best_selection']=True
 bad=copy.copy(run);bad.issues=[];bad.metrics=run.metrics.copy();bad.metrics.loc[bad.metrics.epoch.eq(51),'lr']=.001
 bad.metrics_safe=True;m._validate_metrics(bad,[]);assert not bad.metrics_safe
 report['checks']['wrong_LR_at_epoch51_rejected']=True
 bad=copy.copy(run);bad.issues=[];bad.selection=copy.deepcopy(run.selection);bad.selection['best']['epoch']=100
 bad.selection_safe=True;m._validate_selection(bad);assert not bad.selection_safe
 report['checks']['best_checkpoint_selection_mismatch_rejected']=True
 bad=copy.copy(run);bad.issues=[];bad.scale_blocks=run.scale_blocks.copy();bad.scale_blocks.iloc[0,bad.scale_blocks.columns.get_loc('theta')]=9.1
 bad.scales_safe=True;m._validate_scales(bad,run.history['epochs']);assert not bad.scales_safe
 report['checks']['theta_outside_range_rejected']=True
 bad=copy.copy(partial);bad.report=copy.deepcopy(partial.report);bad.summary=copy.deepcopy(partial.summary)
 bad.report['status']='PASS';bad.report['full_training_completed']=True
 bad.summary['full_training_completed']=bad.summary['full_100_epoch_training_completed']=True
 assert not bad.reference_complete
 report['checks']['completion_flags_cannot_invent_missing_epochs']=True
 with tempfile.TemporaryDirectory(prefix='scale10_reference_corruption_') as tmp:
  import shutil
  d=Path(tmp)/'refs';shutil.copytree(p/'fixed_references',d)
  target=d/'learned_binary_reference100/epoch_metrics.csv'
  target.write_bytes(target.read_bytes()+b'\n')
  corrupted=m.load_fixed_references(run,d)['binary'];assert not corrupted.safe
 report['checks']['changed_reference_bytes_blocked']=True
 assert all(Path(q).is_file() and digest(Path(q))==h for q,h in before.items())
 report['checks']['raw_run_and_reference_inputs_unchanged']=True
 assert 'torch' not in sys.modules
 report['checks']['no_torch_model_checkpoint_import']=True
 report['checks']['blank_delivered_notebook']=True
 report['elapsed_seconds']=time.monotonic()-started;report['status']='PASS'
 (out/'notebook_validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':'PASS','report':str(out/'notebook_validation_report.json'),'scenarios':list(scenarios),'checks':len(report['checks'])}))

if __name__=='__main__':main()
