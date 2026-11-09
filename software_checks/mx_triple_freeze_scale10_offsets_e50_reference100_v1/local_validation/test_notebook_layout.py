from pathlib import Path
import ast, json, sys, importlib.util, hashlib, copy, math, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKG=Path(__file__).resolve().parents[1]
ROOT=PKG.parent
OUT=ROOT/'freeze100_notebook_qa'
OUT.mkdir(parents=True,exist_ok=True)
helper=PKG/'notebook_support.py'
nbpath=PKG/'MX_triple_freeze_scale10_offsets_e50_reference100_resultaten.ipynb'
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (helper,nbpath)}
spec=importlib.util.spec_from_file_location('_freeze100_notebook_qa',helper)
ns=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ns;spec.loader.exec_module(ns)
checks=[]
def check(ok,label):
    checks.append({'test':label,'passed':bool(ok)})
    if not ok:raise AssertionError(label)
def state(offsets):
    result={}; cursor=0
    for layer,size in ns.LAYER_BLOCKS.items():
        channels={'layers.3':16,'layers.6':16,'layers.10':32,'layers.13':32,'layers.16':32,'layers.20':128}[layer]
        result[layer]=np.asarray(offsets[cursor:cursor+size],dtype=float).reshape(channels,-1).tolist();cursor+=size
    assert cursor==4080
    return result
fixed=state([4]*5+[5]*1191+[6]*2745+[7]*139)
early=state([2]*2040+[3]*2040)
def score(epoch,bit,frozen=False):
    if epoch<=50:
        base=94.9-60*math.exp(-(epoch-1)/6)
        if epoch==35:base=95.7
        return round(base-{8:0,6:.14,4:.26}[bit],5)
    base=94.88+.04*math.sin(epoch/4)+.001*epoch
    return round(base-{8:0,6:.19,4:.24}[bit]+({8:0,6:.35,4:.34}[bit] if frozen else 0),5)
def records(epochs,frozen=False):
    rec=[]
    for epoch in epochs:
        record={'epoch':epoch,'scale_state':fixed if epoch>=50 else early,'train':[],'test':[]}
        for phase in ('train','test'):
            for bit in ns.BITS:
                acc=score(epoch,bit,frozen)
                n=10000000
                record[phase].append({'epoch':epoch,'phase':phase,'weight_bits':bit,'top1_percent':acc,
                    'correct1':round(acc*n/100),'n':n,'objective_loss':.08+3/epoch+bit*.005})
        rec.append(record)
    return rec
def scale_frame(records):
    rows=[]
    for rec in records:
        counts=ns._state_counts(rec['scale_state'])
        rows.append({'epoch':rec['epoch'],'blocks':4080,**{f'count_x{f}':counts[f] for f in ns.SCALE_FACTORS},
                     'theta_changed_blocks':0,'choice_flips_during_epoch':0,'unique_blocks_switched_during_epoch':0})
    return pd.DataFrame(rows)
run=ns.Analysis(OUT/'SYNTHETIC_PLOT_FIXTURE_NOT_RESEARCH')
run.report={'status':'PASS','synthetic_fixture_only':True,'continuation_completed':True,'completed_epoch':100}
run.history={'epochs':records(range(51,101),True)}
run.historical_records={'epochs':records(range(1,101))}
run.baseline=run.historical_records['epochs'][49]
run.metrics=ns._metrics_from_history(run.history['epochs'])
run.reference=ns._metrics_from_history(run.historical_records['epochs'])
run.scales=scale_frame(run.history['epochs'])
run.reference_scales=scale_frame(run.historical_records['epochs'])
run.times=pd.DataFrame([{'epoch':e,'total_epoch_seconds':100+math.sin(e)} for e in range(51,101)])
run.reference_times=pd.DataFrame([{'epoch':e,'total_epoch_seconds':101+math.sin(e)} for e in range(1,101)])
(OUT/'SYNTHETIC_plot_fixture_description.json').write_text(json.dumps({
    'synthetic_fixture_only':True,'purpose':'Plot-only software fixture; fabricated values are not research results and are not provenance-validator inputs.',
    'epochs':'historical1–100; continuation51–100','global_best_epoch':35,
    'early_counts':ns._state_counts(early),'frozen_counts':ns._state_counts(fixed)},indent=2))
check(run.best_epoch(global_selection=True)==35,'Synthetic global selector is deliberately before freeze')
check(not ns.factor_distribution_table(run,'global').Groepen.equals(ns.factor_distribution_table(run,'last').Groepen),
      'Global factor distribution uses its own pre-freeze state')
check(ns.factor_distribution_table(run,'last').Groepen.sum()==4080,'All ten factors sum to 4080')
check(ns.factor_distribution_table(run,'last').shape[0]==10,'All ten factor rows included')
check((ns.factor_distribution_table(run,'last').Groepen==0).sum()==6,'Six unused factors retained as zero')

figs={};warnings_seen=[]
with warnings.catch_warnings(record=True) as warning_list:
    warnings.simplefilter('always')
    for role in ('endpoint','postfreeze','global'):
        figs['checkpoint_'+role]=ns.plot_checkpoint_accuracy(run,role)
    limits=[fig.axes[0].get_xlim() for fig in figs.values()]
    check(all(np.allclose(limits[0],lim) for lim in limits),'Checkpoint roles share exactly the same x-limits')
    check(limits[0][0]>90 and limits[0][1]<100,'Checkpoint plots are zoomed to relevant high accuracy range')
    for role in ('endpoint','postfreeze','global'):
        table=ns.checkpoint_table(run,role)
        low,high=figs['checkpoint_'+role].axes[0].get_xlim()
        check(table['Testaccuracy [%]'].between(low,high).all(),role+' checkpoint contains every accuracy')
        check('ingezoomde as' in figs['checkpoint_'+role].axes[0].get_xlabel(),role+' accuracy zoom labelled')
        check(len(figs['checkpoint_'+role].axes[0].collections)==2,role+' uses two paired point series')
    figs['full_history']=ns.plot_accuracy_curves(run,full_history=True)
    figs['after_freeze']=ns.plot_accuracy_curves(run)
    for name in ('full_history','after_freeze'):
        figure=figs[name];ys=[ax.get_ylim() for ax in figure.axes]
        check(all(np.allclose(ys[0],v) for v in ys),name+' precision panels share y-limits')
        start=1 if name=='full_history' else 50
        vals=pd.concat([run.global_test,run.historical_test]).loc[lambda x:x.epoch.between(start,100),'top1_percent']
        check(vals.between(*ys[0]).all(),name+' retains all accuracy values')
        check(all(np.allclose(ax.get_xlim(),[start,100]) for ax in figure.axes),name+' full epoch range')
    check(figs['full_history'].axes[0].get_ylim()[0]<35,'Full history includes intentionally low initial accuracy')
    check(figs['after_freeze'].axes[0].get_ylim()[0]>90,'Post-freeze figure zooms without early values')
    figs['factor_distribution']=ns.plot_factor_distribution(run)
    factorfig=figs['factor_distribution']
    for ax,role in zip(factorfig.axes,('last','postfreeze','global')):
        table=ns.factor_distribution_table(run,role)
        check(len(ax.patches)==10 and len(ax.get_yticklabels())==10,role+' factor chart includes ten categories')
        check(ax.get_xlim()[0]==0,role+' factor count axis starts at zero')
        check(np.allclose([p.get_width() for p in ax.patches],table.Groepen),role+' factor bars match own checkpoint counts')
        annotations=[t.get_text() for t in ax.texts]
        check(len(annotations)==10 and all('|' in t and '%' in t for t in annotations),role+' every factor shows count and percentage')
    check(all(np.allclose(factorfig.axes[0].get_xlim(),ax.get_xlim()) for ax in factorfig.axes),'Factor panels share count scale')
    check('vóór freeze' in factorfig.axes[2].get_title(),'Global factor chart labels pre-freeze selection')
    for name,fig in figs.items():
        fig.savefig(OUT/(name+'.png'),dpi=140)
    warnings_seen.extend(str(w.message) for w in warning_list)

nb=json.loads(nbpath.read_text())
check(len(nb['cells'])==25,'Notebook has 25 cells')
check(all(c.get('outputs')==[] and c.get('execution_count') is None for c in nb['cells'] if c['cell_type']=='code'),'Notebook is unexecuted')
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':
        src=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        compile(src,f'cell-{i}','exec')
check(True,'Every notebook code cell compiles')
context={'ns':ns,'run':run,'figures':{},'display':lambda x:None,'plt':plt,'pd':pd,'SAVE_FIGURES':False}
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and i not in (1,2):
        src=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        exec(compile(src,f'cell-{i}','exec'),context)
check(len(context['figures'])==11,'All 11 analysis figures generated by notebook cells')
partial=copy.copy(run)
partial.history={'epochs':run.history['epochs'][:5]}
partial.metrics=run.metrics.loc[run.metrics.epoch.le(55)].copy()
partial.scales=run.scales.loc[run.scales.epoch.le(55)].copy()
partial.report={**run.report,'continuation_completed':False,'completed_epoch':55,'status':'STOPPED_AT_EPOCH_BOUNDARY'}
check(ns.checkpoint_table(partial,'endpoint').empty,'Partial run has no invented epoch100 endpoint')
ns.plot_checkpoint_accuracy(partial,'endpoint').savefig(OUT/'partial_endpoint_not_available.png',dpi=100)
earlypath=OUT/'empty_check'
earlypath.mkdir(exist_ok=True)
(earlypath/'run_report.json').write_text(json.dumps({'schema':'mx-triple-freeze-scale10-offsets-e50-reference100-report-v1','command':'check','status':'PASS'}))
empty=ns.load_run(earlypath)
context={'ns':ns,'run':empty,'figures':{},'display':lambda x:None,'plt':plt,'pd':pd,'SAVE_FIGURES':False}
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and i not in (1,2):
        src=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        exec(compile(src,f'empty-cell-{i}','exec'),context)
check(True,'All notebook analysis cells handle empty technical check')
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (helper,nbpath)}
check(before==after,'Shared package helper and notebook bytes unchanged')
plt.close('all')
result={'scope':'Read-only graph/notebook QA, synthetic plot-only data; no training or new research results.',
        'checks':checks,'check_count':len(checks),'passed':all(c['passed'] for c in checks),
        'checkpoint_xlim':limits[0],'warnings':warnings_seen,'source_sha256':before,
        'not_tested':'Full artifact provenance and numerical consistency validation belongs to root full fixtures.',
        'visual_review':'PNG files generated; assistant visual review pending.'}
(OUT/'qa_report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'warnings':warnings_seen,'files':[p.name for p in OUT.glob('*.png')]}))
