"""Export-to-inference check of the completed truncation-aware model.

check: synthetic frozen-model tests, no updates.
evaluate: two previously selected roles, two modes, full SVHN test; high source only.
No training, data download, BN calibration or new checkpoint selection.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from zipfile import ZipFile, ZIP_DEFLATED
sys.dont_write_bytecode = True
import svhn_support as s
ROOT = Path(__file__).resolve().parent
ROLES = ('best_E2M3_test', 'last_epoch')
CODE_REUSE = ('mx_format.py','mx_e2m1.py','mx_adapter.py','dual_adapter.py','trunc_adapter.py',
              'svhn_support.py','smoke_support.py','resume_core.py')
ENV_KEYS = ('python','torch','torchvision','numpy','scipy','pillow','cuda_build','cudnn','device_name')
BACKEND_KEYS = ('cudnn_benchmark','cudnn_deterministic','deterministic_algorithms',
                'matmul_fp32_precision','cudnn_conv_fp32_precision','AMP','torch_compile')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def dependencies(out, log):
    for command in ('freeze', 'check'):
        p = subprocess.run([sys.executable, '-m', 'pip', command], capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
        (out/f'pip_{command}.txt').write_text(p.stdout+p.stderr,encoding='utf-8')
        s.require(p.returncode == 0, 'pip ' + command + ' mislukt; geen pakketten vervangen.')
        if command == 'check': log('pip check: ' + p.stdout.strip())


def inspect_reference(ref, rt):
    """Read only JSON/CSV metadata and TWO HIGH export files. No checkpoint IO."""
    import resume_core as core
    required = ('run_report.json','epoch_metrics.csv','selected_metrics.csv','checkpoint_selection.json')
    s.require(ref.is_dir() and all((ref/n).is_file() for n in required),
              'Verwacht de originele truncatiebewuste honderd-epoch _joint_run-map.')
    r = json.loads((ref/'run_report.json').read_text(encoding='utf-8'))
    s.require(r.get('schema') == 'mx-truncation-aware-reference100-report-v1' and r.get('status') == 'PASS'
              and r.get('full_training_completed') is True and r.get('synthetic_fixture_only') is False
              and r.get('completed_epoch') == 100 and r.get('optimizer_steps') == 57300,
              'Geen volledige echte truncatiebewuste honderd-epochrun.')
    expected = json.loads((ROOT/'REFERENCE_PROTOCOL.json').read_text())
    diff = core.differences(r['config'], expected)
    s.require(not diff, 'Afwijkend trainingsprotocol: ' + ', '.join(diff[:12]))
    for k in ENV_KEYS: s.require(r['environment'][k] == rt.environment[k], 'Andere omgeving: '+k)
    for k in BACKEND_KEYS: s.require(r['execution_settings'][k] == rt.settings[k], 'Andere backend: '+k)
    for n in CODE_REUSE:
        s.require(r['package_sha256'].get(n) == core.file_sha256(ROOT/n), 'Decoder/herbruikte code verschilt: '+n)
    s.require(r['source']['archive_sha256'] == json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['archive_sha256'],
              'Andere auteurscode.')
    epochs = csv_rows(ref/'epoch_metrics.csv')
    keys = {(e,p,b) for e in range(1,101) for p in ('train','test') for b in (4,6)}
    s.require(len(epochs) == 400 and {(int(x['epoch']),x['phase'],int(x['weight_bits'])) for x in epochs} == keys,
              'Onvolledige referentiehistorie.')
    for x in epochs:
        s.require(int(x['n']) == (73257 if x['phase']=='train' else 26032), 'Afwijkend sampleaantal.')
        s.require(all(math.isfinite(float(x[k])) for k in ('top1_percent','objective_loss')), 'Niet-eindige metrics.')
        s.require(0 <= int(x['correct1']) <= int(x['n']), 'Ongeldig aantal correct.')
        s.require(abs(float(x['top1_percent']) - 100*int(x['correct1'])/int(x['n'])) < 1e-4,
                  'Accuracy en correcte aantallen inconsistent.')
    high = sorted((x for x in epochs if x['phase']=='test' and int(x['weight_bits'])==6),key=lambda x:int(x['epoch']))
    best = max(high,key=lambda x:float(x['top1_percent']))
    selection = json.loads((ref/'checkpoint_selection.json').read_text())
    s.require(selection == r['checkpoint_selection'] and selection['best']['epoch']==int(best['epoch'])
              and selection['last']['epoch']==100, 'Selectie niet consistent met historie.')
    rows = csv_rows(ref/'selected_metrics.csv')
    s.require(len(rows)==4 and {(x['role'],int(x['weight_bits'])) for x in rows} == {(role,b) for role in ROLES for b in (4,6)},
              'Onvolledige best/last-metrics.')
    for x in rows:
        epoch=selection['best' if x['role']==ROLES[0] else 'last']['epoch']
        original=next(y for y in epochs if int(y['epoch'])==epoch and y['phase']=='test' and y['weight_bits']==x['weight_bits'])
        s.require(x == {'role':x['role'], **original}, 'Best/last verschilt van epochhistorie.')
    fingerprints={n:core.file_sha256(ref/n) for n in required}
    for role in ROLES:
        item=r['mx_exports'][role]['6']
        name=item['file']
        s.require(type(name) is str and Path(name).name==name and not any(c in name for c in '/\\:'), 'Onveilig exportpad.')
        s.require(name in ('mx_source_E2M3_best.pt','mx_source_E2M3_last.pt'), 'Onjuiste hoge bronbestandsnaam.')
        s.require(item['decoded_weights_exact'] is True and item['six_fp32_masters_absent'] is True
                  and item['weight_bits']==6 and item['truncation'] is False, 'Afwijkende hoge export.')
        p=ref/name; marker=p.with_suffix('.pt.sha256')
        s.require(p.is_file() and marker.is_file() and p.stat().st_size==item['bytes'], 'Hoge export/marker ontbreekt.')
        s.require(core.file_sha256(p)==item['sha256']==marker.read_text(encoding='ascii').strip(), 'Exportchecksum faalt.')
        fingerprints[name]=item['sha256']; fingerprints[marker.name]=core.file_sha256(marker)
    return r, rows, fingerprints


def compare_history(metrics, history):
    output=[]
    for new in metrics:
        old=next(r for r in history if r['role']==new['role'] and int(r['weight_bits'])==new['weight_bits'])
        delta=new['correct1']-int(old['correct1'])
        loss_delta=new['hard_cross_entropy']-float(old['objective_loss'])
        count_match=delta==0 and new['correct5']==int(old['correct5'])
        ce_close=math.isclose(new['hard_cross_entropy'],float(old['objective_loss']),rel_tol=1e-5,abs_tol=1e-6)
        output.append({'role':new['role'],'epoch':new['epoch'],'weight_bits':new['weight_bits'],
            'training_correct1':int(old['correct1']),'export_correct1':new['correct1'],
            'training_accuracy_percent':100*int(old['correct1'])/int(old['n']),
            'export_accuracy_percent':new['top1_percent'],'accuracy_difference_pp':100*delta/new['n'],
            'difference_correct1':delta,'training_correct5':int(old['correct5']),'export_correct5':new['correct5'],
            'training_ce':float(old['objective_loss']),'export_ce':new['hard_cross_entropy'],
            'ce_difference':loss_delta,'correct_counts_match':count_match,'ce_close':ce_close,
            'reproduction': 'MATCH_COUNTS_CE_CLOSE' if count_match and ce_close else 'REVIEW_DIFFERENCE',
            'training_logits_available':False})
    return output


def launch_worker(job_file, output, log):
    command=[sys.executable,str(ROOT/'source_worker.py'),'--job',str(job_file),'--output',str(output)]
    # Stream updates instead of an opaque, silent subprocess. sys.executable
    # makes the worker use the exact same venv as the parent on Windows.
    env=os.environ.copy(); env['PYTHONIOENCODING']='utf-8'; env['PYTHONDONTWRITEBYTECODE']='1'
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                       encoding='utf-8',errors='replace',env=env,cwd=job_file.parent)
    try:
        for line in p.stdout: log(line.rstrip())
        rc=p.wait()
    except BaseException:
        p.terminate(); p.wait(); raise
    s.require(rc==0 and (output/'worker_report.json').is_file(), 'Worker faalt; zie worker_report.json en worker.log.')
    report=json.loads((output/'worker_report.json').read_text())
    s.require(report['status']=='PASS' and report['pid']!=os.getpid(), 'Geen geslaagde nieuwe worker.')
    return report


def evaluate(args,out,report,rt,log):
    import resume_core as core
    ref=args.reference_run.resolve()
    r, history, before=inspect_reference(ref,rt)
    report['reference']={'directory':str(ref),'report_sha256':before['run_report.json'],
        'checkpoint_selection':r['checkpoint_selection'],'FP32_checkpoint_read':False,'low_export_read':False}
    copies=out/'reference_inputs'; copies.mkdir()
    for n in ('run_report.json','selected_metrics.csv','checkpoint_selection.json'):
        shutil.copy2(ref/n,copies/n)
    metrics=[]; pairs=[]; workers={}
    for role in ROLES:
        info=r['mx_exports'][role]['6']; suffix='best' if role==ROLES[0] else 'last'
        selection=r['checkpoint_selection'][suffix]
        stage=out/'staged_sources'/suffix; stage.mkdir(parents=True)
        shutil.copy2(ref/info['file'],stage/'source_E2M3.pt')
        shutil.copy2(ref/(info['file']+'.sha256'),stage/'source_E2M3.pt.sha256')
        job={'synthetic_fixture_only':False,'role':role,'epoch':selection['epoch'],
             'seed':r['config']['seed'],'device':str(rt.device),'source_sha256':info['sha256'],
             'checkpoint_sha256':selection['sha256'],'environment':r['environment'],
             'execution_settings':r['execution_settings'],'data_root':str(args.data_root.resolve()),
             'dataset_files':r['dataset_files'],'dataset_selection':r['dataset_selection'],'transforms':r['transforms']}
        # No original model paths and no low exports sent to the worker.
        # Only test metadata is relevant; drop even the train metadata here.
        job['dataset_files']={'files':{'test_32x32.mat':r['dataset_files']['files']['test_32x32.mat']}}
        job['dataset_selection']={'test':r['dataset_selection']['test']}
        job['transforms']={'test':r['transforms']['test']}
        s.write_json(stage/'job.json',job)
        log(f'BRONCONTROLE {role}, epoch {selection["epoch"]}: nieuw proces, alleen hoge export.')
        w=launch_worker(stage/'job.json',out/('worker_'+suffix),log)
        workers[role]=w; metrics.extend(w['metrics'])
        pairs.extend(csv_rows(out/('worker_'+suffix)/'paired_metrics.csv'))
        s.require(core.file_sha256(stage/'source_E2M3.pt')==info['sha256'], 'Staged bron gewijzigd.')
        report['warnings'].extend({**x,'worker_role':role} for x in w['warnings'])
    reproduction=compare_history(metrics,history)
    s.write_csv(out/'metrics.csv',metrics); s.write_csv(out/'historical_reproduction.csv',reproduction)
    s.write_csv(out/'paired_metrics.csv',pairs)
    for n,digest in before.items(): s.require(core.file_sha256(ref/n)==digest, 'Origineel bestand gewijzigd: '+n)
    review=any(x['reproduction']=='REVIEW_DIFFERENCE' for x in reproduction)
    for x in reproduction:
        log(f'{x["role"]}, W{x["weight_bits"]}/A32: {x["export_accuracy_percent"]:.4f}%; '
            f'verschil training {x["difference_correct1"]:+d} correct; {x["reproduction"]}.')
    if review:
        warnings.warn('Uitvoering technisch voltooid, maar minstens een historische score wijkt af. '
                      'Geen toleranties, BN of gewichten aangepast. Deel de feedbackbundel.',RuntimeWarning)
    report.update(workers=workers,metrics=metrics,historical_reproduction=reproduction,
                  input_files_unchanged=True,input_sha256=before,
                  reproduction_status='REVIEW_DIFFERENCE' if review else 'MATCH_COUNTS_CE_CLOSE',
                  all_inference_models_from_high_sources_only=True,
                  training_logits_available=False,synthetic_fixture_only=False,
                  full_test_export_evaluation_completed=True)
    nb=json.loads((ROOT/'MX_bron_only_evaluatie.ipynb').read_text())
    for c in nb['cells']:
        if c['cell_type']=='code':
            text=''.join(c['source']).replace('RUN_DIR = Path(".")  # RUN_DIRECTORY',
                      'RUN_DIR = Path('+json.dumps(str(out))+')  # RUN_DIRECTORY')
            c['source']=text.splitlines(True)
    s.write_json(out/'MX_bron_only_evaluatie.ipynb',nb)


def feedback(out):
    exclude={'source_workcopy','staged_sources'}
    with ZipFile(out/'feedback_bundle.zip','x',compression=ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            rel=p.relative_to(out)
            if not p.is_file() or any(x in exclude for x in rel.parts): continue
            if p.suffix.lower() in ('.pt','.pth','.zip','.py','.pyc','.ipynb'): continue
            if p.suffix.lower() in ('.json','.csv','.txt','.log','.npz','.patch'):
                z.write(p,rel.as_posix())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','evaluate'])
    p.add_argument('--reference-run',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('data/svhn'))
    p.add_argument('--output-root',type=Path,default=Path('mx_runs/mx_truncation_source_only_eval'))
    p.add_argument('--device',choices=['cuda','cuda:0'],default='cuda:0')
    args=p.parse_args()
    if (args.command=='evaluate') != (args.reference_run is not None):
        p.error('evaluate vereist --reference-run; check gebruikt geen getrainde modellen.')
    if args.reference_run and (args.output_root.resolve().is_relative_to(args.reference_run.resolve())
                               or args.output_root.resolve().is_relative_to(args.data_root.resolve())):
        p.error('Uitvoermap moet buiten originele resultaten en data staan.')
    out=(args.output_root/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.command)).resolve()
    out.mkdir(parents=True,exist_ok=False)
    report={'schema':'mx-truncation-source-only-eval-v1','status':'RUNNING','command':args.command,
      'output_directory':str(out),'started_at':s.now(),'argv':sys.argv,'warnings':[],
      'optimizer_steps':0,'new_checkpoint_selection':False,'batchnorm_recalibration':False,
      'synthetic_fixture_only':args.command=='check','train_or_extra_read':False,
      'limits':['No new training, no BN calibration, no new checkpoint selection.',
                'Six MX layers use decoded FP32 buffers; remaining layers and BN stay FP32.',
                'Match of aggregate metrics is NOT proof of identical historical logits.',
                'Old best role was test-selected; epoch100 is the fixed endpoint.',
                'No native MX speed, energy or continual-learning claim.',
                'Guards are software correctness checks, not an OS security sandbox.']}
    log=s.legacy.Log(out/'run.log'); start=time.perf_counter(); rc=1
    def on_warning(message,category,filename,lineno,file=None,line=None):
        row={'category':category.__name__,'message':str(message),'file':filename,'line':lineno}
        report['warnings'].append(row);log('WAARSCHUWING: '+str(message))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('default'); warnings.showwarning=on_warning
            log('RESULTATENMAP: '+str(out))
            log('BRON-ONLY: beide MX-modi uit opgeslagen E2M3; geen training of FP32-checkpoint.')
            report['package_sha256']=s.verify_package()
            dependencies(out,log)
            rt=s.runtime(20261009,args.device)
            report.update(environment=rt.environment,execution_settings=rt.settings,interpreter=sys.executable)
            log('GPU: '+rt.environment['device_name'])
            if args.command=='check':
                import test_source_eval
                report['tests']=test_source_eval.run_tests(out,rt,log)
                log('CHECK PASS: code-afleiding, BN-banken, frozen inferentie en nieuw bron-only proces.')
            else: evaluate(args,out,report,rt,log)
            s.verify_package();report['status']='PASS';rc=0
    except BaseException as exc:
        report.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',
                      error=repr(exc),traceback=traceback.format_exc())
        log(report['status']+': '+str(exc));log(report['traceback'])
        log('Bestaande code, checkpoints en onderzoeksdata niet overschreven.')
    finally:
        report['ended_at']=s.now();report['session_wall_seconds']=time.perf_counter()-start
        log('EINDSTATUS: '+report['status']);log('Rapport: '+str(out/'run_report.json'))
        log.handle.close();s.write_json(out/'run_report.json',report);feedback(out)
    return rc

if __name__=='__main__': raise SystemExit(main())
