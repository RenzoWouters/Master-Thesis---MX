"""Additional source-grounded overview checks. No training, torch or model loading."""
from pathlib import Path
import hashlib
import json
import math
import zipfile
import numpy as np
import pandas as pd
import meeting_data as M
import dedicated_data as DD

DEDICATED_ID = 'dedicated_e2m1_100'


def validate_dedicated(base, data, storage):
    """Validate the received E2M1 run, not the unavailable model/checkpoint bytes."""
    base=Path(base)
    q=DD.load_dedicated(base/'bronnen'/DEDICATED_ID)
    r=q['report']; h=q['history']; cfg=r['config']
    M.require(not q['synthetic'], 'Synthetic dedicated data cannot be reported as measured.')
    M.require(r['training_forward_passes']==57300, 'Incorrect number of dedicated forward passes.')
    M.require(r['selected_best_epoch']==97 and r['selected_last_epoch']==100,
              'The selected epochs differ from the supplied completed run.')
    epochs=h['epochs']
    for e in epochs:
        M.require(e['optimizer_steps']==573*e['epoch'], 'Incorrect cumulative Adam count.')
        M.require(e['test_optimizer_steps']==0, 'Test performed optimizer updates.')
        M.require(e['gradient_checks']['finite_gradient_checks']==573, 'Missing gradient check.')
        M.require(e['adam_checks']['unused_bn_has_no_optimizer_state'], 'Unused BN has Adam state.')
        M.require(e['batchnorm_counters_correct'], 'BN check failed.')
        expected_lr=.001 * .1**sum(e['epoch']>k for k in (50,75,90))
        next_lr=.001 * .1**sum(e['epoch']>=k for k in (50,75,90))
        M.require(math.isclose(e['lr_used'],expected_lr,rel_tol=1e-12,abs_tol=1e-15), 'Incorrect LR.')
        M.require(math.isclose(e['lr_next'],next_lr,rel_tol=1e-12,abs_tol=1e-15), 'Incorrect next LR.')
        for ph in ['train','test']:
            M.require(e['coverage'][ph]['complete_once'], 'Sample coverage incomplete.')
        M.require(len(e['weight_updates'])==6, 'Six layer records required.')
        M.require(all(w['master_dtype']=='torch.float32' for w in e['weight_updates']), 'Master dtype differs.')
    M.require(sum(e['first_epoch_in_process'] for e in epochs)==1, 'Run was not one continuous process.')
    layers=pd.read_csv(base/'bronnen'/DEDICATED_ID/'mx_export_layers_last.csv')
    layout=pd.read_csv(base/'bronnen/trunc100/mx_export_layers_4_last.csv')
    for col in ['layer','n_weights','blocks','padded_slots']:
        M.require(layers[col].tolist()==layout[col].tolist(), 'Dedicated storage layout mismatch: '+col)
    M.require(q['code_scale_bytes']==storage['low_bytes'], 'Inconsistent E2M1 payload.')
    for ex in r['mx_exports'].values():
        M.require(ex['decoded_weights_exact'] and ex['six_fp32_masters_absent'], 'Export check failed.')
    # Read code files only as bytes, never execute code contained in the training archive.
    verified=[]
    with zipfile.ZipFile(base/'bronnen/code_dedicated_e2m1.zip') as z:
        for name,digest in r['package_sha256'].items():
            M.require(name in z.namelist(), 'Code file missing: '+name)
            ok=hashlib.sha256(z.read(name)).hexdigest()==digest
            M.require(ok, 'Training package hash differs: '+name)
            verified.append(name)
    comparisons=[]
    for rid in ['single_fp100','single_mx100','dual100','trunc100']:
        other=data['reports'][rid]; oh=data['histories'][rid]
        order=[epochs[i]['coverage'][p]['order_sha256']==oh['epochs'][i]['coverage'][p]['order_sha256']
               for i in range(100) for p in ['train','test']]
        comparisons.append({
            'run':rid,'initialization_equal':r['initial_model_sha256']==other['initial_model_sha256'],
            'order_hashes_equal':int(sum(order)),'order_hashes_compared':len(order),
            'environment_equal':r['environment']==other['environment'],
            'backend_equal':r['execution_settings']==other['execution_settings'],
            'dataset_manifest_equal':r['dataset_files']==other['dataset_files'],
            'dataset_selection_equal':r['dataset_selection']==other['dataset_selection'],
            'transforms_equal':r['transforms']==other['transforms'],
            'interpretation':'Recorded common setup, not identical loss, compute budget or deterministic CUDA.'})
    times=pd.read_csv(base/'bronnen'/DEDICATED_ID/'epoch_times.csv')
    M.require(math.isclose(times.total_epoch_seconds.sum(),r['training_summary']['measured_completed_epoch_seconds'],rel_tol=0,abs_tol=1e-6), 'Session timing inconsistent.')
    # Check consistency across duplicate sidecar summaries without changing either file.
    cs=M.read_json(base/'bronnen'/DEDICATED_ID/'checkpoint_selection.json')
    M.require(cs==r['checkpoint_selection'], 'Checkpoint selection JSON differs from report.')
    M.require(M.read_json(base/'bronnen'/DEDICATED_ID/'training_summary.json')==r['training_summary'], 'Training summary differs.')
    check={
        'schema':'dedicated-e2m1-overview-audit-v2','status':'PASS (report and byte checks)',
        'epochs':100,'metric_rows':200,'optimizer_steps':57300,'training_forward_passes':57300,
        'best_epoch':r['selected_best_epoch'],'last_epoch':100,
        'last_correct1':int(M.final_row(data,DEDICATED_ID,4).correct1),
        'last_accuracy_percent':float(M.final_row(data,DEDICATED_ID,4).accuracy_percent),
        'package_files_verified':len(verified),'package_files':verified,
        'code_scale_bytes':q['code_scale_bytes'],
        'raw_model_payload_bytes_with_all_preserved_BN':q['code_scale_bytes']+storage['auxiliary_bytes'],
        'reported_physical_export_bytes':r['mx_exports']['last_epoch']['bytes'],
        'full_test_from_export_performed':r['mx_exports']['last_epoch']['full_test_forward_rerun'],
        'measured_epochs_seconds':float(times.total_epoch_seconds.sum()),
        'measured_session_seconds':r['session_wall_seconds'],'pairs':comparisons,
        'warnings':r['warnings'],
        'pip_check':(base/'bronnen'/DEDICATED_ID/'pip_check.txt').read_text().strip(),
        'source_unchanged_reported':r['source_unchanged'],
        'data_unchanged_reported':r['dataset_files_unchanged'],
        'limits':['Model checkpoints, exports and individual predictions are absent from this feedback bundle.',
                  'One seed per method; test selected best checkpoints; different training regimes.',
                  'No statistical equivalence, significance, full export inference or hardware performance tested.']}
    (base/'afgeleid/dedicated_controle.json').write_text(json.dumps(check,indent=2,ensure_ascii=False),encoding='utf8')
    return check


def low_comparison(data):
    """Keep A/B/C from one trained model apart from the two other training regimes."""
    rows=[]
    for path,label in [('A_direct_E2M1','Gezamenlijk direct / direct E2M1 (A)'),
                       ('B_round_from_E2M3','Zelfde model / afronden uit E2M3 (B)'),
                       ('C_truncate_E2M3','Zelfde model / achteraf trunceren (C)')]:
        a=M.gap_row(data,path)
        rows.append(dict(uitvoering=label,path=path,epoch=100,accuracy_percent=float(a.accuracy_percent),
                         correct1=int(a.correct1),n=int(a.n),test_ce=float(a.hard_cross_entropy),
                         bron='bronnen/gap/metrics.csv'))
    for rid,path,label in [('trunc100','TA','Gezamenlijk truncatiebewust / trunceren'),
                           (DEDICATED_ID,'DED','Dedicated / direct E2M1')]:
        a=M.final_row(data,rid,4)
        rows.append(dict(uitvoering=label,path=path,epoch=100,accuracy_percent=float(a.accuracy_percent),
                         correct1=int(a.correct1),n=int(a.n),test_ce=float(a.objective_loss),
                         bron=f'bronnen/{rid}/epoch_metrics.csv'))
    out=pd.DataFrame(rows)
    acc=float(out.loc[out.path=='DED','accuracy_percent'].iloc[0])
    co=int(out.loc[out.path=='DED','correct1'].iloc[0])
    out['delta_vs_dedicated_pp']=out.accuracy_percent-acc
    out['net_correct_difference_vs_dedicated']=out.correct1-co
    return out


def high_comparison(data):
    """Unique full-run E2M3 trajectories; saved-code replays are NOT new experiments."""
    cast=data['cast']
    a=cast[(cast['mode']=='MXFP6_E2M3_A32')&(cast.role=='last')&(cast.epoch==100)]
    b=cast[(cast['mode']=='MXFP6_E2M3_A32')&(cast.role=='best')]
    M.require(len(a)==1 and len(b)==1, 'Missing unique E2M3 conversion endpoint.')
    a=a.iloc[0]; b=b.iloc[0]
    rows=[dict(run_id='cast',traject='Any-Precision vijfmodustraining + E2M3-conversie',
               groep='Conversie zonder MX-hertraining',epoch=100,accuracy_percent=float(a.accuracy_percent),
               correct1=int(a.correct1),test_ce=float(a.hard_cross_entropy),best_epoch=int(b.epoch),
               best_accuracy_percent=float(b.accuracy_percent),selection_policy='W32/A32 brontraining',
               bron='bronnen/cast/metrics.csv')]
    for rid,label in [('single_mx100','Dedicated E2M3/A32'),('dual100','Gezamenlijk direct E2M3/A32'),
                      ('trunc100','Gezamenlijk truncatiebewust E2M3/A32')]:
        a=M.final_row(data,rid,6)
        b=M.final_row(data,rid,6,data['reports'][rid]['selected_best_epoch'])
        rows.append(dict(run_id=rid,traject=label,groep='MX tijdens training',epoch=100,
                         accuracy_percent=float(a.accuracy_percent),correct1=int(a.correct1),
                         test_ce=float(a.objective_loss),best_epoch=int(b.epoch),
                         best_accuracy_percent=float(b.accuracy_percent),selection_policy='E2M3/A32 eigen run',
                         bron=f'bronnen/{rid}/selected_metrics.csv'))
    return pd.DataFrame(rows)
