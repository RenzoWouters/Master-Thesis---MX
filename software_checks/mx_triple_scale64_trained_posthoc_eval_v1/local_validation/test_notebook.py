"""Bounded, read-only notebook validation using generated JSON/CSV/NPZ fixtures.

Never imports torch, training, a dataset or model code. Tests two separate
checkpoint groups, 18 metrics, 24 prediction pairs, 216 weight pairs, 108 layer
rows, 6 historical rows, legitimate ties, B identity and a constant-class route.
"""
from __future__ import annotations

import builtins
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType

sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='scale64_notebook_mpl_'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import notebook_support as ns

NOTEBOOK = 'MX_triple_scale64_trained_posthoc_resultaten.ipynb'
FIGURES = {'accuracy_scale64', 'accuracy_deltas', 'paired_changes',
           'weight_magnitudes', 'weight_differences', 'prediction_diversity',
           'training_comparison_E2M3', 'training_comparison_E2M1'}
BANNED_IMPORTS = {'torch', 'torchvision', 'run_eval', 'evaluation', 'scale64', 'joint_adapter',
                  'joint_support', 'posthoc_truncation', 'resume_core', 'svhn_support', 'mx_adapter',
                  'mx_format', 'mx_e2m1', 'mx_e4m3', 'trunc_adapter', 'trunc_support'}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_hashes(folder):
    return {str(p.relative_to(folder)): digest(p) for p in folder.rglob('*')
            if p.is_file() and not any(x.startswith('analysis_') for x in p.relative_to(folder).parts)}


def make_fixture(folder):
    folder.mkdir(parents=True, exist_ok=False)
    report = {'schema': ns.SCHEMA, 'status': 'PASS', 'command': 'evaluate',
              'synthetic_fixture_only': True, 'optimizer_steps': 0, 'batchnorm_recalibration': False,
              'new_checkpoint_selection': False, 'learned_shared_scale': False, 'shared_scale_compensation': True,
              'training_kinds': list(ns.TRAININGS), 'scale_exponent_offset': 6,
              'input_files_unchanged': True, 'test_file_unchanged': True, 'train_and_extra_read': False,
              'original_checkpoint_selection_unchanged': True,
              'derived_models_all_evaluated_from_saved_and_reloaded_codes': True,
              'historical_correct_counts_all_match': True, 'warnings': [],
              'limitations': ['SYNTHETIC SOFTWARE VALIDATION ONLY: no SVHN, model or training.'],
              'session_wall_seconds': 12.3, 'roles': {}, 'references': {}}
    metrics, pairs, codes, weights, history = [], [], [], [], []
    y = np.tile(np.arange(10, dtype=np.int64), 2); n = len(y)
    indices = np.arange(n, dtype=np.int64)
    for ti,t in enumerate(ns.TRAININGS):
        report['roles'][t] = {};report['references'][t] = {
            'checkpoint_selection': {}, 'checkpoint_files_opened': False,
            'FP32_master_weights_opened': False, 'contract_matches': True,
            'directory': 'SYNTHETIC_' + t, 'report_sha256': hashlib.sha256(t.encode()).hexdigest()}
        (folder/t).mkdir()
        for ri,r in enumerate(ns.ROLES):
            epoch = 100 if r == 'last_epoch' else 42
            checkpoint = hashlib.sha256((t+'/'+r).encode()).hexdigest()
            arrays = {'labels': y, 'indices': indices}
            count = {'source8': 16+ri, 'trained6': 18, 'trained4': 17,
                     'A6': 2, 'A4': 2, 'B6': 18, 'B4': 17, 'C6': 13, 'C4': 11}
            for pi,p in enumerate(ns.PATHS):
                pred = (y + 1 + pi%3)%10
                keep = (np.arange(count[p]) + (pi*3+ri))%n
                pred[keep] = y[keep]
                logits = np.full((n,10),-1,dtype=np.float32);logits[np.arange(n),pred] = 2
                if p == 'A6':
                    logits[:] = 0;pred[:] = 8  # Legal topk choice; NumPy argmax would select 0.
                if p == 'A4':
                    pred[:] = 1;logits[:] = -1;logits[:,1] = 2
                if p in ('B6','B4'):
                    q = 'trained'+str(ns.BITS[p]);pred = arrays[q+'__predictions'].copy();logits = arrays[q+'__logits'].copy()
                ld = logits.astype(np.float64);m=ld.max(axis=1)
                ce = (np.log(np.exp(ld-m[:,None]).sum(axis=1))+m-ld[np.arange(n),y]).astype(np.float32)
                arrays[p+'__predictions'] = pred.astype(np.int64);arrays[p+'__logits'] = logits;arrays[p+'__per_image_ce'] = ce
                c1 = int((pred==y).sum())
                top5 = np.argsort(logits,axis=1)[:,-5:];c5 = int((top5==y[:,None]).any(axis=1).sum())
                metrics.append({'training':t,'role':r,'epoch':epoch,'path':p,'weight_bits':ns.BITS[p],
                    'weight_format':{8:'E4M3',6:'E2M3',4:'E2M1'}[ns.BITS[p]],'source_weight_bits':8,
                    'family':p[0] if p in ns.DERIVED else 'trained','direct_baseline':ns.BASELINES[p],
                    'activation_bits':32,'bn_bank_key':ns.BANKS[ns.BITS[p]],'n':n,'correct1':c1,'correct5':c5,
                    'top1_percent':100*c1/n,'top5_percent':100*c5/n,'hard_cross_entropy':float(ce.astype(np.float64).mean()),
                    'optimizer_steps':0,'batchnorm_recalibrated':False,
                    'classes_predicted':int((np.bincount(pred,minlength=10)>0).sum()),
                    'dominant_predicted_class':int(np.bincount(pred,minlength=10).argmax()),
                    'dominant_prediction_fraction':float(np.bincount(pred,minlength=10).max()/n),
                    'scale_exponent_offset':ns.OFFSET[p]})
                if p in ns.PATHS[:3]:
                    history.append({'training':t,'role':r,'epoch':epoch,'path':p,'historical_correct1':c1,
                        'reevaluated_correct1':c1,'difference_correct1':0,'accuracy_difference_pp':0.,'correct1_matches_history':True})
            for a,b in ns.COMPARISONS:
                ap,bp=arrays[a+'__predictions'],arrays[b+'__predictions'];ac,bc=ap==y,bp==y;changed=ap!=bp
                up,down=int((~ac&bc).sum()),int((ac&~bc).sum());err=arrays[b+'__logits'].astype(np.float64)-arrays[a+'__logits'].astype(np.float64)
                pairs.append({'training':t,'role':r,'epoch':epoch,'comparison':b+'_minus_'+a,'reference_path':a,'other_path':b,'n':n,
                    'reference_correct1':int(ac.sum()),'other_correct1':int(bc.sum()),'accuracy_difference_pp':100*(up-down)/n,
                    'prediction_changes':int(changed.sum()),'wrong_to_correct':up,'correct_to_wrong':down,'both_correct':int((ac&bc).sum()),
                    'both_wrong':int((~ac&~bc).sum()),'wrong_to_other_wrong':int((~ac&~bc&changed).sum()),'net_correct_change':up-down,
                    'mean_abs_logit_difference':float(np.abs(err).mean()),'max_abs_logit_difference':float(np.abs(err).max())})
            ratios={'source8':1.,'trained6':.98,'trained4':.9,'A6':.02,'A4':.018,'B6':1.28,'B4':1.152,'C6':.95,'C4':.90}
            ratios.update(trained6=ratios['B6'],trained4=ratios['B4'])
            for li,layer in enumerate(ns.LAYERS):
                source=np.linspace(-.01,.011,20,dtype=np.float64)/(li+1)
                values={p:source*ratios[p] for p in ns.PATHS}
                for p in ns.PATHS:
                    v=values[p];baseline=values[ns.BASELINES[p]];anchor='source8'
                    ref=values[anchor];offset=ns.OFFSET[p];scale=-12+offset
                    rms=float(np.sqrt(np.square(v).mean()));brms=float(np.sqrt(np.square(baseline).mean()));srms=float(np.sqrt(np.square(ref).mean()))
                    ma=float(np.abs(v).mean());bma=float(np.abs(baseline).mean())
                    codes.append({'training':t,'role':r,'epoch':epoch,'layer':layer,'path':p,'family':p[0] if p in ns.DERIVED else 'trained',
                        'source_bits':8,'target_bits':ns.BITS[p],'bn_bank_key':ns.BANKS[ns.BITS[p]],
                        'n_weights':20,'blocks':1,'padding_slots':12,'direct_baseline':ns.BASELINES[p],'mean_abs':ma,'rms':rms,
                        'mean_abs_ratio_to_direct':ma/bma,'rms_ratio_to_direct':rms/brms,'rms_ratio_to_source':rms/srms,
                        'zero_count':0,'negative_zero_count':0,'new_zero_vs_source':0,'local_subnormal_count':1,
                        'scale_exp_min':scale,'scale_exp_max':scale,'scale_exponent_offset':offset,'scale_anchor_path':anchor,
                        'scales_equal_source':offset==0,'source_scale_delta_counts':json.dumps({str(offset):1}),
                        'scale_delta_to_direct_counts':json.dumps({'0':1}),
                        'nonzero_dropped_exponent_suffix_count':10 if p[0] in 'AB' else None,
                        'nonzero_dropped_mantissa_suffix_count':10 if p in ('A4','B4','C4') else None,
                        'rounded_source_out_of_target_element_range':0 if p=='C6' else None,
                        'code_and_scale_bytes':{8:33,6:25,4:17}[ns.BITS[p]]})
                for a,b in ns.COMPARISONS + tuple(('source8',p) for p in ns.DERIVED):
                    error=values[b]-values[a];same=ns.BITS[a]==ns.BITS[b];same_scale=(ns.OFFSET[a]==ns.OFFSET[b]) or (a in ('trained6','trained4') and b=='B'+a[-1])
                    weights.append({'training':t,'role':r,'epoch':epoch,'layer':layer,'reference_path':a,'other_path':b,
                        'comparison':b+'_minus_'+a,'n_weights':len(source),'code_comparison_same_format':same,
                        'code_differences':(0 if (a[0]=='A' and b[0]=='B') or (a in ('trained6','trained4') and b=='B'+a[-1]) else 10) if same else None,
                        'scale_code_differences':0 if same_scale else 1,'scale_bytes_equal':same_scale,
                        'numeric_differences':int((error!=0).sum()),'mean_signed_difference':float(error.mean()),
                        'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.square(error).mean())),
                        'max_abs_difference':float(np.abs(error).max()),'reference_l2_norm':float(np.linalg.norm(values[a])),
                        'difference_l2_norm':float(np.linalg.norm(error))})
            npz=folder/t/(r+'_per_image_outputs.npz');np.savez_compressed(npz,**arrays)
            report['roles'][t][r]={'epoch':epoch,'checkpoint_sha256':checkpoint,'n_test':n,
                'label_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'per_image_file':str(npz.relative_to(folder)),
                'per_image_sha256':digest(npz),'all_model_and_bn_states_unchanged':True,'all_models_no_grad':True,
                'all_paths_use_frozen_target_precision_bn':True,'six_FP32_master_parameters_absent_all_paths':True,
                'optimizer_steps':0,'batchnorm_recalibrated':False,
                'trained_lower_codes_exact_B_routes':True,'trained_lower_scale_codes_exact_B_routes':True,
                'B_same_frozen_state_as_trained':True,
                'B_same_run_reproduction':{str(bit):{'logits_exact':True,'logits_allclose':True,
                     'predictions_equal':True,'prediction_changes':0,'correct1_difference':0,
                     'max_abs_logit_difference':0.,'ce_exact':True,'max_abs_ce_difference':0.} for bit in (6,4)}}
            report['references'][t]['checkpoint_selection']['last' if r=='last_epoch' else 'best']={'epoch':epoch,'sha256':checkpoint}
    report['metrics']=metrics;report['paired_metrics']=pairs
    write_json(folder/'run_report.json',report)
    for name,rows in [('metrics.csv',metrics),('paired_metrics.csv',pairs),('weight_pair_metrics.csv',weights),
                      ('layer_code_metrics.csv',codes),('historical_reproduction.csv',history)]:
        pd.DataFrame(rows).to_csv(folder/name,index=False)
    shutil.copy2(ROOT/'notebook_support.py',folder/'notebook_support.py');shutil.copy2(ROOT/NOTEBOOK,folder/NOTEBOOK)
    shutil.copytree(ROOT/'prior_posthoc',folder/'prior_posthoc')


def execute_template(folder, expected_confirmed=True, require_synthetic=True):
    notebook=json.loads((folder/NOTEBOOK).read_text(encoding='utf-8'))
    assert sum(c['cell_type']=='code' for c in notebook['cells'])==11
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in notebook['cells'] if c['cell_type']=='code')
    originals={k:sys.modules.get(k) for k in ('IPython','IPython.display')}
    stub=ModuleType('IPython');stub.version_info=(9,0);stub.get_ipython=lambda:None
    display=ModuleType('IPython.display');display.display=lambda *a,**k:None;display.Markdown=lambda s:s
    stub.display=display;sys.modules['IPython']=stub;sys.modules['IPython.display']=display
    env={};original_import=builtins.__import__
    def safe_import(name,*args,**kwargs):
        assert name.split('.')[0] not in BANNED_IMPORTS,'Forbidden notebook import: '+name
        return original_import(name,*args,**kwargs)
    try:
        with redirect_stdout(io.StringIO()):
            builtins.__import__=safe_import
            for i,cell in enumerate(notebook['cells']):
                if cell['cell_type']!='code':continue
                if 'parameters' in cell.get('metadata',{}).get('tags',[]):
                    env.update(Path=Path,RUN_DIR=folder,SAVE_FIGURES=False)
                else:exec(compile(''.join(cell['source']),f'notebook_cell_{i}','exec'),env)
    finally:
        builtins.__import__=original_import
        for name,value in originals.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value
    assert env['run'].confirmed==expected_confirmed,env['run'].issues
    assert set(env['figures'])==FIGURES
    if require_synthetic:assert all('SYNTHETISCHE CONTROLE' in f._suptitle.get_text() for f in env['figures'].values())
    return env


def main(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    fixture=out/'synthetic_complete';make_fixture(fixture);before=raw_hashes(fixture)
    run=ns.load_run(fixture)
    assert run.confirmed,run.issues
    assert len(run.metrics)==18 and len(run.pairs)==24 and len(run.reproduction)==6 and len(run.output_checks)==2
    assert len(run.weights)==216 and len(run.codes)==108
    assert run.epoch('scale64','best_E4M3_test')==42 and run.prior_safe and len(run.prior_metrics)==36
    assert run.epoch('scale64','last_epoch')==100
    assert len(ns.historical_comparison_table(run))==54
    preds=ns.prediction_table(run)
    collapse=preds.loc[(preds.training=='scale64')&(preds.path=='A6')]
    assert collapse.classes_predicted.eq(1).all() and collapse.dominant_fraction.eq(1).all()
    assert 'torch' not in sys.modules
    env=execute_template(fixture);assert raw_hashes(fixture)==before
    saved=ns.save_figures(run,env['figures'])
    assert len(list(saved.glob('*.png')))==len(list(saved.glob('*.pdf')))==8
    assert raw_hashes(fixture)==before and not (saved/NOTEBOOK).exists()
    plt.close('all')
    negative={}
    def copy_fixture(name):
        target=out/name;shutil.copytree(fixture,target,ignore=shutil.ignore_patterns('analysis_*'));return target
    def reject(name,target):
        bad=ns.load_run(target)
        assert not bad.confirmed and any(x['Niveau']=='FOUT' for x in bad.issues),(name,bad.issues)
        negative[name]={'expected_rejection':True,'messages':[x['Melding'] for x in bad.issues if x['Niveau']=='FOUT']}
    csv_changes={
        'missing_route':('metrics.csv',lambda d:d.drop(index=7)),
        'mixed_epoch':('metrics.csv',lambda d:d.assign(epoch=np.where(d.index==6,55,d.epoch))),
        'mixed_training_population':('metrics.csv',lambda d:d.assign(training=np.where(d.index==5,'truncation',d.training))),
        'wrong_target_BN':('metrics.csv',lambda d:d.assign(bn_bank_key=np.where(d.path=='B6',8,d.bn_bank_key))),
        'incorrect_top1_count':('metrics.csv',lambda d:d.assign(correct1=np.where(d.index==0,d.correct1+1,d.correct1))),
        'wrong_B_scale_offset':('layer_code_metrics.csv',lambda d:d.assign(scale_exponent_offset=np.where(d.path=='B6',0,d.scale_exponent_offset))),
        'wrong_B_scale_equality':('layer_code_metrics.csv',lambda d:d.assign(scales_equal_source=np.where(d.path=='B6',True,d.scales_equal_source))),
        'wrong_B_weight_ratio':('layer_code_metrics.csv',lambda d:d.assign(rms=np.where(d.path=='B6',d.rms*1.1,d.rms))),
        'wrong_block_scale_counts':('layer_code_metrics.csv',lambda d:d.assign(source_scale_delta_counts=np.where(d.path=='C6','{"0":1}',d.source_scale_delta_counts))),
        'missing_weight_pair':('weight_pair_metrics.csv',lambda d:d.drop(index=7)),
        'cross_format_codes_falsely_compared':('weight_pair_metrics.csv',lambda d:d.assign(code_differences=np.where(d.reference_path=='source8',0,d.code_differences))),
        'pair_counts_not_matching_predictions':('paired_metrics.csv',lambda d:d.assign(wrong_to_correct=d.wrong_to_correct+1,correct_to_wrong=d.correct_to_wrong+1,prediction_changes=d.prediction_changes+2)),
        'bad_historical_delta':('historical_reproduction.csv',lambda d:d.assign(difference_correct1=1)),
    }
    for name,(filename,change) in csv_changes.items():
        target=copy_fixture(name);change(pd.read_csv(target/filename,float_precision='round_trip')).to_csv(target/filename,index=False);reject(name,target)
    json_changes={
        'NPZ_checksum_mismatch':lambda d:d['roles']['scale64']['last_epoch'].update(per_image_sha256='0'*64),
        'unsafe_relative_NPZ_path':lambda d:d['roles']['scale64']['last_epoch'].update(per_image_file='../outside.npz'),
        'changed_original_checkpoint':lambda d:d['references']['scale64']['checkpoint_selection']['best'].update(sha256='0'*64),
        'missing_provenance':lambda d:d.pop('synthetic_fixture_only'),
        'unknown_schema':lambda d:d.update(schema='old-unsupported-schema'),
        'FP32_master_access':lambda d:d['references']['scale64'].update(FP32_master_weights_opened=True),
        'BN_recalibrated':lambda d:d.update(batchnorm_recalibration=True),
        'learned_scale':lambda d:d.update(learned_shared_scale=True),
        'report_metrics_disagree':lambda d:d['metrics'][0].update(correct1=0),
        'B_code_identity_unconfirmed':lambda d:d['roles']['scale64']['last_epoch'].update(trained_lower_codes_exact_B_routes=False),
        'B_reproduction_report_wrong':lambda d:d['roles']['scale64']['last_epoch']['B_same_run_reproduction']['6'].update(prediction_changes=1),
        'real_run_wrong_test_count':lambda d:d.update(synthetic_fixture_only=False),
    }
    for name,change in json_changes.items():
        target=copy_fixture(name);report=json.loads((target/'run_report.json').read_text());change(report);write_json(target/'run_report.json',report);reject(name,target)
    target=copy_fixture('NPZ_internal_prediction_inconsistent')
    report=json.loads((target/'run_report.json').read_text());role=report['roles']['scale64']['last_epoch'];path=target/role['per_image_file']
    with np.load(path,allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    arrays['source8__predictions'][0]=(arrays['source8__predictions'][0]+1)%10
    np.savez_compressed(path,**arrays);role['per_image_sha256']=digest(path);write_json(target/'run_report.json',report);reject('NPZ_internal_prediction_inconsistent',target)
    target=copy_fixture('prior_metrics_changed')
    prior=target/'prior_posthoc'/'metrics.csv'
    prior.write_text(prior.read_text()+'\n',encoding='utf-8')
    reject('prior_metrics_changed',target)
    target=copy_fixture('prior_reference_wrong_schema')
    path=target/'prior_posthoc'/'REFERENCE.json';manifest=json.loads(path.read_text())
    manifest['schema']='wrong-schema';write_json(path,manifest);reject('prior_reference_wrong_schema',target)
    assert not ns.load_run(fixture,verify_per_image=False).confirmed
    partial=copy_fixture('failed_partial');report=json.loads((partial/'run_report.json').read_text());report.update(status='FAIL',error='SYNTHETIC interruption after one group')
    write_json(partial/'run_report.json',report)
    for name in ('metrics.csv','paired_metrics.csv','weight_pair_metrics.csv','layer_code_metrics.csv','historical_reproduction.csv'):(partial/name).unlink()
    partial_before=raw_hashes(partial);execute_template(partial,expected_confirmed=False);assert partial_before==raw_hashes(partial);plt.close('all')
    check=out/'check_only';check.mkdir();write_json(check/'run_report.json',{'schema':ns.SCHEMA,'status':'PASS','command':'check','synthetic_fixture_only':True})
    shutil.copy2(ROOT/'notebook_support.py',check/'notebook_support.py');shutil.copy2(ROOT/NOTEBOOK,check/NOTEBOOK)
    check_before=raw_hashes(check);execute_template(check,expected_confirmed=False);assert check_before==raw_hashes(check);plt.close('all')
    missing=ns.load_run(out/'missing');assert not missing.confirmed and missing.metrics.empty
    # Historical differences are reported but must not silently overwrite the historical reference.
    drift=copy_fixture('legitimate_historical_drift');d=pd.read_csv(drift/'historical_reproduction.csv',float_precision='round_trip')
    d.loc[0,'historical_correct1']-=1;d.loc[0,'difference_correct1']=1;d.loc[0,'accuracy_difference_pp']=5.;d.loc[0,'correct1_matches_history']=False
    d.to_csv(drift/'historical_reproduction.csv',index=False);report=json.loads((drift/'run_report.json').read_text());report['historical_correct_counts_all_match']=False;write_json(drift/'run_report.json',report)
    drift_run=ns.load_run(drift);assert drift_run.confirmed and any(x['Niveau']=='LET OP' for x in drift_run.issues)
    result={'schema':'mx-triple-scale64-trained-notebook-validation-v1','status':'PASS','synthetic_checks_only':True,
        'scope':'Generated reports and arrays; no SVHN data, model, PyTorch, training or real checkpoint accuracy.',
        'helper_sha256':digest(ROOT/'notebook_support.py'),'notebook_sha256':digest(ROOT/NOTEBOOK),
        'blank_notebook':True,'code_cells_executed_in_memory':11,'figures':sorted(FIGURES),
        'groups':2,'metrics':18,'paired_metrics':24,'historical_reproduction':6,'layer_rows':108,'weight_pair_rows':216,
        'per_image_checks':2,'constant_class_outcome_preserved':True,'non_argmax_tie_supported':True,
        'two_current_checkpoints_and_three_historical_training_cohorts_preserved':True,'raw_inputs_unchanged':True,
        'failure_partial_check_only_and_missing_handled':True,'historical_drift_reported_without_overwriting':True,
        'model_imports_blocked':sorted(BANNED_IMPORTS),'executed_notebook_saved':False,
        'figure_save_png_count':8,'figure_save_pdf_count':8,'preview_directory':str(saved),
        'expected_negative_rejections':negative,'negative_case_count':len(negative),'visual_qa':'pending','prior_summary_sha256_validated':True,'no_prior_NPZ_rerun_or_model_execution':True}
    write_json(ROOT/'local_validation'/'notebook_validation_report.json',result)
    write_json(out/'validation_result.json',result)
    print(json.dumps({'status':'PASS','negative_cases':len(negative),'preview_directory':str(saved)},indent=2))


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python local_validation/test_notebook.py NEW_TEMPORARY_OUTPUT_DIRECTORY')
    main(sys.argv[1])
