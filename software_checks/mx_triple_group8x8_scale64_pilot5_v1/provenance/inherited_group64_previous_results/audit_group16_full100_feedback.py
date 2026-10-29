"""Read-only audit of the real B16 x64 feedback received 2026-10-26."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys

ROOT = Path('/workspace/scratch/4619498905be')
SOURCE = ROOT / 'upload/feedback_bundle(20261026-195319).zip'
P = ROOT / 'work/group16_full100_actual_20261026_195319'
PACKAGE = ROOT / 'work/group64_preparation_sources/mx_triple_group16_scale64_reference100_v1'
OUT = ROOT / 'work/group16_full100_actual_audit'
OUT.mkdir(exist_ok=True)
sys.dont_write_bytecode = True
before = {str(p.relative_to(P)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in P.rglob('*') if p.is_file()}
source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

def read_json(name):
    return json.loads((P / name).read_text())

def read_csv(name):
    with (P / name).open(newline='') as f:
        return list(csv.DictReader(f))

checks = []
def check(ok, name):
    checks.append({'name': name, 'passed': bool(ok)})

def equal_value(csv_value, value):
    if value is None:
        return csv_value == ''
    if isinstance(value, bool):
        return csv_value == str(value)
    if isinstance(value, (int, float)):
        return math.isclose(float(csv_value), value, rel_tol=0, abs_tol=1e-10)
    return csv_value == str(value)

def validate_metrics(rows, epochs, label):
    expected = {(e, phase, b) for e in range(1, epochs+1)
                for phase in ['train', 'test'] for b in [4, 6, 8]}
    actual = {(int(r['epoch']), r['phase'], int(r['weight_bits'])) for r in rows}
    check(actual == expected and len(rows) == len(expected), label + ': all epoch/mode metrics once')
    check(all(int(r['n']) == (73257 if r['phase'] == 'train' else 26032) for r in rows), label + ': sample counts')
    check(all(int(r['activation_bits']) == 32 for r in rows), label + ': A32')
    check(all(r['bn_bank_key'] == {'4': '4', '6': '32', '8': '8'}[r['weight_bits']] for r in rows), label + ': BN banks')
    check(all(math.isfinite(float(r[k])) for r in rows for k in ['top1_percent','top5_percent','objective_loss','lr']), label + ': finite metrics')
    check(all(0 <= int(r['correct1']) <= int(r['correct5']) <= int(r['n']) for r in rows), label + ': integer correct bounds')
    check(all(abs(float(r['top1_percent']) - 100*int(r['correct1'])/int(r['n'])) < 1e-7 for r in rows), label + ': top1 integer reconciliation')
    check(all(abs(float(r['count_top1_percent']) - 100*int(r['correct1'])/int(r['n'])) < 1e-12 for r in rows), label + ': exact count percentages')
    check(all(math.isclose(float(r['lr']), 0.001 * 0.1**sum(int(r['epoch']) > m for m in [50,75,90]), abs_tol=1e-16) for r in rows), label + ': LR schedule')
    return {(int(r['epoch']), r['phase'], int(r['weight_bits'])): r for r in rows}

r = read_json('run_report.json')
summary = read_json('training_summary.json')
selection = read_json('checkpoint_selection.json')
h = read_json('history.json')
rows = read_csv('epoch_metrics.csv')
idx = validate_metrics(rows, 100, 'B16')
rr = read_json('scale64_reference/run_report.json')
rb = read_csv('scale64_reference/epoch_metrics.csv')
bidx = validate_metrics(rb, 100, 'B32 reference')
rp = read_json('pilot_reference/run_report.json')
prows = read_csv('pilot_reference/epoch_metrics.csv')
pidx = validate_metrics(prows, 5, 'B16 pilot')

check(r['status'] == 'PASS' and r['command'] == 'run', 'real completed run PASS')
check(r['synthetic_fixture_only'] is False and summary['synthetic_fixture_only'] is False, 'not synthetic')
check(summary['completed_epochs'] == r['completed_epoch'] == 100, '100 complete epochs')
check(summary['full_training_completed'] is True and summary['full_100_epoch_training_completed'] is True, 'completion flags')
check(r['optimizer_steps'] == summary['optimizer_steps'] == 57300, '57300 optimizer steps')
check(r['training_forward_passes'] == summary['training_forward_passes'] == 171900, '171900 training passes')
check(summary['resumed'] is False and summary['new_initialization_not_previous_experiment_warmstart'] is True, 'fresh100 initialization')
check(summary == r['training_summary'] and selection == r['checkpoint_selection'], 'duplicated summary and selection match')
check(r['source'] == read_json('source_manifest.json'), 'source manifest matches')
dm = read_json('dataset_manifest.json')
check(dm['files'] == r['dataset_files'] and dm['selection'] == r['dataset_selection'] and dm['transforms'] == r['transforms'], 'dataset manifest matches')
check(set(r['dataset_files']['files']) == {'train_32x32.mat','test_32x32.mat'}, 'canonical dataset names only')
check(r['dataset_selection']['extra_used'] is False and r['dataset_selection']['held_out_validation_examples'] == 0, 'train-only no extra/validation')
check(r['source_unchanged'] is True and r['dataset_files_unchanged'] is True, 'recorded immutable source/data')
check(len(h['epochs']) == 100 and [e['epoch'] for e in h['epochs']] == list(range(1,101)), 'history contiguous')
check(h['initial_model_sha256'] == r['initial_model_sha256'] == rr['initial_model_sha256'] == rp['initial_model_sha256'], 'same initial model B16/B32/pilot')
for key in ['execution_settings','environment','source','transforms','dataset_selection']:
    check(r[key] == rr[key], 'B16/B32 identical ' + key)
check(r['dataset_files']['files'] == rr['dataset_files']['files'], 'B16/B32 dataset hashes equal')
expected_diffs = {'comparison_baseline_block_size','mx_block_size','mx_group_size_research_variant','purpose','schema'}
config_diff = {k: {'B16': r['config'].get(k), 'B32': rr['config'].get(k)}
               for k in set(r['config']) | set(rr['config']) if r['config'].get(k) != rr['config'].get(k)}
check(set(config_diff) == expected_diffs, 'only intended group size protocol change')
check(r['config']['mx_block_size'] == 16 and r['config']['scale_exponent_offset'] == 6 and r['config']['scale_factor'] == 64 and r['config']['learned_shared_scale'] is False, 'B16 fixed64 contract')
for flag in ['gradient_checks_before_every_update','fp32_masters_and_moments_verified','one_E4_source_per_layer_per_minibatch_verified','all_lower_scale_codes_equal_current_E4_source_plus6','all_mantissa_scale_bytes_equal_E2M3','source_cache_cleared_before_every_optimizer_update','batchnorm_learned_during_training_eval_frozen']:
    check(summary[flag] is True, 'recorded ' + flag)

predictions = read_csv('test_prediction_counts.csv')
pridx = {(int(x['epoch']),int(x['weight_bits'])): x for x in predictions}
updates = read_csv('weight_updates.csv')
upidx = {(int(x['epoch']),int(x['weight_bits']),x['layer']): x for x in updates}
times = read_csv('epoch_times.csv')
timeidx = {int(x['epoch']): x for x in times}
check(len(pridx) == len(predictions) == 300, '300 prediction diagnostics')
check(len(upidx) == len(updates) == 1800, '1800 weight-update diagnostics')
check(len(timeidx) == len(times) == 100, '100 time rows')
best_score = -1
best_epoch = 0
for e in h['epochs']:
    n = e['epoch']
    check(all(all(equal_value(idx[(n,phase,m['weight_bits'])][k],v) for k,v in m.items()) for phase in ['train','test'] for m in e[phase]), f'e{n}: history metrics exact CSV')
    check(all(e['coverage'][phase]['complete_once'] is True and e['coverage'][phase]['n'] == count for phase,count in [('train',73257),('test',26032)]), f'e{n}: recorded complete train/test coverage')
    check(e['optimizer_steps'] == 573*n and e['test_optimizer_steps'] == 0, f'e{n}: step counts')
    check(math.isclose(e['lr_used'],0.001*0.1**sum(n > m for m in [50,75,90]),abs_tol=1e-16) and math.isclose(e['lr_next'],0.001*0.1**sum(n >= m for m in [50,75,90]),abs_tol=1e-16), f'e{n}: scheduler transition')
    gc = e['gradient_checks']
    check(gc['optimizer_steps_in_phase'] == gc['finite_gradient_checks'] == gc['forward_calls_E2M1'] == gc['forward_calls_E2M3'] == gc['forward_calls_E4M3'] == 573 and gc['detached_teacher_checks'] == 1146 and gc['no_extra_FP32_teacher'] is True and gc['cache_empty_before_all_optimizer_steps'] is True, f'e{n}: gradient/teacher/forward checks')
    for sc,batches,order,label in [(gc['shared_source_checks'],573,[8,6,4],'train'),(e['test_source_checks'],204,[4,6,8],'test')]:
        check(sc['source_triplets'] == batches and sc['high_encodings'] == sc['exponent_derivations'] == sc['mantissa_derivations'] == sc['mantissa_scale_copies_unchanged'] == batches*6 and sc['scale_offsets_plus6'] == batches*12 and sc['scale_exponent_offset'] == 6 and sc['scale_factor'] == 64 and sc['block_size'] == 16 and sc['training_order'] == order and sc['source_cache_cleared'] is True and sc['no_direct_lower_quantizer'] is True and sc['cache_cleared_before_adam'] is True, f'e{n}: {label} shared-source counters')
    check(all(e['adam_checks'][k] is True for k in ['all_active_adam_steps_correct','moments_fp32_finite','unused_bn_has_no_optimizer_state']) and e['batchnorm_counters_correct'] is True, f'e{n}: Adam/BN recorded checks')
    for bits in [4,6,8]:
        pred = pridx[(n,bits)]
        counts = [int(pred[f'predicted_class_{c}']) for c in range(10)]
        check(counts == e['test_prediction_counts'][str(bits)] and sum(counts) == 26032 and sum(x>0 for x in counts) == int(pred['classes_predicted']) and max(range(10),key=lambda c: counts[c]) == int(pred['dominant_predicted_class']) and abs(float(pred['dominant_prediction_fraction'])-max(counts)/26032)<1e-15, f'e{n}: predictions b{bits}')
    check(all(all(equal_value(upidx[(n,u['weight_bits'],u['layer'])][k],v) for k,v in u.items()) for u in e['weight_updates']), f'e{n}: weight updates CSV exact')
    check(all(equal_value(timeidx[n][k], e[k]) for k in ['epoch','train_seconds','test_seconds','checkpoint_seconds','other_seconds','total_epoch_seconds','first_epoch_in_process','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes']), f'e{n}: timing CSV exact')
    score = float(idx[(n,'test',8)]['top1_percent'])
    new_best = score > best_score
    if new_best:
        best_score = score
        best_epoch = n
    check(e['is_best_at_this_epoch'] == new_best and e['best_epoch_so_far'] == best_epoch, f'e{n}: strict first maximum bookkeeping')
check(best_epoch == selection['best']['epoch'] == r['selected_best_epoch'] == summary['best_epoch_E4M3_A32'] == 53, 'selected E4 best53')
check(selection['last']['epoch'] == r['selected_last_epoch'] == summary['last_epoch'] == 100, 'last100')

comparison=[]
for role,be,ae in [('last_epoch',100,100),('best_E4M3_test',42,53)]:
    for bits in [8,6,4]:
        a=idx[(ae,'test',bits)]; b=bidx[(be,'test',bits)]
        comparison.append({'role':role,'weight_bits':bits,'weight_format':a['weight_format'],'epoch_B16':ae,'epoch_B32':be,'B16_top1_percent':float(a['top1_percent']),'B32_top1_percent':float(b['top1_percent']),'delta_B16_minus_B32_pp':float(a['top1_percent'])-float(b['top1_percent']),'B16_correct1':int(a['correct1']),'B32_correct1':int(b['correct1']),'delta_correct1':int(a['correct1'])-int(b['correct1']),'test_n':26032,'B16_hard_CE':float(a['objective_loss']),'B32_hard_CE':float(b['objective_loss'])})
selected=read_csv('selected_metrics.csv')
check(len(selected)==6 and all(all(equal_value(s[k], v) for k,v in idx[(int(s['epoch']),'test',int(s['weight_bits']))].items()) for s in selected), 'selected metrics match exact source rows')
check(all(int(s['epoch']) == (53 if s['role']=='best_E4M3_test' else 100) for s in selected), 'all three modes same selected checkpoint')
check(selection['selected_on_official_test'] is True and selection['independent_test_estimate'] is False, 'selection test limitation recorded')

for folder in ['scale64_reference','pilot_reference']:
    manifest=read_json(folder+'/REFERENCE.json')
    for name,sha in manifest['files_sha256'].items():
        check(hashlib.sha256((P/folder/name).read_bytes()).hexdigest()==sha, folder+': SHA '+name)
present_package_files=[]
for name,sha in r['package_sha256'].items():
    if (P/name).is_file() and not name.endswith('.ipynb'):
        check(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha,'included package SHA '+name)
        present_package_files.append(name)
package_manifest=json.loads((PACKAGE/'PACKAGE_SHA256.json').read_text())
check(package_manifest==r['package_sha256'],'actual run package manifest equals delivered full100 package')
for name,sha in package_manifest.items():
    check(hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()==sha,'delivered package SHA '+name)
check(json.loads((PACKAGE/'PROTOCOL.json').read_text())==r['config'],'actual run config equals delivered protocol')
nb_name='MX_triple_group16_scale64_reference100_resultaten.ipynb'
expected_nb=json.loads((PACKAGE/nb_name).read_text())
for cell in expected_nb['cells']:
    if 'parameters' in cell.get('metadata',{}).get('tags',[]):
        cell['source']=['from pathlib import Path\n',f'RUN_DIR = Path({r["output_directory"]!r})\n','SAVE_FIGURES = False  # Zet op True om figuren in een nieuwe analysemap te bewaren.\n']
check(expected_nb==read_json(nb_name),'run notebook only has documented RUN_DIR/settings substitution')
check(all(not c.get('outputs') and c.get('execution_count') is None for c in expected_nb['cells']),'feedback notebook unexecuted')
unchanged_core=[]
for name,sha in r['package_sha256'].items():
    if '/' not in name and name.endswith('.py') and name not in ['run_train.py','notebook_support.py']:
        check(rp['package_sha256'].get(name)==sha,'same pilot core '+name)
        unchanged_core.append(name)

costs=read_json('group_size_costs.json')
check(costs==r['group_size_costs'],'cost report duplicated exact')
shapes=costs['layer_shapes']
geometries={}
for group in [16,32,64]:
    total_weights=blocks=padding=slots=0
    for shape in shapes.values():
        row=math.prod(shape[1:]); count=shape[0]; nb=count*math.ceil(row/group)
        total_weights+=count*row; blocks+=nb; slots+=nb*group; padding+=nb*group-count*row
    geom={'group_size':group,'n_weights':total_weights,'blocks':blocks,'padded_slots':padding,'stored_weight_slots':slots,'scale_bytes':blocks}
    for bits,fmt in [(8,'e4m3'),(6,'e2m3'),(4,'e2m1')]:
        geom[f'packed_{fmt}_bytes']=slots*bits//8+blocks
    reported=next(x for x in costs['summary'] if x['group_size']==group)
    check(all(reported[k]==v for k,v in geom.items()),f'independent B{group} geometry')
    geometries[group]=geom
for role,short,epoch in [('best_E4M3_test','best',53),('last_epoch','last',100)]:
    for bits in [8,6,4]:
        exp=r['mx_exports'][role][str(bits)]
        check(exp['epoch']==epoch and exp['decoded_weights_exact'] is True and exp['six_fp32_masters_absent'] is True and exp['saved_and_reloaded'] is True and exp['master_weights_read_to_derive'] is False and exp['mx_block_size']==16, f'{short} b{bits}: recorded exact exports')
        check(exp['scale_exponent_offset']==(0 if bits==8 else 6) and exp['scale_factor']==(1 if bits==8 else 64) and exp['derived_from_saved_source']==(bits<8) and exp['source_file_sha256']==r['mx_exports'][role]['8']['sha256'],f'{short} b{bits}: export source provenance')
        ers=read_csv(f'mx_export_layers_{bits}_{short}.csv')
        check(len(ers)==6 and {x['layer'] for x in ers}==set(shapes),f'{short} b{bits}: six-layer export')
        for row in ers:
            shape=shapes[row['layer']]; width=math.prod(shape[1:]); weights=shape[0]*width; nb=shape[0]*math.ceil(width/16); pad=nb*16-weights
            check(int(row['n_weights'])==weights and int(row['blocks'])==nb and int(row['block_size'])==16 and int(row['padded_slots'])==pad and int(row['packed_weight_and_scale_bytes'])==nb*16*bits//8+nb and int(row['fp32_weight_bytes'])==weights*4,f'{short} b{bits}: geometry {row["layer"]}')
            check(all(math.isfinite(float(row[k])) for k in ['mae','rmse','relative_l2','max_abs_error']),f'{short} b{bits}: finite errors {row["layer"]}')
check(not any(P.rglob('*.pt')), 'bundle omits binary checkpoints/exports; report audit only')

timing={}
for label,folder in [('B16',''),('B32','scale64_reference/'),('B16_pilot','pilot_reference/')]:
    ts=read_csv(folder+'epoch_times.csv'); rep=read_json(folder+'run_report.json')
    total=sum(float(x['total_epoch_seconds']) for x in ts)
    timing[label]={'epochs':len(ts),'epoch_sum_seconds':total,'session_wall_seconds':rep['session_wall_seconds'],'median_nonfirst_epoch_seconds':statistics.median(float(x['total_epoch_seconds']) for x in ts[1:])}
check(abs(timing['B16']['epoch_sum_seconds']-summary['measured_completed_epoch_seconds'])<1e-8,'timing sum matches summary')
timing['B16_vs_B32_delta_seconds']=timing['B16']['epoch_sum_seconds']-timing['B32']['epoch_sum_seconds']
timing['B16_vs_B32_delta_percent']=100*(timing['B16']['epoch_sum_seconds']/timing['B32']['epoch_sum_seconds']-1)
trajectory=[]
for epoch in range(1,101):
    for bits in [8,6,4]:
        a=idx[(epoch,'test',bits)];b=bidx[(epoch,'test',bits)]
        trajectory.append({'epoch':epoch,'weight_bits':bits,'B16_top1_percent':float(a['top1_percent']),'B32_top1_percent':float(b['top1_percent']),'delta_pp':float(a['top1_percent'])-float(b['top1_percent'])})
phases=[]
for start,end in [(1,5),(6,50),(51,75),(76,90),(91,100)]:
    for bits in [8,6,4]:
        tr=[x for x in trajectory if start<=x['epoch']<=end and x['weight_bits']==bits]
        phases.append({'first_epoch':start,'last_epoch':end,'weight_bits':bits,'B16_mean_top1':statistics.mean(x['B16_top1_percent'] for x in tr),'B32_mean_top1':statistics.mean(x['B32_top1_percent'] for x in tr),'mean_delta_pp':statistics.mean(x['delta_pp'] for x in tr),'B16_higher_epochs':sum(x['delta_pp']>0 for x in tr),'B16_lower_epochs':sum(x['delta_pp']<0 for x in tr)})
pilot_comparison=[]
for bits in [8,6,4]:
    a=idx[(5,'test',bits)];p=pidx[(5,'test',bits)];b=bidx[(5,'test',bits)]
    pilot_comparison.append({'weight_bits':bits,'B16_full100_epoch5':float(a['top1_percent']),'B16_pilot_epoch5':float(p['top1_percent']),'B32_full100_epoch5':float(b['top1_percent']),'B16_full_minus_pilot_pp':float(a['top1_percent'])-float(p['top1_percent'])})
same_epoch_selected=[]
for epoch in [42,53]:
    for bits in [8,6,4]:
        a=idx[(epoch,'test',bits)];b=bidx[(epoch,'test',bits)]
        same_epoch_selected.append({'epoch':epoch,'weight_bits':bits,'B16_top1_percent':float(a['top1_percent']),'B32_top1_percent':float(b['top1_percent']),'delta_pp':float(a['top1_percent'])-float(b['top1_percent'])})

os.environ['MPLCONFIGDIR']='/tmp/mx_b16_audit_matplotlib'
sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('group16_actual_notebook_support',P/'notebook_support.py')
nb=importlib.util.module_from_spec(spec);sys.modules[spec.name]=nb;spec.loader.exec_module(nb)
analysis=nb.load_run(P); ref=nb.load_scale64_reference(analysis)
nb_result={'metrics_safe':analysis.metrics_safe,'selection_safe':analysis.selection_safe,'prediction_safe':analysis.prediction_safe,'cost_report_verified':analysis.cost_report_verified,'reference_safe':ref.safe,'reference_compatible':ref.compatible,'issues':analysis.issues,'reference_issues':ref.issues}
check(all(nb_result[k] is True for k in ['metrics_safe','selection_safe','prediction_safe','cost_report_verified','reference_safe','reference_compatible']),'existing notebook loaders accept actual completed run')
check(not any(x['Niveau'] in ['FOUT','RUNFOUT'] for x in analysis.issues+ref.issues),'no notebook analysis errors')
after={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file()}
check(before==after,'all extracted input bytes unchanged')
check(source_sha==hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'original uploaded ZIP unchanged')

def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv('checkpoint_comparison.csv',comparison)
write_csv('trajectory_comparison.csv',trajectory)
write_csv('phase_comparison.csv',phases)
write_csv('pilot_comparison.csv',pilot_comparison)
write_csv('same_epoch_selected_diagnostic.csv',same_epoch_selected)
write_csv('group_costs_independent.csv',list(geometries.values()))
(OUT/'notebook_loader_actual.json').write_text(json.dumps(nb_result,indent=2)+'\n')
report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','source_bundle':SOURCE.name,'source_sha256':source_sha,'source_run':r['output_directory'].split('\\')[-1],'synthetic_fixture_only':False,'checks_count':len(checks),'failed_checks':[c for c in checks if not c['passed']],'selected_epoch_B16':53,'selected_epoch_B32':42,'comparison':comparison,'phase_comparison':phases,'pilot_comparison':pilot_comparison,'same_epoch_selected_diagnostic':same_epoch_selected,'timing':timing,'geometries':geometries,'config_differences_B16_B32':config_diff,'unchanged_numerical_modules_vs_pilot':unchanged_core,'included_package_files_sha256_verified':present_package_files,'notebook_loader':nb_result,'input_sha256':before,'limitations':['One fixed seed; CUDA backend is not deterministic.','Selection uses official test; selected checkpoint is not an independent test estimate.','Actual checkpoints, sample-order arrays and packed exports are omitted from this feedback ZIP. No independent model reevaluation or byte-level export reevaluation performed; exact CUDA checks are recorded by the run.','Time comparison is descriptive across separate laptop runs, not a controlled hardware/energy benchmark.','Group-cost counts are six-layer packed payload, including row padding, excluding auxiliary FP32 state and container overhead.','Per-mode best checkpoint scores are not used for either selected comparison.','Last-ten-epoch averages describe a correlated training trajectory, not ten independent repeated runs.'], 'checks':checks}
(OUT/'audit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','checks_count','failed_checks','comparison','timing']},indent=2))
if report['status']!='PASS':
    raise SystemExit(1)
