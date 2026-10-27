"""Read-only consistency audit of the actual B64 x64 pilot received 2026-10-27.

This reads reports only. Binary checkpoints, packed exports, and sample-order
arrays are absent from the feedback bundle and are not re-evaluated here.
"""
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
SOURCE = ROOT / 'upload/feedback_bundle(20261027-161453).zip'
P = ROOT / 'work/group64_pilot5_actual_20261027_161453'
PACKAGE = ROOT / 'output/mx_triple_group64_scale64_pilot5_v1'
OUT = ROOT / 'work/group64_pilot5_actual_audit'
OUT.mkdir(exist_ok=True)
sys.dont_write_bytecode = True


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(name):
    return json.loads((P / name).read_text(encoding='utf-8'))


def read_csv(name):
    with (P / name).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


before = {str(p.relative_to(P)): sha(p) for p in P.rglob('*') if p.is_file()}
source_sha = sha(SOURCE)
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
    check(all(math.isfinite(float(r[k])) for r in rows for k in ['top1_percent', 'top5_percent', 'objective_loss', 'lr']), label + ': finite metrics')
    check(all(0 <= int(r['correct1']) <= int(r['correct5']) <= int(r['n']) for r in rows), label + ': integer correct bounds')
    check(all(abs(float(r['top1_percent']) - 100*int(r['correct1'])/int(r['n'])) < 1e-7 for r in rows), label + ': top1 integer reconciliation')
    check(all(abs(float(r['count_top1_percent']) - 100*int(r['correct1'])/int(r['n'])) < 1e-12 for r in rows), label + ': exact count percentages')
    check(all(math.isclose(float(r['lr']), 0.001 * 0.1**sum(int(r['epoch']) > m for m in [50, 75, 90]), abs_tol=1e-16) for r in rows), label + ': LR schedule')
    return {(int(r['epoch']), r['phase'], int(r['weight_bits'])): r for r in rows}


r = read_json('run_report.json')
summary = read_json('training_summary.json')
selection = read_json('checkpoint_selection.json')
h = read_json('history.json')
idx = validate_metrics(read_csv('epoch_metrics.csv'), 5, 'B64')
reference_reports = {32: read_json('scale64_reference/run_report.json'), 16: read_json('group16_reference/run_report.json')}
reference_indexes = {32: validate_metrics(read_csv('scale64_reference/epoch_metrics.csv'), 100, 'B32 full100 reference'),
                     16: validate_metrics(read_csv('group16_reference/epoch_metrics.csv'), 100, 'B16 full100 reference')}

check(r['status'] == 'PASS' and r['command'] == 'run', 'real completed run PASS')
check(r['synthetic_fixture_only'] is False and summary['synthetic_fixture_only'] is False, 'not synthetic')
check(summary['completed_epochs'] == r['completed_epoch'] == 5, 'five complete epochs')
check(summary['pilot_completed'] is True and r['pilot_completed'] is True, 'pilot completion flags')
check(summary['full_training_completed'] is False and summary['full_100_epoch_training_completed'] is False, 'no full100 claim')
check(r['optimizer_steps'] == summary['optimizer_steps'] == 2865, '2865 optimizer steps')
check(r['training_forward_passes'] == summary['training_forward_passes'] == 8595, '8595 training passes')
check(summary['resumed'] is False and summary['new_initialization_not_previous_experiment_warmstart'] is True, 'fresh initialization')
check(summary == r['training_summary'] and selection == r['checkpoint_selection'], 'duplicated summary and selection match')
check(r['source'] == read_json('source_manifest.json'), 'source manifest matches')
dm = read_json('dataset_manifest.json')
check(dm['files'] == r['dataset_files'] and dm['selection'] == r['dataset_selection'] and dm['transforms'] == r['transforms'], 'dataset manifest matches')
check(set(r['dataset_files']['files']) == {'train_32x32.mat', 'test_32x32.mat'}, 'canonical dataset names only')
check(r['dataset_selection']['extra_used'] is False and r['dataset_selection']['held_out_validation_examples'] == 0, 'train-only no extra or validation')
check(r['source_unchanged'] is True and r['dataset_files_unchanged'] is True, 'recorded immutable source and data')
check(len(h['epochs']) == 5 and [e['epoch'] for e in h['epochs']] == list(range(1, 6)), 'history contiguous')
check(h['initial_model_sha256'] == r['initial_model_sha256'] == 'a8fcd20c40a204ba24cabfa36258f9501d55b9d4d4bdfb8ba742b8c827af127f', 'same recorded initial model')
config_diffs = {}
for block, rr in reference_reports.items():
    for key in ['execution_settings', 'environment', 'source', 'transforms', 'dataset_selection']:
        check(r[key] == rr[key], f'B64/B{block} identical ' + key)
    check(r['dataset_files']['files'] == rr['dataset_files']['files'], f'B64/B{block} dataset hashes equal')
    check(r['initial_model_sha256'] == rr['initial_model_sha256'], f'B64/B{block} same initial model')
    diff = {k: {'B64': r['config'].get(k), 'reference': rr['config'].get(k)}
            for k in set(r['config']) | set(rr['config']) if r['config'].get(k) != rr['config'].get(k)}
    expected = {'purpose', 'schema', 'mx_block_size', 'pilot_stop_epoch', 'reference_end_epoch', 'reporting_policy'}
    if block == 32:
        expected |= {'comparison_baseline_block_size', 'mx_group_size_research_variant'}
    check(set(diff) == expected, f'B64/B{block} only group size and stage protocol differences')
    config_diffs[block] = diff
check(r['config']['mx_block_size'] == 64 and r['config']['scale_exponent_offset'] == 6 and r['config']['scale_factor'] == 64 and r['config']['learned_shared_scale'] is False, 'B64 fixed64 contract')
check(all(r['config'][key] == 'float32' for key in ['master_dtype', 'gradient_dtype', 'adam_moments_dtype']), 'FP32 master gradient Adam contract')
for flag in ['gradient_checks_before_every_update', 'fp32_masters_and_moments_verified', 'one_E4_source_per_layer_per_minibatch_verified', 'all_lower_scale_codes_equal_current_E4_source_plus6', 'all_mantissa_scale_bytes_equal_E2M3', 'source_cache_cleared_before_every_optimizer_update', 'batchnorm_learned_during_training_eval_frozen']:
    check(summary[flag] is True, 'recorded ' + flag)

predictions = read_csv('test_prediction_counts.csv')
pridx = {(int(x['epoch']), int(x['weight_bits'])): x for x in predictions}
updates = read_csv('weight_updates.csv')
upidx = {(int(x['epoch']), int(x['weight_bits']), x['layer']): x for x in updates}
times = read_csv('epoch_times.csv')
timeidx = {int(x['epoch']): x for x in times}
check(len(pridx) == len(predictions) == 15, '15 prediction diagnostics')
check(len(upidx) == len(updates) == 90, '90 weight-update diagnostics')
check(len(timeidx) == len(times) == 5, 'five time rows')
best_score, best_epoch = -1, 0
for e in h['epochs']:
    n = e['epoch']
    check(all(all(equal_value(idx[(n, phase, m['weight_bits'])][k], v) for k, v in m.items()) for phase in ['train', 'test'] for m in e[phase]), f'e{n}: history metrics exact CSV')
    check(all(e['coverage'][phase]['complete_once'] is True and e['coverage'][phase]['n'] == count for phase, count in [('train', 73257), ('test', 26032)]), f'e{n}: recorded complete coverage')
    check(e['optimizer_steps'] == 573*n and e['test_optimizer_steps'] == 0, f'e{n}: step counts')
    check(e['lr_used'] == e['lr_next'] == 0.001, f'e{n}: unchanged pilot LR')
    gc = e['gradient_checks']
    check(gc['optimizer_steps_in_phase'] == gc['finite_gradient_checks'] == gc['forward_calls_E2M1'] == gc['forward_calls_E2M3'] == gc['forward_calls_E4M3'] == 573 and gc['detached_teacher_checks'] == 1146 and gc['no_extra_FP32_teacher'] is True and gc['cache_empty_before_all_optimizer_steps'] is True, f'e{n}: gradient teacher forward checks')
    for sc, batches, order, label in [(gc['shared_source_checks'], 573, [8, 6, 4], 'train'), (e['test_source_checks'], 204, [4, 6, 8], 'test')]:
        check(sc['source_triplets'] == batches and sc['high_encodings'] == sc['exponent_derivations'] == sc['mantissa_derivations'] == sc['mantissa_scale_copies_unchanged'] == batches*6 and sc['scale_offsets_plus6'] == batches*12 and sc['scale_exponent_offset'] == 6 and sc['scale_factor'] == 64 and sc['block_size'] == 64 and sc['training_order'] == order and sc['source_cache_cleared'] is True and sc['no_direct_lower_quantizer'] is True and sc['cache_cleared_before_adam'] is True, f'e{n}: {label} shared-source counters')
    check(all(e['adam_checks'][k] is True for k in ['all_active_adam_steps_correct', 'moments_fp32_finite', 'unused_bn_has_no_optimizer_state']) and e['batchnorm_counters_correct'] is True, f'e{n}: Adam and BN checks')
    for bits in [4, 6, 8]:
        pred = pridx[(n, bits)]
        counts = [int(pred[f'predicted_class_{c}']) for c in range(10)]
        check(counts == e['test_prediction_counts'][str(bits)] and sum(counts) == 26032 and sum(x > 0 for x in counts) == int(pred['classes_predicted']) and max(range(10), key=lambda c: counts[c]) == int(pred['dominant_predicted_class']) and abs(float(pred['dominant_prediction_fraction']) - max(counts)/26032) < 1e-15, f'e{n}: predictions b{bits}')
    check(all(all(equal_value(upidx[(n, u['weight_bits'], u['layer'])][k], v) for k, v in u.items()) for u in e['weight_updates']), f'e{n}: weight updates CSV exact')
    check(all(equal_value(timeidx[n][k], e[k]) for k in ['epoch', 'train_seconds', 'test_seconds', 'checkpoint_seconds', 'other_seconds', 'total_epoch_seconds', 'first_epoch_in_process', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']), f'e{n}: timing CSV exact')
    score = float(idx[(n, 'test', 8)]['top1_percent'])
    new_best = score > best_score
    if new_best:
        best_score, best_epoch = score, n
    check(e['is_best_at_this_epoch'] == new_best and e['best_epoch_so_far'] == best_epoch, f'e{n}: strict first maximum bookkeeping')
check(best_epoch == selection['best']['epoch'] == r['selected_best_epoch'] == summary['best_epoch_E4M3_A32'] == 5, 'selected E4 best5')
check(selection['last']['epoch'] == r['selected_last_epoch'] == summary['last_epoch'] == 5, 'last5')
check(selection['same_checkpoint'] is True and selection['last']['sha256'] == selection['best']['sha256'], 'best and last same checkpoint')
selected = read_csv('selected_metrics.csv')
check(len(selected) == 6 and all(all(equal_value(s[k], v) for k, v in idx[(int(s['epoch']), 'test', int(s['weight_bits']))].items()) for s in selected), 'selected metrics exact source rows')
check(all(int(s['epoch']) == 5 for s in selected), 'all selected modes from epoch5')
check(selection['selected_on_official_test'] is True and selection['independent_test_estimate'] is False, 'selection limitation recorded')

for folder in ['scale64_reference', 'group16_reference']:
    manifest = read_json(folder + '/REFERENCE.json')
    for name, expected_sha in manifest['files_sha256'].items():
        check(sha(P / folder / name) == expected_sha, folder + ': SHA ' + name)
package_manifest = json.loads((PACKAGE / 'PACKAGE_SHA256.json').read_text())
check(package_manifest == r['package_sha256'], 'actual run package manifest equals delivered pilot package')
for name, expected_sha in package_manifest.items():
    check(sha(PACKAGE / name) == expected_sha, 'delivered package SHA ' + name)
    if (P / name).is_file() and not name.endswith('.ipynb'):
        check(sha(P / name) == expected_sha, 'included package SHA ' + name)
check(json.loads((PACKAGE / 'PROTOCOL.json').read_text()) == r['config'], 'actual config equals delivered protocol')
nb_name = 'MX_triple_group64_scale64_pilot5_resultaten.ipynb'
expected_nb = json.loads((PACKAGE / nb_name).read_text())
for cell in expected_nb['cells']:
    if 'parameters' in cell.get('metadata', {}).get('tags', []):
        cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({r["output_directory"]!r})\n', 'SAVE_FIGURES = False  # Zet op True om figuren in een nieuwe analysemap te bewaren.\n']
check(expected_nb == read_json(nb_name), 'feedback notebook only parameter cell substitution')
check(all(not c.get('outputs') and c.get('execution_count') is None for c in expected_nb['cells']), 'feedback notebook unexecuted')

costs = read_json('group_size_costs.json')
check(costs == r['group_size_costs'], 'cost report duplicated exact')
shapes = costs['layer_shapes']
geometries = {}
for group in [16, 32, 64]:
    total_weights = blocks = padding = slots = 0
    for shape in shapes.values():
        row, count = math.prod(shape[1:]), shape[0]
        nb = count * math.ceil(row/group)
        total_weights += count*row; blocks += nb; slots += nb*group; padding += nb*group-count*row
    geom = {'group_size': group, 'n_weights': total_weights, 'blocks': blocks, 'padded_slots': padding, 'stored_weight_slots': slots, 'scale_bytes': blocks}
    for bits, fmt in [(8, 'e4m3'), (6, 'e2m3'), (4, 'e2m1')]:
        geom[f'packed_{fmt}_bytes'] = slots*bits//8 + blocks
    reported = next(x for x in costs['summary'] if x['group_size'] == group)
    check(all(reported[k] == v for k, v in geom.items()), f'independent B{group} geometry')
    geometries[group] = geom
for role, short in [('best_E4M3_test', 'best'), ('last_epoch', 'last')]:
    for bits in [8, 6, 4]:
        exp = r['mx_exports'][role][str(bits)]
        check(exp['epoch'] == 5 and exp['decoded_weights_exact'] is True and exp['six_fp32_masters_absent'] is True and exp['saved_and_reloaded'] is True and exp['master_weights_read_to_derive'] is False and exp['mx_block_size'] == 64, f'{short} b{bits}: recorded exact exports')
        check(exp['scale_exponent_offset'] == (0 if bits == 8 else 6) and exp['scale_factor'] == (1 if bits == 8 else 64) and exp['derived_from_saved_source'] == (bits < 8) and exp['source_file_sha256'] == r['mx_exports'][role]['8']['sha256'], f'{short} b{bits}: source provenance')
        ers = read_csv(f'mx_export_layers_{bits}_{short}.csv')
        check(len(ers) == 6 and {x['layer'] for x in ers} == set(shapes), f'{short} b{bits}: six-layer export')
        for row in ers:
            shape = shapes[row['layer']]; width = math.prod(shape[1:]); weights = shape[0]*width
            nb = shape[0]*math.ceil(width/64); pad = nb*64 - weights
            check(int(row['n_weights']) == weights and int(row['blocks']) == nb and int(row['block_size']) == 64 and int(row['padded_slots']) == pad and int(row['packed_weight_and_scale_bytes']) == nb*64*bits//8 + nb and int(row['fp32_weight_bytes']) == weights*4, f'{short} b{bits}: geometry {row["layer"]}')
            check(all(math.isfinite(float(row[k])) for k in ['mae', 'rmse', 'relative_l2', 'max_abs_error']), f'{short} b{bits}: finite errors {row["layer"]}')
check(not any(P.rglob('*.pt')), 'bundle omits binary checkpoints and exports; report audit only')

all_indexes = {64: idx, **reference_indexes}
best_within5 = {b: max(range(1, 6), key=lambda e: float(index[(e, 'test', 8)]['top1_percent'])) for b, index in all_indexes.items()}
comparison = []
trajectory = []
for epoch in range(1, 6):
    for bits in [8, 6, 4]:
        a = idx[(epoch, 'test', bits)]
        row = {'epoch': epoch, 'weight_bits': bits, 'weight_format': a['weight_format'], 'B64_top1_percent': float(a['top1_percent']), 'B64_correct1': int(a['correct1'])}
        for b in [16, 32]:
            ref = reference_indexes[b][(epoch, 'test', bits)]
            row.update({f'B{b}_top1_percent': float(ref['top1_percent']), f'B{b}_correct1': int(ref['correct1']), f'B64_minus_B{b}_pp': float(a['top1_percent']) - float(ref['top1_percent']), f'B64_minus_B{b}_correct': int(a['correct1']) - int(ref['correct1'])})
        trajectory.append(row)
for role in ['epoch5', 'best_E4M3_within_first5']:
    for bits in [8, 6, 4]:
        es = {b: 5 if role == 'epoch5' else best_within5[b] for b in [64, 16, 32]}
        a = idx[(es[64], 'test', bits)]
        row = {'role': role, 'weight_bits': bits, 'weight_format': a['weight_format'], 'test_n': 26032}
        for b in [64, 16, 32]:
            item = all_indexes[b][(es[b], 'test', bits)]
            row.update({f'epoch_B{b}': es[b], f'B{b}_top1_percent': float(item['top1_percent']), f'B{b}_correct1': int(item['correct1']), f'B{b}_hard_CE': float(item['objective_loss'])})
        for b in [16, 32]:
            row[f'B64_minus_B{b}_pp'] = row['B64_top1_percent'] - row[f'B{b}_top1_percent']
            row[f'B64_minus_B{b}_correct'] = row['B64_correct1'] - row[f'B{b}_correct1']
        comparison.append(row)

timing = {}
for block, folder in [(64, ''), (32, 'scale64_reference/'), (16, 'group16_reference/')]:
    ts = read_csv(folder + 'epoch_times.csv')[:5]
    total = sum(float(x['total_epoch_seconds']) for x in ts)
    median = statistics.median(float(x['total_epoch_seconds']) for x in ts[1:])
    timing[block] = {'epochs': 5, 'epoch_sum_seconds': total, 'median_nonfirst_epoch_seconds': median, 'first_epoch_seconds': float(ts[0]['total_epoch_seconds'])}
    if block == 64:
        timing[block]['session_wall_seconds'] = r['session_wall_seconds']
        timing[block]['estimated_100_epoch_seconds'] = timing[block]['first_epoch_seconds'] + 99*median
estimate = read_json('timing_estimate.json')
check(abs(timing[64]['epoch_sum_seconds'] - summary['measured_completed_epoch_seconds']) < 1e-8, 'timing sum matches summary')
check(abs(timing[64]['estimated_100_epoch_seconds'] - estimate['estimated_100_epoch_seconds']) < 1e-8, 'fresh100 projection recomputed')
check(estimate['measured_100_epoch_seconds'] is None, 'projected100 not measured100')
for b in [16, 32]:
    timing[64][f'epoch_sum_delta_vs_B{b}_seconds'] = timing[64]['epoch_sum_seconds'] - timing[b]['epoch_sum_seconds']
    timing[64][f'epoch_sum_delta_vs_B{b}_percent'] = 100*(timing[64]['epoch_sum_seconds']/timing[b]['epoch_sum_seconds'] - 1)

trajectory_summary = []
for bits in [8, 6, 4]:
    tr = [x for x in trajectory if x['weight_bits'] == bits]
    for b in [16, 32]:
        trajectory_summary.append({'weight_bits': bits, 'reference_group_size': b, 'B64_higher_epochs': sum(x[f'B64_minus_B{b}_pp'] > 0 for x in tr), 'B64_lower_epochs': sum(x[f'B64_minus_B{b}_pp'] < 0 for x in tr), 'mean_delta_first5_pp': statistics.mean(x[f'B64_minus_B{b}_pp'] for x in tr)})

os.environ['MPLCONFIGDIR'] = '/tmp/mx_b64_pilot_audit_matplotlib'
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('group64_actual_notebook_support', P / 'notebook_support.py')
nb = importlib.util.module_from_spec(spec); sys.modules[spec.name] = nb; spec.loader.exec_module(nb)
analysis = nb.load_run(P)
refs = nb.load_group_references(analysis)
nb_result = {'metrics_safe': analysis.metrics_safe, 'selection_safe': analysis.selection_safe, 'prediction_safe': analysis.prediction_safe, 'cost_report_verified': analysis.cost_report_verified, 'pilot_complete': analysis.pilot_complete, 'issues': analysis.issues, 'references': [{'group_size': x.group_size, 'safe': x.safe, 'compatible': x.compatible, 'issues': x.issues} for x in refs]}
check(all(nb_result[k] is True for k in ['metrics_safe', 'selection_safe', 'prediction_safe', 'cost_report_verified', 'pilot_complete']), 'existing notebook loader accepts completed actual pilot')
check(all(x.safe and x.compatible for x in refs), 'both notebook references compatible')
check(not any(x['Niveau'] in ['FOUT', 'RUNFOUT'] for x in analysis.issues), 'no notebook analysis errors')
after = {str(p.relative_to(P)): sha(p) for p in P.rglob('*') if p.is_file()}
check(before == after, 'all extracted input bytes unchanged')
check(source_sha == sha(SOURCE), 'original uploaded ZIP unchanged')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


write_csv('checkpoint_comparison.csv', comparison)
write_csv('trajectory_comparison.csv', trajectory)
write_csv('trajectory_summary.csv', trajectory_summary)
write_csv('group_costs_independent.csv', list(geometries.values()))
(OUT / 'notebook_loader_actual.json').write_text(json.dumps(nb_result, indent=2) + '\n')
report = {'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL', 'source_bundle': SOURCE.name, 'source_sha256': source_sha, 'source_run': r['output_directory'].split('\\')[-1], 'synthetic_fixture_only': False, 'checks_count': len(checks), 'failed_checks': [c for c in checks if not c['passed']], 'best_E4_epochs_within_first5': best_within5, 'comparison': comparison, 'trajectory_summary': trajectory_summary, 'timing': timing, 'geometries': geometries, 'config_differences': config_diffs, 'notebook_loader': nb_result, 'input_sha256': before, 'limitations': ['One fixed seed and nondeterministic CUDA; five epochs are a pilot, not final100 evidence.', 'Reference rows come from first5 of each full100 run, not their global E4-best checkpoints.', 'Actual checkpoints, sample-order arrays and packed exports are omitted. No model or binary export reevaluation; exact CUDA checks are recorded by the run.', 'Timing comparisons are descriptive across separate laptop runs, not controlled native-hardware claims.', 'Group-cost counts cover six-layer packed payload including row padding, excluding FP32 auxiliary state and container overhead.'], 'checks': checks}
(OUT / 'audit_summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['status', 'checks_count', 'failed_checks', 'best_E4_epochs_within_first5', 'comparison', 'trajectory_summary', 'timing']}, indent=2))
if report['status'] != 'PASS':
    raise SystemExit(1)
