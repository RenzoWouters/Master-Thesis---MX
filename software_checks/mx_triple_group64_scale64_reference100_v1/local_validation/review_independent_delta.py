"""Independent static review; no model, dataset, runtime or notebook execution."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import math

ROOT = Path('/workspace/scratch/4619498905be')
FULL = ROOT / 'output/mx_triple_group64_scale64_reference100_v1'
PILOT = ROOT / 'output/mx_triple_group64_scale64_pilot5_v1'
B16_FULL = ROOT / 'work/group64_preparation_sources/mx_triple_group16_scale64_reference100_v1'
ACTUAL = ROOT / 'work/group64_pilot5_actual_20261027_161453'
OUT = ROOT / 'work/group64_full100_independent_delta_audit.json'
CORE = ['trunc_support.py', 'trunc_adapter.py', 'mx_e4m3.py', 'mx_format.py', 'mx_e2m1.py',
        'joint_adapter.py', 'joint_support.py', 'mx_adapter.py', 'posthoc_truncation.py',
        'resume_core.py', 'smoke_support.py', 'svhn_support.py', 'group_config.py',
        'group_costs.py', 'test_mx.py', 'test_e4m3.py', 'test_group_size.py',
        'test_joint.py', 'test_trunc_training.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8-sig'), str(path))


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(module, name):
    return next(x for x in module.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name)


def function_map(module):
    return {x.name: x for x in module.body if isinstance(x, ast.FunctionDef)}


def literal_dict(node):
    return {ast.literal_eval(k): v for k, v in zip(node.keys, node.values)}


checks = []
def check(name, ok, detail=None):
    checks.append({'name': name, 'passed': bool(ok), **({'detail': detail} if detail is not None else {})})


for name in CORE:
    check('unchanged_successful_pilot_core_' + name, (FULL / name).read_bytes() == (PILOT / name).read_bytes())

full_tree, pilot_tree, b16_tree = tree(FULL / 'run_train.py'), tree(PILOT / 'run_train.py'), tree(B16_FULL / 'run_train.py')
full_source = (FULL / 'run_train.py').read_text()
check('main_training_loop_AST_identical_to_proven_B16_full100', dump(function(full_tree, 'run_epochs')) == dump(function(b16_tree, 'run_epochs')))
check('immutability_checks_AST_identical_to_B16_full100', dump(function(full_tree, 'verify_unchanged')) == dump(function(b16_tree, 'verify_unchanged')))


class NormalizeDocumentedB64Delta(ast.NodeTransformer):
    """Normalize ONLY observed B16→B64 identity/guard/reference differences."""
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            value = node.value
            for old, new in [('group64', 'group16'), ('group size64', 'group size16'),
                             ('groepsgrootte64', 'groepsgrootte16'),
                             ('GROEP64', 'GROEP16'), ('groepen van64', 'groepen van16')]:
                value = value.replace(old, new)
            node.value = value
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if isinstance(node.left, ast.Name) and node.left.id == 'BLOCK_SIZE':
            assert len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant)
            assert node.comparators[0].value == 64
            node.comparators[0].value = 16
        return node

    def visit_List(self, node):
        # Exactly two added copy lists: export_results and feedback_bundle.
        if any(isinstance(x, ast.Constant) and x.value == 'group16_reference' for x in node.elts):
            assert [x.value for x in node.elts] == ['scale64_reference', 'group16_reference', 'pilot_reference']
            node.elts = [x for x in node.elts if x.value != 'group16_reference']
        return self.generic_visit(node)


normalized = NormalizeDocumentedB64Delta().visit(copy.deepcopy(full_tree))
check('whole_runner_AST_matches_B16_full100_after_only_documented_differences', dump(normalized) == dump(b16_tree),
      'Normalized B64 experiment text, BLOCK_SIZE guard 64→16, and extra group16_reference list entry; no scale/bit operations normalized.')

full_config = literal_dict(next(x for x in function(full_tree, 'config').body if isinstance(x, ast.Return)).value)
pilot_config = literal_dict(next(x for x in function(pilot_tree, 'config').body if isinstance(x, ast.Return)).value)
config_diffs = {k: {'pilot': ast.unparse(pilot_config[k]) if k in pilot_config else None,
                    'full100': ast.unparse(full_config[k]) if k in full_config else None}
                for k in sorted(set(full_config) | set(pilot_config))
                if k not in full_config or k not in pilot_config or dump(full_config[k]) != dump(pilot_config[k])}
check('protocol_changes_only_stage_and_reporting', set(config_diffs) == {'schema', 'purpose', 'pilot_stop_epoch', 'reference_end_epoch', 'reporting_policy'})
check('full100_config_endpoint', ast.literal_eval(full_config['planned_epochs']) == ast.literal_eval(full_config['reference_end_epoch']) == 100 and 'pilot_stop_epoch' not in full_config)
check('original_LR_schedule', ast.literal_eval(full_config['scheduler_milestones']) == [50, 75, 90] and ast.literal_eval(full_config['lr']) == 0.001)
check('fresh_initialization_unchanged', dump(full_config['initialization_policy']) == dump(pilot_config['initialization_policy']))
check('A32_FP32_and_fixed64', all(ast.literal_eval(full_config[k]) == v for k, v in {'activation_bits': 32, 'master_dtype': 'float32', 'gradient_dtype': 'float32', 'adam_moments_dtype': 'float32', 'scale_factor': 64, 'scale_exponent_offset': 6, 'learned_shared_scale': False}.items()))
check('canonical_dataset_names_retained', "{'train_32x32.mat', 'test_32x32.mat'}" in full_source and '32x64.mat' not in full_source)
check('only_resume_takes_checkpoint', "(args.command == 'resume') != (args.checkpoint is not None)" in full_source)
check('restore_requires_current_full_contract', 'core.restore_checkpoint(state, model, optimizer, scheduler, {},' in full_source and 'device, ctx.contract)' in full_source)
check('completed100_checkpoint_rejected', "0 < state['completed_epoch'] < stop_after" in full_source)
check('default_and_guard_stop100', "parser.add_argument('--stop-after-epoch', type=int, default=100)" in full_source and 'if args.stop_after_epoch != 100:' in full_source)
check('own_full100_report_identity', "'schema': 'mx-triple-group64-scale64-reference100-report-v1'" in full_source)
check('own_full100_completion_identity', "'schema': 'mx-triple-group64-scale64-reference100-completion-v1'" in full_source)
check('own_full100_result_root', "Path('mx_runs/mx_triple_group64_scale64_reference100')" in full_source)
check('full_completion_only_at100', "full_completed = last['epoch'] == c['reference_end_epoch']" in full_source)
check('measured100_only_when_completed', "estimate['measured_100_epoch_seconds'] = elapsed if full_completed else None" in full_source)

shapes_node = next(x.value for x in tree(FULL / 'group_costs.py').body if isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LAYER_SHAPES' for t in x.targets))
shapes = ast.literal_eval(shapes_node)
weights = sum(math.prod(s) for s in shapes.values())
blocks = sum(s[0] * math.ceil(math.prod(s[1:]) / 64) for s in shapes.values())
geometry = {'real_weights': weights, 'blocks': blocks, 'padded_slots': blocks * 64 - weights,
            'payload_bytes': {bits: blocks * (64 * bits // 8 + 1) for bits in [8, 6, 4]}}
check('B64_geometry_retained', geometry == {'real_weights': 129472, 'blocks': 2160, 'padded_slots': 8768, 'payload_bytes': {8: 140400, 6: 105840, 4: 71280}})

for folder in ['scale64_reference', 'group16_reference', 'pilot_reference']:
    manifest = json.loads((FULL / folder / 'REFERENCE.json').read_text())
    check(folder + '_hashes_match', all(sha(FULL / folder / n) == h for n, h in manifest['files_sha256'].items()))
    if folder != 'pilot_reference':
        check(folder + '_full100_comparison', manifest['comparison_endpoint'] == 100)
        check(folder + '_historical_raw_data_unchanged', all((FULL / folder / n).read_bytes() == (PILOT / folder / n).read_bytes() for n in manifest['files_sha256']))
    else:
        check('actual_pilot_reference_unchanged', all((FULL / folder / n).read_bytes() == (ACTUAL / n).read_bytes() for n in manifest['files_sha256']))
        check('actual_pilot_reference_correct_input_sha', manifest['source_bundle_sha256'] == '2e47f6ea1c26d894eaa91b82b277c38f9297fe8c5a7773f48b4040643b4bbda8')

nb_path = FULL / 'MX_triple_group64_scale64_reference100_resultaten.ipynb'
notebook = json.loads(nb_path.read_text())
nb_source = (FULL / 'notebook_support.py').read_text()
nb_tree = tree(FULL / 'notebook_support.py')
pilot_nb_tree = tree(PILOT / 'notebook_support.py')
check('notebook_unexecuted', all(not c.get('outputs') and c.get('execution_count') is None for c in notebook['cells']))
check('notebook_correct_full100_metadata', all(notebook['metadata'].get(k) == v for k, v in {'experiment': 'mx_triple_group64_scale64_reference100_v1', 'mx_block_size': 64, 'scale_factor': 64, 'activation_bits': 32, 'reference_end_epoch': 100, 'read_only_analysis': True}.items()) and 'pilot_stop_epoch' not in notebook['metadata'])
check('notebook_selection_validation_unchanged', dump(function(nb_tree, '_validate_selection')) == dump(function(pilot_nb_tree, '_validate_selection')))
check('notebook_current_complete_requires_all100', "and self.last_epoch == 100" in nb_source and "== list(range(1, 101)) for fmt in FORMATS.values()" in nb_source)
check('notebook_reference_requires_full_history', 'len(probe.metrics) != 600 or len(probe.test) != 300' in nb_source)
check('notebook_reference_global_best_verified', 'expected_best = 53 if group_size == 16 else 42' in nb_source and 'probe.observed_best_epoch != expected_best' in nb_source)
check('notebook_full_endpoint_guard', 'endpoint = 100 if run.reference_complete else None' in nb_source)
check('notebook_partial_common_range', 'through = min(100, run.last_epoch)' in nb_source and 'epoch = ref.best_epoch(through) if select_best else endpoint' in nb_source)
check('notebook_all_modes_from_selected_epoch', 'current = run.metric(current_epoch, FORMATS[bit])' in nb_source and 'metric = ref.metric(epoch, bit)' in nb_source)
check('notebook_no_first5_filter', 'full_metrics.epoch.between(1, 5)' not in nb_source and 'full_metrics.epoch.between(1, 100)' in nb_source)
check('notebook_no_torch_or_training_import', not any(isinstance(x, (ast.Import, ast.ImportFrom)) and ('torch' in ast.unparse(x) or 'run_train' in ast.unparse(x)) for x in ast.walk(nb_tree)))
check('notebook_costs_still_B64', 'expected_group_size=64' in nb_source and 'active_group_size=64' in nb_source)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
check('notebook_code_cells_parse', True)

report = {'status': 'PASS' if all(x['passed'] for x in checks) else 'FAIL',
          'scope': 'Independent static review only; no model, training, notebook or dataset execution.',
          'checks_count': len(checks), 'failed_checks': [x for x in checks if not x['passed']],
          'checks': checks, 'protocol_differences_vs_successful_B64_pilot': config_diffs,
          'unchanged_core_files': CORE, 'B64_geometry': geometry,
          'production_sha256': {n: sha(FULL / n) for n in [*CORE, 'run_train.py', 'notebook_support.py', nb_path.name]},
          'known_limits': ['Root documentation and package manifests are assembled separately.', 'Runtime checks belong to the runner agent; this review adds no new training.', 'The conditional per-mode selection statements are statically reviewed; execution fixtures belong to the notebook validation.']}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['status', 'checks_count', 'failed_checks', 'protocol_differences_vs_successful_B64_pilot', 'B64_geometry']}, indent=2))
if report['status'] != 'PASS':
    raise SystemExit(1)
