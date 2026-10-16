"""CPU checks against real original model modules with synthetic CIFAR fixtures.

Requires a locally created fixture_reference 3-epoch run. Never use research
checkpoints here; test files are created in a NEW --work-root.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse, ast, importlib, json, shutil, subprocess, sys
p = argparse.ArgumentParser()
p.add_argument('--reference-dir', type=Path, required=True)
p.add_argument('--fixture-data', type=Path, required=True)
p.add_argument('--fixture-run-root', type=Path, required=True)
p.add_argument('--fixture-anchor', type=Path, required=True)
p.add_argument('--work-root', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(a.reference_dir.resolve()/'local_validation'))
import fixture_reference as fixture
sys.path.insert(0, str(root))
import select_checkpoints as select
ref = fixture.ref
w = a.work_root.resolve()
w.mkdir(parents=True, exist_ok=False)
checks = {}

args = SimpleNamespace(device='cpu', data_root=a.fixture_data)
context_out = w / 'context'
context_out.mkdir()
ctx = ref.make_context(args, context_out, {}, lambda _: None)
core, torch = ctx['core'], ctx['torch']
anchor = core.load_verified(a.fixture_anchor)
entry = {'epoch':3, 'path':str(a.fixture_anchor), 'sha256':select.sha256(a.fixture_anchor)}
select.restore_model(ctx, ref, anchor, entry)
loader, tf = select.prepare_test_loader(ctx, ref)
ours, details = select.evaluate_bits(ctx, ref, loader, 3, select.BITS)
# Extract the actual forward() function from the immutable supplied train.py.
# Its .cuda() transfer is changed to an identity ONLY during this CPU-only test.
tree = ast.parse((ctx['source']/'train.py').read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
utils = importlib.import_module('utils')
namespace = {'torch':torch, 'args':SimpleNamespace(bit_width_list='1,2,4,8,32', print_freq=20),
             'AverageMeter':utils.AverageMeter, 'accuracy':utils.accuracy}
exec(compile(ast.Module(body=[function],type_ignores=[]), '<original train.py forward>', 'exec'), namespace)
ctx['model'].eval()
with patch.object(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self):
    expected_loss, expected_p1, expected_p5 = namespace['forward'](
        ((x,y) for x,y,ids in loader), ctx['model'], torch.nn.CrossEntropyLoss(), None, 2, False)
checks['exact_loss_and_accuracies_vs_original_forward'] = all(
    r['hard_ce']==lo and r['top1_percent']==p1 and r['top5_percent']==p5
    for r,lo,p1,p5 in zip(ours, expected_loss, expected_p1, expected_p5))
w32, _ = select.evaluate_bits(ctx, ref, loader, 3, [32])
checks['w32_only_equals_w32_in_full_loop_including_logit_hash'] = w32[0] == ours[-1]
checks['model_and_bn_unchanged'] = not core.differences(core.cpu_copy(ctx['model'].state_dict()),anchor['model'])
checks['no_gradients'] = all(t.grad is None for t in ctx['model'].parameters())
records = [{'epoch':e,'row':{'top1_percent':v}} for e,v in [(3,94.0),(1,94.0),(2,93.0)]]
checks['tie_chooses_earliest_epoch'] = select.best_epoch(records)['epoch']==1
records[0]['row']['top1_percent'] = 94.01
checks['strictly_greater_replaces_previous_best'] = select.best_epoch(records)['epoch']==3
try:
    bad = dict(anchor, trace=dict(anchor['trace'], initial_model_sha256='0'*64))
    select.validate_payload(bad, anchor, 3, core, ref.support)
    checks['reject_other_trajectory'] = False
except RuntimeError:
    checks['reject_other_trajectory'] = True
bad_path = w/'corrupt.pt'
shutil.copy2(a.fixture_anchor, bad_path)
shutil.copy2(a.fixture_anchor.with_suffix('.pt.sha256'),bad_path.with_suffix('.pt.sha256'))
with bad_path.open('ab') as f:f.write(b'corrupt')
try:
    core.load_verified(bad_path)
    checks['reject_bad_checksum'] = False
except RuntimeError:checks['reject_bad_checksum'] = True
sample = w/'sample.json'
select.seal_json(sample, {'number':1})
bad=select.read_json(sample);bad['number']=2;sample.write_text(json.dumps(bad))
try:
    select.read_sealed(sample)
    checks['reject_edited_cache'] = False
except RuntimeError:checks['reject_edited_cache'] = True
try:
    select.new_json(sample, {'number':3})
    checks['never_overwrite_result'] = False
except RuntimeError:checks['never_overwrite_result'] = True
# Missing epoch rejection without evaluating any test image.
missing = w/'missing'
for e in (1,3):
    src=a.fixture_anchor.parent/f'epoch_{e:05d}.pt'
    dst=missing/'session'/'checkpoints'/src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst);shutil.copy2(src.with_suffix('.pt.sha256'), dst.with_suffix('.pt.sha256'))
inv=w/'missing_report'; inv.mkdir()
try:
    select.inventory_checkpoints(missing, (missing/'session/checkpoints/epoch_00003.pt').resolve(),
                                 anchor, ref, core, inv, lambda _:None)
    checks['missing_epoch_blocks_selection'] = False
except RuntimeError:
    checks['missing_epoch_blocks_selection'] = select.read_json(inv/'inventory.json')['missing_epochs']==[2]
# Duplicate byte copies are reported, not silently double-counted.
duplicates=w/'duplicates';duplicates.mkdir()
for label in ['session','copy']:
    (duplicates/label/'checkpoints').mkdir(parents=True)
    for src in a.fixture_anchor.parent.glob('epoch_*.pt*'):
        shutil.copy2(src,duplicates/label/'checkpoints'/src.name)
inv=w/'duplicate_report';inv.mkdir()
entries=select.inventory_checkpoints(duplicates, (duplicates/'session/checkpoints/epoch_00003.pt').resolve(),
                                   anchor, ref, core, inv, lambda _:None)
checks['identical_duplicates_reported_once'] = len(entries)==3 and len(select.read_json(inv/'inventory.json')['identical_duplicates'])==3
# Run full, paused/resumed and cache-only sweeps in SEPARATE processes.
original_hashes={str(f):select.sha256(f) for f in a.fixture_run_root.glob('*/checkpoints/epoch_*.pt*')}

def run(label, output, limit):
    command=[sys.executable,'-B',str(root/'local_validation/fixture_selection.py'),'sweep',
             '--reference-dir',str(a.reference_dir),'--run-root',str(a.fixture_run_root),
             '--final-checkpoint',str(a.fixture_anchor),'--data-root',str(a.fixture_data),
             '--output-dir',str(output),'--device','cpu','--max-checkpoints',str(limit)]
    with (w/(label+'.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError((w/(label+'.log')).read_text())
    result_path=sorted((output/'sessions').glob('*/selection_report.json'))[-1]
    return select.read_json(result_path),result_path.parent
full, full_path = run('full',w/'full',0)
part, part_path = run('partial',w/'split',1)
rest, rest_path = run('continued',w/'split',0)
cache, cache_path = run('cached',w/'split',0)
checks['bounded_sweep_pauses'] = part['status']=='PAUSED' and part['completed_checkpoint_count']==1
checks['resume_evaluates_only_remaining'] = rest['newly_evaluated_epochs']==[2,3]
checks['cache_only_rerun_no_new_tests'] = cache['newly_evaluated_epochs']==[]
checks['full_and_resumed_same_selection_and_metrics'] = all(full[k]==rest[k]==cache[k] for k in [
    'selected_epoch','selected_metrics','fixed_final_metrics','github_comparison'])
checks['full_and_resumed_identical_score_csv']=(full_path/'checkpoint_scores.csv').read_bytes()==(rest_path/'checkpoint_scores.csv').read_bytes()
checks['w32_allbits_checks_pass'] = all(r['status']=='PASS' for r in [full,rest,cache])
checks['fixtures_not_presented_as_research_results']=all(r['valid_research_result'] is False for r in [full,part,rest,cache])
checks['original_checkpoints_unchanged']=original_hashes=={f:select.sha256(Path(f)) for f in original_hashes}
checks['no_python_warnings']=all(not r['warnings'] for r in [full,part,rest,cache])
# The lock prevents a second process from opening the same evaluation directory.
lock_root=w/'lock_test';lock_root.mkdir()
with select.exclusive_lock(lock_root):
    code="import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); import select_checkpoints as s;\nwith s.exclusive_lock(Path(sys.argv[2])): print('unexpected')"
    r=subprocess.run([sys.executable,'-B','-c',code,str(root),str(lock_root)],capture_output=True,text=True)
checks['second_process_lock_rejected']=r.returncode!=0 and 'Er draait al' in r.stderr
stop_session=w/'stop_test';stop_session.mkdir();select.new_json(stop_session/'invocation_plan.json',{})
select.request_stop(stop_session)
checks['stop_request_file_created']=(stop_session/'STOP_AFTER_CHECKPOINT.txt').is_file()
select.new_json(stop_session/'selection_report.json',{'status':'PAUSED'})
try:
    select.request_stop(stop_session);checks['closed_session_stop_rejected']=False
except RuntimeError:checks['closed_session_stop_rejected']=True
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'environment':ctx['contract']['environment'],
        'fixture_only':True,'CUDA_tested_here':False,'Windows_tested_here':False,
        'real_CIFAR10_or_user_trained_checkpoints_available_here':False,
        'addon_hashes':select.verify_addon()}
select.new_json(w/'validation_result.json',result)
print(json.dumps(result,indent=2),flush=True)
require=lambda x: (_ for _ in ()).throw(RuntimeError('A local check failed')) if not x else None
require(all(checks.values()))
