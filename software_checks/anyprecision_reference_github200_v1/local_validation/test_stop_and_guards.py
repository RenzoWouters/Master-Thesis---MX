"""Run after test_integration. More local-only synthetic CPU checks."""
from pathlib import Path
import argparse, json, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import resume_core as core
p = argparse.ArgumentParser()
p.add_argument('--work-root', type=Path, required=True)
a = p.parse_args()
w = a.work_root.resolve()
data = w / 'fixture_data'
out = w / 'stop_request_test'
assert not out.exists()
cmd = [sys.executable, '-B', str(ROOT/'local_validation/fixture_reference.py'), 'train',
       '--session-epochs', '3', '--device', 'cpu', '--data-root', str(data), '--output-root', str(out)]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
requested = False
with (w/'stop_request_test.log').open('x') as f:
    for line in process.stdout:
        f.write(line)
        f.flush()
        if 'epoch 1/3, batch 1/3:' in line and not requested:
            session = next(out.iterdir())
            r = subprocess.run([sys.executable, '-B', str(ROOT/'run_reference.py'),
                                'request-stop', '--session-dir', str(session)],
                               capture_output=True, text=True)
            f.write(r.stdout+r.stderr)
            assert r.returncode == 0
            requested = True
assert process.wait() == 0
session = next(out.iterdir())
r = json.loads((session/'reference_report.json').read_text())
a_session = next((w/'uninterrupted').iterdir())
ckpt = session/'checkpoints/epoch_00001.pt'
saved = core.load_verified(ckpt)
checks = {'stop_request_sent': requested,
          'stop_after_epoch1': r['status']=='PAUSED' and r['completed_epoch']==1,
          'stop_request_recorded': r['stopped_by_request'],
          'stop_did_not_change_checkpoint': not core.differences(saved,core.load_verified(a_session/'checkpoints/epoch_00001.pt'))}
oldsha = core.file_sha256(ckpt)
try:
    core.atomic_save(ckpt,saved)
    checks['refuses_overwrite']=False
except RuntimeError:
    checks['refuses_overwrite'] = core.file_sha256(ckpt)==oldsha
# New corrupt copy, never damage the tested source checkpoint.
p = w/'corrupt_test.pt'
p.write_bytes(ckpt.read_bytes())
p.with_suffix('.pt.sha256').write_text(oldsha+'\n')
with p.open('r+b') as f:
    f.seek(-1,2)
    b=f.read(1)
    f.seek(-1,2)
    f.write(bytes([b[0]^1]))
try:
    core.load_verified(p)
    checks['rejects_corruption']=False
except RuntimeError as e:
    checks['rejects_corruption']='checksum' in str(e)
# Contract rejection is checked before any model/scheduler/optimizer mutation.
try:
    core.restore_checkpoint(saved,None,None,None,{},None,{'different_contract':True})
    checks['rejects_contract_change']=False
except RuntimeError as e:
    checks['rejects_contract_change']='verschilt' in str(e)
# Test real milestones independently of tiny fixture milestones.
import torch
parameter=torch.nn.Parameter(torch.tensor([1.0]))
opt=torch.optim.Adam([parameter],lr=0.001,weight_decay=0.0)
sched=torch.optim.lr_scheduler.MultiStepLR(opt,[100,150,180],gamma=0.1)
used=[]
for epoch in range(1,201):
    used.append(opt.param_groups[0]['lr'])
    opt.zero_grad(); parameter.sum().backward(); opt.step(); sched.step()
# Compare the recursively computed floats, not rounded decimal spellings.
e=0.001; expected=[]
for epoch in range(1,201):
    expected.append(e)
    if epoch in [100,150,180]: e*=0.1
checks['real200_scheduler_sequence']=used==expected and sched.last_epoch==200
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'synthetic_CPU_only':True,'power_failure_tested':False,'new_CUDA_runner_tested':False}
(w/'stop_guard_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
assert all(checks.values())
