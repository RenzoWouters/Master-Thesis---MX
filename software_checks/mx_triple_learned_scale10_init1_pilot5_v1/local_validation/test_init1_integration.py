"""Bounded production-loop checks on three synthetic images; no SVHN training."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack,redirect_stdout
from unittest.mock import patch
import sys,json,copy,ast,hashlib,subprocess,time,csv,math
from zipfile import ZipFile
P=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve();OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(P));sys.dont_write_bytecode=True
import run_train as r
import torch
import resume_core as core
s,t,a=r.s,r.t,r.a
MODES=(4,6,8)
torch.set_num_threads(2)
start=time.perf_counter();code_names=['run_train.py','trunc_adapter.py','trunc_support.py','test_learned_scale.py']
code_binding={n:core.file_sha256(P/n) for n in code_names}
package_id={'synthetic_test_code':code_binding,'synthetic_fixture_only':True}
prepared=OUT/'source';prepared.mkdir();source,source_info=s.legacy.prepare_source(prepared);modules=s.import_source(source)
gen=torch.Generator().manual_seed(4837);images=torch.rand(3,3,40,40,generator=gen)*2-1;labels=torch.tensor([0,1,2])
config=r.config();assert config['scale_proxy_init']==0 and config['scale_lr']==.001 and a.THETA_INITIAL==0
reference=r.load_initialization_reference();ctxs={};cases={}
# Linux CPU and historical Windows/CUDA initialization hashes need not match.
# Verify numerical equality against an independently constructed init64 CPU oracle;
# the production guard against the untouched actual reference remains strict.
s.runtime(config['seed'],'cpu',strict=False)
with patch.object(a,'THETA_INITIAL',6.0):
 oracle_model=t.build_model(modules,'joint',torch.device('cpu'))
oracle_base=copy.deepcopy(a.base_state_dict(oracle_model))
oracle_hash=s.legacy.state_digest(oracle_base)
fixture_reference=copy.deepcopy(reference)
fixture_reference['history']['initial_base_model_sha256']=oracle_hash
fixture_reference['report']['initial_base_model_sha256']=oracle_hash
fixture_reference['meta']['synthetic_fixture_only']=True

def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def read_csv(path):
 with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def make_ctx(args,out,report,log,check_only=False):
 assert not check_only and args.stop_after_epoch<=5
 rt=s.runtime(config['seed'],'cpu',strict=False)
 c=r.config();c.update(train_examples=3,test_examples=3,batch_size=2,purpose='SYNTHETIC_CPU_INIT1_PILOT_ONLY')
 model=t.build_model(modules,'joint',rt.device)
 assert not core.differences(a.base_state_dict(model),oracle_base)
 r.validate_initialization(model,c,fixture_reference,report)
 assert all(torch.equal(p,torch.zeros_like(p)) for p in a.scale_parameters(model))
 optimizer=t.build_optimizer(modules,model,c);scheduler=modules.optimizer.get_lr_scheduler('adam',optimizer,c['scheduler_milestones'])
 loaders={phase:s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(torch.utils.data.TensorDataset(images.clone(),labels.clone())),batch_size=2,shuffle=phase=='train',num_workers=0,pin_memory=False,drop_last=False)) for phase in ('train','test')}
 contract={'config':c,'environment':rt.environment,'settings':rt.settings,'source_files':source_info['workcopy_files_sha256'],'package_sha256':package_id,'synthetic_fixture_only':True}
 report.update(config=c,environment=rt.environment,execution_settings=rt.settings,source=source_info,package_sha256=package_id,contract_sha256=core.json_digest(contract),synthetic_fixture_only=True,dataset_downloaded=False)
 ctx=SimpleNamespace(rt=rt,c=c,source=source,source_info=source_info,modules=modules,model=model,optimizer=optimizer,scheduler=scheduler,loaders=loaders,contract=contract,report=report)
 ctxs[str(out)]=ctx
 return ctx
def no_cuda(x,*args,**kwargs):assert x.device.type=='cpu';return x
def no_lower(*args,**kwargs):raise AssertionError('Direct lower quantizer called')
def verify_unchanged(ctx,log):
 assert all(core.file_sha256(source/name)==digest for name,digest in source_info['workcopy_files_sha256'].items())
 assert all(core.file_sha256(P/n)==digest for n,digest in code_binding.items())
 ctx.report.update(source_unchanged=True,dataset_files_unchanged=True,synthetic_input_tensors_unchanged=True)
def run_case(name,stop=5,checkpoint=None):
 destination=OUT/name;argv=[str(P/'run_train.py'),'resume' if checkpoint else 'run','--output-root',str(destination),'--stop-after-epoch',str(stop)]
 if checkpoint:argv+=['--checkpoint',str(checkpoint)]
 with (OUT/(name+'.log')).open('x') as stdout,redirect_stdout(stdout),ExitStack() as stack:
  for obj,field,value in [(sys,'argv',argv),(r,'create_context',make_ctx),(r,'verify_unchanged',verify_unchanged),(torch.Tensor,'cuda',no_cuda)]:stack.enter_context(patch.object(obj,field,value))
  for bit in [4,6]:
   for field in ['encode_tensor','encode_elements']:stack.enter_context(patch.object(a.codec(bit),field,no_lower))
  rc=r.main()
 dirs=list(destination.iterdir());assert len(dirs)==1
 out=dirs[0];report=read(out/'run_report.json');assert rc==0,report.get('traceback')
 assert report['status']==('PASS' if stop==5 else 'STOPPED_AT_EPOCH_BOUNDARY')
 assert not report['full_training_completed'];assert report['pilot_completed']==(stop==5)
 trace=read(out/'history.json');assert [e['epoch'] for e in trace['epochs']]==list(range(1,stop+1))
 assert report['history_export_validation']['exact_trace_match'] is True
 assert report['initialization_reference']['comparison_performed'] is False
 assert all(v==0 for layer in trace['initial_scale_state'].values() for row in layer for v in row)
 for e in trace['epochs']:
  assert e['optimizer_steps']==2*e['epoch'] and e['test_optimizer_steps']==0 and e['batchnorm_counters_correct']
  assert e['lr_used']==e['lr_next']==e['scale_lr_used']==e['scale_lr_next']==.001
  g=e['gradient_checks'];assert g['optimizer_steps_in_phase']==g['finite_gradient_checks']==g['scale_optimizer_steps']==2
  assert g['detached_teacher_checks']==4
  assert all(g['forward_calls_'+f]==2 for f in ['E4M3','E2M3','E2M1'])
  for phase in ['train','test']:assert all(row['n']==3 and row['activation_bits']==32 for row in e[phase])
  for layer in e['scale_state'].values():assert all(0<=v<=9 and math.isfinite(v) for row in layer for v in row)
 with ZipFile(out/'feedback_bundle.zip') as z:
  assert r.NOTEBOOK in z.namelist() and 'notebook_support.py' in z.namelist()
  assert 'init64_reference/history.json' in z.namelist()
  assert not any(n.endswith('.pt') for n in z.namelist())
 cases[name]={'status':report['status'],'last_epoch':stop,'output_directory':str(out)}
 return out,report,trace
full,report,trace=run_case('complete5')
partial,_,_=run_case('partial2',2)
resumed,resume_report,resume_trace=run_case('resume2_to5',5,partial/'checkpoints/epoch_00002.pt')
f=core.load_verified(full/'checkpoints/epoch_00005.pt');g=core.load_verified(resumed/'checkpoints/epoch_00005.pt')
for key in ['model','optimizer','scheduler','rng']:assert not core.differences(f[key],g[key]),key
for x,y in zip(trace['epochs'],resume_trace['epochs']):
 for key in ['train','test','model_sha256','scale_state','scale_epoch_metrics','gradient_checks']:
  assert x[key]==y[key],(key,x['epoch'])
 for phase in ('train','test'):
  for key in ('n','order_sha256','complete_once'):
   assert x['coverage'][phase][key]==y['coverage'][phase][key],('coverage',phase,key,x['epoch'])
assert any(v>0 for layer in trace['epochs'][-1]['scale_state'].values() for row in layer for v in row),'theta failed to leave zero in actual synthetic training'
assert f['optimizer_steps']==10 and f['scheduler']['last_epoch']==5
assert all(pg['lr']==.001 for pg in f['optimizer']['param_groups'])
# Independent source-only export verification, accepting any legitimately learned subset of factors.
def independent_offsets(choice, scale_shape):
    """Decode the documented nibble layout without production helper calls."""
    assert choice['schema'] == 'mx-ten-scale-choices-four-bit-v1'
    assert choice['shape'] == list(scale_shape)
    assert choice['num_blocks'] == scale_shape[0] * scale_shape[1]
    assert choice['bits_per_choice'] == 4
    assert choice['bit_order'] == 'little'
    assert choice['mapping'] == {str(i): i for i in range(10)}
    packed = choice['packed_choices']
    n = choice['num_blocks']
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == ((n + 1) // 2,)
    unpacked = ((packed.to(torch.int64)[:, None] >> torch.tensor([0, 4])) & 15).reshape(-1)
    assert not bool(unpacked[n:].any())
    assert bool(((unpacked[:n] >= 0) & (unpacked[:n] <= 9)).all())
    return unpacked[:n].reshape(scale_shape).to(torch.int32)

def verify_exports(directory, exports, epoch, modules):
    """Read actual files and derive expected fields from stored E4+choice bits."""
    assert set(exports) == {'4', '6', '8'}
    loaded = {}
    for bit in MODES:
        detail = exports[str(bit)]
        assert detail['epoch'] == epoch and detail['weight_bits'] == bit
        assert detail['decoded_weights_exact'] and detail['six_fp32_masters_absent']
        assert detail['truncation'] is (bit != 8)
        assert detail['full_test_forward_rerun'] is False
        export_path = directory / detail['file']
        assert core.file_sha256(export_path) == detail['sha256']
        assert export_path.with_suffix(export_path.suffix + '.sha256').read_text().strip() == detail['sha256']
        loaded[bit] = a.load_verified_payload(export_path)
        aux = loaded[bit]['auxiliary_fp32_state']
        assert all(name + '.weight' not in aux for name in a.TARGET_LAYERS)
        assert not any('.quantize_fn.theta' in name for name in aux)
        assert loaded[bit]['weight_bits'] == bit
        assert loaded[bit]['bn_bank_key'] == a.BANK[bit]
        assert loaded[bit]['provenance']['source_epoch'] == epoch
    choices = loaded[8]['scale_choices']
    assert set(choices) == set(a.TARGET_LAYERS)
    assert sum(v['num_blocks'] for v in choices.values()) == 4080
    assert sum(v['packed_choices'].numel() for v in choices.values()) == 2040
    seen = set()
    for name in a.TARGET_LAYERS:
        h = a.codec(8).MXTensor.from_payload(loaded[8]['mx_layers'][name])
        mid = a.codec(6).MXTensor.from_payload(loaded[6]['mx_layers'][name])
        low = a.codec(4).MXTensor.from_payload(loaded[4]['mx_layers'][name])
        u8 = h.codes.long()
        expected6 = ((u8 >> 7) << 5) | (((u8 >> 5) & 3) << 3) | (u8 & 7)
        assert torch.equal(mid.codes.long(), expected6)
        assert torch.equal(low.codes.long(), expected6 >> 2)
        offsets = independent_offsets(choices[name], tuple(h.scale_codes.shape))
        seen.update(offsets.reshape(-1).tolist())
        shifted = h.scale_codes.to(torch.int32) + offsets
        assert bool((shifted <= 254).all())
        assert torch.equal(mid.scale_codes.to(torch.int32), shifted)
        assert torch.equal(low.scale_codes, mid.scale_codes)
        assert h.shape == mid.shape == low.shape
    assert seen.issubset(set(range(10)))
    for bit in MODES:
        assert exports[str(bit)]['source_scale_bytes_copied'] is (bit == 8 or seen.issubset({0}))
    def forbidden(*args, **kwargs):
        raise AssertionError('Source-only reconstruction attempted encoder or G(W).')
    images = torch.linspace(-1, 1, 2 * 3 * 40 * 40).reshape(2, 3, 40, 40)
    with ExitStack() as stack:
        stack.enter_context(patch.object(a, 'effective_reference_weight', forbidden))
        for bit in MODES:
            stack.enter_context(patch.object(a.codec(bit), 'encode_tensor', forbidden))
            stack.enter_context(patch.object(a.codec(bit), 'encode_elements', forbidden))
        for bit in MODES:
            source_model = a.load_exported_model(modules.models, loaded[8], torch.device('cpu'), bit)
            own_model = a.load_exported_model(modules.models, loaded[bit], torch.device('cpu'), bit)
            before = s.legacy.state_digest(source_model.state_dict())
            assert not any('.quantize_fn.theta' in name for name in source_model.state_dict())
            with torch.no_grad():
                source_logits, own_logits = source_model(images), own_model(images)
            assert source_logits.shape == (2, 10) and torch.isfinite(source_logits).all()
            assert torch.equal(source_logits, own_logits)
            assert all(not module.training for module in source_model.modules())
            assert all(not p.requires_grad and p.grad is None for p in source_model.parameters())
            assert s.legacy.state_digest(source_model.state_dict()) == before
for role,detail in report['mx_exports'].items():
 ep=next(iter(detail.values()))['epoch'];verify_exports(full,detail,ep,modules)
# Isolated source-backed boundary gradients: no zero-gradient deadzone at theta0.
proxy=torch.nn.Parameter(torch.zeros(1,1));effective=torch.ones(1,32,requires_grad=True);opt=torch.optim.Adam([proxy],lr=.001)
high=a.codec(8).encode_tensor(effective.detach());chain=a.code_chain(high,a.hard_offsets(proxy));q=a._CachedCodeSTE.apply(effective,proxy,chain[6].codes,chain[6].scale_codes,6)
(-q.sum()).backward();assert proxy.grad.item()<0;opt.step();assert proxy.item()>0
# The guard detects a different initial theta before actual training.
ctx=ctxs[str(full)];fresh=s.runtime(config['seed'],'cpu',strict=False);model=t.build_model(modules,'joint',fresh.device)
with torch.no_grad():next(iter(a.scale_parameters(model))).fill_(6)
try:r.validate_initialization(model,config,fixture_reference,{})
except (RuntimeError,AssertionError):pass
else:raise AssertionError('Init64 accepted by init1 guard')
invalid={}
for args in [['run','--stop-after-epoch','6'],['run','--stop-after-epoch','100'],['run','--stop-after-epoch','0'],['resume'],['run','--device','cpu']]:
 q=subprocess.run([sys.executable,str(P/'run_train.py'),*args],capture_output=True,text=True)
 assert q.returncode==2,(args,q.stdout,q.stderr);invalid[' '.join(args)]=q.returncode
write(OUT/'training_integration_report.json',{'status':'PASS','synthetic_fixture_only':True,'CUDA_tested':False,'no_SVHN_training':True,'true_initial_theta':0,'train_examples':3,'test_examples':3,'batch_size':2,'cases':cases,'all_modes_A32_FP32_original_passes':True,'resume2_to5_exact_model_Adam_scheduler_RNG':True,'CPU_init1_matches_independent_CPU_init64_base_exactly':True,'CPU_initial_base_sha256':oracle_hash,'historical_CUDA_base_guard_unchanged':True,'zero_start_can_learn_upward':True,'direct_lower_encoders_forbidden':True,'packed_source_only_exports_verified':True,'hard_five_epoch_limit':invalid,'no_research_metrics_compared_to_synthetic':True,'code_binding':code_binding,'duration_seconds':time.perf_counter()-start})
print(json.dumps({'status':'PASS','report':str(OUT/'training_integration_report.json'),'complete_run':str(full),'seconds':time.perf_counter()-start}))
