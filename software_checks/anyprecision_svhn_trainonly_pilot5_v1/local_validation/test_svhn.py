"""Offline CPU software validation. No SVHN accuracy/runtime claims.

Only this test module replaces Tensor.cuda by an identity on CPU and constructs
synthetic fixtures. The production CLI has neither a CPU nor a subset bypass.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import scipy.io
import nbformat
from nbclient import NotebookClient
import svhn_support as s
import run_svhn as runner
import resume_core as core

torch.set_num_threads(1)
ROOTOUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='svhn_tests_'))
ROOTOUT.mkdir(parents=True, exist_ok=True)
log_lines=[]
def log(text):
    log_lines.append(text)

def must_fail(fn):
    try:
        fn()
    except (RuntimeError, FileExistsError):
        return True
    raise AssertionError('Expected rejection did not happen')

source_dir=ROOTOUT/'source'
source_dir.mkdir()
source, source_info=s.legacy.prepare_source(source_dir)
modules=s.import_source(source)
c=runner.config()
c.update(train_examples=6, test_examples=4, batch_size=4)
# Explicit synthetic protocol: production config() remains untouched.
c['purpose']='SYNTHETIC_OFFLINE_TEST_ONLY'

g=torch.Generator().manual_seed(778)
x=torch.rand(6,3,40,40,generator=g)*2-1
y=torch.tensor([0,1,2,3,4,5])
dataset=torch.utils.data.TensorDataset(x,y)
test_dataset=torch.utils.data.TensorDataset(x[:4],y[:4])

def ctx_new(out):
    out.mkdir()
    rt=s.runtime(c['seed'],'cpu',strict=False)
    model=modules.models.svhnq(s.BITS).to(rt.device)
    opt=modules.optimizer.get_optimizer_config(model,'adam',.001,0.0)
    sch=modules.optimizer.get_lr_scheduler('adam',opt,[50,75,90])
    loaders={phase:s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(ds),batch_size=4,
                       shuffle=phase=='train',num_workers=0,pin_memory=False,drop_last=False))
             for phase,ds in [('train',dataset),('test',test_dataset)]}
    contract={'config':c,'environment':rt.environment,'settings':rt.settings,
              'source':source_info['workcopy_files_sha256'],'synthetic_fixture_only':True}
    report={'status':'PASS','schema':'SYNTHETIC_TEST','config':c,'environment':rt.environment,
            'execution_settings':rt.settings,'synthetic_fixture_only':True,'warnings':[],
            'limitations':['SYNTHETIC CPU software test; not SVHN research data']}
    return SimpleNamespace(rt=rt,c=c,source=source,modules=modules,model=model,optimizer=opt,
                           scheduler=sch,loaders=loaders,contract=contract,report=report,
                           source_info=source_info)

results={'test_scope':'OFFLINE SYNTHETIC CPU ONLY','real_svhn_downloaded':False,
         'real_svhn_epochs_run':0,'gpu_tested':False,'python':platform.python_version(),
         'torch':str(torch.__version__),'torchvision':__import__('torchvision').__version__,
         'numpy':np.__version__,'scipy':__import__('scipy').__version__,'tests':{}}

with patch.object(torch.Tensor,'cuda',lambda tensor,*a,**kw:tensor):
    # Independently run uninstrumented upstream function with regular datasets.
    base=ctx_new(ROOTOUT/'baseline')
    base_loaders={phase:torch.utils.data.DataLoader(ds,batch_size=4,shuffle=phase=='train',
                                      num_workers=0,pin_memory=False,drop_last=False)
                  for phase,ds in [('train',dataset),('test',test_dataset)]}
    f,ast_hash=s.load_forward(source,modules.utils,log)
    baseline_metrics=[]
    for ep in range(5):
        base.model.train()
        tr=f(base_loaders['train'],base.model,torch.nn.CrossEntropyLoss(),
             modules.losses.CrossEntropyLossSoft(),ep,True,base.optimizer,None)
        base.model.eval()
        te=f(base_loaders['test'],base.model,torch.nn.CrossEntropyLoss(),
             modules.losses.CrossEntropyLossSoft(),ep,False)
        base.scheduler.step()
        baseline_metrics.append((tr,te))
    expected_model=core.cpu_copy(base.model.state_dict())
    expected_opt=core.cpu_copy(base.optimizer.state_dict())
    expected_sch=core.cpu_copy(base.scheduler.state_dict())
    expected_rng=core.capture_rng({},torch.device('cpu'))

    out=ROOTOUT/'instrumented'
    full=ctx_new(out)
    full_trace=runner.run_epochs(full,out,log,5,install_signal_handler=False)
    assert not core.differences(expected_model,core.cpu_copy(full.model.state_dict()))
    assert not core.differences(expected_opt,core.cpu_copy(full.optimizer.state_dict()))
    assert not core.differences(expected_sch,core.cpu_copy(full.scheduler.state_dict()))
    assert not core.differences(expected_rng,core.capture_rng({},torch.device('cpu')))
    for i,record in enumerate(full_trace['epochs']):
        for phase_index,phase in enumerate(['train','test']):
            for metric_index,key in enumerate(['objective_loss','top1_percent','top5_percent']):
                assert baseline_metrics[i][phase_index][metric_index]==[r[key] for r in record[phase]]
    results['tests']['uninstrumented_upstream_vs_instrumented_5epochs_exact_CPU']={
        'status':'PASS','model':True,'optimizer':True,'scheduler':True,'all_rng_states':True,
        'all_epoch_metrics':True,'forward_ast_sha256':ast_hash}
    runner.export_results(full,out,full_trace,log)
    s.write_json(out/'run_report.json',full.report)

    partial_out=ROOTOUT/'partial'
    partial=ctx_new(partial_out)
    runner.run_epochs(partial,partial_out,log,2,install_signal_handler=False)
    resumed_out=ROOTOUT/'resumed'
    resumed=ctx_new(resumed_out)
    trace=runner.run_epochs(resumed,resumed_out,log,5,partial_out/'checkpoints/epoch_00002.pt',False)
    assert not core.differences(expected_model,core.cpu_copy(resumed.model.state_dict()))
    assert not core.differences(expected_opt,core.cpu_copy(resumed.optimizer.state_dict()))
    assert not core.differences(expected_sch,core.cpu_copy(resumed.scheduler.state_dict()))
    assert not core.differences(expected_rng,core.capture_rng({},torch.device('cpu')))
    for a,b in zip(full_trace['epochs'],trace['epochs']):
        for key in ['train','test','model_sha256','best_epoch_so_far','lr_used','lr_next','optimizer_steps']:
            assert a[key]==b[key],key
        for phase in ['train','test']:
            assert a['coverage'][phase]['order_sha256']==b['coverage'][phase]['order_sha256']
    runner.export_results(resumed,resumed_out,trace,log)
    s.write_json(resumed_out/'run_report.json',resumed.report)
    results['tests']['resume_after_epoch2_matches_5epoch_continuous_CPU_including_dropout']='PASS'
    results['tests']['best_and_last_copies_verified_against_saved_model_and_metrics']='PASS'

    # Guard tests: no changed protocol, no overwrite, no corrupted checkpoint.
    checkpoint=partial_out/'checkpoints/epoch_00002.pt'
    state=core.load_verified(checkpoint)
    bad=copy.deepcopy(resumed.contract)
    bad['config']['batch_size']=3
    assert must_fail(lambda:core.restore_checkpoint(state,resumed.model,resumed.optimizer,
                      resumed.scheduler,{},torch.device('cpu'),bad))
    assert must_fail(lambda:core.atomic_save(checkpoint,state))
    corrupted=ROOTOUT/'corrupt.pt'
    corrupted.write_bytes(checkpoint.read_bytes()+b'changed')
    corrupted.with_suffix('.pt.sha256').write_text(core.file_sha256(checkpoint)+'\n')
    assert must_fail(lambda:core.load_verified(corrupted))
    results['tests']['checkpoint_hash_overwrite_and_contract_guards']='PASS'
    other_split=copy.deepcopy(resumed.contract)
    other_split['config'].update(train_split='train_extra',extra_used=True,train_examples=604388)
    assert must_fail(lambda:core.restore_checkpoint(state,resumed.model,resumed.optimizer,
                       resumed.scheduler,{},torch.device('cpu'),other_split))
    results['tests']['cross_split_resume_contract_is_rejected']='PASS'


    # A controlled stop marker causes a complete epoch boundary, not a partial step.
    stopped_out=ROOTOUT/'stop_test'
    stopped=ctx_new(stopped_out)
    (stopped_out/'STOP_AFTER_EPOCH').touch()
    stopped_trace=runner.run_epochs(stopped,stopped_out,log,5,install_signal_handler=False)
    assert len(stopped_trace['epochs'])==1
    assert core.load_verified(stopped_out/'checkpoints/epoch_00001.pt')['completed_epoch']==1
    results['tests']['stop_marker_finishes_current_epoch']='PASS'

# Train-only loading via the real source class on artificial .mat files.
# No extra file is available initially; the original class must not need it.
matroot=ROOTOUT/'mat_fixture'
matroot.mkdir()
old_splits=copy.deepcopy(modules.data.SVHNMIX.split_list)
old_path=modules.data.data_paths['svhn']
try:
    for split,labels in [('train',[10,2,3]),('test',[10,9])]:
        n=len(labels)
        filename=split+'_32x32.mat'
        raw=np.arange(32*32*3*n,dtype=np.int64).reshape(32,32,3,n).astype(np.uint8)
        scipy.io.savemat(matroot/filename,{'X':raw,'y':np.asarray(labels,dtype=np.uint8)[:,None]})
        modules.data.SVHNMIX.split_list[split][2]=hashlib.md5((matroot/filename).read_bytes()).hexdigest()
    modules.data.data_paths['svhn']=str(matroot)
    info=s.ensure_dataset(matroot,modules.data.SVHNMIX,False,log)
    assert set(info['files'])=={'train_32x32.mat','test_32x32.mat'}
    assert not (matroot/'extra_32x32.mat').exists()
    reads=[]
    original_loadmat=scipy.io.loadmat
    def tracked_loadmat(filename,*a,**kw):
        name=Path(filename).name
        assert name in {'train_32x32.mat','test_32x32.mat'},name
        reads.append(name)
        return original_loadmat(filename,*a,**kw)
    with patch.object(scipy.io,'loadmat',tracked_loadmat):
        ds=modules.data.get_dataset('svhn','train',modules.data.get_transform('svhn','train'),download=False)
        ts=modules.data.get_dataset('svhn','val',modules.data.get_transform('svhn','val'),download=False)
    assert reads==['train_32x32.mat','test_32x32.mat']
    assert ds.split=='train' and ts.split=='test'
    assert ds.labels.tolist()==[0,2,3] and ts.labels.tolist()==[0,9]
    assert ds[0][0].shape==(3,40,40) and np.isfinite(ds[0][0].numpy()).all()
    assert must_fail(lambda:s.validate_dataset(ds,73257,'SVHN train-only'))
    # Even an invalid existing extra file must not be inspected or overwritten.
    sentinel=matroot/'extra_32x32.mat'
    sentinel.write_bytes(b'NOT SVHN: untouched sentinel used only for a software test')
    before=sentinel.read_bytes()
    assert s.ensure_dataset(matroot,modules.data.SVHNMIX,False,log)['files']==info['files']
    assert sentinel.read_bytes()==before
    results['tests']['original_loader_train_only_no_extra_file_required_label10_and_resize40']='PASS'
    results['tests']['extra_present_but_invalid_is_ignored_and_unchanged']='PASS'
    results['tests']['production_rejects_subset_dataset']='PASS'
finally:
    modules.data.SVHNMIX.split_list=old_splits
    modules.data.data_paths['svhn']=old_path

# Dataset download testing is mocked; ensures no accidental overwrite of raw files.
rawroot=ROOTOUT/'download_fixture'
rawroot.mkdir()
content=b'only a mocked download test, never an SVHN sample'
digest=hashlib.md5(content).hexdigest()
class FakeDatasetClass:
    split_list={split:['http://official.invalid/'+split+'.mat',split+'.mat',digest]
                for split in ['train','extra','test']}
class Response(io.BytesIO):
    def geturl(self):return 'http://official.invalid/test'
with patch.object(s,'urlopen',lambda *a,**kw:Response(content)):
    info=s.ensure_dataset(rawroot,FakeDatasetClass,True,log)
    assert len(info['files'])==2
    assert set(info['files'])=={'train.mat','test.mat'}
    assert not (rawroot/'extra.mat').exists()
    assert s.ensure_dataset(rawroot,FakeDatasetClass,False,log)['files']==info['files']
corrupt_raw=rawroot/'train.mat'
corrupt_raw.write_bytes(b'bad existing raw file')  # only our artificial fixture
before=corrupt_raw.read_bytes()
assert must_fail(lambda:s.ensure_dataset(rawroot,FakeDatasetClass,True,log))
assert corrupt_raw.read_bytes()==before
results['tests']['staged_download_and_existing_raw_never_overwritten']='PASS'

# Explicit equal-score rule, plus a low-bit peak that must NOT select a checkpoint.
def rec(ep,score,low):
    return {'epoch':ep,'test':[{'bits_w_and_a':1,'top1_percent':low},
                              {'bits_w_and_a':32,'top1_percent':score}]}
assert s.best_record([rec(1,70,60),rec(2,80,50),rec(3,80,99)])['epoch']==2
results['tests']['selection_strict_W32_earliest_tie_not_low_bit_peak']='PASS'

# Verify the original 100-epoch scheduler, without real training.
op=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=.001)
sch=modules.optimizer.get_lr_scheduler('adam',op,[50,75,90])
lrs=[]
for ep in range(1,101):
    lrs.append(op.param_groups[0]['lr'])
    op.step()
    sch.step()
assert lrs[:50]==[.001]*50
assert lrs[50:75]==[.0001]*25
assert abs(lrs[75]-.00001)<1e-15 and abs(lrs[90]-.000001)<1e-15
results['tests']['LR_milestones_50_75_90_not_compressed_into_pilot']='PASS'
assert runner.config()['planned_epochs']==100 and runner.config()['pilot_stop_epoch']==5
assert runner.config()['pin_memory'] is True and runner.config()['workers']==0
assert runner.config()['train_examples']==73257
assert runner.config()['train_split']=='train' and runner.config()['extra_used'] is False
assert (73257+127)//128==573 and 573*5==2865
results['tests']['production_config_five_full_epochs_no_subset']='PASS'

# Compare executable configuration to the supplied previous real-run config.
previous=json.loads((ROOT/'local_validation/previous_pilot_config.json').read_text())
current=runner.config()
shared_changes={key for key in previous if previous[key]!=current[key]}
assert shared_changes=={'schema','purpose','train_split','train_examples'},shared_changes
assert set(current)-set(previous)=={'extra_used','research_change','initialization_policy'}
results['tests']['previous_protocol_unchanged_except_split_and_explicit_metadata']='PASS'
import csv
with (ROOTOUT/'instrumented/paper_comparison.csv').open(newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==10
assert all(r['comparable_training_budget']=='False' and r['same_training_split']=='False' for r in rows)
results['tests']['paper_comparison_always_marked_as_different_dataset_and_budget']='PASS'

# Notebook must execute from saved reports, without needing the model or dataset.
notebook=nbformat.read(ROOTOUT/'instrumented/SVHN_trainonly_pilot5_resultaten.ipynb',as_version=4)
NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(ROOTOUT/'instrumented')}}).execute()
nbformat.write(notebook,ROOTOUT/'notebook_synthetic_executed.ipynb')
results['tests']['read_only_notebook_executes_from_saved_synthetic_data']='PASS'

# All user source modules still match their per-run manifest.
for name,digest in source_info['workcopy_files_sha256'].items():
    assert core.file_sha256(source/name)==digest
reuse = json.loads((ROOT/'REUSE_MANIFEST.json').read_text())
for name, expected in reuse['byte_identical_files'].items():
    assert core.file_sha256(ROOT/name) == expected
results['tests']['source_and_reused_resume_core_unchanged']='PASS'
results['status']='PASS'
(ROOTOUT/'validation_report.json').write_text(json.dumps(results,indent=2)+'\n')
(ROOTOUT/'test.log').write_text('\n'.join(log_lines)+'\n')
print(json.dumps(results,indent=2))
