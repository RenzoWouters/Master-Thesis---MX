"""Independent ten-choice synthetic CPU fixtures; not SVHN research data."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch
import json, csv
from continuation_bridge import ref as parent, s, t, a, core, torch

def make_context(path: Path, source, source_info, modules, package_hashes):
    rt = s.runtime(20261009, 'cpu', strict=False)
    c = parent.config()
    c.update(train_examples=3, test_examples=3, batch_size=2,
             purpose='SYNTHETIC_CPU_FREEZE50_VALIDATION_PARENT_ONLY',
             synthetic_initial_theta_mixed=True)
    model = t.build_model(modules, 'joint', rt.device)
    with torch.no_grad():
        pattern = torch.tensor([0.0, 0.4999, 0.5, 1.4999, 1.5, 2.4999, 2.5,
                                3.4999, 3.5, 4.4999, 4.5, 5.4999, 5.5,
                                6.4999, 6.5, 7.4999, 7.5, 8.4999, 8.5, 9.0])
        for p in a.scale_parameters(model):
            p.copy_(pattern.repeat((p.numel()+pattern.numel()-1)//pattern.numel())[:p.numel()].reshape_as(p))
    optimizer = t.build_optimizer(modules, model, c)
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(3, 3, 40, 40, generator=generator)*2-1
    labels = torch.tensor([0, 1, 2])
    loaders = {phase: s.TrackedLoader(torch.utils.data.DataLoader(
        s.IndexedDataset(torch.utils.data.TensorDataset(images.clone(), labels.clone())),
        batch_size=2, shuffle=phase=='train', num_workers=0, pin_memory=False,
        drop_last=False)) for phase in ('train','test')}
    contract = {'config':c, 'environment':rt.environment, 'settings':rt.settings,
                'source_files':source_info['workcopy_files_sha256'],
                'package_sha256':package_hashes, 'synthetic_fixture_only':True}
    report = {'config':c,'environment':rt.environment,'execution_settings':rt.settings,
              'source':source_info,'package_sha256':package_hashes,
              'contract_sha256':rt.core.json_digest(contract),
              'synthetic_fixture_only':True,'dataset_downloaded':False}
    return SimpleNamespace(rt=rt,c=c,source=source,source_info=source_info,modules=modules,
        model=model,optimizer=optimizer,scheduler=scheduler,loaders=loaders,
        contract=contract,report=report)

def independent_offsets(torch, choice, shape):
    assert choice['schema']=='mx-ten-scale-choices-four-bit-v1'
    assert choice['shape']==list(shape) and choice['num_blocks']==shape[0]*shape[1]
    assert choice['bit_order']=='little' and choice['bits_per_choice']==4
    assert choice['mapping']=={str(i):i for i in range(10)}
    packed=choice['packed_choices']
    n=shape[0]*shape[1]
    assert packed.dtype==torch.uint8 and packed.numel()==(n+1)//2
    digits=torch.stack((packed.to(torch.int64)&15, packed.to(torch.int64)>>4),dim=1).flatten()
    assert not bool(digits[n:].any())
    assert bool((digits[:n]<=9).all())
    return digits[:n].reshape(shape).to(torch.int32)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare_evidence(r,parent_fixture,directory):
    """Derive fixture expectations from the actually executed parent timeline."""
    directory.mkdir(parents=True,exist_ok=False)
    history=read_json(parent_fixture/'history.json')
    contract=read_json(parent_fixture/'parent_contract.json')
    parent_state=r.core.load_verified(parent_fixture/'checkpoints/epoch_00050.pt')
    baseline=history['epochs'][49]
    comparison={rec['epoch']:rec for rec in history['epochs'][50:55]}
    assert contract['synthetic_fixture_only'] is True
    assert set(comparison)==set(range(51,56))
    write_json(directory/'baseline_epoch50.json',baseline)
    write_json(directory/'comparison_epochs51_55.json',{
        'schema':'mx-freeze50-historical-comparison-v1',
        'source':'SYNTHETIC history.json','epochs':list(comparison.values())})
    write_json(directory/'parent_contract.json',contract)
    original_report=read_json(parent_fixture/'run_report.json')
    original_report.update(status='PASS',schema='mx-triple-learned-scale10-reference100-report-v1',
        completed_epoch=55,synthetic_fixture_only=True,full_training_completed=False,
        output_directory=str(parent_fixture),command='synthetic_fixture')
    write_json(directory/'run_report.json',original_report)
    r.s.write_csv(directory/'epoch_metrics.csv',[
        row for record in history['epochs'] for phase in ('train','test') for row in record[phase]])
    r.s.write_csv(directory/'scale_epoch_metrics.csv',[
        row for record in history['epochs'] for row in record['scale_epoch_metrics']])
    r.s.write_csv(directory/'epoch_times.csv',[
        {'epoch':record['epoch'],**{k:record[k] for k in (
            'train_seconds','test_seconds','checkpoint_seconds','other_seconds','total_epoch_seconds',
            'first_epoch_in_process','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')}}
        for record in history['epochs']])
    meta={
        'schema':'mx-freeze-offsets-e50-parent-reference-v1',
        'source_feedback_filename':'SYNTHETIC_FIXTURE_ONLY',
        'source_feedback_sha256':r.core.file_sha256(parent_fixture/'history.json'),
        'source_history_sha256':r.core.file_sha256(parent_fixture/'history.json'),
        'source_run_directory':str(parent_fixture),
        'source_contract_sha256':r.core.json_digest(contract),
        'source_initial_model_sha256':history['initial_model_sha256'],
        'source_initial_base_model_sha256':history['initial_base_model_sha256'],
        'parent_epoch':50,'parent_optimizer_steps':100,
        'parent_model_sha256':baseline['model_sha256'],
        'parent_checkpoint_sha256':r.core.file_sha256(parent_fixture/'checkpoints/epoch_00050.pt'),
        'parent_checkpoint_trace_sha256':r.core.json_digest(parent_state['trace']),
        'parent_checkpoint_relative_to_project':'SYNTHETIC_FIXTURE_ONLY/checkpoints/epoch_00050.pt',
        'comparison_epochs':list(range(51,56)),'synthetic_fixture_only':True,
        'raw_files_copied_byte_for_byte':[],
        'derived_files':{'all':'Derived from actually executed synthetic parent fixture'},
        'files_sha256':{p.name:r.core.file_sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}}
    write_json(directory/'REFERENCE.json',meta)
    return {'meta':meta,'contract':contract,'baseline':baseline,'comparison':comparison,
            'directory':directory,'reference_sha256':r.core.file_sha256(directory/'REFERENCE.json')}


def verify_exports(r,torch,path,exports,epoch,modules,parent_theta):
    a,core,s=r.a,r.core,r.s
    assert set(exports)=={'4','6','8'}
    payloads={}
    for bit in (4,6,8):
        detail=exports[str(bit)]
        assert detail['epoch']==epoch and detail['weight_bits']==bit
        export=path/detail['file']
        assert core.file_sha256(export)==detail['sha256']
        payloads[bit]=a.load_verified_payload(export)
        auxiliary=payloads[bit]['auxiliary_fp32_state']
        assert not any('.quantize_fn.theta' in n for n in auxiliary)
        assert all(n+'.weight' not in auxiliary for n in a.TARGET_LAYERS)
    total_bits=0
    for name in a.TARGET_LAYERS:
        high=a.codec(8).MXTensor.from_payload(payloads[8]['mx_layers'][name])
        mid=a.codec(6).MXTensor.from_payload(payloads[6]['mx_layers'][name])
        low=a.codec(4).MXTensor.from_payload(payloads[4]['mx_layers'][name])
        offsets=independent_offsets(torch,payloads[8]['scale_choices'][name],tuple(high.scale_codes.shape))
        theta = torch.tensor(parent_theta[name])
        lo = torch.floor(theta)
        expected_offsets=(lo + ((theta-lo)>=0.5)).to(torch.int32)
        assert torch.equal(offsets,expected_offsets)
        total_bits+=offsets.numel()
        u=high.codes.to(torch.int32)
        expected_codes=((u>>7)<<5)|(((u>>5)&3)<<3)|(u&7)
        assert torch.equal(mid.codes.to(torch.int32),expected_codes)
        assert torch.equal(low.codes.to(torch.int32),expected_codes>>2)
        assert torch.equal(mid.scale_codes.to(torch.int32),high.scale_codes.to(torch.int32)+offsets)
        assert torch.equal(low.scale_codes,mid.scale_codes)
    assert total_bits==4080
    def forbidden(*args,**kwargs):
        raise AssertionError('Compact source reconstruction invoked a weight encoder.')
    images=torch.linspace(-1,1,2*3*40*40).reshape(2,3,40,40)
    with ExitStack() as stack:
        stack.enter_context(patch.object(a,'effective_reference_weight',forbidden))
        for bit in (4,6,8):
            stack.enter_context(patch.object(a.codec(bit),'encode_tensor',forbidden))
            stack.enter_context(patch.object(a.codec(bit),'encode_elements',forbidden))
        for bit in (4,6,8):
            source_model=a.load_exported_model(modules.models,payloads[8],torch.device('cpu'),bit)
            derived_model=a.load_exported_model(modules.models,payloads[bit],torch.device('cpu'),bit)
            with torch.no_grad():
                x,y=source_model(images),derived_model(images)
            assert torch.equal(x,y) and bool(torch.isfinite(x).all())
