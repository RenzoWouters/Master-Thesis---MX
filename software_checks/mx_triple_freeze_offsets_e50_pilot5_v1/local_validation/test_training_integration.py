"""CPU integration for the frozen-offset51..55 continuation.

All accuracies and timings produced by this test belong to three synthetic
images. There is no SVHN download/training, CUDA claim or research outcome.
The fixture executes the unchanged parent runner to obtain real Adam/BN/RNG
state at epoch50; the production continuation functions then execute51..55.
"""
from __future__ import annotations
from contextlib import ExitStack, redirect_stdout
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

sys.dont_write_bytecode = True


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


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
    original_report.update(status='PASS',schema='mx-triple-learned-scale32-64-reference100-report-v1',
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
        'parent_checkpoint_trace_sha256':r.core.json_digest(parent_state['trace']),
        'parent_checkpoint_relative_to_project':'SYNTHETIC_FIXTURE_ONLY/checkpoints/epoch_00050.pt',
        'comparison_epochs':list(range(51,56)),'synthetic_fixture_only':True,
        'raw_files_copied_byte_for_byte':[],
        'derived_files':{'all':'Derived from actually executed synthetic parent fixture'},
        'files_sha256':{p.name:r.core.file_sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}}
    write_json(directory/'REFERENCE.json',meta)
    return {'meta':meta,'contract':contract,'baseline':baseline,'comparison':comparison,
            'directory':directory,'reference_sha256':r.core.file_sha256(directory/'REFERENCE.json')}


def independent_offsets(torch,choice,shape):
    assert choice['schema']=='mx-binary-scale-choice-v1'
    assert choice['shape']==list(shape)
    assert choice['bit_order']=='little' and choice['mapping']=={'0':5,'1':6}
    packed=choice['packed_choices']
    assert packed.dtype==torch.uint8 and packed.numel()==(shape[0]*shape[1]+7)//8
    bits=((packed.to(torch.int64)[:,None]>>torch.arange(8))&1).flatten()
    assert not bool(bits[shape[0]*shape[1]:].any())
    return (bits[:shape[0]*shape[1]]+5).reshape(shape).to(torch.int32)


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
        expected_offsets=5+(torch.tensor(parent_theta[name])>=5.5).to(torch.int32)
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


def main(package,out,parent_fixture):
    package,out,parent_fixture=map(lambda p:Path(p).resolve(),(package,out,parent_fixture))
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    sys.path.insert(0,str(package))
    r=load_module('freeze50_runner_validation',package/'run_train.py')
    code_binding={name:r.core.file_sha256(package/name) for name in
        ('run_train.py','freeze_core.py','runtime_bridge.py','test_freeze.py')}
    import torch
    torch.set_num_threads(2)
    fixture_helper=load_module('freeze50_fixture_helper',Path(__file__).with_name('build_parent_fixture.py'))
    fixture_info=read_json(parent_fixture/'fixture.json')
    source=Path(fixture_info['source']);source_info=fixture_info['source_info']
    modules=r.s.import_source(source)
    evidence=prepare_evidence(r,parent_fixture,out/'parent_reference')
    loaded_evidence=r.load_parent_reference(evidence['directory'],evidence['reference_sha256'])
    assert not r.core.differences(evidence['contract'],loaded_evidence['contract'])
    assert evidence['baseline']==loaded_evidence['baseline']
    parent_checkpoint=parent_fixture/'checkpoints/epoch_00050.pt'
    parent_digest=r.core.file_sha256(parent_checkpoint)
    original_package={p.relative_to(package).as_posix():r.core.file_sha256(p)
        for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    cases={};contexts={}
    fixture_package={'fixture_only_test_file_sha256':r.core.file_sha256(Path(__file__))}

    def fresh_context():
        ctx=fixture_helper.make_context(out,source,source_info,modules,r.s.verify_package())
        assert not r.core.differences(ctx.contract,evidence['contract'])
        ctx.branch_package_hashes=fixture_package
        ctx.report.update(schema='mx-triple-freeze-offsets-e50-pilot5-report-v1',
                          status='RUNNING',limitations=['SYNTHETIC CPU FIXTURE ONLY'])
        return ctx

    def cpu_cuda(tensor,*args,**kwargs):
        assert tensor.device.type=='cpu'
        return tensor

    def forbidden_encoder(*args,**kwargs):
        raise AssertionError('Frozen training called a direct lower quantizer.')

    def run_case(name,stop_after=55,resume=None,marker=False):
        path=out/name;path.mkdir()
        ctx=fresh_context();contexts[name]=ctx
        log=r.s.legacy.Log(path/'run.log')
        if marker:(path/'STOP_AFTER_EPOCH').touch()
        with (out/(name+'_stdout.log')).open('x',encoding='utf-8') as stdout:
            with redirect_stdout(stdout),ExitStack() as stack:
                stack.enter_context(patch.object(torch.Tensor,'cuda',cpu_cuda))
                for bit in (4,6):
                    stack.enter_context(patch.object(r.a.codec(bit),'encode_tensor',forbidden_encoder))
                    stack.enter_context(patch.object(r.a.codec(bit),'encode_elements',forbidden_encoder))
                trace=r.run_epochs(ctx,path,log,stop_after=stop_after,parent_checkpoint=parent_checkpoint,
                    evidence=evidence,resume=resume,install_signal_handler=False)
                r.export_results(ctx,path,trace,log,evidence)
        log.handle.close()
        ctx.report['status']='PASS' if ctx.report['continuation_completed'] else 'STOPPED_AT_EPOCH_BOUNDARY'
        write_json(path/'run_report.json',ctx.report)
        r.feedback_bundle(path)
        cases[name]={'output_directory':str(path),'completed_epoch':trace['epochs'][-1]['epoch'],
                     'cumulative_new_optimizer_updates_since50':2*(trace['epochs'][-1]['epoch']-50),
                     'optimizer_updates_in_process':ctx.guard.steps,
                     'status':ctx.report['status']}
        return path,ctx.report,trace

    path,report,trace=run_case('frozen51_to55')
    assert report['continuation_completed'] is True
    assert [rec['epoch'] for rec in trace['epochs']]==list(range(51,56))
    assert trace['initial_scale_state']==evidence['baseline']['scale_state']
    parent_state=r.core.load_verified(parent_checkpoint)
    final_state=r.core.load_verified(path/'checkpoints/epoch_00055.pt')
    for rec in trace['epochs']:
        ep=rec['epoch']
        assert rec['optimizer_steps']==2*ep
        assert rec['lr_used']==rec['lr_next']==.0001
        assert rec['scale_lr_used']==rec['scale_lr_next']==0
        assert rec['scale_state']==evidence['baseline']['scale_state']
        assert rec['batchnorm_counters_correct'] and rec['test_optimizer_steps']==0
        assert all(rec['coverage_historical_match'].values())
        assert rec['freeze_checks']['all_theta_unchanged']
        assert rec['freeze_checks']['all_hard_offsets_unchanged']
        assert rec['freeze_checks']['theta_lr_zero']
        assert rec['freeze_checks']['checked_optimizer_steps']==2
        assert rec['gradient_checks']['optimizer_steps_in_phase']==2
        assert rec['gradient_checks']['detached_teacher_checks']==4
        assert rec['gradient_checks']['scale_optimizer_steps']==2
        assert any(row['master_changed_values']>0 for row in rec['weight_updates'])
        for row in rec['scale_epoch_metrics']:
            assert row['scale_lr']==0
            assert all(row[k]==0 for k in ('theta_changed_blocks','theta_update_l2',
                'net_choice_changes','choice_flips_during_epoch','unique_blocks_switched_during_epoch',
                'projected_values_during_epoch'))
        for phase in ('train','test'):
            assert rec['coverage'][phase]['order_sha256']==evidence['comparison'][ep]['coverage'][phase]['order_sha256']
            assert all(row['n']==3 and row['activation_bits']==32 for row in rec[phase])
    assert any(parent_state['model'][name].ne(final_state['model'][name]).any()
        for name in parent_state['model'] if name.endswith('.weight'))
    for name in r.a.theta_state_keys():
        assert torch.equal(parent_state['model'][name],final_state['model'][name])
    assert final_state['optimizer_steps']==110 and final_state['scheduler']['last_epoch']==55
    assert final_state['optimizer']['param_groups'][0]['lr']==.0001
    assert final_state['optimizer']['param_groups'][1]['lr']==0
    assert final_state['scheduler']['_last_lr']==[.0001,0]
    assert final_state['scheduler']['base_lrs']==[.001,0]
    assert len(read_csv(path/'epoch_metrics.csv'))==30
    assert len(read_csv(path/'scale_block_state.csv'))==6*4080
    best=r.t.best_record(trace['epochs'])['epoch']
    assert 51<=best<=55 and report['selected_best_epoch']==best
    for role,ep in [('best_E4M3_test',best),('last_epoch',55)]:
        verify_exports(r,torch,path,report['mx_exports'][role],ep,modules,evidence['baseline']['scale_state'])
    partial,partial_report,partial_trace=run_case('stop_after52',stop_after=52)
    assert not partial_report['continuation_completed'] and len(partial_trace['epochs'])==2
    resumed,resume_report,resume_trace=run_case('resume52_to55',resume=partial/'checkpoints/epoch_00052.pt')
    resumed_state=r.core.load_verified(resumed/'checkpoints/epoch_00055.pt')
    for key in ('model','optimizer','scheduler','rng'):
        difference=r.core.differences(final_state[key],resumed_state[key])
        assert not difference,(key,difference[:4])
    for before,after in zip(trace['epochs'],resume_trace['epochs']):
        for key in ('train','test','scale_state','scale_epoch_metrics','weight_updates','gradient_checks',
                    'model_sha256','lr_used','lr_next','scale_lr_used','scale_lr_next','freeze_checks',
                    'test_prediction_counts','test_label_counts'):
            assert before[key]==after[key],(before['epoch'],key)
    for name in ('scale_block_state.csv','scale_epoch_metrics.csv'):
        assert (path/name).read_bytes()==(resumed/name).read_bytes()
    marker,marker_report,marker_trace=run_case('softstop51',marker=True)
    assert marker_report['completed_epoch']==51 and len(marker_trace['epochs'])==1
    assert not marker_report['continuation_completed']

    rejections={}
    altered_reference=out/'altered_reference'
    shutil.copytree(evidence['directory'],altered_reference)
    with (altered_reference/'epoch_metrics.csv').open('a') as f:f.write('\n')
    try:r.load_parent_reference(altered_reference,evidence['reference_sha256'])
    except RuntimeError as exc:rejections['altered_historical_reference']={'rejected':True,'message':str(exc)}
    else:raise AssertionError('Modified historical metrics accepted')
    def rejects(name,state,expected=None):
        ctx=fresh_context()
        before=r.s.legacy.state_digest(ctx.model.state_dict())
        try:r.verify_parent_state(ctx,state,evidence)
        except (RuntimeError,AssertionError,ValueError) as exc:
            if expected:assert expected in str(exc),(name,str(exc))
            rejections[name]={'rejected':True,'message':str(exc)}
        else:raise AssertionError('Bad parent accepted: '+name)
        assert r.s.legacy.state_digest(ctx.model.state_dict())==before
    wrong=copy.deepcopy(parent_state);wrong['completed_epoch']=49
    rejects('wrong_parent_epoch',wrong)
    wrong=copy.deepcopy(parent_state);wrong['contract']['config']['seed']+=1
    rejects('wrong_parent_contract',wrong)
    wrong=copy.deepcopy(parent_state)
    name=next(n for n,v in wrong['model'].items() if n.endswith('.weight') and v.dtype==torch.float32)
    wrong['model'][name].reshape(-1)[0]+=1
    rejects('wrong_parent_model',wrong)
    wrong=copy.deepcopy(parent_state);wrong['trace']['epochs'][-1]['lr_used']=.2
    rejects('wrong_parent_trace',wrong)
    wrong=copy.deepcopy(parent_state);wrong['optimizer_steps']-=1
    rejects('wrong_parent_optimizer_steps',wrong)
    wrong=copy.deepcopy(parent_state);wrong['scheduler']['last_epoch']=49
    rejects('wrong_parent_scheduler_epoch',wrong)
    damaged=out/'bad_checksum.pt'
    shutil.copyfile(parent_checkpoint,damaged)
    damaged.with_suffix('.pt.sha256').write_text('0'*64+'\n')
    try:r.core.load_verified(damaged)
    except RuntimeError as exc:rejections['wrong_file_checksum']={'rejected':True,'message':str(exc)}
    else:raise AssertionError('Damaged checksum accepted')

    guard_context=fresh_context()
    r.initialize_branch(guard_context,parent_checkpoint,evidence,fixture_package)
    parameter=r.a.scale_parameters(guard_context.model)[0]
    saved=parameter.detach().clone()
    initial_bits=r.a.hard_offsets(parameter).detach().clone()
    with torch.no_grad():
        flat=parameter.reshape(-1)
        candidates=((flat>5.01)&(flat<5.49))|((flat>5.51)&(flat<5.99))
        index=int(candidates.nonzero()[0])
        flat[index]=torch.nextafter(flat[index],torch.tensor(6.0))
    assert torch.equal(initial_bits,r.a.hard_offsets(parameter))
    try:guard_context.guard.assert_unchanged()
    except RuntimeError as exc:
        rejections['within_bin_theta_drift']={'rejected':True,'message':str(exc)}
    else:raise AssertionError('Theta drift inside an unchanged hard-bin accepted')
    with torch.no_grad():parameter.copy_(saved)
    guard_context.optimizer.param_groups[1]['lr']=.0001
    try:guard_context.guard.assert_unchanged()
    except RuntimeError as exc:
        rejections['nonzero_frozen_lr']={'rejected':True,'message':str(exc)}
    else:raise AssertionError('Nonzero frozen theta LR accepted')
    guard_context.optimizer.param_groups[1]['lr']=0
    guard_context.guard.assert_unchanged()
    guard_context.guard.close()

    unsafe_outputs={}
    for label,destination in (
        ('package',package/'new_result'),
        ('parent_run',parent_fixture/'new_result'),
        ('resume_run',partial/'new_result'),
        ('dataset',out/'synthetic_dataset/new_result')):
        try:r.validate_output_location(destination,parent_checkpoint,
            partial/'checkpoints/epoch_00052.pt',out/'synthetic_dataset')
        except RuntimeError as exc:unsafe_outputs[label]={'rejected':True,'message':str(exc)}
        else:raise AssertionError('Protected output location accepted: '+label)
    r.validate_output_location(out/'safe_new_result',parent_checkpoint,
        partial/'checkpoints/epoch_00052.pt',out/'synthetic_dataset')

    main_cases={}
    def main_fixture_context(args,path,report,log):
        assert not args.download
        ctx=fresh_context()
        report.update(ctx.report)
        ctx.report=report
        return ctx
    def main_verify_inputs(ctx,log):
        assert r.core.file_sha256(parent_checkpoint)==parent_digest
        assert r.s.verify_package()==evidence['contract']['package_sha256']
        ctx.report.update(parent_checkpoint_unchanged=True,source_unchanged=True,
            synthetic_input_tensors_unchanged=True)
    for label,additional,expected_status,expected_epoch in (
        ('main_default55',[],'PASS',55),
        ('main_partial51',['--stop-after-epoch','51'],'STOPPED_AT_EPOCH_BOUNDARY',51)):
        destination=out/label
        argv=[str(package/'run_train.py'),'run','--parent-checkpoint',str(parent_checkpoint),
              '--output-root',str(destination),*additional]
        with (out/(label+'_stdout.log')).open('x',encoding='utf-8') as stdout:
            with redirect_stdout(stdout),ExitStack() as stack:
                stack.enter_context(patch.object(sys,'argv',argv))
                stack.enter_context(patch.object(r,'create_context',main_fixture_context))
                stack.enter_context(patch.object(r,'verify_package',lambda:fixture_package))
                stack.enter_context(patch.object(r,'load_parent_reference',lambda:evidence))
                stack.enter_context(patch.object(r,'verify_unchanged',main_verify_inputs))
                stack.enter_context(patch.object(torch.Tensor,'cuda',cpu_cuda))
                result=r.main()
        directories=list(destination.glob('*_freeze_run'))
        assert len(directories)==1
        main_report=read_json(directories[0]/'run_report.json')
        assert result==0 and main_report['status']==expected_status,main_report.get('traceback')
        assert main_report['completed_epoch']==expected_epoch
        assert main_report['synthetic_fixture_only'] is True
        assert main_report['continuation_completed'] is (expected_epoch==55)
        with ZipFile(directories[0]/'feedback_bundle.zip') as z:
            assert {'run_report.json','history.json','freeze_summary.json','lineage.json',
                'parent_reference/REFERENCE.json',r.NOTEBOOK,'notebook_support.py'}.issubset(z.namelist())
            assert not any(n.endswith('.pt') for n in z.namelist())
        main_cases[label]={'status':main_report['status'],'exit_code':result,
            'completed_epoch':expected_epoch,'feedback_bundle_verified':True}

    cli_rejections={}
    bad_args=[['run','--stop-after-epoch',v] for v in ['0','50','56','55.0']]
    bad_args += [['run','--device','cpu'],['run','--synthetic'],['run','--mode','control'],
        ['resume'],['run','--checkpoint',str(parent_checkpoint)],['run','--download']]
    for arguments in bad_args:
        cli=subprocess.run([sys.executable,str(package/'run_train.py'),*arguments],
            cwd=out,text=True,capture_output=True)
        assert cli.returncode==2,(arguments,cli.stdout,cli.stderr)
        cli_rejections[' '.join(arguments)]={'exit_code':2,'message':cli.stderr.splitlines()[-1]}
    assert not (out/'mx_runs').exists()
    for name,digest in code_binding.items():assert r.core.file_sha256(package/name)==digest

    assert r.core.file_sha256(parent_checkpoint)==parent_digest
    unchanged={p.relative_to(package).as_posix():r.core.file_sha256(p)
        for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    # Notebook/root implementation can be finalized concurrently; frozen runtime
    # and parent reference bytes must remain untouched throughout this test.
    for name,digest in original_package.items():
        if name.startswith(('reference_runtime/','parent_reference/')):
            assert unchanged[name]==digest,name
    result={'status':'PASS','synthetic_fixture_only':True,'CUDA_tested':False,
        'no_SVHN_data_loaded_or_downloaded':True,
        'not_a_SVHN_accuracy_or_GPU_timing_result':True,
        'fixture_changes':{'train_examples':3,'test_examples':3,'batch_size':2,
            'parent_epochs_actually_executed':55,'mixed_initial_theta':True},
        'parent_checkpoint_sha256':parent_digest,
        'parent_state_contract_sha256':r.core.json_digest(evidence['contract']),
        'runner_sha256':r.core.file_sha256(package/'run_train.py'),
        'code_binding':code_binding,
        'test_file_sha256':r.core.file_sha256(Path(__file__)),
        'duration_seconds':time.perf_counter()-started,'cases':cases,'main_cases':main_cases,
        'tests':{'original_parent_restored_before_freeze':True,
            'new_epochs51_55_only':True,'all_theta_and_hardbits_exact_every_update':True,
            'three_precisions_A32_FP32_recursive_losses_and_BN_preserved':True,
            'weight_Adam_and_BN_counters_continue50_to55':True,
            'theta_lr_zero_and_finite_gradients_and_moments_continue':True,
            'matched_historical_minibatch_order_all5epochs':True,
            'CPU_resume52_exact_model_optimizer_scheduler_RNG':True,
            'partial_and_softstop_remain_incomplete':True,
            'best_checkpoint_selected_only51_55':True,
            'independent_packed_choices_and_source_only_export_reconstruction':True,
            'negative_parent_validation_before_model_mutation':rejections,
            'protected_output_directories_refused':unsafe_outputs,
            'CLI_rejections_before_outputs':cli_rejections,
            'real_main_default55_and_partial51_status_and_feedback':True,
            'parent_checkpoint_and_original_runtime_unchanged':True}}
    write_json(out/'training_integration_report.json',result)
    print(json.dumps({'status':'PASS','report':str(out/'training_integration_report.json'),
        'duration_seconds':result['duration_seconds']},indent=2))
    return result


if __name__=='__main__':
    if len(sys.argv)!=4:
        raise SystemExit('Usage: test_training_integration.py PACKAGE OUTPUT_DIRECTORY PARENT_FIXTURE_DIRECTORY')
    main(*sys.argv[1:])
