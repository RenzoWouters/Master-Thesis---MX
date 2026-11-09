"""Complete validation after a notebook-write-only correction.

Read the already executed synthetic freeze/restore/export evidence; prove via
AST that numerical runner functions are unchanged. Re-execute only the failed
production entry path (one synthetic epoch), its early failure, and CLI guards.
This test does not replay the parent or the completed numerical continuation.
"""
from __future__ import annotations
import ast
from contextlib import ExitStack, redirect_stdout
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
from zipfile import ZipFile

sys.dont_write_bytecode=True


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(package,out,parent_fixture,existing,reuse_completed_entry=False):
    package,out,parent_fixture,existing=map(lambda p:Path(p).resolve(),
        (package,out,parent_fixture,existing))
    out.mkdir(parents=True,exist_ok=reuse_completed_entry)
    assert not (out/'training_integration_report.json').exists()
    start=time.perf_counter();sys.path.insert(0,str(package))
    r=load('freeze10_reporting_tail_runner',package/'run_train.py')
    test=load('freeze10_prior_integration_harness',Path(__file__).with_name('test_training_integration.py'))
    fixture=load('freeze10_fixture_helper',Path(__file__).with_name('build_parent_fixture.py'))
    import torch
    torch.set_num_threads(2)
    names=('run_train.py','freeze_core.py','runtime_bridge.py','test_freeze.py')
    old_binding={};new_binding={name:r.core.file_sha256(package/name) for name in names}
    archive=existing/'frozen51_to55/feedback_bundle.zip'
    with ZipFile(archive) as z:
        for name in names:
            body=z.read('runner_source/'+name)
            old_binding[name]=__import__('hashlib').sha256(body).hexdigest()
            if name=='run_train.py':old_runner=body.decode('utf-8')
            else:assert body==(package/name).read_bytes(),name
    before=ast.parse(old_runner);after=ast.parse((package/'run_train.py').read_text())
    old_fn={n.name:ast.dump(n,include_attributes=False) for n in before.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    new_fn={n.name:ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    assert set(old_fn)==set(new_fn)
    changed=[name for name in old_fn if old_fn[name]!=new_fn[name]]
    assert changed==['write_notebook'],changed
    old_other=[ast.dump(n,include_attributes=False) for n in before.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    new_other=[ast.dump(n,include_attributes=False) for n in after.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    assert old_other==new_other
    evidence=r.load_parent_reference(existing/'parent_reference',r.core.file_sha256(existing/'parent_reference/REFERENCE.json'))
    checkpoint=parent_fixture/'checkpoints/epoch_00050.pt';parent_digest=r.core.file_sha256(checkpoint)
    info=read(parent_fixture/'fixture.json');source=Path(info['source']);source_info=info['source_info']
    modules=r.s.import_source(source)
    fixture_package={'fixture_only_test_file_sha256':r.core.file_sha256(Path(__file__))}
    summaries={}
    for name,end,status in [('frozen51_to55',55,'PASS'),('stop_after52',52,'STOPPED_AT_EPOCH_BOUNDARY'),
                            ('resume52_to55',55,'PASS'),('softstop51',51,'STOPPED_AT_EPOCH_BOUNDARY')]:
        path=existing/name;report=read(path/'run_report.json');history=read(path/'history.json')
        assert report['status']==status and report['completed_epoch']==end
        assert [x['epoch'] for x in history['epochs']]==list(range(51,end+1))
        assert report['synthetic_fixture_only'] is True
        assert report['history_export_validation']['sha256']==r.core.file_sha256(path/'history.json')
        assert report['history_export_validation']['epochs']==end-50
        for rec in history['epochs']:
            assert rec['scale_state']==evidence['baseline']['scale_state']
            assert rec['scale_lr_used']==rec['scale_lr_next']==0
            assert rec['lr_used']==rec['lr_next']==.0001
            assert rec['freeze_checks']['checked_optimizer_steps']==2
            assert all(rec['coverage_historical_match'].values())
            for row in rec['scale_epoch_metrics']:
                assert all(row[key]==0 for key in ('theta_changed_blocks','theta_update_l2',
                    'net_choice_changes','choice_flips_during_epoch','unique_blocks_switched_during_epoch',
                    'projected_values_during_epoch','choice_jump_distance_during_epoch','max_choice_jump'))
                expected=test.independent_hard_offsets(torch,evidence['baseline']['scale_state'][row['layer']])
                assert all(row[f'count_x{1<<i}']==int((expected==i).sum()) for i in range(10))
        summaries[name]={'completed_epoch':end,'status':status,
            'history_sha256':r.core.file_sha256(path/'history.json'),
            'original_run_report_sha256':r.core.file_sha256(path/'run_report.json')}
    baseline_state=r.core.load_verified(checkpoint)
    final=r.core.load_verified(existing/'frozen51_to55/checkpoints/epoch_00055.pt')
    resumed=r.core.load_verified(existing/'resume52_to55/checkpoints/epoch_00055.pt')
    for key in ('model','optimizer','scheduler','rng'):assert not r.core.differences(final[key],resumed[key]),key
    for name in r.a.theta_state_keys():assert torch.equal(baseline_state['model'][name],final['model'][name])
    assert any(not torch.equal(baseline_state['model'][n],final['model'][n]) for n in baseline_state['model'] if n.endswith('.weight'))
    for name in ('scale_block_state.csv','scale_epoch_metrics.csv'):
        assert (existing/'frozen51_to55'/name).read_bytes()==(existing/'resume52_to55'/name).read_bytes()
    full_report=read(existing/'frozen51_to55/run_report.json')
    for role,epoch in [('best_E4M3_test',full_report['selected_best_epoch']),('last_epoch',55)]:
        test.verify_exports(r,torch,existing/'frozen51_to55',full_report['mx_exports'][role],epoch,modules,evidence['baseline']['scale_state'])
    def fresh_context(args,path,report,log):
        assert not args.download
        ctx=fixture.make_context(path,source,source_info,modules,r.s.verify_package())
        assert not r.core.differences(ctx.contract,evidence['contract'])
        ctx.branch_package_hashes=fixture_package
        report.update(ctx.report);ctx.report=report;return ctx
    def verify_inputs(ctx,log):
        assert r.core.file_sha256(checkpoint)==parent_digest
        assert r.s.verify_package()==evidence['contract']['package_sha256']
        ctx.report.update(parent_checkpoint_unchanged=True,source_unchanged=True,synthetic_input_tensors_unchanged=True)
    def cpu_cuda(tensor,*args,**kwargs):
        assert tensor.device.type=='cpu';return tensor
    main_cases={}
    for name,parent_arg,extra,expected_exit,expected_status in (
        ('main_partial51',checkpoint,['--stop-after-epoch','51'],0,'STOPPED_AT_EPOCH_BOUNDARY'),
        ('main_missing_parent',out/'absent_source/checkpoints/epoch_00050.pt',[],1,'FAIL')):
        destination=out/name
        argv=[str(package/'run_train.py'),'run','--parent-checkpoint',str(parent_arg),'--output-root',str(destination),*extra]
        if reuse_completed_entry and name=='main_partial51':
            prior=list(destination.glob('*_freeze_run'));assert len(prior)==1
            previous_report=read(prior[0]/'run_report.json')
            assert previous_report['status']=='STOPPED_AT_EPOCH_BOUNDARY' and previous_report['completed_epoch']==51
            with ZipFile(prior[0]/'feedback_bundle.zip') as prior_bundle:
                for filename in names:
                    assert prior_bundle.read('runner_source/'+filename)==(package/filename).read_bytes(),filename
            code=0
        else:
            with (out/(name+('_retry_stdout.log' if reuse_completed_entry else '_stdout.log'))).open('x',encoding='utf-8') as stream:
                with redirect_stdout(stream),ExitStack() as stack:
                    stack.enter_context(patch.object(sys,'argv',argv))
                    stack.enter_context(patch.object(r,'create_context',fresh_context))
                    stack.enter_context(patch.object(r,'verify_package',lambda:fixture_package))
                    stack.enter_context(patch.object(r,'load_parent_reference',lambda:evidence))
                    stack.enter_context(patch.object(r,'verify_unchanged',verify_inputs))
                    stack.enter_context(patch.object(torch.Tensor,'cuda',cpu_cuda))
                    code=r.main()
        dirs=list(destination.glob('*_freeze_run'));assert len(dirs)==1
        report=read(dirs[0]/'run_report.json')
        assert code==expected_exit and report['status']==expected_status,report.get('traceback')
        with ZipFile(dirs[0]/'feedback_bundle.zip') as z:
            assert {'run_report.json',r.NOTEBOOK,'notebook_support.py'}.issubset(z.namelist())
            nb=json.loads(z.read(r.NOTEBOOK))
            assert all(not c.get('outputs') and c.get('execution_count') is None for c in nb['cells'] if c['cell_type']=='code')
            assert not any(n.endswith('.pt') for n in z.namelist())
        if code==0:
            assert report['completed_epoch']==51 and report['continuation_completed'] is False
            assert report['history_export_validation']['exact_trace_match']
            assert report['history_export_validation']['epochs']==1
        else:assert not (dirs[0]/'checkpoints').exists()
        main_cases[name]={'exit_code':code,'status':report['status'],'unexecuted_notebook_in_feedback':True,
            'output_directory':str(dirs[0])}
    cli_checks={}
    for arguments in [['run','--stop-after-epoch','50'],['run','--stop-after-epoch','56'],
                      ['run','--device','cpu'],['run','--synthetic'],['resume']]:
        result=subprocess.run([sys.executable,str(package/'run_train.py'),*arguments],cwd=out,text=True,capture_output=True)
        assert result.returncode==2,(arguments,result.stdout,result.stderr)
        cli_checks[' '.join(arguments)]={'exit_code':2,'message':result.stderr.splitlines()[-1]}
    assert not (out/'mx_runs').exists()
    for name,digest in new_binding.items():assert r.core.file_sha256(package/name)==digest
    assert r.core.file_sha256(checkpoint)==parent_digest
    result={'status':'PASS','synthetic_fixture_only':True,'CUDA_tested':False,
        'no_SVHN_data_loaded_or_downloaded':True,'not_a_SVHN_accuracy_or_GPU_timing_result':True,
        'scope':'Read/revalidate existing numerical integration_v2 evidence; rerun only corrected main entry and early failure.',
        'original_integration_terminated_after_numerical_checks':'Notebook exclusive-create duplicate in main export; corrected only write_notebook.',
        'pre_fix_code_binding':old_binding,'code_binding':new_binding,'changed_AST_functions':changed,
        'all_other_runner_AST_and_core_files_identical':True,
        'parent_checkpoint_sha256':parent_digest,'fixture_epochs_previously_executed':55,
        'new_synthetic_training_epochs_in_successful_main_entry':1,'new_synthetic_optimizer_updates_in_successful_main_entry':2,
        'reporting_tail_reuses_completed_main_entry':reuse_completed_entry,
        'previously_executed_cases_revalidated':summaries,'main_cases':main_cases,'CLI_rejections':cli_checks,
        'tests':{'CPU_resume52_exact_model_optimizer_scheduler_RNG':True,'theta_and_all10_choices_exact_frozen':True,
            'ten_factor_counts_and_zero_jump_metrics':True,'four_bit_independent_export_oracle_and_reconstruction':True,
            'weights_and_BN_continue':True,'partial_and_softstop_incomplete':True,
            'history_export_readback_complete':True,'idempotent_existing_notebook_allows_main_export':True,
            'early_missing_parent_FAIL_keeps_blank_notebook':True,'original_parent_unchanged':True},
        'test_file_sha256':r.core.file_sha256(Path(__file__)),'duration_seconds':time.perf_counter()-start}
    dump(out/'training_integration_report.json',result)
    print(json.dumps({'status':'PASS','report':str(out/'training_integration_report.json'),
        'duration_seconds':result['duration_seconds']},indent=2))


if __name__=='__main__':
    if len(sys.argv) not in (5,6):raise SystemExit('Usage: test_reporting_tail.py PACKAGE OUTPUT_DIRECTORY PARENT_FIXTURE EXISTING_INTEGRATION [--reuse-completed-entry]')
    if len(sys.argv)==6:assert sys.argv[5]=='--reuse-completed-entry'
    main(*sys.argv[1:5],reuse_completed_entry=len(sys.argv)==6)
