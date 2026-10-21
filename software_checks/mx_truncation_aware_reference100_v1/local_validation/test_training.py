"""Offline synthetic tests, never a real SVHN/GPU experiment.

Only here is Tensor.cuda replaced to exercise the original CUDA-only forward
on CPU. The production CLI has no CPU/synthetic bypass. Output must be new.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
import svhn_support as s
import trunc_support as t
import trunc_adapter as a
import run_train as r
import resume_core as core
import test_mx
import test_trunc_training as test_dual


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    logs=[];log=logs.append
    srcdir=out/'source';srcdir.mkdir()
    source, source_info=s.legacy.prepare_source(srcdir)
    modules=s.import_source(source)
    rt=s.runtime(20261009,'cpu',strict=False)
    tests={}
    g=torch.Generator().manual_seed(778)
    x=torch.rand(6,3,40,40,generator=g)*2-1
    y=torch.tensor([0,1,2,3,4,5])
    datasets={'train':torch.utils.data.TensorDataset(x,y),
              'test':torch.utils.data.TensorDataset(x[:4],y[:4])}
    def context(path):
        path.mkdir()
        rt=s.runtime(20261009,'cpu',strict=False)
        c=r.config();c.update(train_examples=6,test_examples=4,batch_size=4)
        c['purpose']='SYNTHETIC_CPU_TEST_ONLY'
        model=t.build_model(modules,'joint',rt.device)
        opt=modules.optimizer.get_optimizer_config(model,'adam',.001,0.)
        scheduler=modules.optimizer.get_lr_scheduler('adam',opt,[50,75,90])
        loaders={phase:s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(ds),batch_size=4,
                  shuffle=phase=='train',num_workers=0,pin_memory=False,drop_last=False))
                  for phase,ds in datasets.items()}
        contract={'config':c,'environment':rt.environment,'settings':rt.settings,
                  'source':source_info['workcopy_files_sha256'],'synthetic_fixture_only':True}
        report={'status':'PASS','schema':'mx-truncation-aware-reference100-report-v1','config':c,
                'environment':rt.environment,'execution_settings':rt.settings,'warnings':[],
                'dataset_files':{'files':{'synthetic_fixture':True}},'package_sha256':{'synthetic_test':True},
                'synthetic_fixture_only':True,'limitations':['SYNTHETIC CPU TEST ONLY']}
        return SimpleNamespace(rt=rt,c=c,source=source,source_info=source_info,modules=modules,
                    model=model,optimizer=opt,scheduler=scheduler,loaders=loaders,contract=contract,report=report)
    def must_fail(fn):
        try: fn()
        except (RuntimeError,ValueError,FileExistsError): return
        raise AssertionError('Expected rejection')
    with patch.object(torch.Tensor,'cuda',lambda tensor,*args,**kw:tensor):
        tests['existing_E2M3_regression']=test_mx.run_tests(modules,'cpu')
        tests['new_E2M1_joint_training_checks']=test_dual.run_tests(modules,rt,source,log)
        # Same numerical forward WITHOUT instrumentation versus full runner.
        base=context(out/'baseline')
        plain={phase:torch.utils.data.DataLoader(ds,batch_size=4,shuffle=phase=='train',num_workers=0,
                     pin_memory=False,drop_last=False) for phase,ds in datasets.items()}
        raw,ast_hash=t.load_dual_forward(source,modules.utils,log)
        expected_metrics=[]
        for epoch in range(100):
            base.model.train()
            train=raw(plain['train'],base.model,torch.nn.CrossEntropyLoss(),modules.losses.CrossEntropyLossSoft(),epoch,
                      True,base.optimizer,None)
            base.model.eval()
            test=raw(plain['test'],base.model,torch.nn.CrossEntropyLoss(),modules.losses.CrossEntropyLossSoft(),epoch,False)
            base.scheduler.step(); expected_metrics.append((train,test))
        expected={'model':core.cpu_copy(base.model.state_dict()),'optimizer':core.cpu_copy(base.optimizer.state_dict()),
                  'scheduler':core.cpu_copy(base.scheduler.state_dict()),'rng':core.capture_rng({},rt.device)}
        path=out/'joint_run';ctx=context(path)
        trace=r.run_epochs(ctx,path,log,100,install_signal_handler=False)
        for key,value in [('model',ctx.model.state_dict()),('optimizer',ctx.optimizer.state_dict()),
                          ('scheduler',ctx.scheduler.state_dict()),('rng',core.capture_rng({},rt.device))]:
            diff=core.differences(expected[key],value);assert not diff,(key,diff[:4])
        for ep,record in enumerate(trace['epochs']):
            for phase_index,phase in enumerate(('train','test')):
                for metric_index,key in enumerate(('objective_loss','top1_percent','top5_percent')):
                    assert expected_metrics[ep][phase_index][metric_index]==[row[key] for row in record[phase]]
        tests['hundred_epochs_uninstrumented_vs_full_runner_model_optimizer_scheduler_RNG_metrics_exact_CPU']=True
        r.export_results(ctx,path,trace,log)
        s.write_json(path/'run_report.json',ctx.report);r.feedback_bundle(path)
        # Resume from completed epoch boundaries of the same contract.
        # Test all three real LR milestones on synthetic data.
        for stop_epoch in (50,75,90):
            resume_path=out/f'resumed_after_{stop_epoch}'
            resumed=context(resume_path)
            checkpoint=path/'checkpoints'/f'epoch_{stop_epoch:05d}.pt'
            traced=r.run_epochs(resumed,resume_path,log,100,checkpoint,False)
            for key,value in [('model',resumed.model.state_dict()),('optimizer',resumed.optimizer.state_dict()),
                              ('scheduler',resumed.scheduler.state_dict()),('rng',core.capture_rng({},rt.device))]:
                assert not core.differences(expected[key],value),(stop_epoch,key)
            for x1,x2 in zip(trace['epochs'],traced['epochs']):
                for key in ('train','test','model_sha256','best_epoch_so_far','weight_updates','gradient_checks','lr_used','lr_next'):
                    assert x1[key]==x2[key],(stop_epoch,key)
                for phase in ('train','test'):
                    assert x1['coverage'][phase]['order_sha256']==x2['coverage'][phase]['order_sha256']
            if stop_epoch == 90:
                r.export_results(resumed,resume_path,traced,log)
                assert resumed.report['training_summary']['resumed']
                assert resumed.report['full_training_completed']
        tests['resume_after_50_75_90_matches_continuous100_CPU_dropout_two_BN_banks']=True
        bad=copy.deepcopy(resumed.contract);bad['config']['weight_bits']=[6]
        state=core.load_verified(checkpoint)
        must_fail(lambda:core.restore_checkpoint(state,resumed.model,resumed.optimizer,resumed.scheduler,{},rt.device,bad))
        must_fail(lambda:core.atomic_save(checkpoint,state))
        corrupt=out/'corrupt.pt';corrupt.write_bytes(checkpoint.read_bytes()+b'corrupt')
        corrupt.with_suffix('.pt.sha256').write_text(core.file_sha256(checkpoint))
        must_fail(lambda:core.load_verified(corrupt))
        old_contract=copy.deepcopy(resumed.contract)
        old_contract['config']['schema']='mx-truncation-aware-pilot5-protocol-v1'
        old_contract['config'].pop('target_final_epoch')
        old_contract['config']['pilot_stop_epoch']=5
        must_fail(lambda:core.restore_checkpoint(state,resumed.model,resumed.optimizer,resumed.scheduler,{},rt.device,old_contract))
        tests['changed_mode_or_pilot_contract_overwrite_corruption_rejected']=True
        stopped_path=out/'stop_test';stopped=context(stopped_path)
        (stopped_path/'STOP_AFTER_EPOCH').touch()
        stopped_trace=r.run_epochs(stopped,stopped_path,log,100,install_signal_handler=False)
        assert len(stopped_trace['epochs'])==1
        r.export_results(stopped,stopped_path,stopped_trace,log)
        assert not stopped.report['full_training_completed']
        assert stopped.report['training_summary']['completed_epochs']==1
        stopped.report['status']='STOPPED_AT_EPOCH_BOUNDARY'
        s.write_json(stopped_path/'run_report.json',stopped.report)
        tests['stop_marker_finishes_epoch_and_never_claims_full_completion']=True
        assert ctx.report['full_training_completed']
        assert ctx.report['optimizer_steps']==200
        assert ctx.report['training_forward_passes']==400
        assert ctx.report['selected_last_epoch']==100
        import csv
        with (path/'epoch_metrics.csv').open() as f: assert len(list(csv.DictReader(f)))==400
        assert len(list((path/'checkpoints').glob('epoch_*.pt')))==100
        assert len(list((path/'sample_order').glob('*.npy')))==200
        tests['hundred_checkpoints_400_metric_rows_200_sample_orders_and_completion']=True
        for key in ('lr_next','missing_mode','changed_scale_check','low_encoder','test_source'):
            badtrace=copy.deepcopy(trace)
            if key=='lr_next': badtrace['epochs'][49]['lr_next']=.001
            elif key=='missing_mode': badtrace['epochs'][2]['test'].pop()
            elif key=='changed_scale_check': badtrace['epochs'][2]['gradient_checks']['shared_source_checks']['scale_copies_unchanged']-=1
            elif key=='low_encoder': badtrace['epochs'][2]['gradient_checks']['shared_source_checks']['no_direct_low_quantizer']=False
            else: badtrace['epochs'][2]['test_source_checks']['high_encodings']-=1
            reject_path=out/('reject_'+key);reject_path.mkdir()
            must_fail(lambda:r.export_results(ctx,reject_path,badtrace,log))
        tests['wrong_next_LR_missing_mode_or_invalid_source_audits_rejected']=True
    # Positive, tie and conflicting low-precision peak selection tests.
    fake=[{'epoch':i+1,'test':[{'weight_bits':4,'top1_percent':lo},{'weight_bits':6,'top1_percent':hi}]}
          for i,(lo,hi) in enumerate([(20,30),(40,80),(99,80),(100,70),(90,79)])]
    assert t.best_record(fake)['epoch']==2
    tests['first_strict_E2M3_peak_not_low_mode_peak']=True
    p=torch.nn.Parameter(torch.tensor([1.]))
    opt=modules.optimizer.get_optimizer_config(torch.nn.ParameterList([p]),'adam',.001,0.)
    scheduler=modules.optimizer.get_lr_scheduler('adam',opt,[50,75,90])
    for epoch in range(1,101):
        lr=.001
        for m in (50,75,90):
            if m<epoch:lr*=.1
        assert opt.param_groups[0]['lr']==lr
        opt.zero_grad();p.grad=torch.ones_like(p);opt.step();scheduler.step()
    tests['unchanged_LR_milestones_50_75_90_parameter_only_test']=True
    # Exports reconstructed in a fresh subprocess without original checkpoints.
    import subprocess
    script="""import sys, torch
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import svhn_support as s
import trunc_adapter as a
folder=Path(sys.argv[2]);source=Path(sys.argv[3])
mods=s.import_source(source)
for file in folder.glob('mx_source_*.pt'):
    p=torch.load(file,weights_only=True,map_location='cpu')
    for bit in (6,4):
        m=a.load_exported_model(mods.models,p,'cpu',bit=bit)
        assert not any(n+'.weight' in dict(m.named_parameters()) for n in a.TARGET_LAYERS)
        with torch.no_grad(): assert torch.isfinite(m(torch.zeros(2,3,40,40))).all()
print('PASS: both modes reconstruct from high exports only in fresh process')
"""
    result=subprocess.run([sys.executable,'-c',script,str(ROOT),str(path),str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    (out/'fresh_process_export_test.txt').write_text(result.stdout+result.stderr)
    tests['both_modes_reconstruct_from_high_only_no_checkpoint']=True
    import nbformat
    from nbclient import NotebookClient
    nb=nbformat.read(path/'MX_truncation_aware_reference100_resultaten.ipynb',as_version=4)
    NotebookClient(nb,timeout=90,kernel_name='python3').execute(cwd=str(path))
    nbformat.write(nb,out/'notebook_executed.ipynb')
    import base64
    for i,cell in enumerate(nb.cells):
        for j,item in enumerate(cell.get('outputs',[])):
            if 'image/png' in item.get('data',{}):
                (out/f'notebook_cell{i}_{j}.png').write_bytes(base64.b64decode(item.data['image/png']))
    tests['read_only_hundred_epoch_notebook_executed']=True
    partial_nb=nbformat.read(stopped_path/'MX_truncation_aware_reference100_resultaten.ipynb',as_version=4)
    NotebookClient(partial_nb,timeout=90,kernel_name='python3').execute(cwd=str(stopped_path))
    nbformat.write(partial_nb,out/'partial_notebook_executed.ipynb')
    tests['partial_notebook_does_not_claim_100_epochs']=True
    result={'status':'PASS','scope':'SYNTHETIC_CPU_ONLY','python':sys.version,
            'torch':str(torch.__version__),'numpy':np.__version__,'real_svhn_training':False,
            'gpu_training_tested':False,'tests':tests}
    s.write_json(out/'validation_report.json',result)
    (out/'validation.log').write_text('\n'.join(logs),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':list(tests)},indent=2))

if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
