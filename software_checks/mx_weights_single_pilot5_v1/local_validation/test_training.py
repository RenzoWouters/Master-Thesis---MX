"""Offline synthetic CPU tests. Only this file patches Tensor.cuda for CPU tests.

Production has no CPU, subset or synthetic bypass. Reports here are NOT SVHN
accuracies, GPU timings or an assertion of GPU bit-exact resumability.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
import svhn_support as s
import single_support as t
import run_train as r
import resume_core as core
import mx_adapter as a
import mx_format as f
import test_mx
import smoke_single


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    logs=[]
    log=logs.append
    srcdir=out/'source';srcdir.mkdir()
    source, source_info=s.legacy.prepare_source(srcdir)
    modules=s.import_source(source)
    rt=s.runtime(20261009,'cpu',strict=False)
    tests={'codec_and_model':test_mx.run_tests(modules,'cpu'),
           'two_updates_per_mode':smoke_single.check(modules,rt)}
    g=torch.Generator().manual_seed(778)
    x=torch.rand(6,3,40,40,generator=g)*2-1
    y=torch.tensor([0,1,2,3,4,5])
    dataset=torch.utils.data.TensorDataset(x,y)
    test_dataset=torch.utils.data.TensorDataset(x[:4],y[:4])
    all_traces={}
    contexts={}
    def context(path,mode):
        path.mkdir()
        rt=s.runtime(20261009,'cpu',strict=False)
        c=r.config(mode);c.update(train_examples=6,test_examples=4,batch_size=4)
        c['purpose']='SYNTHETIC_CPU_TEST_ONLY'
        model=t.build_model(modules,mode,rt.device)
        opt=modules.optimizer.get_optimizer_config(model,'adam',.001,0.)
        scheduler=modules.optimizer.get_lr_scheduler('adam',opt,[50,75,90])
        loaders={phase:s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(ds),batch_size=4,
                  shuffle=phase=='train',num_workers=0,pin_memory=False,drop_last=False))
                  for phase,ds in [('train',dataset),('test',test_dataset)]}
        contract={'config':c,'environment':rt.environment,'settings':rt.settings,
                  'source':source_info['workcopy_files_sha256'],'synthetic_fixture_only':True}
        report={'status':'PASS','schema':'mx-weights-single-pilot5-report-v1','config':c,
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
        for mode in ('control','mx'):
            base=context(out/(mode+'_baseline'),mode)
            # Control: compare against the original W32 model, not our wrapper.
            if mode=='control':
                state=copy.deepcopy(base.model.state_dict())
                original=modules.models.svhnq(s.BITS,expand=2)
                original.load_state_dict(state)
                base.model=original
                base.optimizer=modules.optimizer.get_optimizer_config(original,'adam',.001,0.)
                base.scheduler=modules.optimizer.get_lr_scheduler('adam',base.optimizer,[50,75,90])
                # Restore the post-initialization RNG of build_model. Constructing
                # the temporary original must not shift dropout or sample order.
                reference=context(out/'rng_reference',mode)
                rng=core.capture_rng({},rt.device)
                core.restore_rng(rng,{},rt.device)
            plain_loaders={phase:torch.utils.data.DataLoader(ds,batch_size=4,shuffle=phase=='train',
                            num_workers=0,pin_memory=False,drop_last=False)
                            for phase,ds in [('train',dataset),('test',test_dataset)]}
            raw,ast_hash=t.load_single_forward(source,modules.utils,log)
            metrics=[]
            for epoch in range(5):
                base.model.train()
                train=raw(plain_loaders['train'],base.model,torch.nn.CrossEntropyLoss(),t.ForbiddenDistillation(),epoch,
                          True,base.optimizer,None)
                base.model.eval()
                test=raw(plain_loaders['test'],base.model,torch.nn.CrossEntropyLoss(),t.ForbiddenDistillation(),epoch,False)
                base.scheduler.step()
                metrics.append((train,test))
            expected={'model':core.cpu_copy(base.model.state_dict()),'optimizer':core.cpu_copy(base.optimizer.state_dict()),
                      'scheduler':core.cpu_copy(base.scheduler.state_dict()),'rng':core.capture_rng({},rt.device)}
            path=out/(mode+'_run');ctx=context(path,mode)
            trace=r.run_epochs(ctx,path,log,5,install_signal_handler=False)
            for key,value in [('model',ctx.model.state_dict()),('optimizer',ctx.optimizer.state_dict()),
                              ('scheduler',ctx.scheduler.state_dict()),('rng',core.capture_rng({},rt.device))]:
                diff=core.differences(expected[key],value)
                assert not diff,(mode,key,diff[:4])
            for ep,record in enumerate(trace['epochs']):
                for phase_index,phase in enumerate(('train','test')):
                    for metric_index,key in enumerate(('objective_loss','top1_percent','top5_percent')):
                        assert metrics[ep][phase_index][metric_index]==[row[key] for row in record[phase]]
            tests[mode+'_original_function_vs_instrumented_model_optimizer_scheduler_rng_metrics_exact_CPU']=True
            all_traces[mode]=trace;contexts[mode]=ctx
            r.export_results(ctx,path,trace,log)
            s.write_json(path/'run_report.json',ctx.report)
            r.feedback_bundle(path)
            partial_path=out/(mode+'_partial');partial=context(partial_path,mode)
            r.run_epochs(partial,partial_path,log,2,install_signal_handler=False)
            resume_path=out/(mode+'_resume');resumed=context(resume_path,mode)
            checkpoint=partial_path/'checkpoints/epoch_00002.pt'
            traced=r.run_epochs(resumed,resume_path,log,5,checkpoint,False)
            for key,value in [('model',resumed.model.state_dict()),('optimizer',resumed.optimizer.state_dict()),
                              ('scheduler',resumed.scheduler.state_dict()),('rng',core.capture_rng({},rt.device))]:
                assert not core.differences(expected[key],value),(mode,key)
            for x1,x2 in zip(trace['epochs'],traced['epochs']):
                for key in ('train','test','model_sha256','best_epoch_so_far','weight_updates','gradient_checks'):
                    assert x1[key]==x2[key],(mode,key)
                assert x1['coverage']['train']['order_sha256']==x2['coverage']['train']['order_sha256']
            tests[mode+'_resume_2_plus_3_matches_5_continuous_CPU']=True
            bad=copy.deepcopy(resumed.contract);bad['config']['mode']='another_mode'
            state=core.load_verified(checkpoint)
            must_fail(lambda:core.restore_checkpoint(state,resumed.model,resumed.optimizer,resumed.scheduler,{},rt.device,bad))
            must_fail(lambda:core.atomic_save(checkpoint,state))
            corrupt=out/(mode+'_corrupt.pt');corrupt.write_bytes(checkpoint.read_bytes()+b'corrupt')
            corrupt.with_suffix('.pt.sha256').write_text(core.file_sha256(checkpoint))
            must_fail(lambda:core.load_verified(corrupt))
            tests[mode+'_contract_overwrite_and_corruption_rejected']=True
        assert all_traces['mx']['initial_model_sha256']==all_traces['control']['initial_model_sha256']
        for x1,x2 in zip(all_traces['mx']['epochs'],all_traces['control']['epochs']):
            for phase in ('train','test'):
                assert x1['coverage'][phase]['order_sha256']==x2['coverage'][phase]['order_sha256']
        tests['control_and_mx_same_initialization_and_sample_order']=True
        stopped_path=out/'stop_test';stopped=context(stopped_path,'mx')
        (stopped_path/'STOP_AFTER_EPOCH').touch()
        stopped_trace=r.run_epochs(stopped,stopped_path,log,5,install_signal_handler=False)
        assert len(stopped_trace['epochs'])==1
        tests['stop_marker_finishes_current_epoch']=True
    # Same original scheduler: cheap parameter-only check, no 100-epoch dataset run.
    p=torch.nn.Parameter(torch.tensor([1.]))
    model=torch.nn.ParameterList([p])
    opt=modules.optimizer.get_optimizer_config(model,'adam',.001,0.)
    scheduler=modules.optimizer.get_lr_scheduler('adam',opt,[50,75,90])
    for epoch in range(1,101):
        expected=.001
        for m in [50,75,90]:
            if m<epoch: expected*=.1
        assert opt.param_groups[0]['lr']==expected
        opt.zero_grad();p.grad=torch.ones_like(p);opt.step();scheduler.step()
    tests['original_lr_transitions_50_75_90_not_compressed']=True
    # Independent tie/selection guard.
    fake=[{'epoch':i+1,'test':[{'top1_percent':v}]} for i,v in enumerate([20,80,80,70,79])]
    assert t.best_record(fake)['epoch']==2
    tests['first_strict_maximum_of_active_mode']=True
    # Exported decoded weights and fresh reconstruction after training.
    for suffix in ('best','last'):
        payload=torch.load(out/'mx_run'/f'mx_source_{suffix}.pt',weights_only=True)
        reconstructed=a.load_exported_model(modules.models,payload,'cpu')
        assert not any(n+'.weight' in dict(reconstructed.named_parameters()) for n in a.TARGET_LAYERS)
        with torch.no_grad(): assert torch.isfinite(reconstructed(x[:2])).all()
    tests['trained_mx_source_export_reconstructs_without_target_masters']=True
    # Execute the read-only result notebook, including paired-control comparisons.
    import nbformat
    from nbclient import NotebookClient
    nb=nbformat.read(out/'mx_run'/'MX_single_training_resultaten.ipynb',as_version=4)
    for cell in nb.cells:
        if 'parameters' in cell.metadata.get('tags',[]):
            cell.source=f"from pathlib import Path\nRUN_DIR=Path({str(out/'mx_run')!r})\nCONTROL_DIR=Path({str(out/'control_run')!r})"
    NotebookClient(nb,timeout=90,kernel_name='python3').execute(cwd=str(out/'mx_run'))
    nbformat.write(nb,out/'notebook_executed.ipynb')
    import base64
    for i,cell in enumerate(nb.cells):
        for j,item in enumerate(cell.get('outputs',[])):
            if 'image/png' in item.get('data',{}):
                (out/f'notebook_cell{i}_{j}.png').write_bytes(base64.b64decode(item.data['image/png']))
    tests['notebook_executed_including_optional_control_comparison']=True
    result={'status':'PASS','scope':'SYNTHETIC_CPU_ONLY','python':sys.version,
            'torch':str(torch.__version__),'numpy':np.__version__,'real_svhn_training':False,
            'gpu_training_tested':False,'tests':tests}
    s.write_json(out/'validation_report.json',result)
    (out/'validation.log').write_text('\n'.join(logs),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':list(tests)},indent=2))

if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
