"""Offline synthetic CPU regression; no SVHN results or GPU time estimates.
Only this test patches Tensor.cuda. Production has no CPU/subset bypass.
"""
from __future__ import annotations
import ast, copy, json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
import svhn_support as s
import single_support as t
import run_train as r
import resume_core as core
import e2m1_adapter as a
import test_e2m1
import smoke_single


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    logs=[]; log=logs.append; tests={}
    def passed(label):
        tests[label]=True; print('PASS',label,flush=True)
    def must_fail(fn):
        try: fn()
        except (RuntimeError,ValueError,FileExistsError): return
        raise AssertionError('Expected rejection')
    srcdir=out/'source';srcdir.mkdir()
    source,source_info=s.legacy.prepare_source(srcdir)
    modules=s.import_source(source)
    rt=s.runtime(20261009,'cpu',strict=False)
    s.verify_package()
    manifest=json.loads((ROOT/'TRAINING_CHANGE_MANIFEST.json').read_text())
    for name,digest in manifest['unchanged_files_sha256'].items():
        assert core.file_sha256(ROOT/name)==digest
    def tree(path):
        fn=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='run_epochs')
        return ast.dump(fn,include_attributes=False)
    assert tree(ROOT/'provenance/single_reference100_run_train.py')==tree(ROOT/'run_train.py')
    passed('pilot_epoch_loop_AST_and_shared_modules_unchanged')
    tests['codec_and_model']=test_e2m1.run_tests(modules,'cpu')
    tests['two_updates_each_mode']=smoke_single.check(modules,rt)
    g=torch.Generator().manual_seed(778)
    x=torch.rand(6,3,40,40,generator=g)*2-1
    y=torch.tensor([0,1,2,3,4,5])
    ds=torch.utils.data.TensorDataset(x,y)
    td=torch.utils.data.TensorDataset(x[:4],y[:4])
    def context(path,mode):
        path.mkdir()
        s.write_json(path/'training_change_manifest.json',manifest)
        rt=s.runtime(20261009,'cpu',strict=False)
        c=r.config(mode);c.update(train_examples=6,test_examples=4,batch_size=4)
        c['purpose']='SYNTHETIC_CPU_TEST_ONLY'
        model=t.build_model(modules,mode,rt.device)
        opt=modules.optimizer.get_optimizer_config(model,'adam',.001,0.)
        sch=modules.optimizer.get_lr_scheduler('adam',opt,[50,75,90])
        loaders={ph:s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(data),batch_size=4,
                   shuffle=ph=='train',num_workers=0,pin_memory=False,drop_last=False))
                   for ph,data in [('train',ds),('test',td)]}
        contract={'config':c,'environment':rt.environment,'settings':rt.settings,
                  'source':source_info['workcopy_files_sha256'],'synthetic_fixture_only':True}
        report={'status':'PASS','schema':'mx-e2m1-dedicated-reference100-report-v1','config':c,
                'environment':rt.environment,'execution_settings':rt.settings,'warnings':[],
                'dataset_files':{'files':{'synthetic_fixture':True}},'package_sha256':s.verify_package(),
                'synthetic_fixture_only':True,'limitations':['SYNTHETIC CPU TEST ONLY']}
        return SimpleNamespace(rt=rt,c=c,source=source,source_info=source_info,modules=modules,
            model=model,optimizer=opt,scheduler=sch,loaders=loaders,contract=contract,report=report)
    traces={};contexts={}
    with patch.object(torch.Tensor,'cuda',lambda tensor,*args,**kw:tensor):
        for mode in ('mx',):
            print('BEGIN',mode,'100 mini-epochs',flush=True)
            base=context(out/(mode+'_baseline'),mode)
            if mode=='control':
                rng=core.capture_rng({},base.rt.device)
                original=modules.models.svhnq(s.BITS,expand=2)
                original.load_state_dict(base.model.state_dict())
                base.model=original
                base.optimizer=modules.optimizer.get_optimizer_config(original,'adam',.001,0.)
                base.scheduler=modules.optimizer.get_lr_scheduler('adam',base.optimizer,[50,75,90])
                core.restore_rng(rng,{},base.rt.device)
            plain={ph:torch.utils.data.DataLoader(data,batch_size=4,shuffle=ph=='train',num_workers=0,
                      pin_memory=False,drop_last=False) for ph,data in [('train',ds),('test',td)]}
            raw,ast_hash=t.load_single_forward(source,modules.utils,log)
            metrics=[]
            for ep in range(100):
                base.model.train()
                tr=raw(plain['train'],base.model,torch.nn.CrossEntropyLoss(),t.ForbiddenDistillation(),ep,True,base.optimizer,None)
                base.model.eval()
                te=raw(plain['test'],base.model,torch.nn.CrossEntropyLoss(),t.ForbiddenDistillation(),ep,False)
                base.scheduler.step();metrics.append((tr,te))
            expected={'model':core.cpu_copy(base.model.state_dict()),'optimizer':core.cpu_copy(base.optimizer.state_dict()),
                      'scheduler':core.cpu_copy(base.scheduler.state_dict()),'rng':core.capture_rng({},base.rt.device)}
            path=out/(mode+'_run');ctx=context(path,mode)
            trace=r.run_epochs(ctx,path,log,100,install_signal_handler=False)
            for key,value in [('model',ctx.model.state_dict()),('optimizer',ctx.optimizer.state_dict()),
                              ('scheduler',ctx.scheduler.state_dict()),('rng',core.capture_rng({},ctx.rt.device))]:
                dif=core.differences(expected[key],value);assert not dif,(mode,key,dif[:4])
            for ep,record in enumerate(trace['epochs']):
                for pi,ph in enumerate(('train','test')):
                    for mi,key in enumerate(('objective_loss','top1_percent','top5_percent')):
                        assert metrics[ep][pi][mi]==[v[key] for v in record[ph]]
                assert record['forward_ast_sha256']==ast_hash
            passed(mode+'_100_epochs_raw_vs_instrumented_model_optimizer_scheduler_rng_metrics_exact_CPU')
            traces[mode]=trace;contexts[mode]=ctx
            r.export_results(ctx,path,trace,log)
            s.write_json(path/'run_report.json',ctx.report);r.feedback_bundle(path)
            assert ctx.report['full_training_completed'] and ctx.report['training_summary']['full_100_epoch_training_completed']
            assert ctx.report['completed_epoch']==100 and ctx.report['optimizer_steps']==200
            assert ctx.report['selected_last_epoch']==100 and len(list((path/'checkpoints').glob('epoch_*.pt')))==100
            for role,record in [('best',t.best_record(trace['epochs'])),('last',trace['epochs'][-1])]:
                assert core.file_sha256(path/(role+'_checkpoint.pt'))==core.file_sha256(Path(record['checkpoint_path']))
            assert ctx.report['timing_estimate']['measured_100_epoch_seconds']==sum(z['total_epoch_seconds'] for z in trace['epochs'])
            passed(mode+'_100_checkpoints_best_last_LR_measured_completion')
            for cut in [50,75,90,99]:
                rp=out/f'{mode}_resume_{cut}';re=context(rp,mode)
                checkpoint=path/'checkpoints'/f'epoch_{cut:05d}.pt';before=core.file_sha256(checkpoint)
                ret=r.run_epochs(re,rp,log,cut+1,checkpoint,False)
                wanted=core.load_verified(path/'checkpoints'/f'epoch_{cut+1:05d}.pt')
                for key,value in [('model',re.model.state_dict()),('optimizer',re.optimizer.state_dict()),
                                  ('scheduler',re.scheduler.state_dict()),('rng',core.capture_rng({},re.rt.device))]:
                    assert not core.differences(wanted[key],value),(mode,cut,key)
                for key in ['train','test','model_sha256','best_epoch_so_far','weight_updates','gradient_checks','lr_used','lr_next']:
                    assert ret['epochs'][-1][key]==trace['epochs'][cut][key],(mode,cut,key)
                for ph in ['train','test']:
                    assert ret['epochs'][-1]['coverage'][ph]['order_sha256']==trace['epochs'][cut]['coverage'][ph]['order_sha256']
                assert core.file_sha256(checkpoint)==before
                if cut==99:
                    r.export_results(re,rp,ret,log);assert re.report['full_training_completed']
            passed(mode+'_resume_50_75_90_99_matches_next_continuous_epoch_CPU')
            checkpoint=path/'checkpoints/epoch_00050.pt';state=core.load_verified(checkpoint)
            for field,value in [('mode','other'),('schema','mx-weights-single-pilot5-protocol-v1')]:
                bad=copy.deepcopy(ctx.contract);bad['config'][field]=value
                must_fail(lambda:core.restore_checkpoint(state,ctx.model,ctx.optimizer,ctx.scheduler,{},ctx.rt.device,bad))
            must_fail(lambda:core.atomic_save(checkpoint,state))
            corrupt=out/(mode+'_corrupt.pt');corrupt.write_bytes(checkpoint.read_bytes()+b'corrupt')
            corrupt.with_suffix('.pt.sha256').write_text(core.file_sha256(checkpoint))
            must_fail(lambda:core.load_verified(corrupt))
            passed(mode+'_pilot_other_mode_overwrite_corruption_rejected')
        torch.manual_seed(20261009); m0=t.build_model(modules,'control','cpu')
        torch.manual_seed(20261009); m1=t.build_model(modules,'mx','cpu')
        assert not core.differences(m0.state_dict(),m1.state_dict())
        passed('MX_vs_control_identical_initialization_CPU')
        sp=out/'stop_test';sc=context(sp,'mx');(sp/'STOP_AFTER_EPOCH').touch()
        tr=r.run_epochs(sc,sp,log,100,install_signal_handler=False)
        assert len(tr['epochs'])==1
        r.export_results(sc,sp,tr,log)
        assert sc.report['full_training_completed'] is False
        assert sc.report['training_summary']['full_100_epoch_training_completed'] is False
        passed('soft_stop_not_reported_as_full_training')
    fake=[{'epoch':i+1,'test':[{'top1_percent':v}]} for i,v in enumerate([20,80,80,70,79])]
    assert t.best_record(fake)['epoch']==2
    passed('first_strict_maximum_retains_earlier_tie')
    for suffix in ['best','last']:
        payload=torch.load(out/'mx_run'/f'mx_source_{suffix}.pt',weights_only=True)
        model=a.load_exported_model(modules.models,payload,'cpu')
        assert not any(name+'.weight' in dict(model.named_parameters()) for name in a.TARGET_LAYERS)
        with torch.no_grad(): assert torch.isfinite(model(x[:2])).all()
    passed('E2M1_export_reconstructs_without_six_FP32_masters')
    import nbformat,base64
    from nbclient import NotebookClient
    nb=nbformat.read(out/'mx_run/MX_E2M1_dedicated_resultaten.ipynb',as_version=4)
    for cell in nb.cells:
        if 'parameters' in cell.metadata.get('tags',[]):
            cell.source=f"from pathlib import Path\nRUN_DIR=Path({str(out/'mx_run')!r})\nCONTROL_DIR=None"
    NotebookClient(nb,timeout=90,kernel_name='python3').execute(cwd=str(out/'mx_run'))
    nbformat.write(nb,out/'notebook_executed.ipynb')
    for i,cell in enumerate(nb.cells):
        for j,item in enumerate(cell.get('outputs',[])):
            if 'image/png' in item.get('data',{}):
                (out/f'notebook_cell{i}_{j}.png').write_bytes(base64.b64decode(item.data['image/png']))
    passed('dedicated_notebook_100_epoch_execution')
    result={'status':'PASS','scope':'SYNTHETIC_CPU_ONLY','python':sys.version,'torch':str(torch.__version__),
            'numpy':np.__version__,'real_SVHN_training':False,'gpu_training_tested':False,
            'dedicated_E2M1_epochs':100,'train_examples_per_synthetic_epoch':6,'test_examples_per_synthetic_epoch':4,
            'updates_per_synthetic_epoch':2,'tests':tests}
    s.write_json(out/'validation_report.json',result)
    (out/'validation.log').write_text('\n'.join(logs),encoding='utf-8')
    print(json.dumps({'status':'PASS','checks':list(tests)},indent=2),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
