"""CPU-only synthetic integration for the independent E2 freeze50 full100 run.

The test constructs a genuine 100-epoch learning-control trace on three synthetic
images, pins its epoch50 checkpoint in a separate synthetic reference and tests
50->100 and 50->75->100, including the LR boundaries at 76 and 91. It never loads SVHN or changes real reference files.
All runtime/parent substitutions below are confined to unittest.mock patches in
this test; production parent checks and CUDA requirements remain strict.
"""
from __future__ import annotations
from contextlib import redirect_stdout,ExitStack
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import torch
import run_train as r
import trunc_support as t
import trunc_adapter as a
import svhn_support as s
import resume_core as core
import freeze_support as f
import test_freeze


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self,split):
        self.split=split
        gen=torch.Generator().manual_seed(446 if split=='train' else 887)
        self.data=torch.randn((3,3,40,40),generator=gen)
        self.labels=torch.tensor([0,1,2])
    def __len__(self):return 3
    def __getitem__(self,index):return self.data[index],int(self.labels[index])


def loaders():
    return {name:s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(SyntheticDataset(split)),
            batch_size=2,shuffle=name=='train',num_workers=0,drop_last=False))
            for name,split in [('train','train'),('test','test')]}


def complete_fixture_files(destination,names=('continuous100','stop75','resume75_to100')):
    """Wrap measured direct-run exports in explicitly synthetic notebook metadata."""
    config=r.config();config.update(train_examples=3,test_examples=3,batch_size=2)
    for name in names:
        out=destination/name
        summary=read(out/'training_summary.json');history=read(out/'history.json')
        selection=read(out/'checkpoint_selection.json')
        report={**summary,'schema':'mx-triple-independent-e2-scale-freeze50-reference100-report-v1',
            'status':'PASS' if summary['branch_training_completed'] else 'STOPPED_AT_EPOCH_BOUNDARY',
            'config':config,'command':'resume' if name.startswith('resume') else 'run',
            'output_directory':str(out),'completed_epoch':summary['completed_epochs'],
            'initial_base_model_sha256':history['initial_base_model_sha256'],
            'initial_model_sha256':history['initial_model_sha256'],
            'parent_checkpoint':history['parent_checkpoint'],'freeze_transition':history['freeze_transition'],
            'checkpoint_selection':selection,'training_summary':summary,
            'timing_estimate':read(out/'timing_estimate.json'),'warnings':[],
            'synthetic_fixture_only':True,
            'fixture_metadata_note':'Test-harness wrapper around the measured direct production loop/export; no SVHN run.'}
        write(out/'run_report.json',report)
        r.write_notebook(out)
        r.archive_initialization_reference(out)


def main(destination):
    destination=destination.resolve();destination.mkdir(exist_ok=False,parents=True)
    started=time.perf_counter();torch.set_num_threads(1)
    real_hashes={str(p.relative_to(ROOT)):core.file_sha256(p) for d in r.REFERENCE_DIRECTORIES
                 for p in (ROOT/d).rglob('*') if p.is_file()}
    source_dir=destination/'source';source_dir.mkdir()
    source,info=s.legacy.prepare_source(source_dir)
    modules=s.import_source(source)
    spec=importlib.util.spec_from_file_location('learning_control_runner',ROOT/'local_validation/learning_control_runner.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    parent_config=copy.deepcopy(read(ROOT/'independent_full100_reference/run_report.json')['config'])
    parent_config.update(train_examples=3,test_examples=3,batch_size=2)
    def context(c):
        rt=s.runtime(c['seed'],'cpu',strict=False)
        model=t.build_model(modules,'joint',rt.device)
        optimizer=t.build_optimizer(modules,model,c)
        scheduler=modules.optimizer.get_lr_scheduler('adam',optimizer,c['scheduler_milestones'])
        contract={'config':c,'environment':rt.environment,'settings':rt.settings,
            'source_files':info['workcopy_files_sha256'],'package_sha256':{},
            'data_files':{},'dataset_selection':{},'transforms':{},'synthetic_fixture_only':True}
        return SimpleNamespace(rt=rt,source=source,modules=modules,model=model,optimizer=optimizer,
            scheduler=scheduler,loaders=loaders(),contract=contract,c=c,report={},source_info=info)
    with patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self):
        control=context(parent_config)
        control_dir=destination/'learning_control100';control_dir.mkdir()
        with (destination/'learning_control100.log').open('w') as output,redirect_stdout(output):
            history=old.run_epochs(control,control_dir,lambda _:None,100,install_signal_handler=False)
        # This reference is explicitly synthetic and physically separate from all
        # research data. It contains complete observed history, not extrapolation.
        refroot=destination/'synthetic_reference_root';refdir=refroot/f.REFERENCE_NAME;refdir.mkdir(parents=True)
        parent_path=control_dir/'checkpoints/epoch_00050.pt'
        parent_state=core.load_verified(parent_path)
        report={'status':'PASS','completed_epoch':100,'config':parent_config,'synthetic_fixture_only':True,
                'initial_base_model_sha256':history['initial_base_model_sha256']}
        write(refdir/'run_report.json',report);write(refdir/'history.json',history)
        marker={'parent_epoch':50,'completed_epochs':100,'parent_checkpoint_sha256':core.file_sha256(parent_path),
                'parent_contract_sha256':core.json_digest(parent_state['contract']),
                'default_parent_checkpoint_relative':str(parent_path),
                'files':{n:core.file_sha256(refdir/n) for n in ('run_report.json','history.json')},
                'synthetic_fixture_only':True}
        write(refdir/'REFERENCE.json',marker)
        reference=f.load_reference(refroot)
        verified=f.verify_parent(parent_path,reference)
        assert verified['completed_epoch']==50
        config=r.config();config.update(train_examples=3,test_examples=3,batch_size=2)
        assert config['branch_stop_epoch'] == 100 and config['branch_new_epochs'] == 50
        assert config['branch_start_epoch'] == 51 and config['inherited_epochs'] == 50
        assert config['scheduler_milestones'] == [50,75,90]
        assert config['scale_learning_end_epoch'] == config['scale_freeze_epoch'] == 50
        assert 'pilot_stop_epoch' not in config and 'pilot_new_epochs' not in config
        def child():
            ctx=context(config)
            ctx.parent_state=core.load_verified(parent_path);ctx.parent_reference=reference;ctx.parent_path=parent_path
            ctx.contract['parent_checkpoint']=f.linkage(reference)
            ctx.report['parent_checkpoint']=f.linkage(reference)
            ctx.report['parent_checkpoint_path']=str(parent_path)
            f.verify_environment(ctx,ctx.parent_state)
            return ctx
        checked=child();check_dir=destination/'check';check_dir.mkdir()
        with (destination/'check.log').open('w') as output,redirect_stdout(output):
            check=test_freeze.check_parent_transition(checked,check_dir,lambda _:None)
        write(destination/'check_report.json',check)
        cases={};contexts={};traces={}
        for name,stop,resume in [('continuous100',100,None),('stop75',75,None),
                                ('resume75_to100',100,destination/'stop75/checkpoints/epoch_00075.pt')]:
            out=destination/name;out.mkdir();ctx=child();contexts[name]=ctx
            with (destination/(name+'.log')).open('w') as output,redirect_stdout(output):
                trace=r.run_epochs(ctx,out,lambda _:None,stop,resume,install_signal_handler=False)
                # Preserve the production output functions. References and the
                # notebook export are independently checked by notebook tests.
                with patch.object(r,'archive_initialization_reference',lambda _out:None),\
                     patch.object(r,'write_notebook',lambda _out:None):
                    r.export_results(ctx,out,trace,lambda _:None)
            traces[name]=trace
            cases[name]={'last_epoch':trace['epochs'][-1]['epoch'],'new_epochs':len(trace['epochs'])-50,
                         'output_directory':str(out),'branch_training_completed':ctx.report['branch_training_completed']}
            finished = stop == 100
            assert ctx.report['branch_training_completed'] is finished
            assert ctx.report['full_training_completed'] is finished
            assert ctx.report['full_100_epoch_training_completed'] is finished
            assert 'pilot_completed' not in ctx.report
            summary=read(out/'training_summary.json')
            assert summary['branch_completed_epochs'] == stop - 50
            assert summary['branch_stop_epoch'] == 100
            assert summary['branch_training_completed'] is finished
            assert summary['full_100_epoch_training_completed'] is finished
            timing=read(out/'timing_estimate.json')
            branch_seconds=sum(row['total_epoch_seconds'] for row in trace['epochs'][50:])
            assert math.isclose(timing['measured_new_branch_epoch_seconds'],branch_seconds)
            assert math.isclose(timing['measured_completed_epoch_seconds'],branch_seconds)
            assert timing['new_measured_epochs'] == list(range(51,stop+1))
            assert timing['remaining_epochs_to100'] == 100-stop
            assert timing['full_100_epoch_training_completed'] is finished
            assert timing['measured_100_epoch_seconds'] is None
            assert timing['measured_50_epoch_branch_seconds'] == (branch_seconds if finished else None)
            assert math.isclose(timing['combined_inherited_and_branch_epoch_seconds'],
                                timing['measured_inherited_epoch_seconds']+branch_seconds)
            if finished:
                assert timing['estimated_remaining_to100_seconds'] == 0
            assert trace['epochs'][:50]==history['epochs'][:50]
            for row in trace['epochs'][50:]:
                assert row['scale_state']==history['epochs'][49]['scale_state']
                assert row['scale_lr_used']==row['scale_lr_next']==0
                expected_lr = config['lr'] * config['scheduler_gamma'] ** sum(
                    milestone < row['epoch'] for milestone in config['scheduler_milestones'])
                assert math.isclose(row['lr_used'], expected_lr, rel_tol=1e-12, abs_tol=0), (
                    row['epoch'], row['lr_used'], expected_lr)
                assert row['gradient_checks']['scale_optimizer_steps']==0
                assert row['freeze_checks']['scale_optimizer_state_unchanged']
                for phase in ('train','test'):
                    assert row['coverage'][phase]['order_sha256']==history['epochs'][row['epoch']-1]['coverage'][phase]['order_sha256']
                assert all(x['phi_changed_blocks']==x['net_exponent_changes']==x['exponent_flips_during_epoch']==0
                           and x['scale_gradients_enabled'] is False for x in row['scale_epoch_metrics'])
            assert ctx.report['freeze_comparison']['all_sample_orders_match']
        left=core.load_verified(destination/'continuous100/checkpoints/epoch_00100.pt')
        right=core.load_verified(destination/'resume75_to100/checkpoints/epoch_00100.pt')
        for name in ('model','optimizer','scheduler','rng'):
            assert not core.differences(left[name],right[name]),name
        # The exact CPU equality also crosses both remaining LR milestones.
        by_epoch={row['epoch']:row for row in traces['continuous100']['epochs']}
        for epoch,expected in ((51,1e-4),(75,1e-4),(76,1e-5),(90,1e-5),(91,1e-6),(100,1e-6)):
            assert math.isclose(by_epoch[epoch]['lr_used'],expected,rel_tol=1e-12,abs_tol=0)
        assert left['scheduler']['last_epoch'] == 100
        for before,after in zip(traces['continuous100']['epochs'],traces['resume75_to100']['epochs']):
            for key in ('train','test','model_sha256','scale_state','source_scale_state','scale_epoch_metrics'):
                assert before[key]==after[key],(before['epoch'],key)
        for name in a.TARGET_LAYERS:
            phi=contexts['continuous100'].model.get_submodule(name).quantize_fn.phi
            assert not phi.requires_grad and phi.grad is None
            assert int(contexts['continuous100'].optimizer.state[phi]['step'])==100
        assert int(next(iter(contexts['continuous100'].optimizer.state.values()))['step'])==200
        assert sum(contexts['continuous100'].model.get_submodule(name).quantize_fn.phi.numel()
                   for name in a.TARGET_LAYERS)==4080
        # Invalid target epochs fail before creating output or changing the model.
        for invalid_stop in (50,101):
            rejected_target=child()
            initial_target=s.legacy.state_digest(rejected_target.model.state_dict())
            invalid_out=destination/f'invalid_stop_{invalid_stop}'
            try:r.run_epochs(rejected_target,invalid_out,lambda _:None,invalid_stop,None,False)
            except RuntimeError:pass
            else:raise AssertionError(f'Invalid final epoch accepted: {invalid_stop}')
            assert not invalid_out.exists()
            assert s.legacy.state_digest(rejected_target.model.state_dict())==initial_target
        # Strict wrong-contract rejection before any model restore/update.
        rejected=child();initial=s.legacy.state_digest(rejected.model.state_dict())
        foreign=core.load_verified(destination/'stop75/checkpoints/epoch_00075.pt');foreign['contract']['config']['schema']='wrong-contract'
        bad=destination/'foreign_child.pt';core.atomic_save(bad,foreign)
        (destination/'foreign_rejected').mkdir()
        try:r.run_epochs(rejected,destination/'foreign_rejected',lambda _:None,100,bad,False)
        except RuntimeError:pass
        else:raise AssertionError('foreign child not rejected')
        assert s.legacy.state_digest(rejected.model.state_dict())==initial
        # Wrong parent SHA, then correct hash with tampered model/trace metadata.
        wrong=copy.deepcopy(reference);wrong['meta']['parent_checkpoint_sha256']='0'*64
        try:f.verify_parent(parent_path,wrong)
        except RuntimeError:pass
        else:raise AssertionError('foreign parent accepted')
        malformed=copy.deepcopy(parent_state);malformed['trace']['epochs'][0]['test'][0]['top1_percent']+=1
        bad_parent=destination/'tampered_parent.pt';core.atomic_save(bad_parent,malformed)
        wrong=copy.deepcopy(reference);wrong['meta']['parent_checkpoint_sha256']=core.file_sha256(bad_parent)
        try:f.verify_parent(bad_parent,wrong)
        except RuntimeError:pass
        else:raise AssertionError('altered parent trace accepted')
        assert core.file_sha256(parent_path)==marker['parent_checkpoint_sha256']
    assert all(core.file_sha256(ROOT/name)==digest for name,digest in real_hashes.items())
    result={'status':'PASS','synthetic_fixture_only':True,'no_SVHN_data_or_training':True,
        'scope':'actual learning-control100 CPU trace, pinned parent50, freeze/check/export through100 and resume after75 on three synthetic images',
        'cases':cases,'check_report':check,'parent_checkpoint_read_only':True,
        'parent_phi_and_hard_E2_scales_and_Adam_unchanged_after_every_update':True,
        'weight_and_BN_updates_continue':True,'all_sample_orders_match_learning_control':True,
        'continuous100_equals_stop75_resume100_model_Adam_scheduler_RNG_exact_CPU':True,
        'scale_Adam_steps_remain100_weight_Adam_steps200':True,
        'weight_LR_boundaries_51_76_91_preserved':True,
        'all_4080_scale_parameters_frozen':True,
        'partial75_and_complete100_flags_and_timing_correct':True,
        'invalid_final_epochs50_and101_rejected_before_mutation':True,
        'wrong_contract_parent_hash_and_parent_trace_rejected':True,
        'real_reference_files_unchanged':True,'production_guard_modified':False,
        'reference_root':str(refroot),'test_seconds':time.perf_counter()-started,
        'note':'CPU substitutes only in this harness; not a GPU timing or SVHN accuracy result.'}
    write(destination/'runner_integration_report.json',result)
    complete_fixture_files(destination)
    print(json.dumps({'status':'PASS','report':str(destination/'runner_integration_report.json'),
                      'seconds':result['test_seconds']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    main(parser.parse_args().destination)
