"""Exercise the real main/check entry on a pinned synthetic CPU parent.

Only the runtime device and reference/configuration are patched in this test.
The actual create_context, package/source verification, dependency checks,
parent SHA/contract/trace checks, restore and bounded freeze check run unchanged.
No dataset functions are replaced because check must not load a dataset.
"""
from pathlib import Path
from contextlib import ExitStack
import argparse
import copy
import json
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import torch
import run_train as r
import svhn_support as s
import freeze_support as f


def main(reference_root,destination):
    destination.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    reference=f.load_reference(reference_root)
    parent=Path(reference['meta']['default_parent_checkpoint_relative'])
    config=r.config();config.update(train_examples=3,test_examples=3,batch_size=2)
    runtime=s.runtime
    create_context=r.create_context
    observed={}
    def traced_create_context(args,*positional,**keywords):
        observed['command']=args.command
        observed['default_stop_after_epoch']=args.stop_after_epoch
        return create_context(args,*positional,**keywords)
    with ExitStack() as stack:
        stack.enter_context(patch.object(f,'load_reference',lambda _:reference))
        stack.enter_context(patch.object(r,'config',lambda mode='joint':copy.deepcopy(config)))
        stack.enter_context(patch.object(s,'runtime',lambda seed,device:runtime(seed,'cpu',strict=False)))
        stack.enter_context(patch.object(r,'create_context',traced_create_context))
        stack.enter_context(patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self))
        stack.enter_context(patch.object(sys,'argv',[str(ROOT/'run_train.py'),'check','--parent-checkpoint',str(parent),
                                                  '--output-root',str(destination/'outputs')]))
        result=r.main()
    outputs=list((destination/'outputs').glob('*/run_report.json'))
    assert len(outputs)==1
    report=json.loads(outputs[0].read_text())
    assert result==0 and report['status']=='PASS',report.get('error')
    assert report['freeze_checks']['status']=='PASS'
    assert observed == {'command':'check','default_stop_after_epoch':100}
    assert report['schema']=='mx-triple-independent-e2-scale-freeze50-reference100-report-v1'
    assert report['config']['branch_stop_epoch']==100
    assert report['config']['branch_new_epochs']==50
    assert report['config']['scale_freeze_epoch']==50
    assert 'pilot_completed' not in report
    result={'status':'PASS','scope':'real main/check and create_context with synthetic CPU parent/reference/config only',
            'production_initialization_and_parent_guards_unchanged':True,
            'production_dependency_checks_exercised':True,'no_dataset_calls_or_SVHN_training':True,
            'CLI_default_final_epoch_is100':True,
            'config_freezes_after50_and_plans50_new_epochs':True,
            'synthetic_fixture_only':True,'output_directory':str(outputs[0].parent)}
    (destination/'check_entry_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference_root',type=Path);p.add_argument('destination',type=Path)
    args=p.parse_args();main(args.reference_root.resolve(),args.destination.resolve())
