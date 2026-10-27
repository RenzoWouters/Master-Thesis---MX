"""Exercise real create_context() with tiny files and full-shape virtual datasets.
Only CPU selection and external dataset loading are substituted. No SVHN data,
training epoch or optimizer update is allowed. The fixture reproduces v1 failure
and demonstrates B64 reaches model/optimizer/loader construction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--expect-bug',action='store_true')
args=parser.parse_args()
package=args.package.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(package));sys.dont_write_bytecode=True
import torch
import numpy as np
from PIL import Image
import run_train as runner
import svhn_support as support

torch.set_num_threads(2)
cache=out/'fixture_data';cache.mkdir()
fixture_bytes={'train_32x32.mat':b'SYNTHETIC train metadata fixture; not a MAT dataset\n',
               'test_32x32.mat':b'SYNTHETIC test metadata fixture; not a MAT dataset\n'}
for name,data in fixture_bytes.items():(cache/name).write_bytes(data)
# Deliberately present: production ensure_dataset must leave extra untouched.
extra=cache/'extra_32x32.mat';extra.write_bytes(b'EXTRA MUST NOT BE OPENED BY THIS TRAIN-ONLY TEST\n')
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir()}
real_runtime=support.runtime;real_import=support.import_source;real_hashes=support.hashes
calls={'dataset_splits':[],'hashed_files':[],'optimizer_updates':0,'loader_items':0}

class VirtualSVHN:
    def __init__(self,split,transform):
        self.split=split;self.transform=transform
        self.n=73257 if split=='train' else 26032
        self.data=np.broadcast_to(np.zeros((1,3,32,32),dtype=np.uint8),(self.n,3,32,32))
        self.labels=np.arange(self.n,dtype=np.int64)%10
    def __len__(self):return self.n
    def __getitem__(self,index):
        calls['loader_items']+=1
        value=Image.fromarray(self.data[index].transpose(1,2,0))
        return self.transform(value),int(self.labels[index])

def import_with_dataset_io_fixture(source):
    modules=real_import(source)
    split_list=dict(modules.data.SVHNMIX.split_list)
    for split in ('train','test'):
        original=split_list[split]
        # Canonical names come from the unchanged upstream metadata.
        assert original[1]==split+'_32x32.mat',original
        split_list[split]=(original[0],original[1],hashlib.md5(fixture_bytes[original[1]]).hexdigest())
    modules.data.SVHNMIX.split_list=split_list
    def get_dataset(name,split,transform,download=False):
        assert name=='svhn' and split in ('train','val') and download is False
        calls['dataset_splits'].append(split)
        return VirtualSVHN('train' if split=='train' else 'test',transform)
    modules.data.get_dataset=get_dataset
    return modules

def cpu_runtime(seed,requested):
    assert requested=='cuda:0'
    return real_runtime(seed,'cpu',strict=False)

def tracked_hashes(path):
    assert path.name in fixture_bytes,'Unexpected file read: '+path.name
    calls['hashed_files'].append(path.name)
    return real_hashes(path)

def forbidden(*a,**k):
    raise AssertionError('No network, epochs, or optimizer updates allowed in startup test.')

report={'synthetic_fixture_only':True};logs=[]
request=SimpleNamespace(mode='joint',device='cuda:0',data_root=cache,download=False)
context_dir=out/'context';context_dir.mkdir()
with patch.object(support,'runtime',cpu_runtime),patch.object(support,'import_source',import_with_dataset_io_fixture),\
     patch.object(support,'hashes',tracked_hashes),patch.object(support,'urlopen',forbidden),\
     patch.object(runner,'run_epochs',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
    if args.expect_bug:
        try: runner.create_context(request,context_dir,report,logs.append,check_only=False)
        except RuntimeError as e:
            assert str(e)=='Train-only mag alleen train en test laden of controleren.',repr(e)
            outcome='v1_exact_user_failure_reproduced'
        else:raise AssertionError('Expected old filename guard to reject canonical SVHN names.')
        assert not (context_dir/'dataset_manifest.json').exists()
    else:
        ctx=runner.create_context(request,context_dir,report,logs.append,check_only=False)
        ctx.contract['synthetic_fixture_only']=True
        assert set(ctx.contract['data_files'])==set(fixture_bytes)
        assert ctx.c['mx_block_size']==64 and ctx.c['scale_factor']==64
        assert ctx.c['train_examples']==73257 and ctx.c['test_examples']==26032
        assert len(ctx.loaders['train'])==573 and len(ctx.loaders['test'])==204
        assert sum(p.numel() for p in ctx.model.parameters())==132954
        assert not ctx.optimizer.state and ctx.scheduler.last_epoch==0
        assert all(p.grad is None for p in ctx.model.parameters())
        for phase in ('train','test'):
            images,labels=next(iter(ctx.loaders[phase]))
            assert images.shape==(128,3,40,40) and labels.shape==(128,)
        assert calls['loader_items']==256
        assert not ctx.optimizer.state and all(p.grad is None for p in ctx.model.parameters())
        assert not list(context_dir.rglob('*.pt'))
        outcome='B64_real_create_context_and_first_loader_batches_pass'
assert calls['dataset_splits']==['train','val']
assert calls['hashed_files']==['train_32x32.mat','test_32x32.mat']
assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir()}
result={'status':'PASS','outcome':outcome,'package':str(package),'synthetic_fixture_only':True,
        'real_create_context_executed':True,'real_ensure_dataset_executed':True,
        'real_validate_dataset_executed':not args.expect_bug,
        'real_model_optimizer_scheduler_loaders_constructed':not args.expect_bug,
        'SVHN_downloaded_or_used':False,'training_epochs':0,'optimizer_updates':0,
        'raw_fixture_files_unchanged':True,'extra_file_not_hashed_or_loaded':True,
        'canonical_filenames':list(fixture_bytes),'calls':calls,
        'runtime_torch':str(torch.__version__),
        'runner_sha256':hashlib.sha256((package/'run_train.py').read_bytes()).hexdigest()}
(out/'startup_regression_report.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'startup.log').write_text('\n'.join(logs)+'\n')
print(json.dumps(result,indent=2))
