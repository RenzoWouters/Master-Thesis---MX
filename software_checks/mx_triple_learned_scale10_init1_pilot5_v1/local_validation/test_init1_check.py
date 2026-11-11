"""Exercise actual check entry with CPU routing, without SVHN or CUDA training."""
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack,redirect_stdout
import json,sys,hashlib,time,copy,subprocess
P=Path(sys.argv[1]).resolve();OUT=Path(sys.argv[2]).resolve();OUT.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(P));sys.dont_write_bytecode=True
import run_train as r
import torch
import resume_core as core
torch.set_num_threads(2)
source_binding={n:core.file_sha256(P/n) for n in ['run_train.py','trunc_adapter.py','test_learned_scale.py']}
# Explicit synthetic CPU expectation from the unchanged original architecture.
pre=OUT/'oracle_source';pre.mkdir();source,info=r.s.legacy.prepare_source(pre)
# The oracle uses its own interpreter, because the actual check must import
# models/datasets/optimizer/utils from its newly prepared workcopy. Reusing
# Python's module cache would correctly violate the production source guard.
oracle_code = """
from pathlib import Path
import json,sys
sys.path.insert(0,sys.argv[1]);sys.dont_write_bytecode=True
import run_train as r
import torch
torch.set_num_threads(2)
rt=r.s.runtime(r.config()['seed'],'cpu',strict=False)
modules=r.s.import_source(Path(sys.argv[2]))
original_model=modules.models.svhnq(r.s.BITS,num_classes=10,expand=2).to('cpu')
print(json.dumps({'base_sha256':r.s.legacy.state_digest(original_model.state_dict()),
                 'models_module':str(Path(modules.models.__file__).resolve())}))
"""
oracle_process=subprocess.run([sys.executable,'-c',oracle_code,str(P),str(source)],
    check=True,capture_output=True,text=True,timeout=60)
oracle=json.loads(oracle_process.stdout)
assert Path(oracle['models_module']).is_relative_to(source.resolve())
oracle_hash=oracle['base_sha256']
assert 'models' not in sys.modules and 'datasets.data' not in sys.modules
original_validate=r.validate_initialization
actual_reference=r.load_initialization_reference()
def cpu_validate(model,c,reference,report):
 local=copy.deepcopy(reference)
 local['history']['initial_base_model_sha256']=oracle_hash
 original_validate(model,c,local,report)
 report['initialization_verification']['CPU_fixture_oracle_not_historical_CUDA_hash']=True
original=r.s.runtime
runtime=lambda seed,device='cpu',strict=True:original(seed,'cpu',strict=False)
def cpu_cuda(x,*args,**kwargs):assert x.device.type=='cpu';return x
argv=[str(P/'run_train.py'),'check','--output-root',str(OUT/'results')]
start=time.perf_counter()
with (OUT/'stdout.log').open('x') as handle,redirect_stdout(handle),ExitStack() as stack:
 stack.enter_context(patch.object(r,'validate_initialization',cpu_validate));stack.enter_context(patch.object(sys,'argv',argv));stack.enter_context(patch.object(r.s,'runtime',runtime));stack.enter_context(patch.object(torch.Tensor,'cuda',cpu_cuda))
 rc=r.main()
paths=list((OUT/'results').iterdir());assert len(paths)==1;path=paths[0];report=json.loads((path/'run_report.json').read_text());assert rc==0 and report['status']=='PASS',report.get('traceback')
assert report['codec_tests']['status']=='PASS' and report['synthetic_training_checks']['status']=='PASS'
assert report['initialization_verification']['all4080_groups_start_at_x1'] and report['initialization_verification']['same_base_weights_and_BN_as_init64']
assert not (path/'checkpoints').exists()
assert (path/r.NOTEBOOK).exists() and (path/'notebook_support.py').exists()
for name,digest in source_binding.items():assert core.file_sha256(P/name)==digest
result={'status':'PASS','synthetic_CPU_only':True,'no_SVHN_training':True,'CUDA_tested':False,'actual_check_main':True,'oracle_separate_interpreter':True,'production_source_module_guard_unchanged':True,'oracle':oracle,'codec_tests':report['codec_tests'],'scale_training_tests':report['synthetic_training_checks'],'initialization_verified_against_independent_original_CPU_base':report['initialization_verification'],'code_binding':source_binding,'check_run':str(path),'duration_seconds':time.perf_counter()-start}
(OUT/'check_entry_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','report':str(OUT/'check_entry_report.json'),'check_run':str(path)}))
