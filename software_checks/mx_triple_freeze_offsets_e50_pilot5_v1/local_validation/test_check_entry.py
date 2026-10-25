"""Execute the production check path against a real synthetic CPU parent.

Only CPU device routing is substituted. Original package checks, codec tests,
parent restore, freeze checks and report paths run normally. No SVHN dataset is
loaded; this is explicitly not validation of CUDA-specific numerical behavior.
"""
from __future__ import annotations
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode=True


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main(package,out,parent_fixture,evidence_directory):
    package,out,parent_fixture,evidence_directory=map(lambda p:Path(p).resolve(),
        (package,out,parent_fixture,evidence_directory))
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    sys.path.insert(0,str(package))
    r=load_module('freeze50_check_entry_runner',package/'run_train.py')
    import torch
    torch.set_num_threads(2)
    evidence=r.load_parent_reference(evidence_directory,r.core.file_sha256(evidence_directory/'REFERENCE.json'))
    checkpoint=parent_fixture/'checkpoints/epoch_00050.pt'
    checkpoint_sha=r.core.file_sha256(checkpoint)
    code_binding={name:r.core.file_sha256(package/name) for name in
        ('run_train.py','freeze_core.py','runtime_bridge.py','test_freeze.py')}
    original_runtime=r.s.runtime
    def cpu_runtime(seed,device='cpu',strict=True):
        return original_runtime(seed,'cpu',strict=False)
    def cpu_cuda(tensor,*args,**kwargs):
        assert tensor.device.type=='cpu'
        return tensor
    args=SimpleNamespace(mode='joint',device='cpu',parent_checkpoint=checkpoint)
    report={'schema':'mx-freeze50-check-entry-synthetic-validation-v1',
        'status':'RUNNING','synthetic_fixture_only':True,'warnings':[]}
    log=r.s.legacy.Log(out/'run.log')
    with (out/'stdout.log').open('x',encoding='utf-8') as stdout:
        with redirect_stdout(stdout),patch.object(r.s,'runtime',cpu_runtime),\
                patch.object(torch.Tensor,'cuda',cpu_cuda):
            r.run_check(args,out,report,log,evidence)
    log.handle.close()
    assert report['parent_verification']['parent_model_matches_reference']
    assert report['parent_verification']['parent_contract_matches_reference']
    assert report['parent_verification']['parent_trace_matches_reference']
    assert all(report['parent_verification']['strict_restore_checks'].values())
    inherited=report['inherited_technical_checks']
    assert inherited['codec_tests']['status']=='PASS'
    assert inherited['synthetic_training_checks']['status']=='PASS'
    assert report['freeze_tests']['status']=='PASS'
    assert all(report['freeze_tests']['checks'].values())
    assert report['dataset_downloaded'] is False
    assert report['full_dataset_verified'] is False
    assert report['continuation_completed'] is False
    assert report['synthetic_fixture_only'] is True
    assert not (out/'checkpoints').exists()
    assert r.core.file_sha256(checkpoint)==checkpoint_sha
    assert r.s.verify_package()==evidence['contract']['package_sha256']
    for name,digest in code_binding.items():assert r.core.file_sha256(package/name)==digest

    # Root-level copied last_checkpoint.pt protects that particular run, while
    # allowing a new sibling result directory under the same experiment root.
    checkpoint_copy=out/'existing_run/last_checkpoint.pt'
    r.validate_output_location(out/'new_sibling_run',checkpoint,checkpoint_copy)
    path_checks={}
    for label,destination,parent,resume,data in (
        ('package',package/'unwanted',checkpoint,None,None),
        ('parent_run',parent_fixture/'unwanted',checkpoint,None,None),
        ('resume_checkpoints',out/'existing_run/unwanted',checkpoint,out/'existing_run/checkpoints/epoch_00052.pt',None),
        ('resume_root_copy',out/'existing_run/unwanted',checkpoint,checkpoint_copy,None),
        ('dataset',out/'dataset/unwanted',checkpoint,None,out/'dataset')):
        try:r.validate_output_location(destination,parent,resume,data)
        except RuntimeError as exc:path_checks[label]={'rejected':True,'message':str(exc)}
        else:raise AssertionError('Protected path accepted: '+label)
    report.update(status='PASS',CUDA_tested=False,no_SVHN_data_loaded_or_downloaded=True,
        not_a_research_accuracy_result=True,check_entry_uses_original_codec_and_gradient_checks=True,
        parent_checkpoint_sha256=checkpoint_sha,code_binding=code_binding,
        protected_path_checks=path_checks,root_checkpoint_copy_allows_sibling_output=True,
        duration_seconds=time.perf_counter()-started,
        test_file_sha256=r.core.file_sha256(Path(__file__)))
    r.s.write_json(out/'check_entry_validation_report.json',report)
    print(json.dumps({'status':'PASS','report':str(out/'check_entry_validation_report.json'),
        'duration_seconds':report['duration_seconds'],
        'freeze_checks':len(report['freeze_tests']['checks'])},indent=2))


if __name__=='__main__':
    if len(sys.argv)!=5:
        raise SystemExit('Usage: test_check_entry.py PACKAGE OUTPUT_DIRECTORY PARENT_FIXTURE_DIRECTORY EVIDENCE_DIRECTORY')
    main(*sys.argv[1:])
