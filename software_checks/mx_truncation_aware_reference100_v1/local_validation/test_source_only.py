"""Fresh-process reconstruction from high source only, no checkpoint or low file."""
from pathlib import Path
import copy, json, shutil, subprocess, sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import svhn_support as s
import trunc_adapter as a
import resume_core as core


def main(validation, out):
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    source=validation/'source/source_workcopy'
    modules=s.import_source(source)
    sample=torch.linspace(-1,1,3*3*40*40).reshape(3,3,40,40)
    expected={'input':sample}
    for suffix in ('best','last'):
        file=validation/'joint_run'/f'mx_source_E2M3_{suffix}.pt'
        shutil.copy2(file,out/file.name)
        shutil.copy2(file.with_suffix('.pt.sha256'),out/(file.name+'.sha256'))
        state=core.load_verified(validation/'joint_run'/f'{suffix}_checkpoint.pt')
        model=a.adapt_copy(modules.models.svhnq([1,2,4,8,32],expand=2))
        model.load_state_dict(state['model']);model.eval()
        for bit in (6,4):
            a.select_mode(model,bit)
            with torch.no_grad():expected[f'{suffix}_{bit}']=model(sample)
    torch.save(expected,out/'expected_logits_only.pt')
    script='''import sys, torch
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import trunc_adapter as a
import svhn_support as s
folder, source = Path(sys.argv[2]),Path(sys.argv[3])
torch.set_num_threads(1)
modules=s.import_source(source)
assert not list(folder.glob('*checkpoint*')) and not list(folder.glob('*derived*'))
expected=torch.load(folder/'expected_logits_only.pt',weights_only=True)
original_load=torch.load
def guarded_load(path,*args,**kwargs):
    p=Path(path).resolve()
    assert p.parent==folder.resolve() and p.name.startswith('mx_source_'), 'Forbidden file read'
    return original_load(path,*args,**kwargs)
with patch.object(torch,'load',guarded_load), patch.object(a.high,'encode_tensor',side_effect=AssertionError('No master encoding')), patch.object(a.low,'encode_tensor',side_effect=AssertionError('No direct low encoding')):
    for suffix in ('best','last'):
        payload=a.load_verified_payload(folder/f'mx_source_E2M3_{suffix}.pt')
        for bit in (6,4):
            model=a.load_exported_model(modules.models,payload,'cpu',bit=bit)
            assert not any(n+'.weight' in dict(model.named_parameters()) for n in a.TARGET_LAYERS)
            with torch.no_grad():actual=model(expected['input'])
            assert torch.equal(actual,expected[f'{suffix}_{bit}'])
print('PASS: four exact logit comparisons; high sources only; checkpoint/low-encoder access blocked')
'''
    result=subprocess.run([sys.executable,'-c',script,str(ROOT),str(out),str(source)],capture_output=True,text=True)
    (out/'fresh_process.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    # Corruption and unknown/hidden schema state must be rejected.
    p=a.load_verified_payload(out/'mx_source_E2M3_best.pt')
    bad=copy.deepcopy(p);bad['hidden']=torch.ones(1)
    try:a.validate_source(bad)
    except ValueError:pass
    else:raise AssertionError('Unknown key accepted')
    bad=copy.deepcopy(p);bad['auxiliary_fp32_state'][a.TARGET_LAYERS[0]+'.weight']=torch.ones(1)
    try:a.load_exported_model(modules.models,bad,'cpu')
    except ValueError:pass
    else:raise AssertionError('Hidden master accepted')
    f=out/'corrupt.pt';f.write_bytes((out/'mx_source_E2M3_best.pt').read_bytes()+b'bad')
    f.with_suffix('.pt.sha256').write_text((out/'mx_source_E2M3_best.pt.sha256').read_text())
    try:a.load_verified_payload(f)
    except ValueError:pass
    else:raise AssertionError('Corrupt source accepted')
    results={'status':'PASS','scope':'synthetic_CPU_only','source_only_fresh_process':True,
             'four_logits_exact_against_training_models':True,'encoder_access_blocked':True,
             'checkpoint_access_blocked':True,'corrupt_and_hidden_master_rejected':True}
    s.write_json(out/'checks.json',results)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
