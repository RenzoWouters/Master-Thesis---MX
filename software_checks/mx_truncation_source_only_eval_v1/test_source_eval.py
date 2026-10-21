"""Synthetic correctness tests; no optimizer steps, real data or research scores."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
import trunc_adapter as a
import source_evaluation as ev
import resume_core as core
import svhn_support as s
from run_eval import launch_worker, compare_history


def rejected(call, label):
    try: call()
    except (ValueError,RuntimeError): return True
    raise AssertionError('Ongeldige toestand werd geaccepteerd: '+label)


def run_tests(out, rt, log):
    """Works on production CUDA for check, or CPU in the offline harness."""
    folder=out/'synthetic_validation';folder.mkdir()
    src,info=s.legacy.prepare_source(folder)
    modules=s.import_source(src)
    # Test all finite source patterns, including signed zeros and subnormals.
    h=a.high.MXTensor(torch.arange(64,dtype=torch.uint8).reshape(2,1,32),
                      torch.full((2,1),127,dtype=torch.uint8),(2,32))
    low=a.truncate_codes(h)
    assert torch.equal(low.codes,h.codes>>2)
    assert torch.equal(low.scale_codes,h.scale_codes)
    assert torch.equal(h.codes.long()>>5,low.codes.long()>>3)
    assert torch.equal((h.codes.long()>>3)&3,(low.codes.long()>>1)&3)
    assert torch.equal(torch.signbit(low.decode()),torch.signbit(h.decode()))
    model=a.adapt_copy(modules.models.svhnq([1,2,4,8,32],expand=2)).cpu().eval()
    # Make the two stored BN states intentionally distinguishable.
    with torch.no_grad():
        for name in a.BN_LAYERS:
            bn=model.get_submodule(name)
            bn.bn_dict['4'].running_mean.fill_(0.17)
            bn.bn_dict['4'].running_var.fill_(1.2)
            bn.bn_dict['32'].running_mean.fill_(-0.03)
        payload,_=a.source_payload(model,{'source_epoch':100,'checkpoint_sha256':'0'*64,
                                         'synthetic_fixture_only':True})
    stage=folder/'staged';stage.mkdir()
    saved=core.atomic_save(stage/'source_E2M3.pt',payload)
    x=torch.linspace(-1,1,7*3*40*40,dtype=torch.float32).reshape(7,3,40,40)
    y=torch.arange(7,dtype=torch.int64)%10
    with (stage/'synthetic_input.npz').open('xb') as f:np.savez(f,images=x.numpy(),labels=y.numpy())
    job={'synthetic_fixture_only':True,'seed':20261009,'device':str(rt.device),
         'role':'synthetic_only','epoch':100,'source_sha256':saved['sha256'],'checkpoint_sha256':'0'*64}
    s.write_json(stage/'job.json',job)
    # Same high source locally, with quantizers and update pathways prohibited.
    local=folder/'local';local.mkdir()
    with ev.inference_guards(stage/'source_E2M3.pt') as opened:
        loaded=a.load_verified_payload(stage/'source_E2M3.pt')
        models,checks,layers=ev.build_models(modules.models,loaded,rt.device)
        data=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y,torch.arange(len(y))),
                                        batch_size=3,shuffle=False,num_workers=0)
        metrics,paired,details=ev.evaluate_models(models,data,len(y),rt.device,local,'synthetic_only',100,log)
        assert len(opened)==1
        rejected(lambda: a.high.encode_tensor(torch.zeros(2,32)), 'high encoder guard')
        rejected(lambda: a.low.encode_tensor(torch.zeros(2,32)), 'low encoder guard')
        rejected(lambda: torch.load(stage/'forbidden_checkpoint.pt',weights_only=True), 'master file guard')
        rejected(lambda: models[6].train(), 'train mode guard')
        rejected(lambda: torch.optim.Adam(models[6].parameters()), 'optimizer guard')
        assert all(not m.training for m in models[6].modules())
    w=launch_worker(stage/'job.json',folder/'fresh_process',log)
    comparisons={}
    with np.load(local/'per_image_outputs.npz',allow_pickle=False) as p, np.load(folder/'fresh_process/per_image_outputs.npz',allow_pickle=False) as q:
        for bit in (6,4):
            name=f'E2M{3 if bit==6 else 1}__logits'
            exact=np.array_equal(p[name],q[name])
            close=np.allclose(p[name],q[name],rtol=1e-4,atol=1e-5)
            comparisons[str(bit)]={'exact':bool(exact),'within_tolerance':bool(close),
                     'max_abs_difference':float(np.max(np.abs(p[name]-q[name])))}
            assert close, 'Fresh-process logits outside unchanged rtol=1e-4, atol=1e-5; inspect diagnostics.'
    # Independently check CPU live model -> export model. The arithmetic is
    # deterministic here; no claim of bitexact production-CUDA replay.
    cpu_models,_,_=ev.build_models(modules.models,payload,'cpu')
    cpu_exact={}
    with torch.no_grad():
        for b in (6,4):
            a.select_mode(model,b)
            actual=model(x[:3]); expected=cpu_models[b](x[:3])
            cpu_exact[str(b)]=torch.equal(actual,expected)
            assert cpu_exact[str(b)]
    bad=copy.deepcopy(payload);bad['hidden_master']=torch.ones(1)
    rejected(lambda:a.validate_source(bad),'unknown top-level field')
    bad=copy.deepcopy(payload);bad['auxiliary_fp32_state'][a.TARGET_LAYERS[0]+'.weight']=torch.ones(1)
    rejected(lambda:ev.build_models(modules.models,bad,'cpu'),'master hidden in auxiliary')
    bad=copy.deepcopy(payload);bad['bank_mapping']['4']='32'
    rejected(lambda:ev.build_models(modules.models,bad,'cpu'),'wrong low BN mapping')
    bad=copy.deepcopy(payload);bad['mx_layers'][a.TARGET_LAYERS[0]]['scale_codes'].fill_(255)
    rejected(lambda:a.validate_source(bad),'NaN scale')
    bad=copy.deepcopy(payload);k=next(k for k,v in bad['auxiliary_fp32_state'].items() if v.is_floating_point())
    bad['auxiliary_fp32_state'][k]=bad['auxiliary_fp32_state'][k].double()
    rejected(lambda:ev.build_models(modules.models,bad,'cpu'),'silent auxiliary dtype conversion')
    corrupt=stage/'corrupt.pt';corrupt.write_bytes((stage/'source_E2M3.pt').read_bytes()+b'x')
    corrupt.with_suffix('.pt.sha256').write_text(saved['sha256'])
    rejected(lambda:a.load_verified_payload(corrupt),'corrupt file')
    history=[{'role':r['role'],'epoch':r['epoch'],'weight_bits':r['weight_bits'],'n':r['n'],
              'correct1':r['correct1'],'correct5':r['correct5'],'objective_loss':r['hard_cross_entropy']} for r in metrics]
    assert all(r['reproduction']=='MATCH_COUNTS_CE_CLOSE' for r in compare_history(metrics,history))
    history[0]['correct1']+=1
    assert compare_history(metrics,history)[0]['reproduction']=='REVIEW_DIFFERENCE'
    result={'status':'PASS','synthetic_only':True,'optimizer_steps':0,'all_64_codes_checked':True,
            'two_modes_from_high_only_in_new_process':w['status']=='PASS',
            'fresh_process_logit_comparisons':comparisons,'rtol':1e-4,'atol':1e-5,
            'cpu_live_vs_export_logits_exact':cpu_exact,
            'guards_and_corruption_tests_pass':True,'historical_difference_not_hidden':True,
            'fixture_metrics_are_not_research_results':True}
    s.write_json(folder/'test_results.json',result)
    return result
