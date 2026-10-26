"""Generate a genuine small learned-scale epoch50 checkpoint for CPU tests.

Three synthetic tensors, two batches/epoch, no SVHN data or reported research
accuracy. Every optimizer step from1 through55 is executed by the unchanged
reference runtime. Epochs51..55 provide a matched data-order control fixture.
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

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
if not (PACKAGE/'reference_runtime').is_dir():
    PACKAGE = Path(__file__).resolve().parents[2] / 'output/mx_triple_freeze_offsets_e50_pilot5_v1'
REFERENCE = PACKAGE / 'reference_runtime'
sys.path.insert(0, str(REFERENCE))
spec = importlib.util.spec_from_file_location('freeze50_parent_runtime', REFERENCE/'run_train.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
s, t, a = parent.s, parent.t, parent.a
import torch


def make_context(path: Path, source, source_info, modules, package_hashes):
    rt = s.runtime(20261009, 'cpu', strict=False)
    c = parent.config()
    c.update(train_examples=3, test_examples=3, batch_size=2,
             purpose='SYNTHETIC_CPU_FREEZE50_VALIDATION_PARENT_ONLY',
             synthetic_initial_theta_mixed=True)
    model = t.build_model(modules, 'joint', rt.device)
    with torch.no_grad():
        pattern = torch.tensor([5.000001, 5.4999, 5.5001, 5.999999])
        for p in a.scale_parameters(model):
            p.copy_(pattern.repeat((p.numel()+3)//4)[:p.numel()].reshape_as(p))
    optimizer = t.build_optimizer(modules, model, c)
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
    generator = torch.Generator().manual_seed(4837)
    images = torch.rand(3, 3, 40, 40, generator=generator)*2-1
    labels = torch.tensor([0, 1, 2])
    loaders = {phase: s.TrackedLoader(torch.utils.data.DataLoader(
        s.IndexedDataset(torch.utils.data.TensorDataset(images.clone(), labels.clone())),
        batch_size=2, shuffle=phase=='train', num_workers=0, pin_memory=False,
        drop_last=False)) for phase in ('train','test')}
    contract = {'config':c, 'environment':rt.environment, 'settings':rt.settings,
                'source_files':source_info['workcopy_files_sha256'],
                'package_sha256':package_hashes, 'synthetic_fixture_only':True}
    report = {'config':c,'environment':rt.environment,'execution_settings':rt.settings,
              'source':source_info,'package_sha256':package_hashes,
              'contract_sha256':rt.core.json_digest(contract),
              'synthetic_fixture_only':True,'dataset_downloaded':False}
    return SimpleNamespace(rt=rt,c=c,source=source,source_info=source_info,modules=modules,
        model=model,optimizer=optimizer,scheduler=scheduler,loaders=loaders,
        contract=contract,report=report)


def main(out):
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    source, source_info = s.legacy.prepare_source(out)
    modules = s.import_source(source)
    ctx = make_context(out, source, source_info, modules, s.verify_package())
    log = s.legacy.Log(out/'run.log')
    def cpu_cuda(tensor,*args,**kwargs):
        assert tensor.device.type=='cpu'
        return tensor
    with (out/'stdout.log').open('x',encoding='utf-8') as stdout:
        with redirect_stdout(stdout), patch.object(torch.Tensor,'cuda',cpu_cuda):
            trace = parent.run_epochs(ctx,out,log,55,install_signal_handler=False)
    log.handle.close()
    assert len(trace['epochs'])==55 and trace['epochs'][49]['optimizer_steps']==100
    s.write_json(out/'history.json',trace)
    s.write_json(out/'parent_contract.json',ctx.contract)
    s.write_json(out/'run_report.json',ctx.report)
    result={'status':'PASS','synthetic_fixture_only':True,'no_SVHN_data':True,
            'source':str(source),'source_info':source_info,
            'history':str(out/'history.json'),
            'checkpoint50':str(out/'checkpoints/epoch_00050.pt'),
            'contract_sha256':ctx.report['contract_sha256'],
            'epochs_executed':55,'optimizer_updates':110,
            'train_examples':3,'test_examples':3,'batch_size':2,
            'duration_seconds':time.perf_counter()-started}
    s.write_json(out/'fixture.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main(sys.argv[1])
