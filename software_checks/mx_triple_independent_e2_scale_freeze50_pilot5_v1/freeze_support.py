"""Explicit read-only parent restore and absolute-E2 freeze for the five-epoch branch.

No numerical encoder or STE is changed. Production parent identities are pinned
in independent_full100_reference/REFERENCE.json and verified before torch.load.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
import resume_core as core
import svhn_support as s
import trunc_adapter as a

PARENT_EPOCH = 50
STOP_EPOCH = 55
REFERENCE_NAME = 'independent_full100_reference'


def load_reference(root: Path):
    directory = root / REFERENCE_NAME
    marker = directory / 'REFERENCE.json'
    meta = json.loads(marker.read_text(encoding='utf-8'))
    s.require(meta['parent_epoch'] == 50 and meta['completed_epochs'] == 100,
              'Onjuiste ouderreferentie voor freeze50.')
    for name, digest in meta['files'].items():
        path = (directory / name).resolve()
        s.require(path.is_relative_to(directory.resolve()) and path.is_file()
                  and core.file_sha256(path) == digest, 'Ouderreferentie gewijzigd: ' + name)
    report = json.loads((directory/'run_report.json').read_text(encoding='utf-8'))
    history = json.loads((directory/'history.json').read_text(encoding='utf-8'))
    s.require(report['status']=='PASS' and report['completed_epoch']==100
              and report['config']['scale_freeze_epoch'] is None
              and report['config']['scale_learning_end_epoch']==100,
              'De controle moet de voltooide onafhankelijke training met doorleren zijn.')
    s.require([r['epoch'] for r in history['epochs']]==list(range(1,101)),
              'Onvolledige oudergeschiedenis.')
    return {'directory':directory,'meta':meta,'report':report,'history':history,
            'reference_sha256':core.file_sha256(marker),'kind':'independent_full100'}


def default_parent(root: Path, reference):
    # Installed under PROJECT/software_checks/PACKAGE. The project cwd is also
    # supported, and an explicit CLI path always takes precedence.
    relative = Path(reference['meta']['default_parent_checkpoint_relative'])
    choices = [Path.cwd()/relative, root.parents[1]/relative]
    return next((p.resolve() for p in choices if p.is_file()), choices[0].resolve())


def linkage(reference):
    m=reference['meta']
    return {'epoch':50,'sha256':m['parent_checkpoint_sha256'],
            'contract_sha256':m['parent_contract_sha256'],
            'reference_sha256':reference['reference_sha256']}


def verify_parent(path: Path, reference):
    path=path.resolve(); m=reference['meta']
    s.require(path.is_file(), 'Epoch-50-checkpoint ontbreekt: '+str(path)+
              '. Gebruik --parent-checkpoint met het oorspronkelijke epoch_00050.pt; geen nieuwe training starten.')
    s.require(core.file_sha256(path)==m['parent_checkpoint_sha256'],
              'Dit is niet het vastgelegde onafhankelijke epoch-50-checkpoint.')
    state=core.load_verified(path)
    s.require(core.json_digest(state['contract'])==m['parent_contract_sha256'],
              'Oudercheckpointcontract wijkt af van de vastgelegde run.')
    s.require(state['completed_epoch']==50 and state['next_epoch']==51,
              'De freeze moet starten na een volledig afgeronde epoch50.')
    s.require(state['contract']['config']==reference['report']['config'],
              'Ouderconfiguratie wijkt af van de feedback.')
    trace=state['trace']; expected=reference['history']
    s.require(len(trace['epochs'])==50,'Oudercheckpoint mist de eerste50 epochs.')
    s.require({k:v for k,v in trace.items() if k!='epochs'} ==
              {k:v for k,v in expected.items() if k!='epochs'},
              'Ouderinitialisatie of historieheader wijkt af.')
    for actual, full in zip(trace['epochs'],expected['epochs'][:50]):
        # A checkpoint is saved before its own write duration is known. Only
        # those final timing fields may be completed from the pinned full trace.
        missing=set(full)-set(actual)
        s.require(not (set(actual)-set(full)) and
                  (not missing or (actual['epoch']==50 and missing <=
                    {'checkpoint_seconds','total_epoch_seconds','other_seconds'})),
                  'Oudertrace mist meer dan de laatste checkpointtiming.')
        s.require(all(actual[k]==full[k] for k in actual),'Oudertrace verschilt van de controle.')
    s.require(s.legacy.state_digest(state['model'])==expected['epochs'][49]['model_sha256'],
              'Epoch50-modelhash wijkt af van de oorspronkelijke meting.')
    s.require(state['optimizer_steps']==expected['epochs'][49]['optimizer_steps'],
              'Ouder-Adamstapteller wijkt af.')
    for name in a.TARGET_LAYERS:
        tensor=torch.tensor(expected['epochs'][49]['scale_state'][name],dtype=torch.float32)
        s.require(torch.equal(state['model'][name+'.quantize_fn.phi'],tensor),
                  'Ouder-phi wijkt af van de schaalregistratie.')
    return state


def verify_environment(ctx, parent_state, include_data=True):
    expected=parent_state['contract']
    actual={'environment':ctx.rt.environment,'settings':ctx.rt.settings,
            'source_files':ctx.source_info['workcopy_files_sha256']}
    if include_data:
        actual.update({key:ctx.contract[key] for key in ('data_files','dataset_selection','transforms')})
    for key,value in actual.items():
        mismatch=core.differences(value,expected[key])
        s.require(not mismatch,'Ouderherstel vereist dezelfde '+key+': '+', '.join(mismatch[:6]))
    return {key:True for key in actual}


def _expected_snapshot(model, optimizer, parent_state):
    parent_ids=parent_state['optimizer']['param_groups'][1]['params']
    s.require(len(parent_ids)==len(a.TARGET_LAYERS)==len(optimizer.param_groups[1]['params']),
              'Ongeldige schaaloptimizer-layout.')
    snapshot={}
    for name,pid,parameter in zip(a.TARGET_LAYERS,parent_ids,optimizer.param_groups[1]['params']):
        phi=model.get_submodule(name).quantize_fn.phi
        s.require(parameter is phi,'Schaalparametervolgorde gewijzigd.')
        old=parent_state['optimizer']['state'][pid]
        snapshot[name]={'phi':parent_state['model'][name+'.quantize_fn.phi'].to(phi.device).clone(),
                        'adam':{key:value.to(optimizer.state[phi][key].device).clone()
                                if torch.is_tensor(value) else copy.deepcopy(value)
                                for key,value in old.items()}}
        snapshot[name]['codes']=a.scale_codes(snapshot[name]['phi'])
    return snapshot


def apply_freeze(model, optimizer, scheduler, parent_state):
    """After exact restore, preserve the parent scales/moments and disable updates."""
    before_rng=core.capture_rng({},next(model.parameters()).device)
    before_model=s.legacy.state_digest(model.state_dict())
    before_lr=[float(g['lr']) for g in optimizer.param_groups]
    snapshot=_expected_snapshot(model,optimizer,parent_state)
    for name in a.TARGET_LAYERS:
        phi=model.get_submodule(name).quantize_fn.phi
        s.require(torch.equal(phi.detach(),snapshot[name]['phi']), 'Phi is niet de bewaarde epoch50-waarde.')
        s.require(not core.differences(core.cpu_copy(optimizer.state[phi]),core.cpu_copy(snapshot[name]['adam'])),
                  'Schaal-Adamstaat verschilt van epoch50.')
        phi.requires_grad_(False); phi.grad=None
    optimizer.param_groups[1]['lr']=0.0
    scheduler._last_lr=[float(g['lr']) for g in optimizer.param_groups]
    handle=getattr(optimizer,'_learned_scale_projection_handle',None)
    if handle is not None:handle.remove()
    previous=getattr(optimizer,'_frozen_scale_guard_handle',None)
    if previous is not None:previous.remove()
    model._absolute_e2_frozen=True
    model._frozen_scale_snapshot=snapshot
    model._frozen_scale_parent_steps=int(parent_state['optimizer_steps'])
    def guard(opt,args,kwargs):
        verify_frozen(model,opt)
        model._last_scale_projection={'projected_values':0,
            'projected_by_layer':{name:0 for name in a.TARGET_LAYERS}}
    optimizer._frozen_scale_guard_handle=optimizer.register_step_post_hook(guard)
    guard(optimizer,None,None)
    s.require(s.legacy.state_digest(model.state_dict())==before_model,'Freeze veranderde modelwaarden.')
    s.require(not core.differences(before_rng,core.capture_rng({},next(model.parameters()).device)),
              'Freeze veranderde RNG.')
    return {'epoch':50,'first_frozen_epoch':51,'absolute_E2_scale_frozen':True,
            'E4_scale_remains_dynamic':True,'weights_and_BN_continue_training':True,
            'requires_grad_false':True,'scale_lr_before_transition':before_lr[1],
            'scale_lr_after_transition':0.0,'weight_lr_unchanged':before_lr[0]==optimizer.param_groups[0]['lr'],
            'model_values_unchanged':True,'scale_Adam_state_unchanged':True,'RNG_unchanged':True,
            'parent_scale_Adam_steps':int(parent_state['optimizer_steps']),
            'frozen_phi_sha256':s.legacy.state_digest({n:v['phi'] for n,v in snapshot.items()}),
            'frozen_hard_E2_scale_codes_sha256':s.legacy.state_digest({n:v['codes'] for n,v in snapshot.items()})}


def verify_frozen(model, optimizer):
    s.require(getattr(model,'_absolute_e2_frozen',False),'Absolute E2-freeze ontbreekt.')
    s.require(float(optimizer.param_groups[1]['lr'])==0.0,'Bevroren schaal-LR is niet nul.')
    for name,expected in model._frozen_scale_snapshot.items():
        phi=model.get_submodule(name).quantize_fn.phi
        s.require(not phi.requires_grad and phi.grad is None,'Bevroren phi krijgt gradients: '+name)
        s.require(torch.equal(phi.detach(),expected['phi']),'Bevroren phi is veranderd: '+name)
        s.require(torch.equal(a.scale_codes(phi),expected['codes']),'Bevroren E2-schaalcode is veranderd: '+name)
        actual=optimizer.state[phi]
        s.require(set(actual)==set(expected['adam']),'Bevroren Adamstaat heeft andere velden.')
        for key,value in expected['adam'].items():
            same=torch.equal(actual[key],value) if torch.is_tensor(value) else actual[key]==value
            s.require(same,'Bevroren schaal-Adamstaat is veranderd: '+name+'/'+key)
    return {'phi_unchanged':True,'hard_E2_scales_unchanged':True,
            'scale_optimizer_state_unchanged':True,'scale_gradients_absent':True,
            'frozen_parameter_tensors':6,'frozen_parameters':4080}


def parent_trace(parent_state,reference):
    trace=copy.deepcopy(parent_state['trace'])
    # Complete the last timing row from the exact, pinned full-run trace.
    trace['epochs']=copy.deepcopy(reference['history']['epochs'][:50])
    trace['parent_checkpoint']=linkage(reference)
    return trace


def verify_prefix(trace,reference):
    s.require(trace.get('parent_checkpoint')==linkage(reference),'Verkeerde ouderlink in geschiedenis.')
    s.require(trace['epochs'][:50]==reference['history']['epochs'][:50],
              'De overgenomen eerste50 epochs zijn gewijzigd.')


def resolve_record_checkpoint(record,ctx):
    if record['epoch']<=50:
        path=ctx.parent_path.parent/f"epoch_{record['epoch']:05d}.pt"
    else:path=Path(record['checkpoint_path'])
    s.require(path.is_file(),'Geselecteerd checkpoint ontbreekt: '+str(path))
    return path
