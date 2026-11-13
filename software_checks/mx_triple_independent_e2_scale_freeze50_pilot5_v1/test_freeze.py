"""Bounded frozen-scale check on the verified epoch50 state; no SVHN data."""
from __future__ import annotations
import copy
import torch
import resume_core as core
import svhn_support as s
import trunc_adapter as a
import trunc_support as t
import freeze_support as freeze


def check_parent_transition(ctx,out,log):
    model,opt,scheduler,rt=ctx.model,ctx.optimizer,ctx.scheduler,ctx.rt
    parent=ctx.parent_state
    restored=core.restore_checkpoint(parent,model,opt,scheduler,{},rt.device,parent['contract'])
    before=s.legacy.state_digest(model.state_dict())
    before_weights=t.layer_snapshot(model)
    transition=freeze.apply_freeze(model,opt,scheduler,parent)
    checks={'complete_parent_state_restored':all(restored.values()),
            'transition_preserves_all_model_values_and_RNG':transition['model_values_unchanged'] and transition['RNG_unchanged']}
    # A source-code change is legal with fixed E2 scales: vary source weights in
    # an isolated codec test, without changing the restored production model.
    q=model.get_submodule(a.TARGET_LAYERS[0]).quantize_fn
    value=torch.ones(tuple(model.get_submodule(a.TARGET_LAYERS[0]).weight.shape),dtype=torch.float32,device=rt.device)
    scales=a.scale_codes(q.phi)
    first=a.encode_chain(value,scales);second=a.encode_chain(value*2,scales)
    s.require(not torch.equal(first[8].scale_codes,second[8].scale_codes)
              and torch.equal(first[6].scale_codes,second[6].scale_codes)
              and torch.equal(first[4].scale_codes,second[4].scale_codes),
              'E4 kan niet onafhankelijk veranderen bij vaste absolute E2-schaal.')
    checks['dynamic_E4_scale_independent_of_frozen_E2']=True
    generator=torch.Generator(device='cpu').manual_seed(73211)
    inputs=torch.randn((2,3,40,40),generator=generator).to(rt.device)
    labels=torch.tensor([0,1],device=rt.device)
    target=out/'synthetic_freeze_check';target.mkdir()
    rows,seconds,ast_hash,audit=t.phase(rt,ctx.source,ctx.modules,model,opt,[(inputs,labels)],
                                      51,'train',float(opt.param_groups[0]['lr']),target,log)
    verify=freeze.verify_frozen(model,opt)
    t.validate_adam_state(model,opt,parent['optimizer_steps']+1)
    t.validate_bn(model,parent['optimizer_steps']+1)
    s.require(s.legacy.state_digest(model.state_dict())!=before,'Synthetische update veranderde geen gewichten/BN.')
    weight_updates=t.layer_updates(before_weights,t.layer_snapshot(model),'joint',51)
    s.require(any(r['master_changed_values']>0 for r in weight_updates),'Geen MX-masterupdate in freezecheck.')
    s.require(audit['scale_optimizer_steps']==0 and audit['frozen_scale_checks']==1
              and audit['scale_gradients_enabled'] is False,
              'Freezecontrole registreert toch schaalupdates.')
    # A malformed frozen state must be rejected, including moment-only drift.
    phi=model.get_submodule(a.TARGET_LAYERS[0]).quantize_fn.phi
    for kind in ('phi','moment','gradient'):
        saved_phi=phi.detach().clone();saved_moment=opt.state[phi]['exp_avg'].clone()
        try:
            if kind=='phi':
                with torch.no_grad():phi.view(-1)[0].add_(0.01)
            elif kind=='moment':opt.state[phi]['exp_avg'].view(-1)[0].add_(0.01)
            else:phi.grad=torch.zeros_like(phi)
            rejected=False
            try:freeze.verify_frozen(model,opt)
            except RuntimeError:rejected=True
            s.require(rejected,'Freezecontrole liet wijziging toe: '+kind)
        finally:
            with torch.no_grad():phi.copy_(saved_phi);opt.state[phi]['exp_avg'].copy_(saved_moment)
            phi.grad=None
    checks.update(verify,three_passes_one_Adam_update=True,
                  frozen_scale_Adam_steps_and_moments_preserved=True,
                  weight_and_BN_updates_continue=True,mutated_phi_moment_or_gradient_rejected=True,
                  original_forward_ast_sha256=ast_hash,synthetic_step_seconds=seconds)
    # Check mode is a self-contained process; still restore all learned parent
    # values and RNG so no check update can leak into an eventual continuation.
    for parameter in a.scale_parameters(model):parameter.requires_grad_(True);parameter.grad=None
    restored=core.restore_checkpoint(parent,model,opt,scheduler,{},rt.device,parent['contract'])
    transition2=freeze.apply_freeze(model,opt,scheduler,parent)
    s.require(transition['frozen_phi_sha256']==transition2['frozen_phi_sha256']
              and all(restored.values()),'Herstel na synthetische check onvolledig.')
    checks.update(status='PASS',parent_state_restored_after_check=True,
                  no_SVHN_data_loaded=True,no_SVHN_training=True)
    return checks
