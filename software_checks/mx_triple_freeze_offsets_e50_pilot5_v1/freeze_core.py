"""Verified scientific fork after epoch50; hard choices and theta stay constant.

The unchanged parent phase still computes scale gradients and Adam moments.
Only the theta group's learning rate changes to zero. This diagnostic does
not claim optimized training cost. E4 source codes/scales remain dynamic.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from runtime_bridge import ROOT, ref, s, t, a, core, torch

REFERENCE_SHA256 = '00d89e52af02359ac5d4abc91c3c19c659f45714b7f229bce0bb54af6f77b246'
PARENT_EPOCH, END_EPOCH = 50, 55
TIMING_AFTER_SAVE = ('checkpoint_seconds', 'total_epoch_seconds', 'other_seconds')


def config(parent=None):
    c = copy.deepcopy(ref.config() if parent is None else parent)
    c.update(schema='mx-triple-freeze-offsets-e50-pilot5-protocol-v1',
        purpose='freeze the learned block offset choices after epoch50; continue weights51-55; historical dynamic control',
        start_after_epoch=50, end_epoch=55, planned_new_epochs=5,
        planned_epochs=100, reference_end_epoch=100, selection_epoch_range=[51, 55],
        initialization_policy='strict restore of the verified learned-scale reference checkpoint after epoch50',
        scale_lr=0.0, offset_updates_enabled=False, source_scale_learned=False,
        source_scale_dynamic=True, theta_gradients_computed=True, theta_adam_moments_updated=True,
        freeze_method='theta Adam-group lr=0; requires_grad remains true; scheduler scale group lr/base_lrs/last_lr zero',
        learned_shared_scale=True, scale_choices_provenance='learned in parent epochs1-50, frozen throughout continuation',
        selection_policy='first strict maximum of E4M3 official-test top1 within new epochs51-55 only',
        reporting_policy='primary endpoint55; secondary shared E4M3-best within51-55; all modes at that same epoch',
        historical_control='original learned-scale reference100, observed epochs51-55; no control retraining',
        completion_policy='five new epochs51-55; no restart of the 100-epoch schedule',
        exact_cuda_resume_guaranteed=False)
    return c


def verify_package():
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    s.require(isinstance(manifest, dict) and manifest, 'Pakketmanifest ontbreekt.')
    for name, expected in manifest.items():
        path = (ROOT / name).resolve()
        s.require(path.is_relative_to(ROOT) and path.is_file(), 'Ongeldig pakketpad: ' + name)
        s.require(core.file_sha256(path) == expected, 'Pakketbestand gewijzigd: ' + name)
    s.verify_package()  # The original parent manifest is independently checked.
    return manifest


def load_parent_reference(directory=None, expected_reference_sha256=REFERENCE_SHA256):
    directory = (ROOT / 'parent_reference' if directory is None else Path(directory)).resolve()
    marker = directory / 'REFERENCE.json'
    s.require(core.file_sha256(marker) == expected_reference_sha256, 'Referentie-identiteit wijkt af.')
    meta = json.loads(marker.read_text(encoding='utf-8'))
    for name, digest in meta['files_sha256'].items():
        path = (directory / name).resolve()
        s.require(path.is_relative_to(directory) and core.file_sha256(path) == digest,
                  'Historisch referentiebestand gewijzigd: ' + name)
    read = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
    contract, baseline = read('parent_contract.json'), read('baseline_epoch50.json')
    comparison = {int(r['epoch']): r for r in read('comparison_epochs51_55.json')['epochs']}
    s.require(list(comparison) == list(range(51, 56)), 'Historische controle moet epochs51-55 bevatten.')
    s.require(core.json_digest(contract) == meta['source_contract_sha256'], 'Oorspronkelijk contract klopt niet.')
    s.require(baseline['epoch'] == meta['parent_epoch'] == 50
              and baseline['model_sha256'] == meta['parent_model_sha256'], 'Verkeerd oudercheckpoint.')
    return dict(meta=meta, contract=contract, baseline=baseline, comparison=comparison,
                directory=directory, reference_sha256=expected_reference_sha256)


def verify_parent_state(ctx, state, evidence):
    """Validate the parent before changing any live state; no permissive resume."""
    m, baseline = evidence['meta'], evidence['baseline']
    parent_contract = getattr(ctx, 'parent_contract', ctx.contract)
    s.require(not core.differences(parent_contract, evidence['contract']),
              'Huidige omgeving/data/bron wijkt af van het vastgelegde oorspronkelijke contract.')
    s.require(not core.differences(state['contract'], parent_contract), 'Checkpoint heeft een ander oudercontract.')
    s.require(core.json_digest(state['contract']) == m['source_contract_sha256'], 'Oudercontracthash wijkt af.')
    s.require(state['completed_epoch'] == 50 and state['next_epoch'] == 51
              and state['scheduler']['last_epoch'] == 50, 'Verwacht exact het checkpoint na epoch50.')
    steps_per_epoch = (parent_contract['config']['train_examples'] + parent_contract['config']['batch_size'] - 1) // parent_contract['config']['batch_size']
    s.require(state['optimizer_steps'] == m['parent_optimizer_steps'] == 50 * steps_per_epoch,
              'Onjuiste Adamstapteller in oudercheckpoint.')
    model_hash = s.legacy.state_digest(state['model'])
    s.require(model_hash == m['parent_model_sha256'] == baseline['model_sha256'], 'Verkeerd model bij epoch50.')
    trace = state['trace']
    s.require([r['epoch'] for r in trace['epochs']] == list(range(1, 51)), 'Oudergeschiedenis is onvolledig.')
    s.require(core.json_digest(trace) == m['parent_checkpoint_trace_sha256'], 'Oudergeschiedenis wijkt af van de feedbackbundel.')
    last = trace['epochs'][-1]
    expected_last = copy.deepcopy(baseline)
    for key in TIMING_AFTER_SAVE:
        expected_last.pop(key, None)
    s.require(not core.differences(last, expected_last), 'Epoch50-record wijkt af (exclusief timing na opslaan).')
    s.require(trace['initial_model_sha256'] == m['source_initial_model_sha256']
              and trace['initial_base_model_sha256'] == m['source_initial_base_model_sha256'],
              'Oorspronkelijke initialisatie wijkt af.')
    expected_lr = float(parent_contract['config']['lr']) * float(parent_contract['config']['scheduler_gamma'])
    groups = state['optimizer']['param_groups']
    s.require([g['name'] for g in groups] == ['weights', 'scale_proxy']
              and [float(g['lr']) for g in groups] == [expected_lr, expected_lr]
              and state['scheduler']['_last_lr'] == [expected_lr, expected_lr],
              'Oudercheckpoint staat niet op de oorspronkelijke LR na epoch50.')
    theta_state = {name: state['model'][name + '.quantize_fn.theta'].tolist() for name in a.TARGET_LAYERS}
    s.require(not core.differences(theta_state, baseline['scale_state']), 'Ouder-offsetparameters wijken af.')
    s.require(bool(state['contract']['synthetic_fixture_only']) == bool(m['synthetic_fixture_only']),
              'Synthetische en echte data mogen niet worden vermengd.')
    return dict(parent_epoch=50, parent_model_sha256=model_hash,
        parent_contract_sha256=m['source_contract_sha256'],
        parent_checkpoint_trace_sha256=m['parent_checkpoint_trace_sha256'],
        parent_checkpoint_verified=True, parent_model_matches_reference=True,
        parent_contract_matches_reference=True, parent_trace_matches_reference=True)


def freeze_optimizer(model, optimizer, scheduler):
    """The single deliberate intervention, after the complete parent restore."""
    before_model = s.legacy.state_digest(model.state_dict())
    before_optimizer = core.cpu_copy(optimizer.state_dict())
    before_scheduler = core.cpu_copy(scheduler.state_dict())
    s.require([g['name'] for g in optimizer.param_groups] == ['weights', 'scale_proxy'], 'Verkeerde Adamgroepen.')
    s.require({id(p) for p in optimizer.param_groups[1]['params']} == {id(p) for p in a.scale_parameters(model)},
              'De te bevriezen groep bevat andere parameters.')
    s.require(all(p.requires_grad for p in a.scale_parameters(model)), 'Schaaldiagnostiek vereist requires_grad.')
    optimizer.param_groups[1]['lr'] = 0.0
    optimizer.param_groups[1]['initial_lr'] = 0.0
    scheduler.base_lrs[1] = 0.0
    scheduler._last_lr[1] = 0.0
    expected_optimizer = copy.deepcopy(before_optimizer)
    expected_optimizer['param_groups'][1].update(lr=0.0, initial_lr=0.0)
    expected_scheduler = copy.deepcopy(before_scheduler)
    expected_scheduler['base_lrs'][1] = 0.0
    expected_scheduler['_last_lr'][1] = 0.0
    s.require(not core.differences(optimizer.state_dict(), expected_optimizer), 'Onbedoelde wijziging van Adamstaat.')
    s.require(not core.differences(scheduler.state_dict(), expected_scheduler), 'Onbedoelde wijziging van scheduler.')
    s.require(s.legacy.state_digest(model.state_dict()) == before_model, 'Bevriezen veranderde het model.')
    return dict(theta_lr_before=float(before_optimizer['param_groups'][1]['lr']), theta_lr_after=0.0,
        weights_and_bn_unchanged_at_fork=True, adam_moments_unchanged_at_fork=True,
        weight_learning_rate_unchanged=True, scheduler_epoch_unchanged=True,
        theta_gradients_computed=True, theta_adam_moments_updated=True)


class FrozenOffsetGuard:
    """Compare every theta value and hard choice against the parent after each step."""
    def __init__(self, model, optimizer, parent_state):
        self.model, self.optimizer = model, optimizer
        self.anchor = {name: torch.tensor(parent_state[name], dtype=torch.float32,
                       device=model.get_submodule(name).quantize_fn.theta.device) for name in a.TARGET_LAYERS}
        self.hard = {name: a.hard_offsets(theta) for name, theta in self.anchor.items()}
        self.steps = 0
        self.assert_unchanged()
        self.handle = optimizer.register_step_post_hook(self.after_step)

    @torch.no_grad()
    def assert_unchanged(self):
        s.require(self.optimizer.param_groups[1]['lr'] == 0.0, 'Offset-LR is niet nul.')
        theta_ok, hard_ok = [], []
        for name in a.TARGET_LAYERS:
            value = self.model.get_submodule(name).quantize_fn.theta.detach()
            theta_ok.append((value == self.anchor[name]).all())
            hard_ok.append((a.hard_offsets(value) == self.hard[name]).all())
        s.require(bool(torch.stack(theta_ok).all()), 'Een bevroren theta is gewijzigd; proef gestopt.')
        s.require(bool(torch.stack(hard_ok).all()), 'Een bevroren offsetkeuze is gewijzigd; proef gestopt.')

    def after_step(self, opt, args, kwargs):
        self.assert_unchanged()
        self.steps += 1

    def close(self):
        self.handle.remove()


def initialize_branch(ctx, parent_checkpoint, evidence, package_hashes, resume=None):
    parent_checkpoint = Path(parent_checkpoint).resolve()
    parent_sha = core.file_sha256(parent_checkpoint)
    state = core.load_verified(parent_checkpoint)
    ctx.parent_contract, ctx.parent_c = copy.deepcopy(ctx.contract), copy.deepcopy(ctx.c)
    verified = verify_parent_state(ctx, state, evidence)
    verified['parent_checkpoint_sha256'] = parent_sha
    verified['strict_restore_checks'] = core.restore_checkpoint(state, ctx.model, ctx.optimizer, ctx.scheduler,
                                                               {}, ctx.rt.device, ctx.parent_contract)
    t.validate_bn(ctx.model, state['optimizer_steps'])
    t.validate_adam_state(ctx.model, ctx.optimizer, state['optimizer_steps'])
    rng_before = core.capture_rng({}, ctx.rt.device)
    intervention = freeze_optimizer(ctx.model, ctx.optimizer, ctx.scheduler)
    s.require(not core.differences(core.capture_rng({}, ctx.rt.device), rng_before), 'Bevriezen verbruikte toevalsgetallen.')
    ctx.c = config(ctx.parent_c)
    ctx.contract = dict(schema='mx-triple-freeze-offsets-e50-pilot5-contract-v1', config=ctx.c,
        parent_contract=ctx.parent_contract, parent_contract_sha256=evidence['meta']['source_contract_sha256'],
        parent_checkpoint_sha256=parent_sha, parent_model_sha256=evidence['meta']['parent_model_sha256'],
        parent_reference_sha256=evidence['reference_sha256'], package_sha256=package_hashes,
        synthetic_fixture_only=ctx.parent_contract['synthetic_fixture_only'])
    trace = dict(parent_epoch=50, initial_model_sha256=s.legacy.state_digest(ctx.model.state_dict()),
        initial_base_model_sha256=s.legacy.state_digest(a.base_state_dict(ctx.model)),
        initial_scale_state=t.scale_state(ctx.model), parent_verification=verified, epochs=[])
    done, steps = 50, state['optimizer_steps']
    if resume is not None:
        resumed = core.load_verified(Path(resume).resolve())
        s.require(50 < resumed['completed_epoch'] < 55, 'Hervat alleen een nieuwe proefcheckpoint na epoch51-54.')
        checks = core.restore_checkpoint(resumed, ctx.model, ctx.optimizer, ctx.scheduler, {}, ctx.rt.device, ctx.contract)
        done, steps = resumed['completed_epoch'], resumed['optimizer_steps']
        trace = copy.deepcopy(resumed['trace'])
        ref.restore_timing_sidecar(ctx.rt, trace)
        s.require(trace['parent_epoch'] == 50
                  and trace['initial_model_sha256'] == evidence['meta']['parent_model_sha256']
                  and not core.differences(trace['initial_scale_state'], evidence['baseline']['scale_state']),
                  'Hervatcheckpoint behoort niet tot dezelfde bevroren proef.')
        s.require([r['epoch'] for r in trace['epochs']] == list(range(51, done + 1)), 'Onvolledige nieuwe proefgeschiedenis.')
        s.require(steps == done * len(ctx.loaders['train']), 'Verkeerde hervatstapteller.')
        ctx.report.update(resumed_from=str(Path(resume).resolve()), resume_input_sha256=core.file_sha256(Path(resume)),
                          restore_checks=checks)
    ctx.guard = FrozenOffsetGuard(ctx.model, ctx.optimizer, evidence['baseline']['scale_state'])
    s.require(not core.differences(t.scale_state(ctx.model), evidence['baseline']['scale_state']), 'Offsets niet vast op epoch50.')
    ctx.report.update(config=ctx.c, parent_verification=verified, intervention=intervention,
        contract_sha256=core.json_digest(ctx.contract), parent_checkpoint=str(parent_checkpoint),
        parent_checkpoint_input_sha256=parent_sha, initial_model_sha256=trace['initial_model_sha256'],
        initial_base_model_sha256=trace['initial_base_model_sha256'],
        synthetic_fixture_only=ctx.contract['synthetic_fixture_only'])
    return trace, done, steps


def validate_output_location(output, parent_checkpoint, resume=None, data_root=None):
    output = Path(output).resolve()
    def run_directory(checkpoint):
        parent = Path(checkpoint).resolve().parent
        return parent.parent if parent.name == 'checkpoints' else parent
    protected = [ROOT, run_directory(parent_checkpoint)]
    if resume is not None:
        protected.append(run_directory(resume))
    if data_root is not None:
        protected.append(Path(data_root).resolve())
    for directory in protected:
        s.require(not output.is_relative_to(directory), 'Resultaten mogen niet in een bestaande invoer-/pakketmap: ' + str(directory))

