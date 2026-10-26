"""Strict restoration of the verified frozen epoch55, then continuation to100.

No optimizer intervention is made: theta was already frozen in the pilot.
Only the experiment horizon/lineage contract changes after exact restoration.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from continuation_bridge import ROOT, PILOT_ROOT, pilot, pilot_core, ref, s, t, a, core, torch

PILOT_REFERENCE_SHA256 = 'd9578eb271cf6f1e8af5544f5e1bb6c0cd10bd9fc6805a2be23ebf79a47da5d9'
DYNAMIC_REFERENCE_SHA256 = '6ddae3d3603a6f31c4dec5ae64b8d0c166646bc4576c599903594786b6e85706'


def config(parent=None):
    c = copy.deepcopy(pilot.config() if parent is None else parent)
    c.update(schema='mx-triple-freeze-offsets-e50-reference100-protocol-v1',
        purpose='continue verified frozen pilot55 through100 with unchanged offsets anchored at50',
        start_after_epoch=55, continuation_parent_epoch=55, freeze_anchor_epoch=50,
        end_epoch=100, planned_new_epochs=45, planned_epochs=100, reference_end_epoch=100,
        selection_epoch_range=[51, 100], global_selection_epoch_range=[1, 100],
        initialization_policy='strict restore of actual frozen epoch55 including model, BN, Adam, scheduler and RNG',
        selection_policy='first strict maximum E4M3 official-test top1 within51-100; additional global1-100 selection explicitly labelled by stage',
        reporting_policy='primary endpoint100, shared E4M3-best postfreeze51-100, and shared globalE4best1-100 with prefreeze label when applicable',
        historical_control='existing complete dynamic learned-scale reference100; no new control training',
        completion_policy='45 new epochs56-100; inherited frozen51-55 retained; original schedule not restarted')
    return c


def verify_package():
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    s.require(isinstance(manifest, dict) and manifest, 'Pakketmanifest ontbreekt.')
    for name, digest in manifest.items():
        path = (ROOT / name).resolve()
        s.require(path.is_relative_to(ROOT) and path.is_file() and core.file_sha256(path) == digest,
                  'Pakketbestand ontbreekt of is gewijzigd: ' + name)
    pilot_core.verify_package()
    return manifest


def _load_pinned_reference(directory, digest, external=()):
    directory = Path(directory).resolve()
    marker = directory / 'REFERENCE.json'
    s.require(core.file_sha256(marker) == digest, 'Referentie-identiteit wijkt af: ' + str(directory))
    meta = json.loads(marker.read_text(encoding='utf-8'))
    for name, expected in meta['files_sha256'].items():
        if name in external:
            continue
        path = (directory / name).resolve()
        s.require(path.is_relative_to(directory) and path.is_file() and core.file_sha256(path) == expected,
                  'Referentiebestand ontbreekt of is gewijzigd: ' + name)
    return meta


def load_evidence(dynamic_run, pilot_dir=None, dynamic_dir=None,
                  expected_pilot_sha=PILOT_REFERENCE_SHA256, expected_dynamic_sha=DYNAMIC_REFERENCE_SHA256):
    pilot_dir = (ROOT / 'pilot_reference' if pilot_dir is None else Path(pilot_dir)).resolve()
    dynamic_dir = (ROOT / 'dynamic_reference' if dynamic_dir is None else Path(dynamic_dir)).resolve()
    dynamic_run = Path(dynamic_run).resolve()
    pm = _load_pinned_reference(pilot_dir, expected_pilot_sha)
    dm = _load_pinned_reference(dynamic_dir, expected_dynamic_sha, external=('history.json',))
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    ph, pr, pc = (read(pilot_dir / n) for n in ('history.json', 'run_report.json', 'parent_contract55.json'))
    dr, dc = (read(dynamic_dir / n) for n in ('run_report.json', 'parent_contract.json'))
    history_file = dynamic_run / 'history.json'
    s.require(history_file.is_file(), 'De oorspronkelijke history.json ontbreekt: ' + str(history_file) + '. Bewaar de oorspronkelijke 100-epochresultaten of geef --dynamic-run op.')
    s.require(core.file_sha256(history_file) == dm['source_history_sha256'] == dm['files_sha256']['history.json'],
              'Oorspronkelijke 100-epochgeschiedenis wijkt af van de vastgelegde feedback.')
    dh = read(history_file)
    s.require(core.json_digest(pc) == pm['source_contract_sha256'] == pr['contract_sha256'], 'Pilot55-contract klopt niet.')
    s.require(core.json_digest(dc) == dm['source_contract_sha256'] == dr['contract_sha256']
              and not core.differences(pc['parent_contract'], dc), 'Oorspronkelijk contract of afstamming klopt niet.')
    s.require(pr['status'] == 'PASS' and pr['continuation_completed'] and pr['completed_epoch'] == 55,
              'Verwacht de volledig afgeronde bevroren pilot na epoch55.')
    s.require(dr['status'] == 'PASS' and dr['completed_epoch'] == 100 and dr['full_training_completed'],
              'Verwacht de volledig afgeronde dynamische 100-epochreferentie.')
    s.require([r['epoch'] for r in ph['epochs']] == list(range(51, 56))
              and [r['epoch'] for r in dh['epochs']] == list(range(1, 101)), 'Onvolledige historische epochs.')
    baseline = read(pilot_dir / 'parent_reference' / 'baseline_epoch50.json')
    s.require(not core.differences(baseline, dh['epochs'][49])
              and ph['initial_model_sha256'] == baseline['model_sha256']
              and not core.differences(ph['initial_scale_state'], baseline['scale_state']),
              'De twee trajecten hebben niet hetzelfde startcheckpoint na epoch50.')
    s.require(ph['epochs'][-1]['model_sha256'] == pm['parent_model_sha256'], 'Verkeerde pilot55-modelidentiteit.')
    flags = [pc['synthetic_fixture_only'], dc['synthetic_fixture_only'], pm['synthetic_fixture_only'], dm['synthetic_fixture_only']]
    s.require(all(type(x) is bool for x in flags) and len(set(flags)) == 1, 'Echte en synthetische referenties vermengd.')
    return dict(pilot_meta=pm, pilot_history=ph, pilot_report=pr, pilot_contract=pc,
        pilot_dir=pilot_dir, pilot_reference_sha256=expected_pilot_sha,
        dynamic_meta=dm, dynamic_history=dh, dynamic_report=dr, dynamic_contract=dc,
        dynamic_dir=dynamic_dir, dynamic_run=dynamic_run, dynamic_reference_sha256=expected_dynamic_sha,
        baseline=baseline, comparison={r['epoch']: r for r in dh['epochs']})


def verify_parent55_state(ctx, state, evidence):
    meta, expected = evidence['pilot_meta'], evidence['pilot_contract']
    original = getattr(ctx, 'original_contract', ctx.contract)
    s.require(not core.differences(original, evidence['dynamic_contract']),
              'Huidige omgeving, data of oorspronkelijke runtime wijkt af.')
    s.require(not core.differences(state['contract'], expected)
              and core.json_digest(state['contract']) == meta['source_contract_sha256'], 'Verkeerd pilot55-contract.')
    s.require(state['completed_epoch'] == 55 and state['next_epoch'] == 56
              and state['scheduler']['last_epoch'] == 55, 'Verwacht exact het bevroren checkpoint na epoch55.')
    c = expected['config']
    batches = (c['train_examples'] + c['batch_size'] - 1) // c['batch_size']
    s.require(state['optimizer_steps'] == meta['parent_optimizer_steps'] == 55 * batches, 'Verkeerd aantal geërfde Adamupdates.')
    model_hash = s.legacy.state_digest(state['model'])
    s.require(model_hash == meta['parent_model_sha256'], 'Onjuist bevroren model na epoch55.')
    s.require(core.json_digest(state['trace']) == meta['parent_checkpoint_trace_sha256'], 'Pilot55-checkpointgeschiedenis wijkt af.')
    expected_trace = copy.deepcopy(evidence['pilot_history'])
    for key in pilot_core.TIMING_AFTER_SAVE:
        expected_trace['epochs'][-1].pop(key, None)
    s.require(not core.differences(state['trace'], expected_trace), 'Pilot55-geschiedenis is niet de geleverde werkelijke geschiedenis.')
    groups = state['optimizer']['param_groups']
    lr = float(c['lr']) * float(c['scheduler_gamma'])
    s.require([g['name'] for g in groups] == ['weights', 'scale_proxy']
              and [float(g['lr']) for g in groups] == [lr, 0.0]
              and groups[1]['initial_lr'] == 0.0
              and state['scheduler']['_last_lr'] == [lr, 0.0]
              and state['scheduler']['base_lrs'][1] == 0.0, 'Pilot55 is niet exact bevroren met het oorspronkelijke LR-schema.')
    theta = {name: state['model'][name + '.quantize_fn.theta'].tolist() for name in a.TARGET_LAYERS}
    s.require(not core.differences(theta, evidence['baseline']['scale_state']), 'Theta is niet gelijk aan de vaste epoch50-keuzes.')
    return dict(parent_epoch=55, parent_model_sha256=model_hash,
        parent_contract_sha256=meta['source_contract_sha256'],
        parent_checkpoint_trace_sha256=meta['parent_checkpoint_trace_sha256'],
        parent_checkpoint_verified=True, parent_model_matches_reference=True,
        parent_contract_matches_reference=True, parent_trace_matches_reference=True)


def initialize_continuation(ctx, parent_checkpoint55, evidence, package_hashes, resume=None):
    parent_checkpoint55 = Path(parent_checkpoint55).resolve()
    digest = core.file_sha256(parent_checkpoint55)
    s.require(digest == evidence['pilot_meta']['parent_checkpoint_sha256'], 'Dit is niet het vastgelegde werkelijke checkpoint na epoch55.')
    state = core.load_verified(parent_checkpoint55)
    ctx.original_contract = copy.deepcopy(ctx.contract)
    verified = verify_parent55_state(ctx, state, evidence)
    verified['parent_checkpoint_sha256'] = digest
    verified['strict_restore_checks'] = core.restore_checkpoint(state, ctx.model, ctx.optimizer, ctx.scheduler,
                                                               {}, ctx.rt.device, evidence['pilot_contract'])
    t.validate_bn(ctx.model, state['optimizer_steps'])
    t.validate_adam_state(ctx.model, ctx.optimizer, state['optimizer_steps'])
    ctx.pilot_contract = copy.deepcopy(evidence['pilot_contract'])
    ctx.c = config(ctx.pilot_contract['config'])
    ctx.contract = dict(schema='mx-triple-freeze-offsets-e50-reference100-contract-v1', config=ctx.c,
        pilot_contract=ctx.pilot_contract, parent55_contract_sha256=evidence['pilot_meta']['source_contract_sha256'],
        parent55_checkpoint_sha256=digest, parent55_model_sha256=verified['parent_model_sha256'],
        pilot_reference_sha256=evidence['pilot_reference_sha256'],
        dynamic_reference_sha256=evidence['dynamic_reference_sha256'],
        dynamic_history_sha256=evidence['dynamic_meta']['source_history_sha256'],
        package_sha256=package_hashes, synthetic_fixture_only=ctx.pilot_contract['synthetic_fixture_only'])
    # The pinned feedback contains the final timing fields written after saving
    # parent55. Preserve those actual measurements without depending on a sidecar.
    trace = copy.deepcopy(evidence['pilot_history'])
    trace.update(continuation_parent_epoch=55, parent55_verification=verified,
                 continuation_initial_model_sha256=verified['parent_model_sha256'])
    done, steps = 55, state['optimizer_steps']
    if resume is not None:
        restored = core.load_verified(Path(resume).resolve())
        s.require(55 < restored['completed_epoch'] < 100, 'Hervat een nieuwe voortzettingscheckpoint na epoch56-99.')
        checks = core.restore_checkpoint(restored, ctx.model, ctx.optimizer, ctx.scheduler, {}, ctx.rt.device, ctx.contract)
        done, steps = restored['completed_epoch'], restored['optimizer_steps']
        trace = copy.deepcopy(restored['trace'])
        ref.restore_timing_sidecar(ctx.rt, trace)
        s.require(trace['continuation_parent_epoch'] == 55
                  and trace['continuation_initial_model_sha256'] == verified['parent_model_sha256']
                  and not core.differences(trace['epochs'][:5], evidence['pilot_history']['epochs'])
                  and [r['epoch'] for r in trace['epochs']] == list(range(51, done + 1)), 'Verkeerde voortzettingsgeschiedenis.')
        s.require(steps == done * len(ctx.loaders['train']), 'Verkeerde hervatstapteller.')
        ctx.report.update(resumed_from=str(Path(resume).resolve()), resume_input_sha256=core.file_sha256(Path(resume)),
                          restore_checks=checks)
    ctx.guard = pilot_core.FrozenOffsetGuard(ctx.model, ctx.optimizer, evidence['baseline']['scale_state'])
    ctx.parent55_path = parent_checkpoint55
    ctx.pilot_run = run_directory(parent_checkpoint55)
    ctx.evidence = evidence
    ctx.report.update(config=ctx.c, parent_verification=trace['parent_verification'], parent55_verification=verified,
        continuation_parent_epoch=55, parent_checkpoint=str(parent_checkpoint55), parent_checkpoint_input_sha256=digest,
        contract_sha256=core.json_digest(ctx.contract), initial_model_sha256=trace['initial_model_sha256'],
        initial_base_model_sha256=trace['initial_base_model_sha256'], continuation_initial_model_sha256=verified['parent_model_sha256'],
        synthetic_fixture_only=ctx.contract['synthetic_fixture_only'],
        intervention=dict(offset_freeze_already_active=True, optimizer_and_scheduler_unchanged_at_continuation=True,
                          theta_lr=0.0, source_scale_dynamic=True, only_horizon_extended_to100=True))
    return trace, done, steps


def select_records(trace, evidence):
    records = trace['epochs']
    s.require(records and [r['epoch'] for r in records] == list(range(51, records[-1]['epoch'] + 1)), 'Onvolledige bevroren geschiedenis.')
    whole = evidence['dynamic_history']['epochs'][:50] + records
    return {'last_epoch': records[-1], 'best_E4M3_postfreeze': t.best_record(records),
            'best_E4M3_global': t.best_record(whole)}


def source_stage(epoch):
    return 'shared_before_freeze' if epoch <= 50 else ('frozen_pilot51_55' if epoch <= 55 else 'frozen_continuation56_100')


def run_directory(checkpoint):
    parent = Path(checkpoint).resolve().parent
    return parent.parent if parent.name == 'checkpoints' else parent


def checkpoint_for_record(ctx, record, evidence):
    epoch = record['epoch']
    if epoch <= 50:
        directory, expected = evidence['dynamic_run'], evidence['dynamic_contract']
    elif epoch <= 55:
        directory, expected = ctx.pilot_run, evidence['pilot_contract']
    else:
        path = Path(record['checkpoint_path'])
        s.require(path.is_file(), 'Voortzettingscheckpoint ontbreekt: ' + str(path))
        directory, expected = run_directory(path), ctx.contract
    if epoch <= 55:
        candidates = [directory / 'checkpoints' / f'epoch_{epoch:05d}.pt']
        candidates.append(directory / ('last_checkpoint.pt' if epoch == 55 else 'best_checkpoint.pt'))
        path = next((p for p in candidates if p.is_file()), candidates[0])
    state = core.load_verified(path)
    s.require(not core.differences(state['contract'], expected), 'Geselecteerd checkpoint heeft een ander stadiumcontract.')
    s.require(state['completed_epoch'] == epoch and s.legacy.state_digest(state['model']) == record['model_sha256']
              and not core.differences(state['trace']['epochs'][-1]['test'], record['test']),
              'Geselecteerd checkpoint strookt niet met zijn historische metrics.')
    theta = {name: state['model'][name + '.quantize_fn.theta'].tolist() for name in a.TARGET_LAYERS}
    expected_theta = record['scale_state'] if epoch <= 50 else evidence['baseline']['scale_state']
    s.require(not core.differences(theta, expected_theta), 'Geselecteerde checkpoint heeft onjuiste stadiumgebonden theta.')
    return path, state


def preflight_selected_inputs(ctx, evidence):
    records = [t.best_record(evidence['dynamic_history']['epochs'][:50]), t.best_record(evidence['pilot_history']['epochs'])]
    result = []
    for record in records:
        path, state = checkpoint_for_record(ctx, record, evidence)
        result.append(dict(epoch=record['epoch'], source_stage=source_stage(record['epoch']),
                           path=str(path), sha256=core.file_sha256(path), verified=True))
    return result


def validate_output_location(output, parent_checkpoint55, dynamic_run, resume=None, data_root=None):
    output = Path(output).resolve()
    protected = [ROOT, run_directory(parent_checkpoint55), Path(dynamic_run).resolve()]
    if resume is not None:
        protected.append(run_directory(resume))
    if data_root is not None:
        protected.append(Path(data_root).resolve())
    for directory in protected:
        s.require(not output.is_relative_to(directory), 'Resultaten mogen niet in een bestaande invoer-/pakketmap: ' + str(directory))

