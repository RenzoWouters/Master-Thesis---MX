"""Five new epochs after restoring the independent-E2 epoch50 checkpoint.

Absolute E2M3/E2M1 scales and their Adam state are frozen. E4 amax scales,
FP32 weights, three BN banks and the original recursive-loss recipe continue.
Only epochs51..55 are allowed; the existing learn-through100 run is the control.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
import svhn_support as s
import trunc_support as t
import trunc_adapter as a
import freeze_support as freeze

ROOT = Path(__file__).resolve().parent


def config(mode: str = 'joint') -> dict:
    s.require(mode == 'joint', 'Deze proef heeft exact drie gezamenlijke MX-modi.')
    c = json.loads((ROOT/'independent_full100_reference/run_report.json').read_text(encoding='utf-8'))['config']
    c = copy.deepcopy(c)
    c.update(schema='mx-triple-independent-e2-scale-freeze50-pilot5-protocol-v1',
        purpose='restore the pinned independent-E2 epoch50 checkpoint; freeze the absolute E2 scale and continue weights/BN for epochs51..55 only; reuse the existing learning control',
        initialization_policy='restore complete independent-E2 epoch50 state; never restart training from initialization',
        planned_epochs=100, reference_end_epoch=100, pilot_stop_epoch=55,
        inherited_epochs=50, branch_start_epoch=51, pilot_new_epochs=5,
        scale_learning_end_epoch=50, scale_freeze_epoch=50,
        frozen_scale_lr=0.0, freeze_absolute_E2_scale=True,
        scale_projection='only while learned through50; no projection after freeze',
        scale_gradient_after50='disabled: requires_grad=False and grad=None; retained scale Adam state does not advance',
        selection_policy='first strict maximum of E4M3/A32 official-test top1; report global and branch-only maxima separately',
        reporting_policy='last observed epoch up to55, shared global best E4 and branch-only best E4; controls use the same available epoch window; no epoch100 completion claim',
        checkpoint_policy='read-only epoch50 parent; immutable child state each new epoch; strict child resume',
        parent_reference='independent_full100_reference')
    return c



NOTEBOOK = 'MX_triple_independent_e2_scale_freeze50_pilot5_resultaten.ipynb'


REFERENCE_DIRECTORIES = ('fixed_scale64_reference', 'relative_scale10_reference',
                         'relative_scale10_full100_reference', 'independent_pilot_reference', 'independent_full100_reference')


def load_initialization_reference(directory=None):
    """Pinned historical relative-scale run; raw fields remain unchanged."""
    directory = Path(directory) if directory is not None else ROOT / 'relative_scale10_full100_reference'
    expected_epochs = 100 if directory.name == 'relative_scale10_full100_reference' else 5
    meta = json.loads((directory / 'REFERENCE.json').read_text(encoding='utf-8'))
    for name, digest in meta['files_sha256'].items():
        path = (directory / name).resolve()
        s.require(path.is_relative_to(directory.resolve()) and path.is_file()
                  and s.legacy.sha256(path) == digest, 'Historische referentie gewijzigd: ' + name)
    report = json.loads((directory / 'run_report.json').read_text(encoding='utf-8'))
    history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
    s.require(report['status'] == 'PASS' and report['config']['scale_proxy_init'] == 6.0,
              'Verwacht de geslaagde historische relatieve tienkeuzerun met start x64.')
    s.require([r['epoch'] for r in history['epochs']] == list(range(1, expected_epochs + 1)),
              'Historische referentie heeft onvolledige of onverwachte epochs.')
    s.require(history['initial_base_model_sha256'] == report['initial_base_model_sha256'],
              'Historische referentie heeft verschillende oorspronkelijke modelidentiteiten.')
    if expected_epochs == 100:
        c = report['config']
        s.require(report.get('full_training_completed') is True
                  and c['planned_epochs'] == c['scale_learning_end_epoch'] == 100
                  and c['scale_freeze_epoch'] is None,
                  'De volledige relatieve referentie moet100 epochs met leerbare schalen bevatten.')
    return dict(directory=directory, meta=meta, report=report, history=history,
                kind='relative10_full100' if expected_epochs == 100 else 'relative10_pilot5')


def archive_initialization_reference(out):
    for name in REFERENCE_DIRECTORIES:
        source, destination = ROOT / name, out / name
        s.require(source.is_dir(), 'Een vastgelegde historische referentie ontbreekt: ' + name)
        if destination.exists():
            for path in source.rglob('*'):
                if path.is_file():
                    other = destination / path.relative_to(source)
                    s.require(other.is_file() and other.read_bytes() == path.read_bytes(),
                              'Bestaande gearchiveerde referentie wijkt af: ' + str(other))
        else:
            shutil.copytree(source, destination)


def capture_source_scale_state(model):
    """Detached diagnostics only; never write or reinitialize phi."""
    diagnostics = a.scale_diagnostics(model)
    return {name: [[int(e) + 127 for e in row] for row in diagnostics[name]['e4_exponents']]
            for name in a.TARGET_LAYERS}


def validate_initialization(model, c, reference, report):
    import torch
    s.require(c['initial_relative_offset'] == a.INITIAL_RELATIVE_OFFSET == 6,
              'Deze proef moet eenmaal beginnen bij de aanvankelijke E4-schaal x64.')
    state = a.validate_scale_parameters(model)
    diagnostics = a.scale_diagnostics(model)
    s.require(state['total_blocks'] == 4080 and state['parameter_tensors'] == 6,
              'Ongeldig aantal onafhankelijke E2-schaalparameters.')
    for name, record in diagnostics.items():
        phi = torch.tensor(record['phi'], dtype=torch.float64)
        source = torch.tensor(record['e4_exponents'], dtype=torch.int32)
        lower = torch.tensor(record['e2_exponents'], dtype=torch.int32)
        s.require(bool((phi == source.to(torch.float64) + 6).all())
                  and bool((lower == source + 6).all()),
                  'De aanvankelijke absolute E2-schaal is niet exact de E4-beginschaal x64: ' + name)
    base = s.legacy.state_digest(a.base_state_dict(model))
    s.require(base == reference['history']['initial_base_model_sha256'],
              'FP32-gewichten of BN-initialisatie wijken af van de historische x64-referentie.')
    report['initialization_verification'] = dict(
        initial_relative_offset=6, initial_relative_factor=64,
        construction_only_all4080_groups_at_initial_E4_scale_x64=True,
        fresh_construction_discarded_before_training=True, experiment_restores_parent_epoch50=True,
        initial_base_model_sha256=base, same_base_weights_and_BN_as_relative_reference=True,
        scale_parameterization_is_the_intended_training_change=False,
        intended_training_change='freeze absolute E2 scales after epoch50',
        source_scale_is_still_dynamic=True, absolute_E2_scale_independent_after_initialization=True,
        relative_factor_is_not_constrained=True)


def write_initialization_comparison(ctx, out, trace):
    if ctx.contract['synthetic_fixture_only'] and not hasattr(ctx, 'initialization_reference'):
        result = dict(synthetic_fixture_only=True, comparison_performed=False,
                      reason='Synthetic software fixture is not compared with real research data.')
        s.write_json(out / 'scale_parameterization_reference_report.json', result)
        ctx.report['scale_parameterization_reference'] = result
        return result
    reference = getattr(ctx, 'initialization_reference', None) or load_initialization_reference()
    old = reference['history']
    s.require(trace['initial_base_model_sha256'] == old['initial_base_model_sha256'],
              'De twee runs vertrekken niet van dezelfde basisgewichten/BN.')
    rows, matched, unmatched = [], [], []
    old_by_epoch = {record['epoch']: record for record in old['epochs']}
    for record in trace['epochs']:
        counterpart = old_by_epoch.get(record['epoch'])
        if counterpart is None:
            unmatched.append(record['epoch'])
            continue
        matched.append(record['epoch'])
        for phase in ('train', 'test'):
            s.require(record['coverage'][phase]['order_sha256'] == counterpart['coverage'][phase]['order_sha256'],
                      'Samplevolgorde verschilt van de historische x64-referentie.')
        bybit = {row['weight_bits']: row for row in counterpart['test']}
        for row in record['test']:
            other = bybit[row['weight_bits']]
            s.require(row['n'] == other['n'], 'Verschillend aantal testbeelden.')
            rows.append(dict(epoch=record['epoch'], weight_bits=row['weight_bits'],
                weight_format=row['weight_format'], independent_top1_percent=row['top1_percent'],
                relative_scale10_top1_percent=other['top1_percent'],
                difference_percentage_points=row['top1_percent']-other['top1_percent'],
                independent_correct1=row['correct1'], relative_scale10_correct1=other['correct1'],
                test_examples=row['n']))
    s.require(bool(rows), 'Geen gemeenschappelijke historische epochs voor de vergelijking.')
    s.write_csv(out / 'scale_parameterization_comparison.csv', rows)
    result = dict(synthetic_fixture_only=ctx.contract['synthetic_fixture_only'], comparison_performed=True,
        matched_epochs=matched, unmatched_epochs=unmatched,
        historical_reference_kind=reference['kind'], historical_reference_completed_epochs=len(old['epochs']),
        historical_reference_directory=reference['directory'].name,
        missing_historical_epochs_are_not_extrapolated=True,
        initial_base_model_sha256=trace['initial_base_model_sha256'],
        initial_base_weights_and_BN_match=True, all_sample_orders_match=True,
        initial_relative_factor_both_runs=64, parameterization_changed=True,
        new_parameterization='absolute E2 exponent independent of current E4 scale',
        reference_parameterization='relative E2 offset0..9 from current E4 scale',
        comparison_is_historical=True, exact_CUDA_repeat_not_claimed=True,
        source_reference_sha256=s.legacy.sha256(reference['directory']/'REFERENCE.json'))
    s.write_json(out / 'scale_parameterization_reference_report.json', result)
    ctx.report['scale_parameterization_reference'] = result
    return result


def write_freeze_comparison(ctx, out, trace):
    reference=ctx.parent_reference
    freeze.verify_prefix(trace,reference)
    control={r['epoch']:r for r in reference['history']['epochs']}
    rows=[]
    for record in trace['epochs']:
        other=control[record['epoch']]
        for phase in ('train','test'):
            s.require(record['coverage'][phase]['order_sha256']==other['coverage'][phase]['order_sha256'],
                      'Samplevolgorde verschilt van de oorspronkelijke doorleercontrole.')
        bybit={r['weight_bits']:r for r in other['test']}
        for row in record['test']:
            ref=bybit[row['weight_bits']]
            s.require(row['n']==ref['n'],'Verschillend aantal testbeelden in freezevergelijking.')
            rows.append({'epoch':record['epoch'],'weight_bits':row['weight_bits'],
                'weight_format':row['weight_format'],'frozen_top1_percent':row['top1_percent'],
                'learning_top1_percent':ref['top1_percent'],
                'difference_percentage_points':row['top1_percent']-ref['top1_percent'],
                'frozen_correct1':row['correct1'],'learning_correct1':ref['correct1'],
                'test_examples':row['n'],'segment':'inherited' if record['epoch']<=50 else 'branch'})
    s.write_csv(out/'freeze_comparison.csv',rows)
    result={'comparison_performed':True,'parent_checkpoint':trace['parent_checkpoint'],
            'parent_control_retrained':False,'inherited_epochs_identical':True,
            'inherited_epochs':50,'matched_epochs':[r['epoch'] for r in trace['epochs']],
            'branch_epochs':[r['epoch'] for r in trace['epochs'] if r['epoch']>50],
            'all_sample_orders_match':True,'exact_CUDA_repeat_not_claimed':True,
            'historical_reference_directory':reference['directory'].name,
            'reference_sha256':reference['reference_sha256'],
            'synthetic_fixture_only':ctx.contract['synthetic_fixture_only']}
    s.write_json(out/'freeze_comparison_report.json',result)
    ctx.report['freeze_comparison']=result
    return result


def write_notebook(out):
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n',
                              'SAVE_FIGURES = False\n']
    target = out / NOTEBOOK
    if target.exists():
        s.require(json.loads(target.read_text(encoding='utf-8')) == notebook,
                  'Bestaande notebook verschilt; niets overschrijven.')
    else:
        s.write_json(target, notebook)
    raw = (ROOT / 'notebook_support.py').read_bytes()
    target = out / 'notebook_support.py'
    if target.exists():
        s.require(target.read_bytes() == raw, 'Bestaande notebookhelper verschilt; niets overschrijven.')
    else:
        with target.open('xb') as handle: handle.write(raw)


def dependencies(out: Path, log) -> None:
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    (out / 'pip_freeze.txt').write_text(freeze.stdout + freeze.stderr, encoding='utf-8')
    s.require(freeze.returncode == 0, 'pip freeze mislukt; zie pip_freeze.txt.')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
    (out / 'pip_check.txt').write_text(check.stdout + check.stderr, encoding='utf-8')
    s.require(check.returncode == 0, 'Afhankelijkheidsconflict; zie pip_check.txt. Geen automatische upgrades.')
    log('pip check: ' + check.stdout.strip())


def create_context(args, out: Path, report: dict, log, check_only: bool = False):
    package_hashes = s.verify_package()
    c = config(args.mode)
    reference = freeze.load_reference(ROOT)
    parent_path = (args.parent_checkpoint.resolve() if args.parent_checkpoint is not None
                   else freeze.default_parent(ROOT, reference))
    parent_state = freeze.verify_parent(parent_path, reference)
    archive_initialization_reference(out)
    source, source_info = s.legacy.prepare_source(out)
    report.update(config=c, source=source_info, package_sha256=package_hashes,
                  interpreter=sys.executable, parent_checkpoint=freeze.linkage(reference),
                  parent_checkpoint_path=str(parent_path))
    s.write_json(out / 'training_change_manifest.json',
                 json.loads((ROOT / 'TRAINING_CHANGE_MANIFEST.json').read_text(encoding='utf-8')))
    dependencies(out, log)
    rt = s.runtime(c['seed'], args.device)
    modules = s.import_source(source)
    report.update(environment=rt.environment, execution_settings=rt.settings)
    log('GPU: ' + rt.environment['device_name'])
    log('Originele backend: cudnn.benchmark=True; geen bitexacte CUDA-hervatgarantie.')
    model = t.build_model(modules, c['mode'], rt.device)
    s.require(model.expand == c['expand'] and sum(p.numel() for p in model.parameters()) == c['parameters'],
              'Onverwacht model; verwacht svhnq expand=2.')
    validate_initialization(model, c, reference, report)
    optimizer = t.build_optimizer(modules, model, c)
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, c['scheduler_milestones'])
    ctx = SimpleNamespace(rt=rt, source=source, modules=modules, model=model, optimizer=optimizer,
        scheduler=scheduler, c=c, report=report, source_info=source_info,
        parent_reference=reference, parent_path=parent_path, parent_state=parent_state)
    report['parent_environment_checks'] = freeze.verify_environment(ctx,parent_state,include_data=False)
    if check_only:
        import test_freeze
        report['freeze_checks'] = test_freeze.check_parent_transition(ctx,out,log)
        s.write_json(out/'freeze_check_report.json',report['freeze_checks'])
        report.update(dataset_downloaded=False, synthetic_checks_only=True,
                      no_SVHN_training=True, parent_checkpoint_unchanged=True)
        s.require(rt.core.file_sha256(parent_path)==reference['meta']['parent_checkpoint_sha256'],
                  'Invoercheckpoint gewijzigd tijdens check.')
        log('CHECK PASS: oudercheckpoint exact hersteld; absolute E2-schalen en schaal-Adam bevroren; gewichten/BN kunnen bijwerken; E4-schaal dynamisch. Geen SVHN-training.')
        return None

    start_data = time.perf_counter()
    info = s.ensure_dataset(args.data_root.resolve(), modules.data.SVHNMIX, args.download, log)
    report['data_preparation_seconds_including_download_and_checksums'] = time.perf_counter() - start_data
    report['dataset_files'] = info
    modules.data.data_paths['svhn'] = str(args.data_root.resolve())
    start_load = time.perf_counter()
    train_tf = modules.data.get_transform('svhn', 'train')
    test_tf = modules.data.get_transform('svhn', 'val')
    s.require(c['train_split'] == 'train' and c['extra_used'] is False,
              'Deze runner is uitsluitend voor de volledige train-split, zonder extra.')
    train = modules.data.get_dataset('svhn', c['train_split'], train_tf, download=False)
    test = modules.data.get_dataset('svhn', 'val', test_tf, download=False)
    s.require(train.split == 'train' and test.split == 'test', 'Onverwachte datasetsplit.')
    s.require(set(info['files']) == {'train_32x32.mat', 'test_32x32.mat'},
              'Train-only mag alleen train en test laden of controleren.')
    selection = {'train': s.validate_dataset(train, c['train_examples'], 'SVHN train-only'),
                 'test': s.validate_dataset(test, c['test_examples'], 'SVHN test'),
                 'train_split_order': ['train'], 'extra_used': False,
                 'held_out_validation_examples': 0}
    report['dataset_load_seconds'] = time.perf_counter() - start_load
    report['dataset_selection'] = selection
    report['transforms'] = {'train': repr(train_tf), 'test': repr(test_tf)}
    s.write_json(out / 'dataset_manifest.json', {'files': info, 'selection': selection,
                                              'transforms': report['transforms']})
    log(f'DATA PASS: train={len(train)}, test={len(test)}; extra NIET gebruikt; '
        'invoer 40x40; volledige train-split; geen validatiesplit.')
    # No explicit generators: match the global RNG policy of the source loaders.
    torch = rt.torch
    loaders = {
        'train': s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(train),
                  batch_size=c['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=False)),
        'test': s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(test),
                  batch_size=c['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=False)),
    }
    log(f'MINIBATCHES per epoch: train={len(loaders["train"])}; test={len(loaders["test"])}; '
        f'batch={c["batch_size"]}, drop_last=False.')

    contract = {
        'config': c, 'environment': rt.environment, 'settings': rt.settings,
        'source_files': source_info['workcopy_files_sha256'], 'package_sha256': package_hashes,
        'data_files': info['files'], 'dataset_selection': selection, 'transforms': report['transforms'],
        'synthetic_fixture_only': False, 'parent_checkpoint': freeze.linkage(reference),
    }
    ctx.contract, ctx.loaders = contract, loaders
    report['parent_environment_checks'] = freeze.verify_environment(ctx,parent_state,include_data=True)
    report['contract_sha256'] = rt.core.json_digest(contract)
    return ctx


def restore_timing_sidecar(rt, trace: dict) -> None:
    # The last checkpoint is written before its own write duration is known.
    # The immutable sidecar completes that timing without altering the checkpoint.
    if not trace['epochs']:
        return
    last = trace['epochs'][-1]
    path = Path(last['checkpoint_path']).parent.parent / 'epoch_summaries' / f"epoch_{last['epoch']:05d}.json"
    marker = path.with_suffix('.json.sha256')
    if path.is_file() and marker.is_file():
        s.require(rt.core.file_sha256(path) == marker.read_text().strip(), 'Epoch-sidecar checksum wijkt af.')
        full = json.loads(path.read_text(encoding='utf-8'))
        for key, value in last.items():
            s.require(value == full.get(key), 'Epoch-sidecar strookt niet met checkpoint: ' + key)
        trace['epochs'][-1] = full


def run_epochs(ctx, out: Path, log, stop_after: int, resume: Path | None = None,
               install_signal_handler: bool = True) -> dict:
    rt, c = ctx.rt, ctx.c
    s.require(c['planned_epochs'] == c['reference_end_epoch'] == 100
              and c['scale_freeze_epoch'] == c['scale_learning_end_epoch'] == 50,
              'Onjuist freeze50-contract.')
    s.require(type(stop_after) is int and 51 <= stop_after <= 55,
              'Deze pilot voert alleen epochs51..55 uit.')
    torch, core, device = rt.torch, rt.core, rt.device
    model, optimizer, scheduler, loaders = ctx.model, ctx.optimizer, ctx.scheduler, ctx.loaders
    for directory in ['checkpoints', 'epoch_summaries', 'sample_order', 'batch_metrics']:
        (out / directory).mkdir(exist_ok=True)
    expected_steps = len(loaders['train'])
    s.require(expected_steps == math.ceil(c['train_examples'] / c['batch_size']), 'Verkeerd aantal minibatches.')
    resume_hash = None
    if resume is not None:
        resume = resume.resolve()
        state = core.load_verified(resume)
        s.require(50 < state['completed_epoch'] < stop_after,
                  'Dit moet een hervatbaar checkpoint uit deze freezevertakking zijn.')
        # core checks the complete child contract before loading any state.
        ctx.report['restore_checks'] = core.restore_checkpoint(state, model, optimizer, scheduler, {},
                                                              device, ctx.contract)
        trace = copy.deepcopy(state['trace'])
        restore_timing_sidecar(rt, trace)
        freeze.verify_prefix(trace,ctx.parent_reference)
        done, steps = state['completed_epoch'], state['optimizer_steps']
        resume_hash = core.file_sha256(resume)
        ctx.report.update(resumed_from=str(resume),resume_input_sha256=resume_hash)
        log(f'HERVAT freezevertakking na epoch {done}; dezelfde ouder en bevroren E2-schalen.')
    else:
        state=ctx.parent_state
        ctx.report['restore_checks'] = core.restore_checkpoint(state,model,optimizer,scheduler,{},
                                                               device,state['contract'])
        trace=freeze.parent_trace(state,ctx.parent_reference)
        done,steps=50,state['optimizer_steps']
        log('OUDERCHECKPOINT epoch50 volledig hersteld; geen hertraining van epochs1..50.')
    s.require(steps == done * expected_steps and len(trace['epochs'])==done,
              'Opgeslagen stappen of epochgeschiedenis onjuist.')
    transition=freeze.apply_freeze(model,optimizer,scheduler,ctx.parent_state)
    if resume is None:trace['freeze_transition']=transition
    else:
        s.require(trace['freeze_transition']['frozen_phi_sha256']==transition['frozen_phi_sha256']
                  and trace['freeze_transition']['frozen_hard_E2_scale_codes_sha256']==transition['frozen_hard_E2_scale_codes_sha256'],
                  'Hervatting wijkt af van oorspronkelijke freeze.')
    ctx.report['freeze_transition']=trace['freeze_transition']
    freeze.verify_prefix(trace,ctx.parent_reference)
    first_epoch = done + 1
    soft_stop = {'requested': False}
    old_handler = None
    def handler(signum, frame):
        if soft_stop['requested']:
            raise KeyboardInterrupt('Tweede Ctrl+C: stop direct; alleen volledige checkpoints zijn hervatbaar.')
        soft_stop['requested'] = True
        log('STOP gevraagd: huidige epoch wordt volledig afgewerkt en opgeslagen. Tweede Ctrl+C stopt direct.')
    if install_signal_handler:
        old_handler = signal.signal(signal.SIGINT, handler)
    try:
        for epoch in range(done + 1, stop_after + 1):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            lr = float(optimizer.param_groups[0]['lr'])
            scale_lr = float(optimizer.param_groups[1]['lr'])
            expected_lr = c['lr']
            for milestone in c['scheduler_milestones']:
                if milestone < epoch:
                    expected_lr *= c['scheduler_gamma']
            s.require(lr == expected_lr, f'LR-schema wijkt af in epoch {epoch}: {lr} vs {expected_lr}')
            s.require(scale_lr == 0.0, 'De bevroren schaal-LR moet nul blijven.')
            freeze.verify_frozen(model,optimizer)
            log(f'EPOCH {epoch}/{stop_after}; MODE={c["mode"]}; LR={lr:g}; oorspronkelijk schema={c["reference_end_epoch"]}')
            before_weights = t.layer_snapshot(model, c['mode'])
            train_rows, train_s, ast_hash, grad_audit = t.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['train'], epoch, 'train', lr, out / 'batch_metrics', log, c['mode'])
            steps += expected_steps
            t.validate_gradients(model)
            update_rows = t.layer_updates(before_weights, t.layer_snapshot(model, c['mode']), c['mode'], epoch)
            if not any(row['master_changed_values'] > 0 for row in update_rows):
                log('DIAGNOSE: geen gewijzigde MX-masterwaarden in deze epoch; eindige gradients en Adamstappen worden afzonderlijk gecontroleerd.')
            adam_check = t.validate_adam_state(model, optimizer, steps)
            s.legacy.finite_state(model)
            state_before_eval = s.legacy.state_digest(model.state_dict())
            test_rows, test_s, test_ast_hash, test_audit = t.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['test'], epoch, 'test', lr, out / 'batch_metrics', log, c['mode'])
            s.require(ast_hash == test_ast_hash, 'Train- en evaluatiefunctie zijn niet identiek.')
            s.require(s.legacy.state_digest(model.state_dict()) == state_before_eval,
                      'Evaluatie heeft modelgewichten of BatchNorm-statistieken gewijzigd.')
            coverage = {name: loader.validate(out / 'sample_order' / f'{name}_{epoch:05d}.npy')
                        for name, loader in loaders.items()}
            for rows, expected in [(train_rows, c['train_examples']), (test_rows, c['test_examples'])]:
                s.require(all(r['n'] == expected for r in rows), 'Metric bevat niet de volledige dataset.')
            bn_check = t.validate_bn(model, steps)
            scheduler.step()  # EXACT original position: after train and test.
            s.require(scheduler.last_epoch == epoch, 'Scheduler-epoch wijkt af.')
            checkpoint = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
            record = {
                'epoch': epoch, 'train': train_rows, 'test': test_rows,
                'lr_used': lr, 'lr_next': float(optimizer.param_groups[0]['lr']),
                'scale_lr_used': scale_lr, 'scale_lr_next': float(optimizer.param_groups[1]['lr']),
                'scale_state': t.scale_state(model), 'source_scale_state': capture_source_scale_state(model),
                'scale_epoch_metrics': grad_audit['scale_epoch_metrics'],
                'freeze_checks': freeze.verify_frozen(model,optimizer),
                'optimizer_steps': steps, 'coverage': coverage,
                'batchnorm_counters_correct': bn_check['all_counters_correct'],
                'gradient_checks': grad_audit, 'adam_checks': adam_check, 'weight_updates': update_rows,
                'test_optimizer_steps': test_audit['optimizer_steps_in_phase'],
                'test_source_checks': test_audit['shared_source_checks'],
                'test_prediction_counts': test_audit['prediction_counts'],
                'test_label_counts': test_audit['label_counts'],
                'model_sha256': state_before_eval, 'checkpoint_path': str(checkpoint.resolve()),
                'forward_ast_sha256': ast_hash, 'train_seconds': train_s, 'test_seconds': test_s,
                'first_epoch_in_process': epoch == first_epoch,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            }
            trace['epochs'].append(record)
            best = t.best_record(trace['epochs'])
            record['is_best_at_this_epoch'] = best['epoch'] == epoch
            record['best_epoch_so_far'] = best['epoch']
            t_save = time.perf_counter()
            payload = core.capture_checkpoint(model, optimizer, scheduler, {}, device, epoch, steps,
                                              ctx.contract, trace, {})
            # Empty validation_logits is intentional: scores are in trace; there
            # is NO reason to copy all test logits into every training checkpoint.
            payload['best_prec1'] = best['test'][-1]['top1_percent']
            core.atomic_save(checkpoint, payload)
            record['checkpoint_seconds'] = time.perf_counter() - t_save
            record['total_epoch_seconds'] = time.perf_counter() - t0
            record['other_seconds'] = record['total_epoch_seconds'] - train_s - test_s - record['checkpoint_seconds']
            epoch_file = out / 'epoch_summaries' / f'epoch_{epoch:05d}.json'
            s.write_json(epoch_file, record)
            epoch_file.with_suffix('.json.sha256').write_text(core.file_sha256(epoch_file) + '\n', encoding='ascii')
            ctx.report['completed_epoch'] = epoch
            log(f'KLAAR epoch {epoch}: train={train_s:.1f}s; test={test_s:.1f}s; '
                f'totaal={record["total_epoch_seconds"]:.1f}s; beste epoch={best["epoch"]}')
            log('  TEST ' + ' | '.join(f'{r["weight_format"]}/A32: '
                                    f'{r["top1_percent"]:.4f}%' for r in test_rows))
            for bit, counts in record['test_prediction_counts'].items():
                if max(counts) == sum(counts):
                    log(f'  DIAGNOSE {a.FORMATS[int(bit)]}: alle testbeelden krijgen klasse {counts.index(max(counts))}; meetresultaat, geen technische FAIL.')
            scale_rows = record['scale_epoch_metrics']
            flips = sum(row['exponent_flips_during_epoch'] for row in scale_rows)
            changed = sum(row['phi_changed_blocks'] for row in scale_rows)
            minimum = min(row['e2_exponent_min'] for row in scale_rows)
            maximum = max(row['e2_exponent_max'] for row in scale_rows)
            rel_min = min(row['observed_relative_offset_min'] for row in scale_rows)
            rel_max = max(row['observed_relative_offset_max'] for row in scale_rows)
            log(f'  E2-SCHALEN: absolute exponenten {minimum}..{maximum}; '
                f'gewijzigde FP32-hulpwaarden={changed}; omschakelingen in epoch={flips}; '
                f'geobserveerd verschil met actuele E4-exponent={rel_min}..{rel_max}.')
            log(f'  FREEZEPILOT: {epoch-50}/5 nieuwe epochs voltooid; E2-schalen en hun Adamstaat exact behouden.')
            if soft_stop['requested'] or (out / 'STOP_AFTER_EPOCH').exists():
                log('Gecontroleerd gestopt op epochgrens; brondata en checkpoints blijven behouden.')
                break
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    if resume is not None:
        s.require(core.file_sha256(resume) == resume_hash, 'Invoercheckpoint is gewijzigd.')
    freeze.verify_prefix(trace,ctx.parent_reference)
    s.require(core.file_sha256(ctx.parent_path)==ctx.parent_reference['meta']['parent_checkpoint_sha256'],
              'Oudercheckpoint is gewijzigd.')
    return trace


def write_scale_results(ctx, out, trace):
    """Validate absolute exponents, then report relative offsets only as diagnostics."""
    import torch
    records = trace['epochs']
    c_initial = int(ctx.c['initial_relative_offset'])
    rows = [row for record in records for row in record['scale_epoch_metrics']]
    block_rows = []
    states = [(0, trace['initial_scale_state'], trace['initial_source_scale_state'])]
    states += [(r['epoch'], r['scale_state'], r['source_scale_state']) for r in records]
    def exponents(value):
        # Exact FP32 values are representable in FP64, including either side of a half-tie.
        lower = torch.floor(value)
        return (lower + (value - lower >= .5)).to(torch.int64)
    previous = previous_source = None
    source_change_rows = []
    for epoch, state, source_state in states:
        s.require(tuple(state) == a.TARGET_LAYERS and tuple(source_state) == a.TARGET_LAYERS,
                  'Schaalstaat mist lagen of verandert volgorde.')
        for name in a.TARGET_LAYERS:
            value = torch.tensor(state[name], dtype=torch.float64)
            source_code = torch.tensor(source_state[name], dtype=torch.int64)
            expected_shape = a.block_shape(tuple(ctx.model.get_submodule(name).weight.shape))
            s.require(tuple(value.shape) == tuple(source_code.shape) == expected_shape
                      and bool(torch.isfinite(value).all())
                      and bool(((value >= a.PHI_MIN) & (value <= a.PHI_MAX)).all())
                      and bool(((source_code >= 0) & (source_code <= 254)).all()),
                      'Ongeldige opgeslagen absolute schaalstaat: ' + name)
            lower_exp = exponents(value)
            source_exp = source_code - 127
            relative = lower_exp - source_exp
            if epoch == 0 and not ctx.contract['synthetic_fixture_only']:
                s.require(bool((value == source_exp + c_initial).all()),
                          'Nieuwe run moet eenmaal beginnen bij de aanvankelijke E4-schaal x64.')
            for channel, values in enumerate(state[name]):
                for index, phi in enumerate(values):
                    lower = int(lower_exp[channel, index])
                    source = int(source_exp[channel, index])
                    delta = lower - source
                    block_rows.append({'epoch': epoch, 'layer': name, 'output_channel': channel,
                        'block_index': index, 'phi': phi, 'e2_exponent': lower,
                        'e2_scale_code': lower + 127, 'source_exponent': source,
                        'source_scale_code': source + 127, 'relative_offset': delta,
                        'relative_factor': math.ldexp(1.0, delta)})
            if epoch:
                layer_rows = [r for r in records[epoch-1]['scale_epoch_metrics'] if r['layer'] == name]
                s.require(len(layer_rows) == 1, 'Schaaldiagnostiek mist of dupliceert een laag.')
                row = layer_rows[0]
                before = torch.tensor(previous[name], dtype=torch.float64)
                before_exp = exponents(before)
                before_source = torch.tensor(previous_source[name], dtype=torch.int64) - 127
                counts = {str(k): int((lower_exp == k).sum()) for k in range(-127, 128)}
                relative_counts = {str(int(k)): int((relative == k).sum()) for k in torch.unique(relative)}
                net = int((lower_exp != before_exp).sum())
                s.require(row['epoch'] == epoch and row['blocks'] == value.numel()
                          and sum(counts.values()) == value.numel()
                          and row['e2_exponent_counts'] == counts
                          and row['observed_relative_offset_counts'] == relative_counts
                          and row['phi_changed_blocks'] == int((value != before).sum())
                          and row['net_exponent_changes'] == net
                          and net <= row['unique_blocks_switched_during_epoch'] <= value.numel()
                          and row['exponent_flips_during_epoch'] >= row['unique_blocks_switched_during_epoch'],
                          'Schaaldiagnostiek strookt niet met opgeslagen absolute blokstaat.')
                expected = {'phi_min': float(value.min()), 'phi_max': float(value.max()),
                            'phi_mean': float(value.mean()), 'phi_std': float(value.std(unbiased=False)),
                            'phi_update_l2': float((value - before).norm()),
                            'e2_exponent_min': int(lower_exp.min()), 'e2_exponent_max': int(lower_exp.max()),
                            'e2_exponent_mean': float(lower_exp.to(torch.float64).mean()),
                            'observed_relative_offset_min': int(relative.min()),
                            'observed_relative_offset_max': int(relative.max()),
                            'observed_relative_offset_mean': float(relative.to(torch.float64).mean())}
                s.require(all(math.isclose(row[key], val, rel_tol=1e-12, abs_tol=1e-12)
                              for key, val in expected.items()), 'Onjuiste phi/exponentstatistieken.')
                s.require(0 <= row['scale_grad_l2_min'] <= row['scale_grad_l2_max'] < float('inf')
                          and 0 <= row['scale_grad_nonzero_blocks_max'] <= value.numel()
                          and row['projected_values_during_epoch'] >= 0,
                          'Niet-eindige of ongeldige schaalgradientdiagnostiek.')
                s.require(row['phi_at_min_blocks'] == int((value == a.PHI_MIN).sum())
                          and row['phi_at_max_blocks'] == int((value == a.PHI_MAX).sum()),
                          'Onjuiste telling aan de absolute phi-grenzen.')
                s.require(row['exponent_jump_distance_during_epoch'] >= row['exponent_flips_during_epoch']
                          and row['exponent_jump_distance_during_epoch'] >= int((lower_exp - before_exp).abs().sum())
                          and 0 <= row['max_exponent_jump'] <= 254,
                          'Onjuiste spronggrootte van absolute exponenten.')
                source_change_rows.append({'epoch': epoch, 'layer': name, 'blocks': value.numel(),
                    'source_exponents_changed_since_previous_epoch': int((source_exp != before_source).sum()),
                    'e2_exponents_changed_since_previous_epoch': net,
                    'observed_relative_offsets_changed_since_previous_epoch':
                        int((relative != before_exp - before_source).sum())})
        previous, previous_source = state, source_state
    # Count dictionaries are JSON within CSV, not implementation-dependent Python repr.
    csv_rows = [{key: json.dumps(val, sort_keys=True) if isinstance(val, dict) else val
                 for key, val in {**row, 'scale_gradients_enabled': row.get('scale_gradients_enabled', True)}.items()} for row in rows]
    s.write_csv(out / 'scale_epoch_metrics.csv', csv_rows)
    s.write_csv(out / 'scale_block_state.csv', block_rows)
    s.write_csv(out / 'scale_source_changes.csv', source_change_rows)
    best, last = t.best_record(records), records[-1]
    selected = {}
    for role, record in [('best_E4M3_test', best), ('last_epoch', last)]:
        group = [r for r in block_rows if r['epoch'] == record['epoch']]
        e2_counts, relative_counts = {}, {}
        for block in group:
            e, d = str(block['e2_exponent']), str(block['relative_offset'])
            e2_counts[e] = e2_counts.get(e, 0) + 1
            relative_counts[d] = relative_counts.get(d, 0) + 1
        selected[role] = {'epoch': record['epoch'], 'blocks': len(group),
                         'e2_exponent_counts': e2_counts,
                         'observed_relative_offset_counts': relative_counts,
                         'relative_offsets_are_diagnostics_not_choices': True}
    scale_bytes = sum(math.prod(a.block_shape(tuple(ctx.model.get_submodule(n).weight.shape)))
                      for n in a.TARGET_LAYERS)
    result = {'schema': 'mx-triple-independent-e2-scale-freeze50-pilot5-scale-learning-v1',
              'learned_scale_parameters': ctx.c['learned_scale_parameters'],
              'initialization': ctx.c['scale_proxy_initialization'],
              'initial_relative_offset': 6, 'initial_relative_factor': 64,
              'tie_break': 'higher_exponent', 'phi_bounds': [a.PHI_MIN, a.PHI_MAX],
              'relative_factor_constrained_after_initialization': False,
              'scale_gradient': ctx.c['scale_gradient'], 'gradient_normalization': 'none',
              'scale_lr': ctx.c['scale_lr'], 'frozen_scale_lr':0.0,
              'projection_after_every_Adam_update': False, 'scale_gradients_enabled_after50':False,
              'scale_learning_end_epoch': ctx.c['scale_learning_end_epoch'],
              'scale_freeze_epoch': ctx.c['scale_freeze_epoch'],
              'completed_optimizer_updates': last['optimizer_steps'],
              'phi_changed_in_any_epoch': any(r['phi_changed_blocks'] > 0 for r in rows),
              'nonzero_scale_gradient_observed': any(r['scale_grad_nonzero_blocks_max'] > 0 for r in rows),
              'total_exponent_flips_including_repeat_flips': sum(r['exponent_flips_during_epoch'] for r in rows),
              'total_exponent_jump_distance': sum(r['exponent_jump_distance_during_epoch'] for r in rows),
              'max_exponent_jump': max(r['max_exponent_jump'] for r in rows),
              'total_projected_values_including_repeats': sum(r['projected_values_during_epoch'] for r in rows),
              'hard_switch_required_for_technical_PASS': False,
              'zero_finite_scale_gradients_are_valid_diagnostics': True,
              'e4_scale_bytes': scale_bytes, 'independent_e2_scale_bytes': scale_bytes,
              'combined_scale_bytes': 2 * scale_bytes,
              'extra_export_bytes_vs_fixed_x64': scale_bytes,
              'extra_export_bytes_vs_fourbit_relative10': scale_bytes - (scale_bytes // 2),
              'analytical_FP32_phi_gradient_Adam_bytes': ctx.c['learned_scale_parameters'] * 16,
              'analytical_bytes_exclude_framework_and_temporaries': True,
              'selected': selected, 'synthetic_fixture_only': ctx.contract['synthetic_fixture_only'],
              'branch_exponent_flips':sum(r['exponent_flips_during_epoch'] for r in rows if r['epoch']>50),
              'branch_phi_changed_blocks':sum(r['phi_changed_blocks'] for r in rows if r['epoch']>50),
              'freeze_checks':freeze.verify_frozen(ctx.model,ctx.optimizer)}
    s.write_json(out / 'scale_learning_summary.json', result)
    ctx.report['scale_learning_summary'] = result
    return result


def validate_history_export(path: Path, expected_trace: dict, core) -> dict:
    """Read back the complete saved trace before an export can succeed.

    This is an export-integrity check only; it changes no training computation.
    Read once so the digest describes exactly the bytes whose contents matched.
    """
    try:
        raw = path.read_bytes()
        restored_trace = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            'Opgeslagen history.json is onleesbaar of onvolledig; export wordt niet '
            'als geslaagd gerapporteerd: ' + str(path) + ' (' + str(exc) + ')') from exc
    mismatch = core.differences(expected_trace, restored_trace)
    s.require(not mismatch,
              'Opgeslagen history.json wijkt af van de volledige epochgeschiedenis: '
              + ', '.join(mismatch[:8]))
    return {'file': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'epochs': len(restored_trace['epochs']), 'exact_trace_match': True}


def export_results(ctx, out: Path, trace: dict, log):
    import torch
    records, c, core = trace['epochs'], ctx.c, ctx.rt.core
    s.require(records and [r['epoch'] for r in records] == list(range(1, records[-1]['epoch'] + 1)),
              'Onvolledige epochgeschiedenis.')
    expected_steps = math.ceil(c['train_examples'] / c['batch_size'])
    s.require(c['planned_epochs'] == c['reference_end_epoch'] == 100
              and c['scale_learning_end_epoch'] == c['scale_freeze_epoch'] == 50,
              'Onjuist freeze50-contract.')
    s.require(50 < records[-1]['epoch'] <= 55,'Deze pilot stopt uiterlijk op epoch55.')
    freeze.verify_prefix(trace,ctx.parent_reference)
    for record in records:
        epoch = record['epoch']
        s.require(record['optimizer_steps'] == expected_steps * epoch, 'Verkeerd aantal updates.')
        expected_lr = c['lr']
        for milestone in c['scheduler_milestones']:
            if milestone < epoch: expected_lr *= c['scheduler_gamma']
        s.require(record['lr_used'] == expected_lr, 'Onjuist LR-schema.')
        expected_next = c['lr']
        for milestone in c['scheduler_milestones']:
            if milestone <= epoch: expected_next *= c['scheduler_gamma']
        s.require(record['lr_next'] == expected_next, 'Onjuist LR voor de volgende epoch.')
        s.require(record['batchnorm_counters_correct'], 'BN-tellers niet correct.')
        s.require(record['test_optimizer_steps'] == 0, 'Evaluatie deed optimizerupdates.')
        s.require(record['adam_checks']['all_active_adam_steps_correct']
                  and record['adam_checks']['moments_fp32_finite']
                  and record['adam_checks']['unused_bn_has_no_optimizer_state'],
                  'Onjuiste Adamstaat.')
        audit = record['gradient_checks']
        s.require(audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == expected_steps,
                  'Onvolledige gradientcontroles.')
        s.require(audit['detached_teacher_checks'] == 2 * expected_steps,
                  'Er moeten twee losgekoppelde teachercontroles per minibatch zijn.')
        s.require(audit['forward_calls_E4M3'] == audit['forward_calls_E2M3']
                  == audit['forward_calls_E2M1'] == expected_steps, 'Verkeerd aantal forwards.')
        s.require(audit['cache_empty_before_all_optimizer_steps'],
                  'Broncache niet leeg voor optimizerupdate.')
        if epoch <= 50:
            s.require(audit['scale_projection_after_every_update']
                      and audit['scale_optimizer_steps']==expected_steps,'Onvolledige geërfde schaalupdates.')
        else:
            s.require(audit['scale_projection_after_every_update'] is False
                      and audit['scale_optimizer_steps']==0 and audit['frozen_scale_checks']==expected_steps
                      and audit['scale_gradients_enabled'] is False
                      and all(record['freeze_checks'][k] for k in
                        ('phi_unchanged','hard_E2_scales_unchanged','scale_optimizer_state_unchanged','scale_gradients_absent')),
                      'Onvolledige of onjuiste freezecontrole.')
            s.require(record['scale_state']==records[49]['scale_state'],'Absolute phi veranderde na50.')
        s.require(len(record['scale_epoch_metrics'])==len(a.TARGET_LAYERS),'Schaaldiagnostiek ontbreekt.')
        expected_scale = c['scale_lr']
        for milestone in c['scheduler_milestones']:
            if milestone < epoch: expected_scale *= c['scheduler_gamma']
        expected_scale_next = c['scale_lr']
        for milestone in c['scheduler_milestones']:
            if milestone <= epoch: expected_scale_next *= c['scheduler_gamma']
        if epoch>50: expected_scale=expected_scale_next=0.0
        s.require(record['scale_lr_used'] == expected_scale
                  and record['scale_lr_next'] == expected_scale_next,
                  'Schaaloptimizer volgt niet het vastgelegde LR-schema.')
        for phase_name, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            source_audit = (audit['shared_source_checks'] if phase_name == 'train'
                            else record['test_source_checks'])
            batches = math.ceil(count / c['batch_size'])
            s.require(source_audit['source_triplets'] == batches
                      and source_audit['high_encodings'] == source_audit['exponent_derivations']
                      == source_audit['mantissa_derivations'] == batches * len(a.TARGET_LAYERS)
                      and source_audit['learned_absolute_scales_applied'] == 2 * batches * len(a.TARGET_LAYERS)
                      and source_audit['mantissa_scale_copies_unchanged'] == batches * len(a.TARGET_LAYERS)
                      and source_audit['e2_exponent_bounds'] == [-127, 127]
                      and source_audit['independent_absolute_lower_scale']
                      and source_audit['learned_shared_scale']
                      and source_audit['source_cache_cleared']
                      and source_audit['no_direct_lower_quantizer']
                      and source_audit['cache_cleared_before_adam'],
                      'Onvolledige gedeelde E4M3-broncontrole: ' + phase_name)
        s.require(set(record['test_prediction_counts']) == {'8', '6', '4'},
                  'Voorspellingsverdelingen ontbreken.')
        for counts in [record['test_label_counts'], *record['test_prediction_counts'].values()]:
            s.require(len(counts) == 10 and all(type(n) is int and n >= 0 for n in counts)
                      and sum(counts) == c['test_examples'], 'Onvolledige klasseverdeling.')
        for phase, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            s.require([row['weight_bits'] for row in record[phase]] == list(a.MODES), 'Onvolledige modi.')
            for row in record[phase]:
                bit = row['weight_bits']
                expected_loss = 'recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy'
                expected_teacher = a.FORMATS[8 if bit == 6 else 6] if phase == 'train' and bit != 8 else None
                s.require(row['n'] == count and row['activation_bits'] == 32 and row['bn_bank_key'] == a.BANK[bit]
                          and row['weight_format'] == a.FORMATS[bit] and row['objective_type'] == expected_loss
                          and row['teacher_weight_format'] == expected_teacher,
                          'Onjuiste modus, loss of datadekking.')
                s.require(abs(row['top1_percent'] - row['count_top1_percent']) < 1e-4,
                          'Accuracy strookt niet met correcte aantallen.')
    best, last = t.best_record(records), records[-1]
    branch_best=t.best_record(records[50:])
    selected, details, mx_exports = [], {}, {}
    for role, record in [('best_E4M3_test', best), ('last_epoch', last), ('branch_best_E4M3_test',branch_best)]:
        path = freeze.resolve_record_checkpoint(record,ctx)
        state = core.load_verified(path)
        expected_contract=ctx.parent_state['contract'] if record['epoch']<=50 else ctx.contract
        s.require(not core.differences(state['contract'], expected_contract), 'Checkpointcontract wijkt af.')
        s.require(state['completed_epoch'] == record['epoch'], 'Verkeerde checkpointepoch.')
        s.require(s.legacy.state_digest(state['model']) == record['model_sha256'], 'Modelhash wijkt af.')
        s.require(not core.differences(state['trace']['epochs'][-1]['test'], record['test']), 'Testmetrics verschillen.')
        suffix = {'best_E4M3_test':'best','last_epoch':'last','branch_best_E4M3_test':'branch_best'}[role]
        filename = suffix + '_checkpoint.pt'
        target = out / filename
        with path.open('rb') as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
        digest = core.file_sha256(target)
        s.require(digest == core.file_sha256(path), 'Checkpointkopie wijkt af.')
        target.with_suffix('.pt.sha256').write_text(digest + '\n', encoding='ascii')
        details[role] = {'epoch': record['epoch'], 'file': filename, 'sha256': digest,
                         'original_checkpoint': str(path), 'selection_score_E4M3': record['test'][-1]['top1_percent']}
        selected.extend({'role': role, **row} for row in record['test'])
        # Encode and compare on the training device: G(W) can differ after moving to CPU.
        model = copy.deepcopy(ctx.model).to(ctx.rt.device).eval()
        model.load_state_dict(state['model'], strict=True)
        exports_for_role = {}
        provenance = {
            'training': 'joint independent absolute E2 block scale; E4 source -> exponent-MSB E2M3 -> mantissa E2M1; identity master STE; absolute E2 scale frozen after50; A32',
            'absolute_E2_scale_frozen_at_this_checkpoint':record['epoch']>50,
            'source_epoch': record['epoch'], 'checkpoint_sha256': digest,
            'encoding_device': str(ctx.rt.device),
            'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
        source_payload, all_layer_rows = a.source_payload(model, provenance)
        source_file = out / f'mx_source_MXFP8_E4M3_{suffix}.pt'
        source_details = core.atomic_save(source_file, source_payload)
        # Derivation starts from the verified saved source. The compact source
        # has all BN/FP32 auxiliary state but no masters for the six MX layers.
        loaded_source = a.load_verified_payload(source_file)
        a.validate_source(loaded_source)
        source_digest = core.file_sha256(source_file)
        all_equal_scales = all(torch.equal(loaded_source['lower_scale_codes'][name].cpu(),
                                           loaded_source['mx_layers'][name]['scale_codes'].cpu())
                               for name in a.TARGET_LAYERS)
        for bit in (8, 6, 4):
            f = a.codec(bit)
            if bit == 8:
                mx_file, mx_details, loaded = source_file, source_details, loaded_source
            else:
                payload = a.derive_payload(loaded_source, bit, source_digest)
                mx_file = out / f'mx_derived_{a.FORMATS[bit]}_{suffix}.pt'
                mx_details = core.atomic_save(mx_file, payload)
                loaded = a.load_verified_payload(mx_file)
            a.select_mode(model, bit)
            for name in a.TARGET_LAYERS:
                encoded = f.MXTensor.from_payload(loaded['mx_layers'][name], ctx.rt.device)
                restored = encoded.decode()
                layer = model.get_submodule(name)
                with torch.no_grad():
                    live = layer.quantize_fn(layer.weight.detach())
                s.require(torch.equal(restored, live), 'Export verschilt van dezelfde trainingsroute.')
                source_scale = loaded_source['mx_layers'][name]['scale_codes']
                expected_scale = source_scale if bit == 8 else loaded_source['lower_scale_codes'][name]
                s.require(expected_scale.dtype == torch.uint8 and bool((expected_scale <= 254).all())
                          and torch.equal(encoded.scale_codes.cpu(), expected_scale.cpu()),
                          'Export volgt niet de opgeslagen absolute schaalbank voor zijn exponentbreedte.')
                s.require(name + '.weight' not in loaded['auxiliary_fp32_state'], 'FP32-master in export.')
            exports_for_role[str(bit)] = {**mx_details, 'epoch': record['epoch'], 'weight_bits': bit,
                                         'decoded_weights_exact': True, 'six_fp32_masters_absent': True,
                                         'full_test_forward_rerun': False, 'truncation': bit != 8,
                                         'source_weight_bits': 8, 'is_source_export': bit == 8,
                                         'derived_from_saved_source': bit != 8,
                                         'source_file_sha256': source_digest,
                                         'source_scale_bytes_copied': bit == 8 or all_equal_scales,
                                         'independent_absolute_lower_scales_in_source': True,
                                         'independent_e2_scale_bytes': sum(x.numel() for x in loaded_source['lower_scale_codes'].values()),
                                         'e2_scale_format': 'E8M0',
                                         'relative_factor_constrained': False,
                                         'fp32_scale_proxies_absent': True,
                                         'learned_shared_scale': bit != 8,
                                         'master_weights_read_to_derive': False,
                                         'master_weights_read_to_encode_source': bit == 8,
                                         'saved_and_reloaded': True,
                                         'encoding_and_comparison_device': str(ctx.rt.device),
                                         'export_validation_scope': 'exact weights versus live shared-source route on training device; no full test rerun'}
            layer_rows = [row for row in all_layer_rows if row['weight_bits'] == bit]
            s.require(len(layer_rows) == len(a.TARGET_LAYERS), 'Onvolledige exportdiagnostiek.')
            s.write_csv(out / f'mx_export_layers_{bit}_{suffix}.csv', layer_rows)
        s.require(core.file_sha256(source_file) == source_digest, 'Opgeslagen E4M3-bron gewijzigd tijdens afleiding.')
        mx_exports[role] = exports_for_role
    s.write_csv(out / 'selected_metrics.csv', selected)
    s.write_csv(out / 'epoch_metrics.csv', [row for r in records for phase in ('train', 'test') for row in r[phase]])
    s.write_csv(out / 'weight_updates.csv', [row for r in records for row in r['weight_updates']])
    selection = {'policy': c['selection_policy'], 'best': details['best_E4M3_test'], 'last': details['last_epoch'],
                 'branch_best':details['branch_best_E4M3_test'],
                 'branch_selection_window':[51,last['epoch']], 'global_selection_window':[1,last['epoch']],
                 'same_checkpoint': best['epoch'] == last['epoch'], 'selected_on_official_test': True,
                 'independent_test_estimate': False, 'selection_mode': 'MXFP8_E4M3/A32',
                 'all_modes_from_same_selected_checkpoint': True}
    s.write_json(out / 'checkpoint_selection.json', selection)
    s.write_json(out / 'history.json', trace)
    prediction_rows = []
    for record in records:
        for bit in a.MODES:
            counts = record['test_prediction_counts'][str(bit)]
            prediction_rows.append({'epoch': record['epoch'], 'weight_bits': bit,
                'weight_format': a.FORMATS[bit], 'n': sum(counts),
                **{f'predicted_class_{k}': count for k, count in enumerate(counts)},
                'classes_predicted': sum(n > 0 for n in counts),
                'dominant_predicted_class': counts.index(max(counts)),
                'dominant_prediction_fraction': max(counts) / sum(counts)})
    s.write_csv(out / 'test_prediction_counts.csv', prediction_rows)
    scale_summary = write_scale_results(ctx, out, trace)
    elapsed = sum(r['total_epoch_seconds'] for r in records)
    pilot_completed = last['epoch']==55
    full_completed = False
    branch_records=records[50:]
    branch_elapsed=sum(r['total_epoch_seconds'] for r in branch_records)
    epoch50_checkpoint = {'available': False, 'epoch': 50,
        'reason': 'Epoch50 is not yet present in this run history.'}
    if last['epoch'] >= 50:
        record50 = records[49]
        path50 = ctx.parent_path
        s.require(record50['epoch'] == 50 and path50.is_file(),
                  'Het volledige checkpoint van epoch50 ontbreekt.')
        digest50 = core.file_sha256(path50)
        marker50 = path50.with_suffix('.pt.sha256')
        s.require(marker50.is_file() and marker50.read_text(encoding='ascii').strip() == digest50,
                  'Checksum van het bewaarde epoch50-checkpoint wijkt af.')
        epoch50_checkpoint = {'available': True, 'epoch': 50, 'original_checkpoint': str(path50),
            'sha256': digest50, 'optimizer_steps': record50['optimizer_steps'],
            'scale_lr_used': record50['scale_lr_used'], 'scale_lr_next': record50['scale_lr_next'],
            'full_training_state': True, 'checkpoint_in_feedback_bundle': False,
            'state_includes': ['FP32 weights', 'FP32 absolute scale proxies phi', 'both Adam groups',
                              'scheduler', 'Python/NumPy/Torch CPU/CUDA RNG', 'full epoch history'],
            'purpose': 'Read-only branch origin; absolute E2 scale frozen from epoch51.'}
    s.write_json(out / 'epoch50_checkpoint.json', epoch50_checkpoint)
    summary = {'schema': 'mx-triple-independent-e2-scale-freeze50-pilot5-completion-v1',
               'planned_epochs': c['planned_epochs'], 'reference_end_epoch': c['reference_end_epoch'],
               'completed_epochs': last['epoch'], 'inherited_epochs':50,
               'branch_completed_epochs':last['epoch']-50,'pilot_stop_epoch':55,
               'parent_checkpoint':trace['parent_checkpoint'],'freeze_transition':trace['freeze_transition'],
               'freeze_checks':freeze.verify_frozen(ctx.model,ctx.optimizer),
               'branch_best_epoch_E4M3_A32':branch_best['epoch'],
               'pilot_completed': pilot_completed,
               'full_training_completed': full_completed,
               'full_100_epoch_training_completed': full_completed,
               'optimizer_steps': last['optimizer_steps'],
               'expected_optimizer_steps_at_reference_completion': expected_steps * c['reference_end_epoch'],
               'expected_training_forward_passes_at_reference_completion': expected_steps * c['reference_end_epoch'] * 3,
               'best_epoch_E4M3_A32': best['epoch'], 'last_epoch': last['epoch'],
               'lr_schedule_verified': True, 'all_epoch_modes_complete': True,
               'resumed': 'resumed_from' in ctx.report,
               'new_initialization_not_previous_experiment_warmstart': False,
               'restored_complete_epoch50_state':True,
               'epoch50_checkpoint': epoch50_checkpoint,
               'training_forward_passes': last['optimizer_steps'] * 3,
               'train_examples': c['train_examples'], 'test_examples': c['test_examples'],
               'mode': c['mode'], 'weight_formats': c['weight_formats'], 'activation_bits': 32,
               'distillation': True, 'truncation': True, 'source_weight_bits': 8,
               'exponent_truncation': True, 'mantissa_truncation': True,
               'independent_E2_shared_scale': True, 'learned_shared_scale': True,
               'scale_parameterization': c['scale_parameterization'],
               'scale_proxy_initialization': c['scale_proxy_initialization'],
               'initial_relative_factor': 64, 'scale_proxy_bounds': [-127.0, 127.0],
               'relative_factor_constrained_after_initialization': False,
               'learned_scale_parameters': c['learned_scale_parameters'],
               'scale_gradient': c['scale_gradient'], 'scale_gradient_normalization': 'none',
               'scale_projection_after_every_update': False, 'scale_lr': c['scale_lr'],
               'frozen_scale_lr':0.0,'scale_gradients_enabled_after50':False,
               'scale_learning_end_epoch': c['scale_learning_end_epoch'],
               'scale_freeze_epoch': c['scale_freeze_epoch'],
               'scale_learning_summary': scale_summary,
               'quantization_source': c['quantization_source'],
               'initial_model_sha256': trace['initial_model_sha256'],
               'initial_base_model_sha256': trace['initial_base_model_sha256'],
               'measured_completed_epoch_seconds': elapsed,
               'measured_new_branch_epoch_seconds':branch_elapsed,
               'new_branch_optimizer_steps':(last['epoch']-50)*expected_steps,
               'gradient_checks_before_every_update': True, 'fp32_masters_and_moments_verified': True,
               'one_E4_source_per_layer_per_minibatch_verified': True,
               'lower_scale_codes_depend_only_on_phi_after_initialization': True,
               'e4_source_scale_unchanged_policy': True,
               'all_mantissa_scale_bytes_equal_E2M3': True,
               'source_cache_cleared_before_every_optimizer_update': True,
               'batchnorm_learned_during_training_eval_frozen': True,
               'synthetic_fixture_only': ctx.contract['synthetic_fixture_only']}
    s.write_json(out / 'training_summary.json', summary)
    import statistics
    steady=[r['total_epoch_seconds'] for r in branch_records if not r['first_epoch_in_process']]
    typical=statistics.median(steady or [r['total_epoch_seconds'] for r in branch_records])
    estimate={'available':True,'pilot_completed':pilot_completed,'full_training_completed':False,
              'full_100_epoch_training_completed':False,'inherited_epochs':50,
              'new_measured_epochs':[r['epoch'] for r in branch_records],
              'measured_new_branch_epoch_seconds':branch_elapsed,
              'measured_inherited_epoch_seconds':elapsed-branch_elapsed,
              'measured_completed_epoch_seconds':elapsed,'measured_100_epoch_seconds':None,
              'median_branch_epoch_seconds':typical,'remaining_epochs_to100':100-last['epoch'],
              'estimated_remaining_to100_seconds':typical*(100-last['epoch']),
              'interpretation':'Only epochs51..last were executed by the freeze branch; remaining time to100 is a projection. This pilot never executes beyond55.',
              'limitations':['Projection is not measured future duration; excludes loading and final exports.',
                             'Inherited epochs1..50 are existing evidence, not new computation.']}
    s.write_json(out / 'timing_estimate.json', estimate)
    s.write_csv(out / 'epoch_times.csv', [
        {'epoch': r['epoch'], **{k: r.get(k) for k in ['train_seconds', 'test_seconds', 'checkpoint_seconds',
          'other_seconds', 'total_epoch_seconds', 'first_epoch_in_process',
          'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']}} for r in records])
    ctx.report.update(completed_epoch=last['epoch'], optimizer_steps=last['optimizer_steps'],
                      selected_best_epoch=best['epoch'], selected_last_epoch=last['epoch'],
                      initial_model_sha256=trace['initial_model_sha256'],
                      initial_base_model_sha256=trace['initial_base_model_sha256'], checkpoint_selection=selection,
                      training_summary=summary, timing_estimate=estimate, mx_exports=mx_exports,
                      synthetic_fixture_only=ctx.contract['synthetic_fixture_only'],
                      full_training_completed=full_completed, pilot_completed=pilot_completed,
                      full_100_epoch_training_completed=full_completed,
                      epoch50_checkpoint=epoch50_checkpoint,
                      inherited_epochs=50, branch_completed_epochs=last['epoch']-50,
                      freeze_checks=freeze.verify_frozen(ctx.model,ctx.optimizer),
                      selected_branch_best_epoch=branch_best['epoch'],
                      training_forward_passes=last['optimizer_steps'] * 3)
    write_freeze_comparison(ctx, out, trace)
    write_notebook(out)
    archive_initialization_reference(out)
    ctx.report['history_export_validation'] = validate_history_export(
        out / 'history.json', trace, core)
    log(f'BEHOUDEN: alle drie modi uit beste E4M3-checkpoint {best["epoch"]}; laatste {last["epoch"]}.')
    log(f'FREEZE50-PILOT: {last["epoch"]-50}/5 nieuwe epochs; laatsteepoch{last["epoch"]}; '
        f'nieuwe gemeten epochtijd={branch_elapsed/60:.2f} min; absolute E2-schalen vast.')


def verify_unchanged(ctx, log):
    for name, expected in ctx.source_info['workcopy_files_sha256'].items():
        s.require(ctx.rt.core.file_sha256(ctx.source / name) == expected, 'Bronbestand gewijzigd: ' + name)
    s.verify_package()
    for name, expected in ctx.report['dataset_files']['files'].items():
        actual = s.hashes(Path(ctx.report['dataset_files']['root']) / name)
        s.require(all(actual[key] == expected[key] for key in actual), 'Ruwe dataset gewijzigd: ' + name)
    ctx.report.update(source_unchanged=True, dataset_files_unchanged=True)
    s.require(ctx.rt.core.file_sha256(ctx.parent_path)==ctx.parent_reference['meta']['parent_checkpoint_sha256'],
              'Oudercheckpoint gewijzigd.')
    ctx.report['parent_checkpoint_unchanged']=True
    log('Eindcontrole: bronbestanden, ruwe data en oudercheckpoint ongewijzigd.')


def feedback_bundle(out):
    names = ['run_report.json', 'epoch_metrics.csv', 'selected_metrics.csv', 'weight_updates.csv',
             'checkpoint_selection.json', 'timing_estimate.json', 'epoch_times.csv', 'history.json',
             'dataset_manifest.json', 'source_manifest.json', 'compatibility.patch',
             'pip_freeze.txt', 'pip_check.txt', 'run.log', 'warnings.log',
             'training_change_manifest.json', 'training_summary.json', 'epoch50_checkpoint.json',
             'scale_parameterization_reference_report.json', 'scale_parameterization_comparison.csv',
             'test_prediction_counts.csv', 'scale_epoch_metrics.csv', 'scale_block_state.csv', 'scale_learning_summary.json',
             'scale_source_changes.csv', 'freeze_check_report.json',
             'freeze_comparison_report.json','freeze_comparison.csv',
             'codec_report.json', 'trunc_training_check_report.json',
             'MX_triple_independent_e2_scale_freeze50_pilot5_resultaten.ipynb', 'notebook_support.py']
    names += [f'mx_export_layers_{bit}_{role}.csv' for bit in a.MODES for role in ('best', 'last','branch_best')]
    for reference_directory in REFERENCE_DIRECTORIES:
        if (out / reference_directory).is_dir():
            names += [str(path.relative_to(out)).replace('\\', '/')
                      for path in sorted((out / reference_directory).rglob('*')) if path.is_file()]
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as z:
        added = set()
        for name in names:
            path = out / name
            if path.is_file():
                z.write(path, name)
                added.add(name)
        # Keep the analysis notebook available even when check/run fails early.
        for name in ['MX_triple_independent_e2_scale_freeze50_pilot5_resultaten.ipynb', 'notebook_support.py']:
            if name not in added:
                z.write(ROOT / name, name)
                added.add(name)
        for reference_directory in REFERENCE_DIRECTORIES:
            if not (out / reference_directory).is_dir():
                for path in sorted((ROOT / reference_directory).rglob('*')):
                    if path.is_file():
                        name = str(path.relative_to(ROOT)).replace('\\', '/')
                        z.write(path, name)
                        added.add(name)
        # Small, reviewable runner snapshot; no checkpoints, dataset or optimizer tensors.
        source_names = [p.name for p in ROOT.iterdir() if p.is_file()
                        and p.suffix in ('.py', '.json', '.md', '.ipynb')]
        source_names.append('upstream/Any-Precision-DNNs-master.zip')
        for name in sorted(source_names):
            z.write(ROOT / name, 'runner_source/' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'resume'])
    parser.set_defaults(mode='joint')
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_triple_independent_e2_scale_freeze50_pilot5'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=55)
    parser.add_argument('--parent-checkpoint', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if not 51 <= args.stop_after_epoch <= 55:
        parser.error('Deze freeze50-pilot kan alleen stoppen op epoch51..55.')
    if (args.command == 'resume') != (args.checkpoint is not None):
        parser.error('Alleen resume vereist --checkpoint uit deze freezevertakking; run herstelt de ouder op50.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (args.output_root / (stamp + '_' + args.mode + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-triple-independent-e2-scale-freeze50-pilot5-report-v1', 'status': 'RUNNING',
              'package_revision': 'mx_triple_independent_e2_scale_freeze50_pilot5_v1',
              'command': args.command, 'mode': args.mode, 'started_at': s.now(),
              'output_directory': str(out), 'argv': sys.argv, 'warnings': [],
              'limitations': ['train only; shared E4M3 source and two code-derived MX modes; this is not a paper reproduction',
                              'FP32 update master/gradients/Adam; A32; exponent MSB and mantissa truncation; both E2 modes share an independently learned absolute E8M0 scale per block',
                              'three forward/backwards: E4M3 hard CE, E2M3/E2M1 recursive detached soft CE, one Adam step',
                              'BN banks learn during training; absolute E2 log-scale STE; E4 source scale unchanged; independent scales do not imply independent element quantization',
                              'restore pinned epoch50 parent; only epochs51..55 newly executed; absolute E2 scales and their Adam state frozen; weights/BN continue; E4 source scale remains dynamic',
                              'official test used for checkpoint selection, not an independent test estimate',
                              'numerical weight-only MX emulation; no native MX kernel or energy claim',
                              'GPU bit-exact repeat/resume not guaranteed; fixed seed is not proof of determinism',
                              'one seed and one training run; no proof of general superiority or convergence']}
    log = s.legacy.Log(out / 'run.log')
    started = time.perf_counter()
    exit_code = 1
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        entry = {'category': category.__name__, 'message': str(message), 'file': filename, 'line': lineno}
        report['warnings'].append(entry)
        log('WARNING ' + str(message))
        with (out / 'warnings.log').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.showwarning = warning_handler
            log('RESULTATENMAP: ' + str(out))
            log('FREEZE50-PILOT: epoch50 herstellen; absolute E2-schalen vast; gewichten vijf epochs verder tot55; dynamische E4-bron; B32; E4M3 -> E2M3 -> E2M1; A32; FP32-masters.')
            write_notebook(out)
            ctx = create_context(args, out, report, log, check_only=args.command == 'check')
            if ctx is None:
                report['status'] = 'PASS'
            else:
                trace = run_epochs(ctx, out, log, args.stop_after_epoch, args.checkpoint)
                export_results(ctx, out, trace, log)
                verify_unchanged(ctx, log)
                report['status'] = 'PASS' if report['pilot_completed'] else 'STOPPED_AT_EPOCH_BOUNDARY'
            exit_code = 0
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report['error'], report['traceback'] = repr(exc), traceback.format_exc()
        log(report['status'] + ': ' + str(exc))
        log(report['traceback'])
        log('Geen bestaande data of eerdere resultaten gewijzigd. Deel de feedback_bundle.zip uit deze map.')
    finally:
        report['ended_at'] = s.now()
        report['session_wall_seconds'] = time.perf_counter() - started
        log('EINDSTATUS: ' + report['status'])
        log('Rapport: ' + str(out / 'run_report.json'))
        log.handle.close()
        s.write_json(out / 'run_report.json', report)
        feedback_bundle(out)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
