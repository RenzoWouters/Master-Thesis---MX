"""Five new epochs51-55 with learned offset choices frozen at parent epoch50.

The dynamic control is the existing full100 run. No control retraining occurs.
A32, FP32 masters and the original three-pass loss/quantization code are reused.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import warnings
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
from runtime_bridge import ROOT, ref, s, t, a, core, torch
from freeze_core import (config, verify_package, load_parent_reference, verify_parent_state,
                         freeze_optimizer, initialize_branch, validate_output_location)

NOTEBOOK = 'MX_triple_freeze_offsets_e50_pilot5_resultaten.ipynb'


def run_epochs(ctx, out, log, stop_after=55, parent_checkpoint=None, evidence=None,
               resume=None, install_signal_handler=True):
    s.require(51 <= stop_after <= 55, 'Deze proef traint uitsluitend nieuwe epochs51-55.')
    s.require(parent_checkpoint is not None and evidence is not None, 'Oudercheckpoint en referentie vereist.')
    out = Path(out)
    for directory in ['checkpoints', 'epoch_summaries', 'sample_order', 'batch_metrics']:
        (out / directory).mkdir(exist_ok=True)
    package_hashes = getattr(ctx, 'branch_package_hashes', None)
    if package_hashes is None:
        package_hashes = verify_package()
    trace, done, steps = initialize_branch(ctx, parent_checkpoint, evidence, package_hashes, resume)
    s.require(done < stop_after, 'Dit checkpoint laat geen nieuwe proefepochs toe.')
    rt, c, device = ctx.rt, ctx.c, ctx.rt.device
    model, optimizer, scheduler, loaders = ctx.model, ctx.optimizer, ctx.scheduler, ctx.loaders
    expected_steps = len(loaders['train'])
    s.require(expected_steps == math.ceil(c['train_examples'] / c['batch_size']), 'Verkeerd aantal minibatches.')
    resume_hash = core.file_sha256(Path(resume)) if resume is not None else None
    log(f'VERTREK na epoch {done}; offsetkeuzes vast op epoch50; theta-LR=0; gewichts-LR={optimizer.param_groups[0]["lr"]:g}.')
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
            expected_scale_lr = c['scale_lr']
            for milestone in c['scheduler_milestones']:
                if milestone < epoch: expected_scale_lr *= c['scheduler_gamma']
            s.require(scale_lr == expected_scale_lr, 'Schaal-LR wijkt af van het vastgelegde schema.')
            log(f'EPOCH {epoch}/{stop_after}; MODE={c["mode"]}; LR={lr:g}; volledig schema={c["planned_epochs"]}')
            frozen_steps_before = ctx.guard.steps
            before_weights = t.layer_snapshot(model, c['mode'])
            train_rows, train_s, ast_hash, grad_audit = t.phase(rt, ctx.source, ctx.modules, model, optimizer,
                  loaders['train'], epoch, 'train', lr, out / 'batch_metrics', log, c['mode'])
            steps += expected_steps
            t.validate_gradients(model)
            ctx.guard.assert_unchanged()
            s.require(ctx.guard.steps - frozen_steps_before == expected_steps, 'Niet elke Adamupdate is op vaste offsets gecontroleerd.')
            scale_rows = grad_audit['scale_epoch_metrics']
            s.require(all(row['theta_changed_blocks'] == row['net_choice_changes'] == row['choice_flips_during_epoch'] == row['unique_blocks_switched_during_epoch'] == row['projected_values_during_epoch'] == 0 for row in scale_rows), 'Bevroren schaal is toch gewijzigd.')
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
            historical = evidence['comparison'][epoch]
            coverage_match = {name: value['order_sha256'] == historical['coverage'][name]['order_sha256'] for name, value in coverage.items()}
            s.require(all(coverage_match.values()), 'Beeldvolgorde verschilt van historische controle: ' + str(coverage_match))
            for rows, expected in [(train_rows, c['train_examples']), (test_rows, c['test_examples'])]:
                s.require(all(r['n'] == expected for r in rows), 'Metric bevat niet de volledige dataset.')
            bn_check = t.validate_bn(model, steps)
            scheduler.step()  # EXACT original position: after train and test.
            s.require(scheduler.last_epoch == epoch, 'Scheduler-epoch wijkt af.')
            checkpoint = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
            record = {
                'epoch': epoch, 'train': train_rows, 'test': test_rows,
                'freeze_checks': {'all_theta_unchanged': True, 'all_hard_offsets_unchanged': True, 'checked_optimizer_steps': ctx.guard.steps - frozen_steps_before, 'theta_lr_zero': True},
                'coverage_historical_match': coverage_match,
                'lr_used': lr, 'lr_next': float(optimizer.param_groups[0]['lr']),
                'scale_lr_used': scale_lr, 'scale_lr_next': float(optimizer.param_groups[1]['lr']),
                'scale_state': t.scale_state(model), 'scale_epoch_metrics': grad_audit['scale_epoch_metrics'],
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
            count32 = sum(row['count_x32'] for row in scale_rows)
            flips = sum(row['choice_flips_during_epoch'] for row in scale_rows)
            changed = sum(row['theta_changed_blocks'] for row in scale_rows)
            log(f'  SCHALEN: x32={count32}/4080; x64={4080-count32}/4080; '
                f'gewijzigde FP32-hulpwaarden={changed}; omschakelingen in epoch={flips}.')
            log(f'  VOORTZETTING: {epoch-50}/5 nieuwe epochs voltooid; historische controle is de bestaande training.')
            if soft_stop['requested'] or (out / 'STOP_AFTER_EPOCH').exists():
                log('Gecontroleerd gestopt op epochgrens; brondata en checkpoints blijven behouden.')
                break
    finally:
        ctx.guard.close()
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
    if resume is not None:
        s.require(core.file_sha256(resume) == resume_hash, 'Invoercheckpoint is gewijzigd.')
    s.require(core.file_sha256(Path(parent_checkpoint)) == ctx.report['parent_checkpoint_input_sha256'], 'Oudercheckpoint is gewijzigd.')
    return trace


def write_scale_results(ctx, out, trace):
    records = trace['epochs']
    rows = [row for record in records for row in record['scale_epoch_metrics']]
    anchor = trace['initial_scale_state']
    block_rows = []
    for epoch, state in [(50, anchor)] + [(r['epoch'], r['scale_state']) for r in records]:
        s.require(not core.differences(state, anchor), 'Opgeslagen theta wijkt af van epoch50.')
        for name in a.TARGET_LAYERS:
            value = torch.tensor(state[name], dtype=torch.float64)
            shape = a.block_shape(tuple(ctx.model.get_submodule(name).weight.shape))
            s.require(tuple(value.shape) == shape and bool(torch.isfinite(value).all())
                      and bool(((value >= 5) & (value <= 6)).all()), 'Ongeldige blokstaat.')
            for channel, values in enumerate(state[name]):
                for index, theta in enumerate(values):
                    offset = 5 if theta < 5.5 else 6
                    block_rows.append(dict(epoch=epoch, layer=name, output_channel=channel,
                        block_index=index, theta=theta, offset=offset, factor=1 << offset))
    for row in rows:
        s.require(row['theta_changed_blocks'] == row['net_choice_changes']
                  == row['choice_flips_during_epoch'] == row['unique_blocks_switched_during_epoch']
                  == row['projected_values_during_epoch'] == row['theta_update_l2'] == 0,
                  'Bevroren schaaldiagnostiek is niet nul.')
        value = torch.tensor(anchor[row['layer']], dtype=torch.float64)
        count32 = int((value < 5.5).sum())
        s.require(row['blocks'] == value.numel() and row['count_x32'] == count32
                  and row['count_x64'] == value.numel() - count32 and row['scale_lr'] == 0.0,
                  'Bevroren schaalverdeling wijkt af.')
        for key, expected in dict(theta_min=float(value.min()), theta_max=float(value.max()),
                theta_mean=float(value.mean()), theta_std=float(value.std(unbiased=False)),
                fraction_x32=count32 / value.numel()).items():
            s.require(math.isclose(row[key], expected, rel_tol=1e-12, abs_tol=1e-12), 'Onjuiste schaalstatistiek.')
    s.write_csv(out / 'scale_epoch_metrics.csv', rows)
    s.write_csv(out / 'scale_block_state.csv', block_rows)
    count32 = sum(theta < 5.5 for matrix in anchor.values() for channel in matrix for theta in channel)
    blocks = sum(len(channel) for matrix in anchor.values() for channel in matrix)
    result = dict(schema='mx-triple-freeze-offsets-e50-pilot5-freeze-v1', parent_epoch=50,
        all_theta_unchanged=True, all_hard_offsets_unchanged=True,
        per_update_checks=sum(r['freeze_checks']['checked_optimizer_steps'] for r in records),
        theta_lr=0.0, theta_gradients_computed=True, theta_adam_moments_updated=True,
        total_choice_flips=0, total_theta_changed_blocks=0, blocks=blocks,
        count_x32=count32, count_x64=blocks - count32, source_scale_dynamic=True,
        offset_updates_enabled=False, packed_choice_bytes=(blocks + 7) // 8,
        legacy_scale_audit_names='learned_shared_scale and scale_offsets_learned identify the inherited route; the offset parameters are fixed in this continuation',
        cost_interpretation='diagnostic freeze; scale gradients/moments still computed; no optimized cost claim')
    s.write_json(out / 'freeze_summary.json', result)
    ctx.report['freeze_summary'] = result
    return result


def validate_records(ctx, trace, evidence):
    records, c = trace['epochs'], ctx.c
    s.require(records and [r['epoch'] for r in records] == list(range(51, records[-1]['epoch'] + 1))
              and records[-1]['epoch'] <= 55, 'Alleen volledige aaneengesloten nieuwe epochs51-55 zijn geldig.')
    batches = math.ceil(c['train_examples'] / c['batch_size'])
    for record in records:
        epoch, audit = record['epoch'], record['gradient_checks']
        expected_lr = c['lr'] * c['scheduler_gamma']
        s.require(record['optimizer_steps'] == epoch * batches
                  and record['lr_used'] == record['lr_next'] == expected_lr
                  and record['scale_lr_used'] == record['scale_lr_next'] == 0.0,
                  'Proef heeft het oorspronkelijke tijdschema niet behouden.')
        s.require(record['batchnorm_counters_correct'] and record['test_optimizer_steps'] == 0,
                  'Onjuiste BN- of evaluatiestappen.')
        s.require(record['adam_checks']['all_active_adam_steps_correct']
                  and record['adam_checks']['moments_fp32_finite']
                  and record['adam_checks']['unused_bn_has_no_optimizer_state'], 'Onjuiste Adamstaat.')
        s.require(audit['optimizer_steps_in_phase'] == audit['finite_gradient_checks'] == batches
                  and audit['detached_teacher_checks'] == 2 * batches
                  and audit['forward_calls_E4M3'] == audit['forward_calls_E2M3']
                  == audit['forward_calls_E2M1'] == batches, 'Onjuist aantal trainingspasses.')
        s.require(audit['cache_empty_before_all_optimizer_steps']
                  and audit['scale_projection_after_every_update'] and audit['scale_optimizer_steps'] == batches,
                  'Oorspronkelijke trainingsroute is niet volledig gecontroleerd.')
        freeze = record['freeze_checks']
        s.require(freeze['all_theta_unchanged'] and freeze['all_hard_offsets_unchanged']
                  and freeze['theta_lr_zero'] and freeze['checked_optimizer_steps'] == batches,
                  'Freezecontrole ontbreekt of faalt.')
        s.require(not core.differences(record['scale_state'], evidence['baseline']['scale_state']),
                  'Blokkeuzes wijken af van het startcheckpoint.')
        for phase, count in [('train', c['train_examples']), ('test', c['test_examples'])]:
            s.require(record['coverage'][phase]['complete_once']
                      and record['coverage'][phase]['order_sha256'] == evidence['comparison'][epoch]['coverage'][phase]['order_sha256']
                      and record['coverage_historical_match'][phase], 'Beeldvolgorde verschilt van de historische controle.')
            source = audit['shared_source_checks'] if phase == 'train' else record['test_source_checks']
            phase_batches = math.ceil(count / c['batch_size'])
            s.require(source['source_triplets'] == phase_batches and source['high_encodings']
                      == source['exponent_derivations'] == source['mantissa_derivations'] == phase_batches * 6
                      and source['scale_offsets_learned'] == phase_batches * 12
                      and source['mantissa_scale_copies_unchanged'] == phase_batches * 6
                      and source['same_hard_offsets_across_lower_modes'] and source['source_cache_cleared']
                      and source['no_direct_lower_quantizer'] and source['cache_cleared_before_adam'],
                      'Onjuiste bron- of truncatieroute.')
            s.require([r['weight_bits'] for r in record[phase]] == list(a.MODES), 'Ontbrekende precisie.')
            for row in record[phase]:
                bit = row['weight_bits']
                objective = 'recursive_soft_cross_entropy' if phase == 'train' and bit != 8 else 'hard_cross_entropy'
                teacher = a.FORMATS[8 if bit == 6 else 6] if phase == 'train' and bit != 8 else None
                s.require(row['n'] == count and row['activation_bits'] == 32 and row['weight_format'] == a.FORMATS[bit]
                          and row['bn_bank_key'] == a.BANK[bit] and row['objective_type'] == objective
                          and row['teacher_weight_format'] == teacher and math.isfinite(row['objective_loss'])
                          and abs(row['top1_percent'] - 100 * row['correct1'] / count) < 1e-4,
                          'Onjuiste datasetdekking, precisie, loss of accuracy.')
        for counts in [record['test_label_counts'], *record['test_prediction_counts'].values()]:
            s.require(len(counts) == 10 and all(type(n) is int and n >= 0 for n in counts)
                      and sum(counts) == c['test_examples'], 'Onjuiste voorspellingstellingen.')




def export_results(ctx, out, trace, log, evidence):
    records, c = trace['epochs'], ctx.c
    validate_records(ctx, trace, evidence)
    best, last = t.best_record(records), records[-1]
    selected, details, mx_exports = [], {}, {}
    for role, record in [('best_E4M3_test', best), ('last_epoch', last)]:
        path = Path(record['checkpoint_path'])
        state = core.load_verified(path)
        s.require(not core.differences(state['contract'], ctx.contract), 'Checkpointcontract wijkt af.')
        s.require(state['completed_epoch'] == record['epoch'], 'Verkeerde checkpointepoch.')
        s.require(s.legacy.state_digest(state['model']) == record['model_sha256'], 'Modelhash wijkt af.')
        s.require(not core.differences(state['trace']['epochs'][-1]['test'], record['test']), 'Testmetrics verschillen.')
        suffix = 'best' if role == 'best_E4M3_test' else 'last'
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
            'training': 'continuation51-55 with theta/offset choices fixed after epoch50; dynamic E4 source -> exponent-MSB E2M3 -> mantissa E2M1; A32',
            'offset_updates_enabled': False, 'parent_epoch': 50, 'theta_lr': 0.0,
            'parent_checkpoint_sha256': ctx.report['parent_checkpoint_input_sha256'],
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
                offsets = a.unpack_scale_choices(loaded_source['scale_choices'][name], 'cpu')
                expected_scale = source_scale.cpu().to(torch.int32) + (0 if bit == 8 else offsets)
                s.require(bool((expected_scale <= 254).all())
                          and torch.equal(encoded.scale_codes.cpu().to(torch.int32), expected_scale),
                          'Export volgt niet de opgeslagen E4-bronschaal en harde blokkeuze.')
                s.require(name + '.weight' not in loaded['auxiliary_fp32_state'], 'FP32-master in export.')
            exports_for_role[str(bit)] = {**mx_details, 'epoch': record['epoch'], 'weight_bits': bit,
                                         'decoded_weights_exact': True, 'six_fp32_masters_absent': True,
                                         'full_test_forward_rerun': False, 'truncation': bit != 8,
                                         'source_weight_bits': 8, 'is_source_export': bit == 8,
                                         'derived_from_saved_source': bit != 8,
                                         'source_file_sha256': source_digest,
                                         'source_scale_bytes_copied': bit == 8,
                                         'scale_exponent_offset': 0 if bit == 8 else None,
                                         'scale_factor': 1 if bit == 8 else None,
                                         'scale_exponent_offset_choices': [0] if bit == 8 else [5, 6],
                                         'scale_factor_choices': [1] if bit == 8 else [32, 64],
                                         'packed_scale_choices_in_source': True,
                                         'packed_scale_choices_bytes': sum(x['packed_choices'].numel() for x in loaded_source['scale_choices'].values()),
                                         'fp32_scale_proxies_absent': True,
                                         'shared_scale_compensation': bit != 8,
                                         'learned_shared_scale': bit != 8,
                                         'offsets_frozen_at_epoch': 50, 'offset_updates_enabled': False,
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
                 'same_checkpoint': best['epoch'] == last['epoch'], 'selected_on_official_test': True,
                 'independent_test_estimate': False, 'selection_mode': 'MXFP8_E4M3/A32',
                 'all_modes_from_same_selected_checkpoint': True, 'selection_epoch_range': [51, last['epoch']],
                 'planned_selection_epoch_range': [51, 55], 'primary_endpoint': 55}
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
    freeze_summary = write_scale_results(ctx, out, trace)
    elapsed = sum(r['total_epoch_seconds'] for r in records)
    new_steps = sum(r['gradient_checks']['optimizer_steps_in_phase'] for r in records)
    complete = last['epoch'] == 55 and len(records) == 5
    summary = dict(schema='mx-triple-freeze-offsets-e50-pilot5-completion-v1',
        parent_epoch=50, start_after_epoch=50, end_epoch=55, planned_new_epochs=5,
        completed_new_epochs=len(records), completed_epoch=last['epoch'], last_epoch=last['epoch'],
        continuation_completed=complete, best_epoch_E4M3_A32=best['epoch'],
        selection_epoch_range=[51, last['epoch']], primary_endpoint=55,
        optimizer_steps=last['optimizer_steps'], new_optimizer_steps=new_steps,
        new_training_forward_passes=3 * new_steps, weight_lr=records[0]['lr_used'], theta_lr=0.0,
        theta_gradients_computed=True, theta_adam_moments_updated=True,
        parent_checkpoint_sha256=ctx.report['parent_checkpoint_input_sha256'],
        parent_model_sha256=trace['initial_model_sha256'], parent_contract_sha256=evidence['meta']['source_contract_sha256'],
        source_scale_dynamic=True, offset_updates_enabled=False, all_offset_parameters_exactly_frozen=True,
        all_sample_orders_match_historical=True, train_examples=c['train_examples'], test_examples=c['test_examples'],
        activation_bits=32, master_dtype='float32', exponent_truncation=True, mantissa_truncation=True,
        measured_completed_epoch_seconds=elapsed, historical_control_retrained=False,
        synthetic_fixture_only=ctx.contract['synthetic_fixture_only'], freeze_summary=freeze_summary)
    s.write_json(out / 'training_summary.json', summary)
    s.write_csv(out / 'epoch_times.csv', [dict(epoch=r['epoch'], **{k: r.get(k) for k in
        ['train_seconds', 'test_seconds', 'checkpoint_seconds', 'other_seconds', 'total_epoch_seconds',
         'first_epoch_in_process', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes']}) for r in records])
    comparison_rows = []
    for record in records:
        original = {row['weight_bits']: row for row in evidence['comparison'][record['epoch']]['test']}
        for row in record['test']:
            baseline = original[row['weight_bits']]
            comparison_rows.append(dict(epoch=record['epoch'], weight_bits=row['weight_bits'],
                weight_format=row['weight_format'], frozen_top1_percent=row['top1_percent'],
                historical_dynamic_top1_percent=baseline['top1_percent'],
                difference_percentage_points=row['top1_percent'] - baseline['top1_percent'],
                frozen_correct1=row['correct1'], historical_correct1=baseline['correct1'],
                test_examples=row['n'], historical_control=True))
    s.write_csv(out / 'comparison_with_historical.csv', comparison_rows)
    ctx.report.update(training_summary=summary, continuation_completed=complete,
        completed_epoch=last['epoch'], optimizer_steps=last['optimizer_steps'], new_optimizer_steps=new_steps,
        new_training_forward_passes=3 * new_steps, selected_best_epoch=best['epoch'], selected_last_epoch=last['epoch'],
        checkpoint_selection=selection, mx_exports=mx_exports,
        measured_completed_epoch_seconds=elapsed, all_sample_orders_match_historical=True)
    s.write_json(out / 'lineage.json', dict(parent_verification=ctx.report['parent_verification'],
        intervention=ctx.report['intervention'], branch_contract_sha256=ctx.report['contract_sha256'],
        historical_reference_sha256=evidence['reference_sha256'], parent_reference='parent_reference/REFERENCE.json',
        original_training_preserved=True, historical_control_retrained=False))
    if not (out / 'parent_reference').exists():
        shutil.copytree(evidence['directory'], out / 'parent_reference')
    write_notebook(out)
    log(f'KLAAR: {len(records)} nieuwe epochs; laatste={last["epoch"]}; E4M3-beste binnen de nieuwe epochs={best["epoch"]}; tijd={elapsed/60:.2f} min.')
    log('VERGELIJKING: historische epochs51-55; geen nieuwe controletraining. Bekijk de notebook in deze resultatenmap.')


def write_notebook(out):
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = ['from pathlib import Path\n', f'RUN_DIR = Path({str(out.resolve())!r})\n',
                              'SAVE_FIGURES = False  # True: bewaar figuren in een nieuwe analysemap.\n']
        if cell.get('cell_type') == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    s.write_json(out / NOTEBOOK, notebook)
    with (ROOT / 'notebook_support.py').open('rb') as src, (out / 'notebook_support.py').open('xb') as dst:
        shutil.copyfileobj(src, dst)


def create_context(args, out, report, log):
    original_report = {}
    ctx = ref.create_context(args, out, original_report, log, check_only=False)
    archive_parent_manifest(out)
    report.update({k: v for k, v in original_report.items() if k not in ('config', 'package_sha256', 'contract_sha256')})
    report['parent_runtime_config'] = original_report['config']
    report['parent_runtime_package_sha256'] = original_report['package_sha256']
    ctx.report = report
    return ctx


def archive_parent_manifest(out):
    source = out / 'training_change_manifest.json'
    target = out / 'parent_training_change_manifest.json'
    s.require(source.is_file() and not target.exists(), 'Oorspronkelijk wijzigingsmanifest ontbreekt of is al gearchiveerd.')
    source.rename(target)


def run_check(args, out, report, log, evidence):
    """Original codec/gradient checks, then strict real-parent restore, no SVHN training."""
    import os
    parent_hash = core.file_sha256(args.parent_checkpoint)
    state = core.load_verified(args.parent_checkpoint)
    old_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    original_report = {}
    try:
        if old_workspace not in (':4096:8', ':16:8'):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        ref.create_context(args, out, original_report, log, check_only=True)
        archive_parent_manifest(out)
    finally:
        if old_workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = old_workspace
    report['inherited_technical_checks'] = original_report
    rt = s.runtime(ref.config()['seed'], args.device)
    s.require(not core.differences(rt.environment, evidence['contract']['environment']),
              'Huidige software/GPU-omgeving wijkt af van de oorspronkelijke training.')
    s.require(not core.differences(rt.settings, evidence['contract']['settings']),
              'Huidige backendinstellingen wijken af van de oorspronkelijke training.')
    s.require(original_report['source']['workcopy_files_sha256'] == evidence['contract']['source_files']
              and s.verify_package() == evidence['contract']['package_sha256'], 'Oorspronkelijke runtimebron wijkt af.')
    source = out / 'source_workcopy'
    modules = s.import_source(source)
    model = t.build_model(modules, 'joint', rt.device)
    optimizer = t.build_optimizer(modules, model, ref.config())
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, ref.config()['scheduler_milestones'])
    ctx = SimpleNamespace(rt=rt, model=model, optimizer=optimizer, scheduler=scheduler, c=ref.config(),
                          contract=evidence['contract'], report=report)
    verification = verify_parent_state(ctx, state, evidence)
    verification['parent_checkpoint_sha256'] = parent_hash
    verification['strict_restore_checks'] = core.restore_checkpoint(state, model, optimizer, scheduler, {}, rt.device, evidence['contract'])
    t.validate_bn(model, state['optimizer_steps'])
    t.validate_adam_state(model, optimizer, state['optimizer_steps'])
    report['intervention'] = freeze_optimizer(model, optimizer, scheduler)
    from freeze_core import FrozenOffsetGuard
    guard = FrozenOffsetGuard(model, optimizer, evidence['baseline']['scale_state'])
    guard.assert_unchanged()
    guard.close()
    report.update(parent_verification=verification, environment=rt.environment, execution_settings=rt.settings,
                  dataset_downloaded=False, full_dataset_verified=False, check_only=True,
                  synthetic_fixture_only=bool(evidence['contract']['synthetic_fixture_only']), continuation_completed=False)
    s.require(core.file_sha256(args.parent_checkpoint) == parent_hash, 'Oudercheckpoint is gewijzigd.')
    from test_freeze import run_tests
    report['freeze_tests'] = run_tests(modules, rt)
    s.write_json(out / 'freeze_check_report.json', report['freeze_tests'])
    log('CHECK PASS: originele codec/gradientchecks; echt checkpoint50 volledig hersteld; freeze en controles geslaagd. Geen SVHN-training.')


def verify_unchanged(ctx, log):
    ref.verify_unchanged(ctx, log)
    verify_package()
    s.require(core.file_sha256(Path(ctx.report['parent_checkpoint'])) == ctx.report['parent_checkpoint_input_sha256'],
              'Oudercheckpoint is gewijzigd.')
    ctx.report['parent_checkpoint_unchanged'] = True


def feedback_bundle(out):
    names = ['run_report.json', 'epoch_metrics.csv', 'selected_metrics.csv', 'weight_updates.csv',
        'checkpoint_selection.json', 'epoch_times.csv', 'history.json', 'dataset_manifest.json',
        'source_manifest.json', 'compatibility.patch', 'pip_freeze.txt', 'pip_check.txt', 'run.log',
        'warnings.log', 'training_change_manifest.json', 'parent_training_change_manifest.json', 'training_summary.json', 'test_prediction_counts.csv',
        'scale_epoch_metrics.csv', 'scale_block_state.csv', 'freeze_summary.json', 'lineage.json',
        'comparison_with_historical.csv', 'codec_report.json', 'trunc_training_check_report.json',
        'freeze_check_report.json', NOTEBOOK, 'notebook_support.py']
    names += [f'mx_export_layers_{bit}_{role}.csv' for bit in a.MODES for role in ('best', 'last')]
    for path in sorted((out / 'parent_reference').rglob('*')):
        if path.is_file():
            names.append(path.relative_to(out).as_posix())
    with ZipFile(out / 'feedback_bundle.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for name in names:
            path = out / name
            if path.is_file():
                archive.write(path, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'resume'])
    parser.set_defaults(mode='joint')
    default_parent = Path('mx_runs/mx_triple_learned_scale32_64_reference100/20261025_150724_515132_joint_run/checkpoints/epoch_00050.pt')
    parser.add_argument('--parent-checkpoint', type=Path, default=default_parent)
    parser.add_argument('--checkpoint', type=Path, help='Alleen resume: checkpoint51-54 uit deze nieuwe proef.')
    parser.add_argument('--data-root', type=Path, default=Path('data/svhn'))
    parser.add_argument('--output-root', type=Path, default=Path('mx_runs/mx_triple_freeze_offsets_e50_pilot5'))
    parser.add_argument('--device', choices=['cuda', 'cuda:0'], default='cuda:0')
    parser.add_argument('--stop-after-epoch', type=int, default=55)
    parser.set_defaults(download=False)
    args = parser.parse_args()
    if not 51 <= args.stop_after_epoch <= 55:
        parser.error('--stop-after-epoch moet 51 tot en met 55 zijn; standaard55.')
    if (args.command == 'resume') != (args.checkpoint is not None):
        parser.error('Alleen resume vereist --checkpoint; run vertrekt altijd van het oudercheckpoint50.')
    args.parent_checkpoint = args.parent_checkpoint.resolve()
    out = (args.output_root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_freeze_' + args.command)).resolve()
    validate_output_location(out, args.parent_checkpoint, args.checkpoint, args.data_root)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='mx-triple-freeze-offsets-e50-pilot5-report-v1', status='RUNNING',
        package_revision='mx_triple_freeze_offsets_e50_pilot5_v1', command=args.command,
        started_at=s.now(), output_directory=str(out), argv=sys.argv, config=config(), warnings=[],
        continuation_completed=False,
        limitations=['Historical dynamic control; no additional control training.',
                     'CUDA repeat/resume is not bitexact; a small difference is not proof of superiority.',
                     'Official test used for E4 checkpoint selection; no independent validation estimate.',
                     'Theta LR is zero; gradients and Adam moments still computed. No optimized runtime/energy claim.',
                     'Only new epochs51-55; this does not establish an optimal freeze epoch.',
                     'A32 activations and FP32 masters; numerical MX emulation.'])
    log = s.legacy.Log(out / 'run.log')
    started, exit_code = time.perf_counter(), 1
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        item = dict(category=category.__name__, message=str(message), file=filename, line=lineno)
        report['warnings'].append(item)
        log('WARNING ' + str(message))
        with (out / 'warnings.log').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('default')
            warnings.showwarning = warning_handler
            log('RESULTATENMAP: ' + str(out))
            log('VASTE OFFSETS NA EPOCH50: nieuwe epochs51-55; drie MX-precisies; A32; FP32-master; bestaande dynamische controle.')
            package_hashes = verify_package()
            report['package_sha256'] = package_hashes
            evidence = load_parent_reference()
            s.require(args.parent_checkpoint.is_file() and args.parent_checkpoint.with_suffix('.pt.sha256').is_file(),
                      'Checkpoint50 of checksum ontbreekt: ' + str(args.parent_checkpoint) + '. Gebruik de projectmap als werkmap of geef --parent-checkpoint op.')
            shutil.copytree(evidence['directory'], out / 'parent_reference')
            if args.command == 'check':
                run_check(args, out, report, log, evidence)
                verify_package()
                report['status'] = 'PASS'
            else:
                ctx = create_context(args, out, report, log)
                ctx.branch_package_hashes = package_hashes
                trace = run_epochs(ctx, out, log, args.stop_after_epoch, args.parent_checkpoint,
                                   evidence, args.checkpoint)
                export_results(ctx, out, trace, log, evidence)
                verify_unchanged(ctx, log)
                report['status'] = 'PASS' if report['continuation_completed'] else 'STOPPED_AT_EPOCH_BOUNDARY'
            s.write_json(out / 'training_change_manifest.json', json.loads((ROOT / 'TRAINING_CHANGE_MANIFEST.json').read_text(encoding='utf-8')))
            exit_code = 0
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report['error'], report['traceback'] = repr(exc), traceback.format_exc()
        log(report['status'] + ': ' + str(exc))
        log(report['traceback'])
        log('Bestaande resultaten blijven behouden. Deel feedback_bundle.zip uit deze nieuwe map.')
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
