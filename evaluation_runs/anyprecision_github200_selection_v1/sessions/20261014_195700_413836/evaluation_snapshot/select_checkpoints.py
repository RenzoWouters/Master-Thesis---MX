"""Read-only checkpoint sweep for the supplied Any-Precision GitHub200 runner.

Select exactly one epoch using W32/A32 test top-1, strict greater-than, earliest
on ties. Evaluate all five precisions of that model AND the fixed final model.
No training, calibration, source modification, downloads, or package installs.
Use only your own trusted checkpoints. See PROTOCOL.md for the test-selection caveat.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
SCHEMA = 'anyprecision-github200-checkpoint-selection-v1'
BITS = [1, 2, 4, 8, 32]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def new_json(path: Path, value: Any) -> None:
    """Immutable output; an interrupted pending file is not mistaken for a result."""
    require(not path.exists(), 'Bestaand bestand wordt niet overschreven: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + datetime.now().strftime('%Y%m%d%H%M%S%f') + '.pending')
    with pending.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    require(not path.exists(), 'Uitvoerpad bestaat inmiddels: ' + str(path))
    os.replace(pending, path)


def seal_json(path: Path, value: Any) -> None:
    payload = dict(value)
    payload['content_sha256'] = digest(value)
    new_json(path, payload)


def read_sealed(path: Path) -> dict:
    payload = read_json(path)
    expected = payload.pop('content_sha256')
    require(digest(payload) == expected, 'Resultaatbestand gewijzigd/onvolledig: ' + str(path))
    return payload


def new_csv(path: Path, rows: list[dict]) -> None:
    with path.open('x', encoding='utf-8', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


@contextmanager
def exclusive_lock(root: Path):
    """OS releases the lock when the process exits, even after a crash."""
    p = root / '.evaluation.lock'
    with p.open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Er draait al een evaluatie in deze uitvoermap.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def verify_addon() -> dict[str, str]:
    m = read_json(ROOT / 'ADDON_SHA256.json')
    for name, expected in m.items():
        require(sha256(ROOT / name) == expected, 'Evaluatiepakket gewijzigd: ' + name)
    return m


def bind_reference(path: Path):
    expected = read_json(ROOT / 'REFERENCE_SHA256.json')
    for name, value in expected.items():
        require((path / name).is_file() and sha256(path / name) == value,
                'De vereiste oorspronkelijke GitHub200-module ontbreekt of is gewijzigd: ' + name)
    sys.path.insert(0, str(path))
    ref = importlib.import_module('run_reference')
    require(Path(ref.__file__).resolve() == path / 'run_reference.py', 'Verkeerde referentiemodule geladen.')
    ref.verify_package()
    return ref


def validate_payload(state: dict, anchor: dict, epoch: int, core: Any, support: Any) -> None:
    require(state['completed_epoch'] == epoch and state['next_epoch'] == epoch + 1,
            f'Checkpointnaam/teller verschilt bij epoch {epoch}.')
    require(not core.differences(state['contract'], anchor['contract']),
            f'Ander protocol/andere omgeving bij epoch {epoch}.')
    expected_trace = dict(anchor['trace'], epochs=anchor['trace']['epochs'][:epoch])
    require(not core.differences(state['trace'], expected_trace),
            f'Checkpoint epoch {epoch} hoort niet bij het traject van het eindcheckpoint.')
    c = anchor['contract']['config']
    steps = epoch * math.ceil(c['train_examples'] / c['batch_size'])
    require(state['optimizer_steps'] == steps and state['scheduler']['last_epoch'] == epoch,
            f'Onjuiste stap/schedulerteller bij epoch {epoch}.')
    expected_model = (anchor['trace']['initial_model_sha256'] if epoch == 0
                      else anchor['trace']['epochs'][epoch - 1]['model_sha256'])
    require(support.state_digest(state['model']) == expected_model,
            f'Modelinhoud past niet bij de opgeslagen eindhistorie: epoch {epoch}.')
    require(state['validation_logits'] == {}, 'Geen train-only-referentiecheckpoint.')
    for v in state['model'].values():
        if v.is_floating_point():
            require(bool(core.torch.isfinite(v).all()), f'Niet-eindige modelwaarde: epoch {epoch}.')


def inventory_checkpoints(run_root: Path, anchor_path: Path, anchor: dict,
                          ref: Any, core: Any, out: Path, log: Any) -> list[dict]:
    """Use the final checkpoint's complete trace to exclude unrelated branches."""
    planned = anchor['contract']['config']['planned_epochs']
    require(anchor['completed_epoch'] == planned, 'Geen volledig eindcheckpoint van deze run.')
    require(len(anchor['trace']['epochs']) == planned, 'Eindcheckpoint mist epochhistorie.')
    selected, excluded, duplicates = {}, [], []
    paths = sorted(run_root.glob('*/checkpoints/epoch_*.pt'))
    require(anchor_path in [p.resolve() for p in paths], 'Eindcheckpoint ligt niet onder --run-root.')
    log(f'Inventarisatie: {len(paths)} bestanden gevonden; alle epochs 1-{planned} worden gecontroleerd.')
    for path in paths:
        match = re.fullmatch(r'epoch_(\d{5})\.pt', path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch == 0 or epoch > planned:
            continue
        try:
            state = core.load_verified(path)
            validate_payload(state, anchor, epoch, core, ref.support)
            entry = {'epoch': epoch, 'path': str(path.resolve()),
                     'sha256': path.with_suffix('.pt.sha256').read_text(encoding='ascii').strip(),
                     'bytes': path.stat().st_size,
                     'model_sha256': anchor['trace']['epochs'][epoch - 1]['model_sha256']}
            if epoch in selected:
                previous = core.load_verified(Path(selected[epoch]['path']))
                require(not core.differences(previous, state),
                        f'Meerdere verschillende geldige checkpoints voor epoch {epoch}; eerst onderzoeken.')
                duplicates.append({'epoch': epoch, 'path': str(path), 'same_entire_content': True})
                # Always keep the user-specified anchor at the final epoch.
                if path.resolve() == anchor_path:
                    selected[epoch] = entry
                del previous
            else:
                selected[epoch] = entry
            del state
        except Exception as exc:
            if 'Meerdere verschillende geldige' in str(exc):
                raise
            excluded.append({'path': str(path.resolve()), 'epoch': epoch, 'reason': str(exc)})
        if epoch % 25 == 0:
            log(f'  gecontroleerd tot kandidaat-epoch {epoch}; {len(selected)} geldige epochs.')
    missing = sorted(set(range(1, planned + 1)) - set(selected))
    info = {'anchor_checkpoint': str(anchor_path), 'expected_epochs': list(range(1, planned + 1)),
            'selected': [selected[e] for e in sorted(selected)], 'excluded': excluded,
            'identical_duplicates': duplicates, 'missing_epochs': missing,
            'membership_check': 'same contract, full trace prefix, model hash and epoch/step counters',
            'no_selection_based_on_test_score_during_inventory': True}
    new_json(out / 'inventory.json', info)
    require(not missing, 'Niet alle checkpoints zijn beschikbaar/geldig. Ontbrekend: '
            + ', '.join(map(str, missing)) + '. Zie inventory.json; geen onvolledige sweep als volledig rapporteren.')
    require(selected[planned]['path'] == str(anchor_path), 'Eindcheckpoint niet juist geselecteerd.')
    if excluded:
        log(f'LET OP: {len(excluded)} andere/ongeldige kandidaten uitgesloten; zie inventory.json.')
    if duplicates:
        log(f'{len(duplicates)} identieke dubbele checkpoints herkend; geen dubbele evaluatie.')
    log(f'Inventarisatie OK: alle {planned} epochcheckpoints horen bij het opgegeven trainingstraject.')
    return info['selected']


def prepare_test_loader(ctx: dict, ref: Any):
    torch, c = ctx['torch'], ctx['config']
    dm = importlib.import_module('datasets.data')
    tf = dm.get_transform('cifar10', 'val')
    data = dm.get_dataset('cifar10', 'val', tf, download=False)
    require(len(data) == c['test_examples'], 'Onvolledige testset.')
    require(data.split in ('val', 'test') and hasattr(data, 'test_data') and not hasattr(data, 'train_data'),
            'De geselecteerde loader gebruikt niet de officiele testtak.')
    if not ctx['contract']['synthetic_fixture_only']:
        require(len(data) == 10000 and [data.test_labels.count(i) for i in range(10)] == [1000] * 10,
                'Onjuiste officiele testset/klasseverdeling.')
    view = ref.pilot.IndexedSubset(data, list(range(len(data))))
    gen = torch.Generator().manual_seed(c['seed'] + 12)
    loader = torch.utils.data.DataLoader(view, batch_size=c['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, generator=gen)
    return loader, repr(tf)


def evaluate_bits(ctx: dict, ref: Any, loader: Any, epoch: int, bits: list[int]) -> tuple[list[dict], dict]:
    """Same forward/eval operations as upstream; also keep integer counts and hashes."""
    torch, model, device = ctx['torch'], ctx['model'], ctx['device']
    require(bits in ([32], BITS), 'Alleen W32-selectie of volledige precisievector toegestaan.')
    model.eval()
    before = ref.support.state_digest(model.state_dict())
    for p in model.parameters():
        p.grad = None
    # Use supplied accuracy + AverageMeter to reproduce its score computation.
    utils = importlib.import_module('utils')
    meters = {b: [utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()] for b in bits}
    correct = {b: [0, 0] for b in bits}
    logit_hash = {b: hashlib.sha256() for b in bits}
    labels_hash, input_hash = hashlib.sha256(), hashlib.sha256()
    order, count = [], 0
    ref.sync(torch, device)
    start = time.perf_counter()
    with torch.no_grad():
        for x, y, ids in loader:
            order.extend(ids.tolist())
            input_hash.update(x.contiguous().numpy().tobytes())
            labels_hash.update(y.contiguous().numpy().tobytes())
            x, y = x.to(device), y.to(device)
            for bit in bits:
                ref.support.select_precision(model, bit)
                output = model(x)
                require(tuple(output.shape) == (len(y), 10) and bool(torch.isfinite(output).all()),
                        'Verkeerde/niet-eindige testuitvoer.')
                loss = float(torch.nn.functional.cross_entropy(output, y).item())
                require(math.isfinite(loss), 'Niet-eindige testloss.')
                p1, p5 = utils.accuracy(output, y, topk=(1, 5))
                preds = output.topk(5, 1, True, True).indices
                correct[bit][0] += int((preds[:, 0] == y).sum().item())
                correct[bit][1] += int((preds == y[:, None]).any(dim=1).sum().item())
                for meter, value in zip(meters[bit], [loss, float(p1.item()), float(p5.item())]):
                    meter.update(value, len(y))
                logit_hash[bit].update(output.detach().cpu().contiguous().numpy().tobytes())
            count += len(y)
    ref.sync(torch, device)
    duration = time.perf_counter() - start
    require(order == list(range(len(loader.dataset))) and count == ctx['config']['test_examples'],
            'Testbeelden ontbreken, zijn dubbel gebruikt of staan in een andere volgorde.')
    require(ref.support.state_digest(model.state_dict()) == before, 'Evaluatie veranderde model/BatchNorm.')
    require(all(p.grad is None for p in model.parameters()), 'Evaluatie heeft gradients aangemaakt.')
    rows = []
    for b in bits:
        loss, m1, m5 = meters[b]
        count_pct = 100.0 * correct[b][0] / count
        # CIFAR10 test batches are 128 or 16, so these percentages match exactly.
        require(abs(m1.avg - count_pct) < 1e-5, 'Accuracyhelper en gehele telling wijken af.')
        rows.append({'epoch': epoch, 'phase': 'official_test', 'bits_w_and_a': b,
                     'n': count, 'correct1': correct[b][0], 'correct5': correct[b][1],
                     'top1_percent': m1.avg, 'top5_percent': m5.avg, 'hard_ce': loss.avg,
                     'count_top1_percent': count_pct, 'logits_sha256': logit_hash[b].hexdigest()})
    details = {'seconds': duration, 'ordered_test_indices': f'range(0,{count})',
               'inputs_sha256': input_hash.hexdigest(), 'labels_sha256': labels_hash.hexdigest(),
               'model_state_sha256': before, 'model_and_batchnorm_unchanged': True,
               'no_gradients_or_optimizer_updates': True}
    return rows, details


def restore_model(ctx: dict, ref: Any, anchor: dict, entry: dict) -> None:
    p = Path(entry['path'])
    require(sha256(p) == entry['sha256'], 'Checkpoint gewijzigd sinds inventarisatie: ' + str(p))
    state = ctx['core'].load_verified(p)
    validate_payload(state, anchor, entry['epoch'], ctx['core'], ref.support)
    ctx['model'].load_state_dict(state['model'], strict=True)
    require(not ctx['core'].differences(ctx['core'].cpu_copy(ctx['model'].state_dict()), state['model']),
            'Model of BatchNorm niet exact geladen.')
    ctx['model'].eval()
    del state


def best_epoch(records: list[dict]) -> dict:
    require(bool(records), 'Geen testresultaten voor selectie.')
    best = None
    for r in sorted(records, key=lambda a: a['epoch']):
        if best is None or r['row']['top1_percent'] > best['row']['top1_percent']:
            best = r
    return best


def compare_targets(ref: Any, rows: list[dict], role: str) -> list[dict]:
    targets = read_json(ref.ROOT / 'github_targets.json')
    require(sha256(ref.ROOT / 'source_notes/upstream_README.md') == targets['readme_sha256'],
            'GitHub-README-referentie is gewijzigd.')
    target = dict(zip(targets['bits'], targets['top1_percent']))
    require(sorted(r['bits_w_and_a'] for r in rows) == BITS and len({r['epoch'] for r in rows}) == 1,
            'Geen volledige precisievector van een enkel checkpoint.')
    return [{'role': role, 'epoch': r['epoch'], 'bits_w_and_a': r['bits_w_and_a'],
             'n': r['n'], 'our_top1_percent': r['top1_percent'],
             'github_top1_percent': target[r['bits_w_and_a']],
             'difference_percentage_points': r['top1_percent'] - target[r['bits_w_and_a']]}
            for r in rows]


def sweep(args: Any, invocation: Path, report: dict, log: Any) -> None:
    addon = verify_addon()
    ref = bind_reference(args.reference_dir.resolve())
    require(args.run_root.resolve() != args.output_dir.resolve(), 'Gebruik een afzonderlijke evaluatiemap.')
    require(not args.output_dir.resolve().is_relative_to(args.run_root.resolve()),
            'Zet de evaluatiemap buiten de trainingsmappen.')
    # Reuse the existing source/runtime/config/data guards. No training function is called.
    ctx = ref.make_context(args, invocation, report, log)
    require(not ctx['contract']['synthetic_fixture_only'] or
            ctx['config']['purpose'].startswith('LOCAL_SYNTHETIC_FIXTURE_ONLY'),
            'Ongeldige fixturemarkering.')
    core, c = ctx['core'], ctx['config']
    anchor_path = args.final_checkpoint.resolve()
    anchor = core.load_verified(anchor_path)
    require(not core.differences(anchor['contract'], ctx['contract']),
            'Eindcheckpoint past niet bij code/configuratie/apparaat/omgeving. Verander niets; deel de foutmelding.')
    require(anchor['completed_epoch'] == c['planned_epochs'], 'Niet het laatste geplande checkpoint.')
    validate_payload(anchor, anchor, c['planned_epochs'], core, ref.support)
    anchor_hash = sha256(anchor_path)
    entries = inventory_checkpoints(args.run_root.resolve(), anchor_path, anchor, ref, core, invocation, log)
    by_epoch = {e['epoch']: e for e in entries}
    loader, transform = prepare_test_loader(ctx, ref)
    plan = {'schema': SCHEMA, 'anchor_checkpoint_sha256': anchor_hash,
            'anchor_trace_sha256': core.json_digest(anchor['trace']),
            'training_contract_sha256': core.json_digest(anchor['contract']),
            'expected_epochs': list(range(1, c['planned_epochs'] + 1)),
            'selection_precision': 32, 'selection_metric': 'top1_percent_from_supplied_accuracy_AverageMeter',
            'selection_rule': 'strict > in ascending epoch order; retain earlier epoch on exact tie',
            'final_vectors': 'all five precisions of SAME selected checkpoint, plus fixed final epoch',
            'test_examples': c['test_examples'], 'test_batch_size': c['batch_size'],
            'test_transform': transform, 'no_test_augmentation_or_recalibration': True,
            'evaluation_addon_hashes': addon,
            'selected_by_training_history_before_test': [{'epoch': e['epoch'], 'sha256': e['sha256']} for e in entries],
            'source_training_policy_unchanged': c['evaluation_policy'],
            'evaluation_policy': 'posthoc_test_selected_checkpoint_for_github_protocol_comparison',
            'not_independent_test_of_selected_model': True,
            'synthetic_fixture_only': bool(ctx['contract']['synthetic_fixture_only'])}
    plan_path = args.output_dir.resolve() / 'evaluation_plan.json'
    if plan_path.exists():
        existing = read_sealed(plan_path)
        require(existing == plan, 'Bestaande evaluatiemap hoort bij andere instellingen/checkpoints/code. Niet overschrijven.')
    else:
        seal_json(plan_path, plan)
    plan_hash = digest(plan)
    report.update(evaluation_plan_sha256=plan_hash, evaluation_policy=plan['evaluation_policy'],
                  not_independent_test_of_selected_model=True, final_checkpoint=str(anchor_path),
                  anchor_sha256=anchor_hash, expected_checkpoint_count=len(entries),
                  official_test_examples_per_evaluated_checkpoint=c['test_examples'],
                  synthetic_fixture_only=plan['synthetic_fixture_only'],
                  valid_research_result=not plan['synthetic_fixture_only'], newly_evaluated_epochs=[])
    log(f'TEST: {len(loader.dataset)} beelden per checkpoint; batch {c["batch_size"]}; alleen evaluatie.')
    log('Selectie: W32/A32 top-1, hoogste score; bij exact gelijk behoud vroegste epoch.')
    log('Geen losse beste epoch per bitbreedte. We trainen of kalibreren NIET verder.')
    cache = args.output_dir.resolve() / 'per_epoch'
    cache.mkdir(exist_ok=True)
    results, session_durations = [], []
    inputs_fingerprint = None
    for entry in entries:
        epoch = entry['epoch']
        target = cache / f'epoch_{epoch:05d}_w32.json'
        cached = target.exists()
        if cached:
            r = read_sealed(target)
            require(r['plan_sha256'] == plan_hash and r['checkpoint_sha256'] == entry['sha256']
                    and r['epoch'] == epoch and r['row']['bits_w_and_a'] == 32
                    and r['row']['n'] == c['test_examples'], 'Bestaand resultaat past niet bij deze evaluatie.')
            # Validated here even though this epoch does not require another forward pass.
            log(f'  epoch {epoch}: eerder resultaat hergebruikt, W32/A32={r["row"]["top1_percent"]:.2f}%')
        else:
            if ((args.max_checkpoints and len(report['newly_evaluated_epochs']) >= args.max_checkpoints)
                    or (invocation / 'STOP_AFTER_CHECKPOINT.txt').exists()):
                break
            t0 = time.perf_counter()
            restore_model(ctx, ref, anchor, entry)
            rows, details = evaluate_bits(ctx, ref, loader, epoch, [32])
            require(sha256(Path(entry['path'])) == entry['sha256'], 'Invoercheckpoint gewijzigd tijdens evaluatie.')
            r = {'epoch': epoch, 'checkpoint_sha256': entry['sha256'], 'plan_sha256': plan_hash,
                 'row': rows[0], 'details': details, 'load_and_test_seconds': time.perf_counter() - t0}
            seal_json(target, r)
            report['newly_evaluated_epochs'].append(epoch)
            session_durations.append(r['load_and_test_seconds'])
            log(f'  epoch {epoch}/{c["planned_epochs"]}: W32/A32 top-1={r["row"]["top1_percent"]:.2f}%; '
                f'test+laden={r["load_and_test_seconds"]:.2f}s; resultaat opgeslagen.')
        fingerprint = (r['details']['inputs_sha256'], r['details']['labels_sha256'])
        if inputs_fingerprint is None:
            inputs_fingerprint = fingerprint
        require(fingerprint == inputs_fingerprint, 'Testinvoer verschilt tussen checkpoints.')
        results.append(r)
        if session_durations and not cached:
            measured = session_durations[1:] or session_durations
            remaining = (len(entries) - len(results)) * statistics.mean(measured) / 60
            log(f'    Raming resterende W32-sweep: {remaining:.1f} minuten, exclusief afsluitende controles.')
    # Always export the completed per-epoch table for THIS invocation, never overwrite old reports.
    new_csv(invocation / 'checkpoint_scores.csv', [dict(r['row'], checkpoint_sha256=r['checkpoint_sha256']) for r in results])
    report['completed_checkpoint_count'] = len(results)
    if len(results) != len(entries):
        report['status'] = 'PAUSED'
        report['selection_final'] = False
        log(f'Evaluatie gepauzeerd: {len(results)}/{len(entries)} checkpoints beschikbaar. Geen definitieve beste epoch.')
    else:
        chosen = best_epoch(results)
        report.update(selected_epoch=chosen['epoch'], selection_final=True,
                      selected_checkpoint=by_epoch[chosen['epoch']]['path'],
                      selected_checkpoint_sha256=chosen['checkpoint_sha256'],
                      tied_epochs_at_best_score=[r['epoch'] for r in results
                          if r['row']['top1_percent'] == chosen['row']['top1_percent']])
        all_vectors = {}
        vector_root = args.output_dir.resolve() / 'all_precision_results'
        vector_root.mkdir(exist_ok=True)
        for epoch in sorted({chosen['epoch'], c['planned_epochs']}):
            file = vector_root / f'epoch_{epoch:05d}_all_bits.json'
            entry = by_epoch[epoch]
            if file.exists():
                v = read_sealed(file)
                require(v['plan_sha256'] == plan_hash and v['checkpoint_sha256'] == entry['sha256'],
                        'Volledige precisievector hoort bij een andere evaluatie.')
            else:
                restore_model(ctx, ref, anchor, entry)
                rows, details = evaluate_bits(ctx, ref, loader, epoch, BITS)
                require((details['inputs_sha256'], details['labels_sha256']) == inputs_fingerprint,
                        'Testinvoer is veranderd voor de volledige precisievector.')
                v = {'plan_sha256': plan_hash, 'checkpoint_sha256': entry['sha256'],
                     'rows': rows, 'details': details}
                seal_json(file, v)
            # Verify single-precision scan and all-precision loop produce identical W32 output.
            w32 = next(row for row in v['rows'] if row['bits_w_and_a'] == 32)
            scanned = next(r['row'] for r in results if r['epoch'] == epoch)
            require(w32 == scanned, 'W32-only en volledige precisielus verschillen; onderzoek vereist.')
            all_vectors[epoch] = v['rows']
            for row in v['rows']:
                log(f'  TEST epoch {epoch}, W{row["bits_w_and_a"]}/A{row["bits_w_and_a"]}: '
                    f'{row["top1_percent"]:.2f}%')
        selected_rows, final_rows = all_vectors[chosen['epoch']], all_vectors[c['planned_epochs']]
        new_csv(invocation / 'selected_checkpoint_metrics.csv', selected_rows)
        new_csv(invocation / 'final_checkpoint_metrics.csv', final_rows)
        comparison = compare_targets(ref, selected_rows, 'github_style_test_selected') + compare_targets(
            ref, final_rows, 'fixed_final_epoch')
        new_csv(invocation / 'github_comparison.csv', comparison)
        report.update(selected_metrics=selected_rows, fixed_final_metrics=final_rows,
                      github_comparison=comparison, github_target_source=read_json(ref.ROOT/'github_targets.json'),
                      historical_readme_run_provenance_confirmed=False, status='PASS')
        seal_json(invocation / 'best_checkpoint.json', {
            'checkpoint': report['selected_checkpoint'], 'sha256': report['selected_checkpoint_sha256'],
            'epoch': chosen['epoch'], 'top1_w32_percent': chosen['row']['top1_percent'],
            'tie_epochs': report['tied_epochs_at_best_score'], 'policy': plan['evaluation_policy'],
            'not_independent_test': True, 'training_not_modified': True})
        log(f'Beste checkpoint volgens W32/A32: epoch {chosen["epoch"]}; '
            f'top-1={chosen["row"]["top1_percent"]:.2f}%.')
    ref.check_sources(ctx)
    # Full read-only evidence: all selected input files are hashed again after the sweep.
    for entry in entries:
        require(sha256(Path(entry['path'])) == entry['sha256'], 'Een trainingscheckpoint is gewijzigd.')
    report.update(all_input_checkpoints_unchanged=True, source_unchanged=True,
                  optimizer_steps_during_evaluation=0, recalibration_performed=False)


def request_stop(path: Path) -> None:
    path = path.resolve()
    require((path / 'invocation_plan.json').is_file(), 'Geen evaluatiesessie op dit pad.')
    require(not (path / 'selection_report.json').exists(), 'Deze evaluatiesessie is al afgesloten.')
    target = path / 'STOP_AFTER_CHECKPOINT.txt'
    if not target.exists():
        with target.open('x', encoding='ascii') as f:
            f.write('Stop after the current complete checkpoint evaluation.\n')
    print('Stopverzoek geplaatst. Wacht op STATUS: PAUSED en de terminalprompt.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('sweep', help='Inventariseer, test alle epochs op W32, kies een model, test alle bits.')
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--final-checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--max-checkpoints', type=int, default=0, help='Maximaal NIEUWE W32-tests per sessie; 0=alle.')
    stop = sub.add_parser('request-stop', help='Stop na de huidige checkpointtest.')
    stop.add_argument('--session-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'request-stop':
        try:
            request_stop(args.session_dir)
            return 0
        except Exception as exc:
            print('FOUT: ' + str(exc))
            return 1
    if args.max_checkpoints < 0:
        parser.error('--max-checkpoints mag niet negatief zijn.')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_lock(args.output_dir):
            invocation = args.output_dir / 'sessions' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            invocation.mkdir(parents=True, exist_ok=False)
            new_json(invocation / 'invocation_plan.json', {
                'argv': sys.argv, 'max_new_checkpoints': args.max_checkpoints,
                'stop_request': str(invocation / 'STOP_AFTER_CHECKPOINT.txt')})
            report = {'status': 'FAIL', 'schema': SCHEMA, 'command': 'sweep',
                      'started_at': datetime.now().astimezone().isoformat(),
                      'session_directory': str(invocation), 'argv': sys.argv,
                      'limitations': ['Test-selected score is not independent of checkpoint selection.',
                        'Original GitHub selection rule is reproduced on OUR saved training trajectory.',
                        'No proof that the historical README numbers used this exact run/seed.',
                        'No retraining, calibration, MX, stored-code truncation or hardware measurements.']}
            start = time.perf_counter()
            with (invocation / 'run.log').open('x', encoding='utf-8') as handle:
                def log(text: str):
                    print(text, flush=True)
                    handle.write(text + '\n')
                    handle.flush()
                log('ANY-PRECISION CHECKPOINTSELECTIE - geen training, geen MX')
                log('Evaluatiemap: ' + str(args.output_dir))
                log('Sessie-/rapportmap: ' + str(invocation))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        addon = verify_addon()
                        snapshot = invocation / 'evaluation_snapshot'
                        for name in addon:
                            dst = snapshot / name
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(ROOT / name, dst)
                        shutil.copy2(ROOT / 'ADDON_SHA256.json', snapshot / 'ADDON_SHA256.json')
                        sweep(args, invocation, report, log)
                    except (Exception, KeyboardInterrupt) as exc:
                        report.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL',
                                      error=str(exc) or type(exc).__name__, traceback=traceback.format_exc())
                        log('FOUT/ONDERBREKING: ' + report['error'])
                        log(report['traceback'])
                    finally:
                        report['warnings'] = [{'category': w.category.__name__, 'message': str(w.message),
                            'filename': w.filename, 'line': w.lineno} for w in caught]
                        report['elapsed_seconds'] = time.perf_counter() - start
                        report['ended_at'] = datetime.now().astimezone().isoformat()
                        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                                                capture_output=True, text=True, check=False)
                        (invocation / 'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
                        report['pip_freeze_returncode'] = freeze.returncode
                        new_json(invocation / 'selection_report.json', report)
                        log(f'Waarschuwingen: {len(report["warnings"])}; zie selection_report.json.')
                        log('STATUS: ' + report['status'])
                        log('Rapportmap: ' + str(invocation))
                        log('Geen downloads, installaties, training of overschrijving van trainingsbestanden.')
                return 0 if report['status'] in ['PASS', 'PAUSED'] else 1
    except Exception as exc:
        print('FOUT: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
