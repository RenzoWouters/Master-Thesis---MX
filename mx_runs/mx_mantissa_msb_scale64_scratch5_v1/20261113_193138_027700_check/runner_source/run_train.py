"""One fresh MSB training run: five epochs, original dynamic x64 scales.

The historical fixed64 LSB epochs 1--5 are read-only reference measurements.
The public CLI requires the existing CUDA environment and full SVHN splits;
it never substitutes CPU execution, synthetic data or a trained parent model.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import warnings
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'vendor' / 'legacy'))
NAME = 'mx_mantissa_msb_scale64_scratch5_v1'
SCHEMA = 'mx-msb-scale64-scratch5-v1'
METHOD = 'msb_bestexp'
METRIC_FIELDS = ['method', 'epoch', 'phase', 'weight_bits', 'weight_format', 'n',
                 'correct1', 'correct5', 'top1_percent', 'top5_percent', 'loss', 'lr']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now().astimezone().isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Log:
    def __init__(self, path):
        self.handle = Path(path).open('x', encoding='utf-8', newline='\n')

    def __call__(self, message):
        print(message, flush=True)
        self.handle.write(message + '\n')
        self.handle.flush()


def project_root():
    return ROOT.parent.parent if ROOT.parent.name == 'software_checks' else Path.cwd()


def protocol():
    # Copy the recorded historical configuration, then enumerate the deliberate
    # method/budget/reporting changes. No training hyperparameter is retuned.
    c = copy.deepcopy(read_json(ROOT / 'reference_lsb_scale64' / 'run_report.json')['config'])
    changes = {
        'schema': SCHEMA + '-protocol',
        'purpose': 'exploratory five-epoch MSB mantissa pilot from fresh initialization with original dynamic x64 scales',
        'planned_epochs': 5,
        'reference_end_epoch': 100,
        'mantissa_mapping': 'retain E2M3 m0; Eprime=min(3,E+int(m==6)); preserve sign and E2 scale',
        'mx_rounding': 'unchanged E4M3 nearest ties-to-even encoder; E4-to-E2M3 exact exponent mapping; E2M3-to-E2M1 constrained minimum absolute-error exponent with retained m0',
        'selection_policy': 'fixed epoch5 endpoint; no best-test checkpoint selection',
        'reporting_policy': 'new MSB epochs1-5 versus recorded historical LSB x64 epochs1-5; endpoint5 for both',
        'checkpoint_policy': 'immutable complete FP32 model/Adam/scheduler/RNG checkpoint at each completed epoch',
        'export_policy': 'full resume checkpoints with explicit MSB method contract; no legacy LSB compact exports',
    }
    c.update(changes)
    c.update(method=METHOD, train_from_scratch=True, trained_parent_used=False,
             reference_method='lsb_trunc', historical_reference=True,
             historical_initial_rng_verified=False, historical_sample_order_verified=False,
             comparison_endpoint=5, deliberate_changes=list(changes),
             limitations=[
                 'one new MSB run and one historical LSB run; no repeated seeds',
                 'five early epochs do not establish convergence or optimal training budget',
                 'historical model initialization hash is available; historical initial RNG and sample-order files are not',
                 'same seed and source policy do not prove identical historical sample order or bit-exact CUDA execution',
                 'shared masters and three trainable BN banks can change W4, W6 and W8 results',
                 'official test reused for reporting; fixed epoch5 endpoint, no best-test selection',
                 'previous independent-scale posthoc scores are not a matched baseline for this pilot',
                 'new epoch timings include additional mapping/decode/gradient audits and are not a controlled speed comparison with historical LSB',
                 'FP32 emulation and diagnostic timing do not measure native low-bit hardware cost'])
    return c


def verify_package():
    values = read_json(ROOT / 'PACKAGE_SHA256.json')
    for name, expected in values.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha256(path) == expected,
                'Pakketbestand ontbreekt of is gewijzigd: ' + name)
    return values


def verify_reference():
    ref = read_json(ROOT / 'REFERENCE.json')
    folder = ROOT / 'reference_lsb_scale64'
    require(ref['factor'] == 64 and ref['pilot_endpoint'] == 5
            and ref['new_experiment_method'] == METHOD and ref['train_from_scratch'] is True,
            'Onjuiste vastgelegde referentie.')
    for name, expected in ref['files_sha256'].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder) and path.is_file() and sha256(path) == expected,
                'Historische referentie gewijzigd of ontbrekend: ' + name)
    history = read_json(folder / 'run_report.json')
    require(history['status'] == 'PASS' and history['completed_epoch'] == 100
            and history['synthetic_fixture_only'] is False
            and history['initial_model_sha256'] == ref['initial_model_sha256'],
            'Historische run is niet de vastgelegde volledige LSB x64-meting.')
    for name, expected in read_json(ROOT / 'VENDOR_SHA256.json').items():
        path = (ROOT / 'vendor' / 'legacy' / name).resolve()
        require(path.is_relative_to(ROOT / 'vendor' / 'legacy') and sha256(path) == expected
                and history['package_sha256'].get(name) == expected,
                'Legacy-code wijkt af van het historische fixed64-pakket: ' + name)
    with (folder / 'epoch_metrics.csv').open(newline='', encoding='utf-8') as handle:
        rows = [normalized_row(row, 'lsb_trunc', int(row['epoch']), 'test')
                for row in csv.DictReader(handle)
                if row['phase'] == 'test' and 1 <= int(row['epoch']) <= 5]
    rows.sort(key=lambda row: (row['epoch'], row['weight_bits']))
    require(len(rows) == 15 and {(r['epoch'], r['weight_bits']) for r in rows}
            == {(epoch, bit) for epoch in range(1, 6) for bit in (4, 6, 8)},
            'Onvolledige historische eerste vijf epochs.')
    require(all(r['n'] == 26032 and r['lr'] == .001 for r in rows),
            'Historische dekking of vroege LR wijkt af.')
    return ref, history, rows


def dependencies(out, log):
    for command in ('freeze', 'check'):
        result = subprocess.run([sys.executable, '-m', 'pip', command], capture_output=True,
                                text=True, encoding='utf-8', errors='replace')
        (out / ('pip_' + command + '.txt')).write_text(result.stdout + result.stderr, encoding='utf-8')
        require(result.returncode == 0, 'pip ' + command + ' mislukt; zie het feedbackrapport. Niets automatisch gewijzigd.')
    log('AFHANKELIJKHEDEN PASS: bestaande omgeving gecontroleerd.')


def runtime_comparison(rt, history):
    # The OS build is recorded, but does not by itself invalidate the comparison.
    # Numerical-library builds, device and execution policy must match.
    keys = ('python', 'torch', 'torchvision', 'numpy', 'scipy', 'pillow',
            'cuda_build', 'cudnn', 'device', 'device_name')
    differences = {key: {'new': rt.environment.get(key), 'reference': history['environment'].get(key)}
                   for key in keys if rt.environment.get(key) != history['environment'].get(key)}
    for key, value in history['execution_settings'].items():
        if rt.settings.get(key) != value:
            differences['settings.' + key] = {'new': rt.settings.get(key), 'reference': value}
    return differences


def prepare_context(args, out, report, log, *, strict=True, dependency_check=True):
    import svhn_support as s
    import resume_core as core
    import trunc_adapter as a
    import msb64_adapter as mapping
    import training_engine as engine
    package = verify_package()
    ref, history, reference_rows = verify_reference()
    c = protocol()
    write_json(out / 'protocol.json', c)
    write_json(out / 'REFERENCE.json', ref)
    write_csv(out / 'reference_lsb.csv', reference_rows, METRIC_FIELDS)
    source, source_info = s.legacy.prepare_source(out)
    report.update(package_sha256=package, source=source_info, interpreter=sys.executable)
    if dependency_check:
        dependencies(out, log)
    # Seed and source import occur in exactly the original sequence. DataLoader
    # construction follows next, BEFORE model construction in prepare_state.
    rt = s.runtime(c['seed'], args.device, strict=strict)
    modules = s.import_source(source)
    source_matches = source_info == history['source']
    mismatch = runtime_comparison(rt, history)
    report.update(environment=rt.environment, execution_settings=rt.settings,
                  runtime_differences=mismatch,
                  synthetic_fixture_only=not strict,
                  reference_comparison={
                      'initialization_hash_matches': False,
                      'runtime_matches_reference': not mismatch,
                      'source_matches_reference': source_matches,
                      'dataset_matches_reference': None,
                      'historical_sample_order_verified': False,
                      'historical_initial_rng_verified': False,
                      'historical_model_sha256': ref['initial_model_sha256'],
                      'reference_run': ref['reference_run'],
                      'basis': 'recorded historical LSB x64 epochs1-5; no new LSB training'})
    require(source_matches, 'Auteurbron/compatibiliteitspatch verschilt van de historische fixed64-run.')
    require(not mismatch or not strict,
            'Omgeving verschilt van de historische x64-run: ' + ', '.join(mismatch) +
            '. Er is nog geen training gestart; stuur de feedbackbundel voor diagnose.')
    before_rules = core.capture_rng({}, rt.device)
    numeric = mapping.verify_rules(rt.device)
    require(not core.differences(before_rules, core.capture_rng({}, rt.device)),
            'Numerieke controles mogen de globale trainings-RNG niet wijzigen.')
    report['numeric_checks'] = numeric
    contract = {'schema': SCHEMA + '-contract', 'method': METHOD, 'protocol': c,
                'reference_sha256': sha256(ROOT / 'REFERENCE.json'), 'package_sha256': package,
                'source_files': source_info['workcopy_files_sha256'],
                'environment': rt.environment, 'execution_settings': rt.settings,
                'synthetic_fixture_only': not strict}
    ctx = SimpleNamespace(rt=rt, core=core, s=s, a=a, mapping=mapping, engine=engine,
                          source=source, source_info=source_info, modules=modules,
                          reference=ref, history=history, reference_rows=reference_rows,
                          protocol=c, contract=contract, report=report, data_info=None,
                          state=None, trace=None, completed=0)
    log('BRON PASS: oorspronkelijke fixed64-code en historische LSB-epochs 1–5 geverifieerd.')
    log('NUMERIEK PASS: alle 64 E2M3-broncodes; behouden m0 en optimale beperkte exponentkeuze.')
    return ctx


def prepare_data(ctx, data_root, out, log):
    s, modules, c = ctx.s, ctx.modules, ctx.protocol
    start = time.perf_counter()
    info = s.ensure_dataset(data_root.resolve(), modules.data.SVHNMIX, False, log)
    require(info['files'] == ctx.history['dataset_files']['files'],
            'Ruwe SVHN-bestanden verschillen van de historische referentie.')
    modules.data.data_paths['svhn'] = str(data_root.resolve())
    train_tf = modules.data.get_transform('svhn', 'train')
    test_tf = modules.data.get_transform('svhn', 'val')
    train = modules.data.get_dataset('svhn', 'train', train_tf, download=False)
    test = modules.data.get_dataset('svhn', 'val', test_tf, download=False)
    require(train.split == 'train' and test.split == 'test', 'Onjuiste datasetsplits.')
    selection = {'train': s.validate_dataset(train, c['train_examples'], 'SVHN train-only'),
                 'test': s.validate_dataset(test, c['test_examples'], 'SVHN test'),
                 'train_split_order': ['train'], 'extra_used': False, 'held_out_validation_examples': 0}
    transforms = {'train': repr(train_tf), 'test': repr(test_tf)}
    require(selection == ctx.history['dataset_selection'] and transforms == ctx.history['transforms'],
            'Datasetselectie, labels of preprocessing verschillen van de historische run.')
    torch = ctx.rt.torch
    # Original global RNG policy; no explicit DataLoader generators.
    ctx.loaders = {name: s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(dataset),
                    batch_size=c['batch_size'], shuffle=name == 'train', num_workers=0,
                    pin_memory=True, drop_last=False))
                   for name, dataset in [('train', train), ('test', test)]}
    ctx.data_info = info
    ctx.contract.update(data_files=info['files'], dataset_selection=selection, transforms=transforms)
    ctx.report.update(dataset_files=info, dataset_selection=selection, transforms=transforms,
                      data_preparation_seconds=time.perf_counter() - start)
    ctx.report['reference_comparison']['dataset_matches_reference'] = True
    write_json(out / 'dataset_manifest.json', {'files': info, 'selection': selection, 'transforms': transforms})
    log('DATA PASS: 73.257 trainbeelden en 26.032 testbeelden, zonder extra-split.')


def rng_hash(state):
    torch = sys.modules['torch']
    digest = hashlib.sha256()
    def update(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((str(value.dtype), list(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode()); update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode())
            for item in value:
                update(item)
        else:
            digest.update(json.dumps(value, allow_nan=False).encode())
    update(state)
    return digest.hexdigest()


def prepare_state(ctx, out, log):
    state = ctx.engine.build_state(ctx.rt, ctx.modules, ctx.protocol,
                                  expected_initial_hash=ctx.reference['initial_model_sha256'])
    ctx.state = state
    ctx.trace = {'schema': SCHEMA + '-history', 'method': METHOD, 'train_from_scratch': True,
                 'initial_model_sha256': state.initial_model_sha256,
                 'initial_rng_sha256': rng_hash(state.initial_rng), 'epochs': []}
    ctx.report['reference_comparison']['initialization_hash_matches'] = True
    ctx.report.update(initial_model_sha256=state.initial_model_sha256,
                      initial_rng_sha256=ctx.trace['initial_rng_sha256'],
                      installation_audit=state.installation_audit,
                      contract_sha256=ctx.core.json_digest(ctx.contract))
    write_json(out / 'initialization.json', {key: ctx.trace[key] for key in
               ('method', 'train_from_scratch', 'initial_model_sha256', 'initial_rng_sha256')})
    write_json(out / 'run_context.json', {'schema': SCHEMA, 'contract': ctx.contract})
    log('INIT PASS: exact dezelfde ongetrainde modelwaarden als de historische LSB x64-run; geen phi.')


def normalized_row(row, method, epoch, phase):
    n = int(row['n'])
    return {'method': method, 'epoch': epoch, 'phase': phase,
            'weight_bits': int(row['weight_bits']), 'weight_format': row['weight_format'],
            'n': n, 'correct1': int(row['correct1']), 'correct5': int(row['correct5']),
            'top1_percent': 100 * int(row['correct1']) / n,
            'top5_percent': 100 * int(row['correct5']) / n,
            'loss': float(row['objective_loss']), 'lr': float(row['lr'])}


def restore(ctx, checkpoint, stop_after, log):
    path = checkpoint.resolve()
    payload = ctx.core.load_verified(path)
    mismatch = ctx.core.differences(payload['contract'], ctx.contract)
    require(not mismatch, 'Hervatcontract verschilt: ' + ', '.join(mismatch[:8]))
    completed, trace = payload['completed_epoch'], payload['trace']
    require(type(completed) is int and 1 <= completed <= stop_after <= 5,
            'Hervatcheckpoint of stop ligt buiten de geplande epochs1–5.')
    require(trace['schema'] == SCHEMA + '-history' and trace['method'] == METHOD
            and trace['train_from_scratch'] is True
            and len(trace['epochs']) == completed
            and [record['epoch'] for record in trace['epochs']] == list(range(1, completed + 1)),
            'Hervatcheckpoint heeft geen volledige MSB x64-geschiedenis vanaf nul.')
    require(trace['initial_model_sha256'] == ctx.trace['initial_model_sha256']
            and trace['initial_rng_sha256'] == ctx.trace['initial_rng_sha256'],
            'Beginmodel of geregistreerde RNG wijkt af van de nieuwe proef.')
    require(payload['optimizer_steps'] == completed * len(ctx.loaders['train']),
            'Onjuist aantal Adamupdates in hervatcheckpoint.')
    for record in trace['epochs']:
        require(record['method'] == METHOD and record['optimizer_steps'] == record['epoch'] * len(ctx.loaders['train']),
                'Hervatgeschiedenis heeft een verkeerde methode of stapteller.')
    state = ctx.state
    checks = ctx.core.restore_checkpoint(payload, state.model, state.optimizer, state.scheduler,
                                         {}, ctx.rt.device, ctx.contract)
    ctx.engine.verify_state(state, payload['optimizer_steps'])
    ctx.trace = copy.deepcopy(trace)
    ctx.completed = completed
    ctx.report.update(resume_checks=checks, inherited_epochs=completed,
                      network_evaluated=True, last_checkpoint={'path': str(path), 'sha256': sha256(path)},
                      resumed_from={'path': str(path), 'sha256': sha256(path), 'completed_epoch': completed})
    log(f'HERSTELD: MSB x64, epoch {completed}/5, inclusief Adam, scheduler en RNG.')


def run_epochs(ctx, out, log, stop_after=5, checkpoint=None):
    require(1 <= stop_after <= 5 and ctx.state is not None, 'Onjuist pilotbudget of ontbrekende initialisatie.')
    if checkpoint is not None:
        restore(ctx, checkpoint, stop_after, log)
    state = ctx.state
    steps = ctx.completed * len(ctx.loaders['train'])
    stop = {'requested': False}
    old_handler = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        if stop['requested']:
            raise KeyboardInterrupt('Tweede Ctrl+C: hervat vanaf de laatste volledig opgeslagen epoch.')
        stop['requested'] = True
        log('Stop aangevraagd: de lopende epoch wordt afgewerkt, geëvalueerd en opgeslagen.')
    signal.signal(signal.SIGINT, interrupt)
    try:
        for epoch in range(ctx.completed + 1, stop_after + 1):
            log(f'TRAIN MSB x64: epoch {epoch}/5 vanaf nul; Adam lr=0.001.')
            folder = out / 'epochs' / f'epoch_{epoch:05d}'
            train_out, test_out = folder / 'train', folder / 'test'
            train_out.mkdir(parents=True); test_out.mkdir()
            started = time.perf_counter()
            training = ctx.engine.run_phase(ctx, state, ctx.loaders['train'], epoch, True, train_out, log)
            train_order = ctx.loaders['train'].validate(train_out / 'sample_order.npy')
            steps += len(ctx.loaders['train'])
            ctx.engine.verify_state(state, steps)
            testing = ctx.engine.run_phase(ctx, state, ctx.loaders['test'], epoch, False, test_out, log)
            ctx.report['network_evaluated'] = True
            test_order = ctx.loaders['test'].validate(test_out / 'sample_order.npy')
            state.scheduler.step()
            require(state.scheduler.last_epoch == epoch and state.optimizer.param_groups[0]['lr'] == .001,
                    'Vroege LR of oorspronkelijke scheduler wijkt af.')
            checks = ctx.engine.verify_state(state, steps)
            target = out / 'checkpoints' / f'epoch_{epoch:05d}.pt'
            record = {'epoch': epoch, 'method': METHOD, 'optimizer_steps': steps,
                      'train': [normalized_row(row, METHOD, epoch, 'train') for row in training['rows']],
                      'test': [normalized_row(row, METHOD, epoch, 'test') for row in testing['rows']],
                      'train_audit': training['audit'], 'test_audit': testing['audit'], 'checks': checks,
                      'train_order': train_order, 'test_order': test_order,
                      'train_seconds': training['seconds'], 'test_seconds': testing['seconds'],
                      'epoch_seconds_before_checkpoint': time.perf_counter() - started,
                      'checkpoint_path': str(target.resolve())}
            candidate = {**ctx.trace, 'epochs': [*ctx.trace['epochs'], record]}
            write_csv(test_out / 'predictions.csv', testing['predictions'])
            payload = ctx.core.capture_checkpoint(state.model, state.optimizer, state.scheduler,
                       {}, ctx.rt.device, epoch, steps, ctx.contract, candidate, {})
            saved = ctx.core.atomic_save(target, payload)
            # Publish history only AFTER both checkpoint and checksum succeed.
            # A later report-write error must not discard this valid boundary.
            ctx.trace, ctx.completed = candidate, epoch
            ctx.report.update(completed_epochs=epoch,
                              last_checkpoint={'path': str(target.resolve()), 'sha256': saved['sha256']})
            write_json(folder / 'epoch_summary.json', record)
            for row in sorted(record['test'], key=lambda value: value['weight_bits'], reverse=True):
                log(f"  TEST W{row['weight_bits']}: {row['correct1']}/{row['n']} = {row['top1_percent']:.6f}%")
            if stop['requested']:
                return
    finally:
        signal.signal(signal.SIGINT, old_handler)


def synthetic_check(ctx, out, log):
    torch = ctx.rt.torch
    # Local generator; the original initial model has already been constructed.
    generator = torch.Generator().manual_seed(20261113)
    dataset = torch.utils.data.TensorDataset(torch.randn(3, 3, 40, 40, generator=generator),
                                             torch.tensor([0, 1, 2], dtype=torch.int64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    folder = out / 'synthetic_checks'
    train_out, test_out = folder / 'train', folder / 'test'
    train_out.mkdir(parents=True); test_out.mkdir()
    training = ctx.engine.run_phase(ctx, ctx.state, loader, 1, True, train_out, log)
    testing = ctx.engine.run_phase(ctx, ctx.state, loader, 1, False, test_out, log)
    checks = ctx.engine.verify_state(ctx.state, len(loader))
    ctx.report.update(synthetic_checks={'train_audit': training['audit'], 'test_audit': testing['audit'],
                                       'state_checks': checks},
                      synthetic_training_steps=2, training_performed=False,
                      network_evaluated=False, synthetic_only=True)
    log('CHECK PASS: verse initialisatie, twee synthetische Adamupdates, echte MSB-forward/backward en dynamische x64-schalen. Geen SVHN-training of SVHN-accuracy.')


def export_results(ctx, out):
    records = [] if ctx.trace is None else ctx.trace['epochs']
    if not (out / 'accuracy.csv').exists():
        write_csv(out / 'accuracy.csv', [row for record in records for row in record['test']], METRIC_FIELDS)
    if not (out / 'training_metrics.csv').exists():
        write_csv(out / 'training_metrics.csv', [row for record in records for row in record['train']], METRIC_FIELDS)
    if not (out / 'history.json').exists():
        write_json(out / 'history.json', ctx.trace or {'schema': SCHEMA + '-history', 'method': METHOD, 'epochs': []})
    ctx.report.update(completed_epochs=ctx.completed, training_completed=ctx.completed == 5,
                      optimizer_steps_in_completed_epochs=(records[-1]['optimizer_steps'] if records else 0),
                      new_completed_epochs=ctx.completed - ctx.report.get('inherited_epochs', 0))
    for name, expected in ctx.source_info['workcopy_files_sha256'].items():
        require(sha256(ctx.source / name) == expected, 'Auteurbron gewijzigd tijdens de run: ' + name)
    ctx.report['upstream_source_unchanged'] = True
    if ctx.data_info is not None:
        for name, expected in ctx.data_info['files'].items():
            require(sha256(Path(ctx.data_info['root']) / name) == expected['sha256'], 'Ruwe data gewijzigd: ' + name)
        ctx.report['raw_dataset_unchanged'] = True


def feedback_bundle(out):
    target = out / 'feedback_bundle.zip'
    excluded = {'source_workcopy', 'checkpoints'}
    with ZipFile(target, 'x', ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            relative = path.relative_to(out)
            if (path.is_file() and path != target and not excluded.intersection(relative.parts)
                    and '__pycache__' not in relative.parts and path.suffix not in ('.pt', '.pyc')):
                archive.write(path, relative.as_posix())
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'resume'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--stop-after-epoch', type=int, default=5, choices=range(1, 6))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if (args.command == 'resume') != (args.checkpoint is not None):
        parser.error('Alleen resume vereist --checkpoint uit deze nieuwe MSB x64-proef.')
    out = project_root() / 'mx_runs' / NAME / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.command)
    out.mkdir(parents=True, exist_ok=False)
    log = Log(out / 'run.log')
    report = {'schema': SCHEMA, 'status': 'RUNNING', 'command': args.command, 'started_at': now(),
              'protocol': protocol(), 'warnings': [], 'limitations': protocol()['limitations'],
              'training_performed': False, 'network_evaluated': False, 'training_completed': False,
              'synthetic_only': args.command == 'check', 'synthetic_fixture_only': False,
              'completed_epochs': 0, 'inherited_epochs': 0, 'actual_new_optimizer_steps': 0,
              'output_directory': str(out.resolve()), 'last_checkpoint': None,
              'checkpoint_argument': None if args.checkpoint is None else str(args.checkpoint.resolve())}
    ctx = None
    started = time.perf_counter()
    exit_code = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            ctx = prepare_context(args, out, report, log)
            if args.command != 'check':
                prepare_data(ctx, args.data_root or project_root() / 'data' / 'svhn', out, log)
            prepare_state(ctx, out, log)
            if args.command == 'check':
                synthetic_check(ctx, out, log)
            else:
                run_epochs(ctx, out, log, args.stop_after_epoch, args.checkpoint)
            export_results(ctx, out)
            report['status'] = 'PASS' if args.command == 'check' or report['training_completed'] else 'STOPPED'
            exit_code = 0
        except KeyboardInterrupt:
            report['status'] = 'STOPPED'
            report['warnings'].append('Direct onderbroken; alleen volledig opgeslagen epochs zijn hervatbaar.')
            log('ONDERBROKEN: de laatste volledige checkpoints blijven beschikbaar.')
            exit_code = 130
        except Exception as exc:
            report.update(status='FAIL', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            log('FAIL: ' + str(exc))
        finally:
            if ctx is not None and not (out / 'history.json').exists():
                try:
                    export_results(ctx, out)
                except Exception as exc:
                    report.setdefault('export_errors', []).append(str(exc))
                    report['status'] = 'FAIL'
                    exit_code = 1
            report['warnings'].extend(str(item.message) for item in caught)
            report.update(ended_at=now(), session_wall_seconds=time.perf_counter() - started)
            write_json(out / 'run_report.json', report)
            for name in ('training_report.py', 'REFERENCE.json'):
                if not (out / name).exists():
                    shutil.copyfile(ROOT / name, out / name)
            runner = out / 'runner_source'; runner.mkdir()
            for name in ('run_train.py', 'training_engine.py', 'msb64_adapter.py', 'mantissa_rules.py',
                         'PACKAGE_SHA256.json', 'VENDOR_SHA256.json', 'PROTOCOL.json', 'BUILD_PROVENANCE.json'):
                if (ROOT / name).is_file():
                    shutil.copyfile(ROOT / name, runner / name)
            try:
                import training_report
                rendered = training_report.write_outputs(out)
                if rendered.get('errors'):
                    write_json(out / 'report_render_warnings.json', rendered)
                    log('Resultaatweergave meldt: ' + '; '.join(rendered['errors']))
            except Exception as exc:
                write_json(out / 'report_render_warnings.json', {'error': str(exc)})
                log('Notebookweergave kon niet worden opgebouwd: ' + str(exc))
            log('Resultaten: ' + str(out.resolve()))
            log('Stuur dit bestand terug: ' + str(out.resolve() / 'feedback_bundle.zip'))
            log.handle.close()
            feedback_bundle(out)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
