"""Five full SVHN epochs: five direct MX weight formats, A32, branched KD.

All heavy imports are inside main's error handling: a missing torch dependency
must still produce a feedback bundle and a final global FAIL in the terminal.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PACKAGE = 'mx_weights_five_direct_pilot5_v1'
END_EPOCH = 5


def config():
    formats = ['E4M3', 'E5M2', 'E2M3', 'E3M2', 'E2M1']
    return {
        'schema': 'mx-five-direct-pilot5-protocol-v1', 'package': PACKAGE, 'mode': 'joint',
        'weight_formats': formats, 'element_bits': dict(zip(formats, [8, 8, 6, 6, 4])),
        'activation_bits': 32, 'mx_block_size': 32, 'mx_scale_format': 'E8M0',
        'weight_preprocess': 'original G(W)=tanh(W)/max(abs(tanh(W)))*mean(abs(W)).detach(), with zero guard',
        'quantization_source': 'each format directly from the same current G(W); no inter-format code conversion',
        'scale_emax_subtracted': dict(zip(formats, [8, 15, 2, 4, 2])),
        'scale_rule': 'floor(log2(block_amax))-format_emax; clip[-127,127]; zero block exponent0',
        'scale_grouping': 'within Cout over flattened Cin*Kh*Kw; positive zero padding to B32',
        'rounding': 'nearest ties-to-even, finite saturation; reject non-finite input',
        'fp32_scalar_subnormals': 'flush to signed zero; local MX subnormals supported',
        'truncation': False, 'mantissa_truncation': False, 'exponent_truncation': False,
        'shared_scale_compensation': False, 'learned_shared_scale': False,
        'master_dtype': 'float32', 'gradient_dtype': 'float32', 'adam_moments_dtype': 'float32',
        'ste_policy': 'identity through quantizer; original G(W) derivative retained',
        'training_mode_order': formats, 'test_mode_order': ['E2M1', 'E2M3', 'E3M2', 'E4M3', 'E5M2'],
        'teacher_formats': {'E4M3': [], 'E5M2': [], 'E2M3': ['E4M3'],
                            'E3M2': ['E5M2'], 'E2M1': ['E2M3', 'E3M2']},
        'teacher_mixture_weights': {'E2M1': [0.5, 0.5]},
        'teacher_mixture_domain': 'post-softmax probabilities, detached',
        'temperature': 1.0, 'loss_weights': {f: 1.0 for f in formats}, 'loss_reduction': 'sum',
        'loss_batch_reduction': 'mean', 'extra_hard_student_loss': False,
        'passes_per_training_batch': 5, 'optimizer_steps_per_training_batch': 1,
        'extra_FP32_teacher_forward': False,
        'bn_bank_mapping': {f: f for f in formats},
        'bn_original_bank_mapping': dict(zip(formats, ['8', '1', '32', '2', '4'])),
        'legacy_abits_and_wbits': 32, 'bn_active_banks_per_layer': 5,
        'dataset': 'svhn', 'train_split': 'train', 'test_split': 'test', 'extra_used': False,
        'train_examples': 73257, 'test_examples': 26032, 'validation_examples': 0,
        'model': 'svhnq', 'expand': 2, 'parameters': 132954,
        'initialization_policy': 'fresh seed20261009; no old three-format or MSB checkpoint',
        'seed': 20261009, 'planned_epochs': 5, 'reference_schedule_epochs': 100,
        'primary_endpoint': 'epoch5, all five formats from one joint checkpoint',
        'checkpoint_selection_on_test': False,
        'batch_size': 128, 'optimizer': 'adam', 'lr': 0.001, 'weight_decay': 0.0,
        'scheduler_milestones': [50, 75, 90], 'scheduler_gamma': 0.1,
        'workers': 0, 'pin_memory': True, 'drop_last': False, 'train_shuffle': True,
        'test_shuffle': False, 'image_shape': [3, 40, 40],
        'backend_policy': 'original cudnn.benchmark=True; no AMP/compile or TF32 override',
        'rng_policy': 'global upstream streams; sequential dropout draws per format',
        'export_policy': 'five independent direct exports at epoch5; all five BN banks retained',
        'resume_policy': 'same pilot contract only; completed epochs1..4; new output directory',
        'scope': 'exploratory weight-only numerical emulation; no native MX or hardware performance claim',
    }


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition: raise RuntimeError(message)


class Log:
    def __init__(self, path): self.handle = path.open('x', encoding='utf-8', newline='\n')
    def __call__(self, text):
        print(text, flush=True)
        self.handle.write(text + '\n'); self.handle.flush()


def verify_package():
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    for name, digest in manifest.items():
        require(sha256(ROOT / name) == digest, 'Pakketbestand wijkt af: ' + name)
    reuse = json.loads((ROOT / 'REUSE_MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in reuse['byte_identical_files'].items():
        require(sha256(ROOT / name) == digest, 'Hergebruikte bron gewijzigd: ' + name)
    return manifest, reuse


def dependencies(out, log, local_validation=False):
    if local_validation:
        log('LOKALE CPU-VALIDATIE: geen gebruikersomgeving of SVHN-training.')
        return {'scope': 'internal CPU check; strict user-version and pip-check gates not exercised'}
    for verb in ('freeze', 'check'):
        result = subprocess.run([sys.executable, '-m', 'pip', verb], capture_output=True,
                                text=True, encoding='utf-8', errors='replace')
        (out / ('pip_' + verb + '.txt')).write_text(result.stdout + result.stderr, encoding='utf-8')
        require(result.returncode == 0, f'pip {verb} mislukt; zie pip_{verb}.txt. Er wordt niets geïnstalleerd.')
    log('Afhankelijkheden: pip check geslaagd.')
    return {'pip_freeze': 'PASS', 'pip_check': 'PASS'}


def create_context(args, out, report, log):
    manifest, reuse = verify_package()
    report.update(config=config(), package_sha256=manifest, source_reuse=reuse, interpreter=sys.executable)
    report['dependencies'] = dependencies(out, log, args.local_validation)
    import svhn_support as s
    import joint_adapter as a
    import joint_support as t
    source, source_info = s.legacy.prepare_source(out)
    report['source'] = source_info
    rt = s.runtime(report['config']['seed'], args.device, strict=not args.local_validation)
    if args.local_validation:
        rt.torch.set_num_threads(2)
        rt.settings['torch_num_threads'] = rt.torch.get_num_threads()
    modules = s.import_source(source)
    report.update(environment=rt.environment, execution_settings=rt.settings,
                  local_cpu_validation_only=bool(args.local_validation))
    log('Device: ' + rt.environment['device_name'])
    return SimpleNamespace(rt=rt, s=s, a=a, t=t, modules=modules, source=source, source_info=source_info,
                           c=report['config'], report=report, manifest=manifest)


def run_check(ctx, out, log):
    import test_new_codecs
    import test_joint
    report = ctx.report
    report.update(synthetic_checks_only=True, dataset_downloaded=False, svhn_accuracy_measured=False)
    codec_report, joint_report = {}, {}
    try:
        test_new_codecs.run_tests(str(ctx.rt.device), report=codec_report)
    finally:
        write_json(out / 'codec_report.json', codec_report)
    report['codec_tests'] = codec_report
    log('NUMERIEK PASS: E5M2 en E3M2; eindige codes, afronding, schalen, opslag en STE.')
    try:
        test_joint.run_tests(ctx.modules, ctx.rt, ctx.source, log, checks=joint_report)
    finally:
        write_json(out / 'joint_check_report.json', joint_report)
    report['synthetic_training_checks'] = joint_report
    verify_unchanged(ctx)
    log('CHECK PASS: vijf directe formaten, afzonderlijke BN-banken, afgesproken distillatie en één update per minibatch.')
    log('Deze check gebruikt synthetische invoer; er is geen SVHN-accuracy gemeten.')


def prepare_training(ctx, args, out, log):
    rt, s, c = ctx.rt, ctx.s, ctx.c
    torch, modules = rt.torch, ctx.modules
    start = time.perf_counter()
    info = s.ensure_dataset(args.data_root.resolve(), modules.data.SVHNMIX, args.download, log)
    modules.data.data_paths['svhn'] = str(args.data_root.resolve())
    train_tf = modules.data.get_transform('svhn', 'train')
    test_tf = modules.data.get_transform('svhn', 'val')
    train = modules.data.get_dataset('svhn', 'train', train_tf, download=False)
    test = modules.data.get_dataset('svhn', 'val', test_tf, download=False)
    require(train.split == 'train' and test.split == 'test', 'Onverwachte SVHN-splits.')
    selection = {'train': s.validate_dataset(train, c['train_examples'], 'SVHN train'),
                 'test': s.validate_dataset(test, c['test_examples'], 'SVHN test'), 'extra_used': False}
    transforms = {'train': repr(train_tf), 'test': repr(test_tf)}
    ctx.report.update(dataset_files=info, dataset_selection=selection, transforms=transforms,
                      data_preparation_seconds=time.perf_counter() - start,
                      synthetic_checks_only=False, svhn_accuracy_measured=False)
    write_json(out / 'dataset_manifest.json', {'files': info, 'selection': selection, 'transforms': transforms})
    ctx.loaders = {
        phase: s.TrackedLoader(torch.utils.data.DataLoader(s.IndexedDataset(data),
            batch_size=c['batch_size'], shuffle=(phase == 'train'), num_workers=c['workers'],
            pin_memory=c['pin_memory'], drop_last=False))
        for phase, data in [('train', train), ('test', test)]}
    ctx.model = ctx.t.build_model(modules, 'joint', rt.device)
    require(sum(p.numel() for p in ctx.model.parameters()) == c['parameters'], 'Modelparameteraantal wijkt af.')
    ctx.optimizer = modules.optimizer.get_optimizer_config(ctx.model, 'adam', c['lr'], c['weight_decay'])
    ctx.scheduler = modules.optimizer.get_lr_scheduler('adam', ctx.optimizer, c['scheduler_milestones'])
    ctx.contract = {'config': c, 'environment': rt.environment, 'settings': rt.settings,
                    'source_files': ctx.source_info['workcopy_files_sha256'],
                    'package_sha256': ctx.manifest, 'data_files': info['files'],
                    'dataset_selection': selection, 'transforms': transforms, 'synthetic_fixture_only': False}
    ctx.report['contract_sha256'] = rt.core.json_digest(ctx.contract)
    log(f'DATA PASS: {len(train)} trainbeelden en {len(test)} testbeelden; geen extra-split.')
    log(f'Per epoch: {len(ctx.loaders["train"])} optimizerupdates; vijf gewichtsmodi per minibatch.')


def copy_resume_evidence(state, checkpoint, out, core):
    """Bring immutable earlier pilot evidence into the new run, never edit it."""
    previous = checkpoint.parent.parent
    copied = []
    for epoch in state['trace']['epochs']:
        for relative, digest in epoch['artifact_hashes'].items():
            source, destination = previous / relative, out / relative
            require(source.is_file() and sha256(source) == digest, 'Ontbrekende/afwijkende hervatdata: ' + str(source))
            require(not destination.exists(), 'Dubbele hervatbestemming.')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative)
        relative = epoch['checkpoint_relative']
        source, destination = previous / relative, out / relative
        marker = source.with_suffix(source.suffix + '.sha256')
        require(source.is_file() and marker.is_file() and sha256(source) == marker.read_text().strip(),
                'Ouder pilotcheckpoint ontbreekt of wijkt af: ' + str(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), 'Checkpointbestemming bestaat al.')
        shutil.copy2(source, destination)
        shutil.copy2(marker, destination.with_suffix(destination.suffix + '.sha256'))
    return {'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': core.file_sha256(checkpoint),
            'copied_immutable_artifacts': copied}


def run_epochs(ctx, args, out, log):
    rt, s, t, c = ctx.rt, ctx.s, ctx.t, ctx.c
    core, torch = rt.core, rt.torch
    trace = {'schema': 'mx-five-direct-pilot5-trace-v1', 'metrics': [], 'epochs': [], 'layer_updates': [],
             'initialization_state_sha256': s.legacy.state_digest(ctx.model.state_dict()), 'run_history': []}
    completed, total_steps = 0, 0
    if args.resume:
        checkpoint = args.resume.resolve()
        state = core.load_verified(checkpoint)
        require(state['contract']['config']['schema'] == c['schema'], 'Alleen dezelfde vijfformaatpilot kan hervatten.')
        completed = int(state['completed_epoch'])
        require(1 <= completed < END_EPOCH, 'Hervatten vereist een voltooid pilotepoch1..4; geen oude of voltooide run.')
        require(args.stop_after_epoch > completed, 'Stop-epoch moet na het hervatcheckpoint liggen.')
        ctx.report['resume_checks'] = core.restore_checkpoint(state, ctx.model, ctx.optimizer, ctx.scheduler,
                                                               {}, rt.device, ctx.contract)
        ctx.report['resume_evidence'] = copy_resume_evidence(state, checkpoint, out, core)
        trace = copy.deepcopy(state['trace'])
        total_steps = int(state['optimizer_steps'])
        require(total_steps == completed * len(ctx.loaders['train']), 'Hervat-stapteller klopt niet.')
        log(f'HERVAT: epochs1..{completed} overgenomen; eerstvolgend epoch {completed + 1}.')
    trace['run_history'].append({'output_directory': str(out), 'start_completed_epoch': completed})
    ctx.trace = trace
    ctx.report['completed_epochs'] = completed
    ctx.report['initialization_state_sha256'] = trace['initialization_state_sha256']
    stop = {'requested': False}
    def on_interrupt(signum, frame):
        if stop['requested']: raise KeyboardInterrupt('Tweede Ctrl+C: onmiddellijk afgebroken.')
        stop['requested'] = True
        log('STOP AANGEVRAAGD: het huidige volledige epoch wordt nog opgeslagen; tweede Ctrl+C breekt direct af.')
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, on_interrupt)
    try:
        for epoch in range(completed + 1, END_EPOCH + 1):
            before = t.layer_snapshot(ctx.model)
            lr = float(ctx.optimizer.param_groups[0]['lr'])
            require(lr == c['lr'], 'Pilotepochs1..5 moeten learningrate0.001 behouden.')
            train_rows, train_seconds, train_audit = t.phase(rt, ctx.model, ctx.optimizer, ctx.loaders['train'],
                                                            epoch, 'train', lr, out, log)
            train_order = ctx.loaders['train'].validate(out / f'train_order_epoch_{epoch:05d}.npy')
            bn_before_eval = {k: v.clone() for k, v in ctx.model.state_dict().items() if '.bn_dict.' in k}
            test_rows, test_seconds, test_audit = t.phase(rt, ctx.model, ctx.optimizer, ctx.loaders['test'],
                                                         epoch, 'test', lr, out, log)
            test_order = ctx.loaders['test'].validate(out / f'test_order_epoch_{epoch:05d}.npy')
            require(ctx.loaders['test'].order == list(range(c['test_examples'])), 'Testvolgorde is gewijzigd.')
            require(all(torch.equal(v, ctx.model.state_dict()[k]) for k, v in bn_before_eval.items()),
                    'Evaluatie heeft BatchNorm-staat gewijzigd.')
            total_steps += len(ctx.loaders['train'])
            bn_audit = t.validate_bn(ctx.model, total_steps)
            adam_audit = t.validate_adam_state(ctx.model, ctx.optimizer, total_steps)
            updates = t.layer_updates(before, t.layer_snapshot(ctx.model), epoch)
            metrics = train_rows + test_rows
            for row in metrics:
                require(row['n'] == c[row['phase'] + '_examples'], 'Een formaat heeft geen volledige dataset gezien.')
            s.write_csv(out / f'metrics_epoch_{epoch:05d}.csv', metrics)
            s.write_csv(out / f'layer_updates_epoch_{epoch:05d}.csv', updates)
            ctx.scheduler.step()
            require(ctx.scheduler.last_epoch == epoch, 'Scheduler-epochteller wijkt af.')
            record = {'epoch': epoch, 'lr_used': lr, 'train_seconds': train_seconds,
                      'test_seconds': test_seconds, 'epoch_seconds': train_seconds + test_seconds,
                      'optimizer_steps_total': total_steps, 'train_audit': train_audit, 'test_audit': test_audit,
                      'train_coverage': train_order, 'test_coverage': test_order,
                      'bn_audit': bn_audit, 'adam_audit': adam_audit,
                      'checkpoint_relative': f'checkpoints/epoch_{epoch:05d}.pt'}
            write_json(out / f'epoch_audit_{epoch:05d}.json', record)
            artifacts = [f'train_batches_epoch_{epoch:05d}.jsonl', f'test_batches_epoch_{epoch:05d}.jsonl',
                         f'train_order_epoch_{epoch:05d}.npy', f'test_order_epoch_{epoch:05d}.npy',
                         f'test_predictions_epoch_{epoch:05d}.npz', f'metrics_epoch_{epoch:05d}.csv',
                         f'layer_updates_epoch_{epoch:05d}.csv', f'epoch_audit_{epoch:05d}.json']
            record['artifact_hashes'] = {name: sha256(out / name) for name in artifacts}
            # Publish the epoch in the aggregate history only after its checkpoint
            # is safely written. Raw per-epoch evidence is retained on a save error.
            candidate_trace = copy.deepcopy(trace)
            candidate_trace['metrics'].extend(metrics)
            candidate_trace['layer_updates'].extend(updates)
            candidate_trace['epochs'].append(record)
            payload = core.capture_checkpoint(ctx.model, ctx.optimizer, ctx.scheduler, {}, rt.device,
                                              epoch, total_steps, ctx.contract, candidate_trace, {})
            core.atomic_save(out / record['checkpoint_relative'], payload)
            trace = candidate_trace
            ctx.trace = trace
            completed = epoch
            ctx.report.update(completed_epochs=completed, optimizer_steps=total_steps, svhn_accuracy_measured=True)
            for row in test_rows:
                log(f'Epoch {epoch}/5 {row["weight_format"]}: {row["correct1"]}/{row["n"]} = {row["top1_percent"]:.6f}%')
            log('CHECKPOINT: ' + str(out / record['checkpoint_relative']))
            if stop['requested'] or epoch >= args.stop_after_epoch: break
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        if trace['metrics']: s.write_csv(out / 'metrics.csv', trace['metrics'])
        if trace['layer_updates']: s.write_csv(out / 'layer_updates.csv', trace['layer_updates'])
        write_json(out / 'training_history.json', trace)
        seconds = [e['epoch_seconds'] for e in trace['epochs']]
        ctx.report['timing'] = {'epoch_seconds': seconds, 'measured_total_epoch_seconds': sum(seconds),
            'estimated_100_epoch_seconds': 100 * sum(seconds) / len(seconds) if seconds else None,
            'basis': 'train+test phases including instrumentation; excludes data preparation, checks, checkpoint writes and exports',
            'estimate_is_not_measured_100_epoch_training': True}
    ctx.report['pilot_completed'] = completed == END_EPOCH
    return completed


def export_results(ctx, out, log):
    torch, a, core = ctx.rt.torch, ctx.a, ctx.rt.core
    ctx.model.eval()
    before = core.cpu_copy(ctx.model.state_dict())
    probe = torch.linspace(-1, 1, 2 * 3 * 40 * 40, device=ctx.rt.device).reshape(2, 3, 40, 40)
    rows, reports = [], {}
    with torch.no_grad():
        for name in a.MODES:
            a.select_mode(ctx.model, name)
            logits = ctx.model(probe)
            payload, diagnostics = a.export_payload(ctx.model, name,
                {'epoch': END_EPOCH, 'config_sha256': core.json_digest(ctx.c),
                 'checkpoint': f'checkpoints/epoch_{END_EPOCH:05d}.pt'})
            path = out / 'exports' / ('epoch00005_' + name + '.pt')
            file_info = core.atomic_save(path, payload)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            rebuilt = a.load_exported_model(ctx.modules.models, loaded, ctx.rt.device)
            for layer in a.TARGET_LAYERS:
                expected = a.codec(name).fake_quantize_mx(a.effective_reference_weight(ctx.model.get_submodule(layer).weight))
                require(torch.equal(expected, rebuilt.get_submodule(layer).decoded_weight), 'Exportgewicht wijkt af.')
            reconstructed_logits = rebuilt(probe)
            torch.testing.assert_close(logits, reconstructed_logits, rtol=1e-4, atol=1e-5)
            reports[name] = {'file': str(path.relative_to(out)), **file_info,
                             'decoded_weights_exact': True, 'synthetic_logits_close': True,
                             'max_abs_logit_difference': float((logits - reconstructed_logits).abs().max()),
                             'full_test_from_export_measured': False}
            rows.extend(diagnostics)
            del rebuilt, loaded, payload
    require(not core.differences(before, core.cpu_copy(ctx.model.state_dict())), 'Export wijzigde modelstaat.')
    ctx.s.write_csv(out / 'quantization_diagnostics_epoch00005.csv', rows)
    write_json(out / 'export_validation.json', reports)
    ctx.report['exports'] = reports
    log('EXPORT PASS: vijf afzonderlijke compacte modellen; gewichten exact gecontroleerd en synthetische logits vergeleken.')


def verify_unchanged(ctx):
    for name, digest in ctx.source_info['workcopy_files_sha256'].items():
        require(sha256(ctx.source / name) == digest, 'Uitgevoerde bron gewijzigd: ' + name)
    verify_package()
    ctx.report['source_and_package_unchanged'] = True


def feedback_bundle(out):
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'feedback_bundle.zip'
                   and '__pycache__' not in p.parts and not p.name.endswith('.pending.pt'))
    manifest = {p.relative_to(out).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files}
    write_json(out / 'feedback_manifest.json', manifest)
    path = out / 'feedback_bundle.zip'
    with ZipFile(path, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files + [out / 'feedback_manifest.json']:
            archive.write(file, file.relative_to(out).as_posix())
    with ZipFile(path) as archive:
        require(archive.testzip() is None, 'Feedback-ZIP is beschadigd.')
    return path


def parse_args(argv=None):
    project = ROOT.parent.parent if ROOT.parent.name == 'software_checks' else Path.cwd()
    parser = argparse.ArgumentParser(description='Vijf-epochpilot: vijf directe MX-gewichtsformaten, A32, geen truncatie.')
    parser.add_argument('command', choices=['check', 'train'])
    parser.add_argument('--data-root', type=Path, default=project / 'data' / 'svhn')
    parser.add_argument('--output-root', type=Path, default=project / 'mx_runs' / PACKAGE)
    parser.add_argument('--device', choices=['cuda', 'cuda:0', 'cpu'], default='cuda:0')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-epoch', type=int, default=END_EPOCH)
    parser.add_argument('--local-validation', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.local_validation and (args.command != 'check' or args.device != 'cpu'):
        parser.error('--local-validation is uitsluitend een interne CPU-check, geen datatraining.')
    if args.device == 'cpu' and not args.local_validation:
        parser.error('De gebruikersproef vereist CUDA; er is geen automatische CPU-fallback.')
    if args.command == 'check' and (args.resume or args.download or args.stop_after_epoch != END_EPOCH):
        parser.error('check gebruikt geen dataset, resume of afwijkend eindepoch.')
    if not 1 <= args.stop_after_epoch <= END_EPOCH:
        parser.error('Deze pilot eindigt uiterlijk op epoch5.')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'check' and args.device.startswith('cuda'):
        # This process only. Training retains the user's unchanged backend policy.
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    out = (args.output_root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mx-five-direct-pilot5-report-v1', 'status': 'RUNNING', 'command': args.command,
              'package': PACKAGE, 'started_at': datetime.now().astimezone().isoformat(),
              'output_directory': str(out), 'completed_epochs': 0, 'warnings': [],
              'limitations': ['One exploratory fixed-seed five-epoch pilot; no final accuracy conclusion.',
                              'Numerical FP32 emulator with MX weights and A32; no native MX speed/energy measurement.',
                              'Fixed seed does not guarantee bit-identical CUDA runs or resumes.',
                              'No matched accuracy comparison with the historical three-format model.',
                              'All five format losses have weight1; equal coefficients do not imply equal gradient norms.']}
    log, started, exit_code = Log(out / 'run.log'), time.perf_counter(), 1
    grouped_warnings = {}
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        key = (category.__name__, str(message), str(filename), int(lineno))
        if key not in grouped_warnings:
            grouped_warnings[key] = {'category': key[0], 'message': key[1], 'file': key[2], 'line': key[3], 'count': 0}
            log('WAARSCHUWING (herhalingen worden geteld): ' + key[1])
        grouped_warnings[key]['count'] += 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = warning_handler
            log('RESULTATENMAP: ' + str(out))
            log('VIJF DIRECTE MX-FORMATEN: E4M3, E5M2, E2M3, E3M2, E2M1; A32; vijf epochs vanaf nul.')
            ctx = create_context(args, out, report, log)
            if args.command == 'check':
                run_check(ctx, out, log)
                report['status'], exit_code = 'PASS', 0
            else:
                prepare_training(ctx, args, out, log)
                completed = run_epochs(ctx, args, out, log)
                if completed == END_EPOCH:
                    export_results(ctx, out, log)
                    report['status'], exit_code = 'PASS', 0
                else:
                    report['status'], exit_code = 'STOPPED_AT_EPOCH_BOUNDARY', 2
                    log('PILOT NIET VOLTOOID: checkpoint opgeslagen; hervatten kan tot epoch5.')
                verify_unchanged(ctx)
    except BaseException as exc:
        report['status'] = 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL'
        report['error'], report['traceback'] = repr(exc), traceback.format_exc()
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        log(report['status'] + ': ' + str(exc)); log(report['traceback'])
    finally:
        report['warnings'] = list(grouped_warnings.values())
        report['ended_at'] = datetime.now().astimezone().isoformat()
        report['session_wall_seconds'] = time.perf_counter() - started
        try:
            from notebook_support import write_notebook
            write_notebook(out)
            shutil.copy2(ROOT / 'README.md', out / 'README_experiment.md')
            shutil.copy2(ROOT / 'PROTOCOL.md', out / 'PROTOCOL_experiment.md')
        except BaseException as exc:
            report['reporting_error'] = repr(exc)
            report['status'], exit_code = 'FAIL', 1
            log('RAPPORTAGE FAIL: ' + str(exc))
        log('EINDSTATUS: ' + report['status'])
        log.handle.close()
        write_json(out / 'warnings_summary.json', report['warnings'])
        write_json(out / 'report.json', report)
        bundle = feedback_bundle(out)
        print('Resultaten: ' + str(out), flush=True)
        print('Stuur dit bestand terug: ' + str(bundle), flush=True)
    return exit_code


if __name__ == '__main__':
    code = 1
    try:
        code = main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except BaseException:
        traceback.print_exc()
        code = 1
    finally:
        print('GLOBAAL RESULTAAT: ' + ('PASS' if code == 0 else 'FAIL'), flush=True)
    raise SystemExit(code)
