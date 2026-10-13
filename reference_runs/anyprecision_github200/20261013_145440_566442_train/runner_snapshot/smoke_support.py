"""Technical Any-Precision smoke test. No CIFAR-10, MX or research accuracy.

Uses the user's archived implementation, not a replacement network. Only a fresh
per-run copy of utils.py is patched. See README.md and CHANGELOG.md.
Run with the dedicated .venv-train interpreter. Requires torch 2.10.x and numpy.
"""
from __future__ import annotations

import argparse
import difflib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from typing import Any
from zipfile import ZipFile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BITS = [1, 2, 4, 8, 32]
SEED_DEFAULT = 20261009
# Imports of torch/numpy are delayed until inside main's error/report handling.
torch: Any = None
np: Any = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


class Log:
    def __init__(self, path: Path) -> None:
        self.handle = path.open('x', encoding='utf-8', newline='\n')

    def __call__(self, text: str) -> None:
        print(text, flush=True)
        self.handle.write(text + '\n')
        self.handle.flush()


def prepare_source(out: Path) -> tuple[Path, dict[str, Any]]:
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    archive = ROOT / manifest['archive_relpath']
    require(sha256(archive) == manifest['archive_sha256'], 'Bron-ZIP heeft een afwijkende SHA-256.')
    source = out / 'source_workcopy'
    source.mkdir()
    files: dict[str, str] = {}
    with ZipFile(archive) as z:
        for item in z.infolist():
            p = PurePosixPath(item.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in item.filename,
                    'Onveilig pad in bron-ZIP.')
            require(p.parts[0] == 'Any-Precision-DNNs-master', 'Onverwachte archiefstructuur.')
            if item.is_dir():
                continue
            relative = Path(*p.parts[1:])
            require(len(p.parts) > 1 and ':' not in str(relative), 'Ongeldig bronpad.')
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(z.read(item))
            files[relative.as_posix()] = sha256(target)
    require(files == manifest['files_sha256'], 'Bestanden wijken af van het inspectiemanifest.')
    target = source / 'utils.py'
    before = target.read_text(encoding='utf-8')
    require(before.count('import gpustat\n') == 1, 'gpustat-patch past niet exact.')
    require(before.count('    stats = gpustat.GPUStatCollection.new_query()') == 1,
            'GPU-selectiepatch past niet exact.')
    require(before.count('correct[:k].view(-1)') == 1, 'Accuracy-patch past niet exact.')
    after = before.replace('import gpustat\n', '', 1)
    after = after.replace('    stats = gpustat.GPUStatCollection.new_query()',
                          '    import gpustat  # Optional: only needed for setup_gpus().\n'
                          '    stats = gpustat.GPUStatCollection.new_query()', 1)
    after = after.replace('correct[:k].view(-1)', 'correct[:k].reshape(-1)', 1)
    # This is a newly extracted work copy, never the supplied archive or user code.
    with target.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(after)
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                   fromfile='upstream/utils.py', tofile='source_workcopy/utils.py'))
    (out / 'compatibility.patch').write_text(diff, encoding='utf-8')
    changed = [name for name, digest in files.items() if sha256(source / name) != digest]
    require(changed == ['utils.py'], 'Meer bronbestanden gewijzigd dan voorzien.')
    result = {'archive_sha256': manifest['archive_sha256'], 'changed_files': changed,
              'original_files_sha256': files,
              'workcopy_files_sha256': {n: sha256(source / n) for n in files}}
    json_write(out / 'source_manifest.json', result)
    return source, result


def import_source(source: Path) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(source))
    modules = [importlib.import_module(name) for name in
               ['models', 'models.losses', 'optimizer', 'utils']]
    for module in modules:
        require(Path(module.__file__).resolve().is_relative_to(source.resolve()),
                f'Verkeerde importlocatie voor {module.__name__}.')
    return modules[0].resnet20q, modules[1].CrossEntropyLossSoft, modules[2].get_optimizer_config, modules[3].accuracy


def configure(seed: int, device: Any, threads: int) -> dict[str, Any]:
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Only the new precision API is used; do not mix with legacy allow_tf32 flags.
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    reseed(seed)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA is niet beschikbaar; geen automatische CPU-fallback.')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    return {'seed': seed, 'torch_num_threads': threads, 'deterministic_algorithms': True,
            'deterministic_warn_only': False, 'cudnn_benchmark': False,
            'cudnn_deterministic': True, 'matmul_fp32_precision': 'ieee',
            'cudnn_conv_fp32_precision': 'ieee', 'AMP': False,
            'torch_compile': False, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}


def reseed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def differences(a: Any, b: Any, prefix: str = '') -> list[str]:
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()) else [prefix]
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [prefix + ':keys']
        return [p for k in a for p in differences(a[k], b[k], f'{prefix}/{k}')]
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return [prefix + ':length']
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differences(x, y, f'{prefix}/{i}')]
    return [] if a == b else [prefix]


def state_digest(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        h.update(name.encode('utf-8') + b'\0')
        h.update(str(tensor.dtype).encode('ascii') + b'\0')
        h.update(str(tuple(tensor.shape)).encode('ascii') + b'\0')
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def select_precision(model: Any, bit: int) -> None:
    # Same selection mechanism as upstream train.py: all modules receive both attrs.
    model.apply(lambda m: setattr(m, 'wbit', bit))
    model.apply(lambda m: setattr(m, 'abit', bit))


def finite_gradients(model: Any, require_all: bool = False) -> int:
    grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
    require(bool(grads), 'Geen gradients aangemaakt.')
    good = torch.stack([torch.isfinite(g).all() for _, g in grads]).all().item()
    require(bool(good), 'Niet-eindige gradients gevonden.')
    if require_all:
        missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        require(not missing, f'Ontbrekende gradients na alle precisies: {missing[:10]}')
    return len(grads)


def finite_state(model: Any) -> None:
    values = [v for v in model.state_dict().values() if v.is_floating_point()]
    require(bool(torch.stack([torch.isfinite(v).all() for v in values]).all().item()),
            'Niet-eindige modelparameters of BatchNorm-statistieken.')


def train_step(model: Any, optimizer: Any, x: Any, y: Any, soft_loss: Any,
               accuracy_fn: Any, step: int, trial: int, log: Log) -> list[dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=False)
    # Monitor an actual quantized convolution's shared floating-point master weight.
    watched = model.layers[0].conv0.weight
    before = watched.detach().clone()
    soft_target = None
    entries = []
    for bit in reversed(BITS):
        select_precision(model, bit)
        output = model(x)
        require(tuple(output.shape) == (x.shape[0], 10), 'Onverwachte outputvorm.')
        require(bool(torch.isfinite(output).all().item()), f'Niet-eindige uitvoer bij {bit} bit.')
        loss = torch.nn.functional.cross_entropy(output, y) if soft_target is None else soft_loss(output, soft_target)
        require(bool(torch.isfinite(loss).item()), f'Niet-eindige loss bij {bit} bit.')
        loss.backward()
        # This is accumulated gradient state, as in the supplied training algorithm.
        count = finite_gradients(model, require_all=bit == BITS[0])
        soft_target = torch.nn.functional.softmax(output.detach(), dim=1)
        # Exercise the patched metric code, but do not report random-label accuracy.
        for metric in accuracy_fn(output.detach(), y, topk=(1, 5)):
            require(math.isfinite(metric.item()) and 0 <= metric.item() <= 100,
                    'Ongeldige uitvoer van accuracy().')
        value = float(loss.detach().item())
        log(f'  herhaling {trial}, batch {step}: W{bit}/A{bit}, loss={value:.7f}, gradients=OK')
        entries.append({'batch': step, 'bits_w_and_a': bit,
                        'loss_type': 'hard_cross_entropy' if bit == 32 else 'recursive_soft_cross_entropy',
                        'loss': value, 'finite_gradients': True,
                        'parameters_with_accumulated_gradients': count})
    optimizer.step()
    finite_state(model)
    delta = float((watched.detach() - before).abs().max().item())
    require(delta > 0, 'De gecontroleerde mastergewichten zijn niet gewijzigd.')
    log(f'  optimizerstap {step}: OK; max |delta mastergewicht|={delta:.7g}')
    entries[-1]['watched_weight_max_abs_update'] = delta
    return entries


def evaluate(model: Any, x: Any, y: Any, accuracy_fn: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    outputs, entries = {}, []
    with torch.no_grad():
        for bit in BITS:
            select_precision(model, bit)
            out = model(x)
            require(bool(torch.isfinite(out).all().item()), 'Niet-eindige evaluatie-uitvoer.')
            loss = float(torch.nn.functional.cross_entropy(out, y).item())
            require(math.isfinite(loss), 'Niet-eindige evaluatieloss.')
            for metric in accuracy_fn(out, y, topk=(1, 5)):
                require(math.isfinite(metric.item()), 'Niet-eindige metric.')
            outputs[str(bit)] = out.detach().cpu().clone()
            entries.append({'bits_w_and_a': bit, 'synthetic_cross_entropy': loss, 'finite_output': True})
    return outputs, entries


def bn_check(model: Any, steps: int) -> dict[str, Any]:
    counts = {name: int(value.item()) for name, value in model.state_dict().items()
              if name.endswith('num_batches_tracked')}
    require(bool(counts), 'Geen BatchNorm-tellers gevonden.')
    for name, value in counts.items():
        expected = steps if '.bn_dict.' in name else steps * len(BITS)
        require(value == expected, f'BatchNorm-teller {name}: {value}, verwacht {expected}.')
    return {'all_counters_correct': True, 'counts': counts,
            'note': 'Switch-BN: eenmaal per batch per precisie. Gedeelde skip-BN: vijfmaal per batch.'}


def run_test(args: Any, out: Path, report: dict[str, Any], log: Log) -> None:
    global torch, np
    log('ANY-PRECISION TECHNISCHE UITVOERTEST - geen referentietraining')
    source, source_info = prepare_source(out)
    report['source'] = source_info
    log('Bron-ZIP: gecontroleerd; alleen utils.py aangepast in een nieuwe werkkopie.')
    # Set before importing/initializing CUDA. These settings affect this process only.
    report['previous_CUBLAS_WORKSPACE_CONFIG'] = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch as torch_import
    import numpy as numpy_import
    torch, np = torch_import, numpy_import
    require(str(torch.__version__).split('+')[0].startswith('2.10.'),
            f'Deze test is gericht op PyTorch 2.10.x; gevonden {torch.__version__}.')
    device = torch.device('cuda:0' if args.device == 'cuda' else args.device)
    report['environment'] = {
        'python': sys.version, 'interpreter': sys.executable, 'platform': platform.platform(),
        'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'device_requested': str(device),
        'torchvision': importlib.metadata.version('torchvision')}
    report['execution_settings'] = configure(args.seed, device, args.threads)
    report['config'] = {'model': 'resnet20q', 'expand': 5, 'classes': 10, 'bits': BITS,
                        'batch_size': args.batch_size, 'batches_per_repetition': args.steps,
                        'fresh_repetitions': 2, 'optimizer': 'Adam', 'lr': 0.001, 'weight_decay': 0.0,
                        'scheduler': None, 'input': 'synthetic N(0,1), NCHW float32, 3x32x32',
                        'labels': 'synthetic uniform integers 0..9, independent of inputs',
                        'synthetic_data_seed': args.seed + 1, 'model_seed': args.seed,
                        'dataset_download': False, 'original_train_py_called': False,
                        'original_cifar10_loader_called': False, 'data_loader_workers': 0}
    if device.type == 'cuda':
        report['environment']['device_name'] = torch.cuda.get_device_name(device)
    log(f"Apparaat: {report['environment'].get('device_name', str(device))}")
    log(f'Model: resnet20q, expand=5; {args.steps} batches van {args.batch_size}; twee identieke herhalingen.')
    log('Alle verliezen zijn van synthetische data; geen accuracy-resultaat.')
    make_model, SoftCE, get_optimizer, accuracy_fn = import_source(source)
    # Known-answer test for the .reshape compatibility fix (not model accuracy).
    scores = torch.tensor([[6., 5., 4., 3., 2., 1.], [6., 5., 4., 3., 2., 1.]], device=device)
    expected = accuracy_fn(scores, torch.tensor([0, 1], device=device), topk=(1, 5))
    require([float(v.item()) for v in expected] == [50., 100.], 'Accuracy-unit-test mislukt.')
    report['checks'] = {'accuracy_helper_known_answer': True}
    gen = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    data = {'inputs': torch.randn(args.steps * args.batch_size, 3, 32, 32, generator=gen),
            'targets': torch.randint(0, 10, (args.steps * args.batch_size,), generator=gen),
            'eval_inputs': torch.randn(args.batch_size, 3, 32, 32, generator=gen),
            'eval_targets': torch.randint(0, 10, (args.batch_size,), generator=gen)}
    with (out / 'synthetic_inputs.pt').open('xb') as handle:
        torch.save(data, handle)
    report['synthetic_inputs_sha256'] = sha256(out / 'synthetic_inputs.pt')
    baseline = None
    report['trials'] = []
    for trial in [1, 2]:
        reseed(args.seed)
        model = make_model(BITS).to(device)
        optimizer = get_optimizer(model, 'adam', 0.001, 0.0)
        soft_loss = SoftCE()
        require(model.expand == 5, 'Modelbreedte is onverwacht gewijzigd.')
        require(all(p.dtype == torch.float32 for p in model.parameters()), 'Mastergewichten zijn niet FP32.')
        parameter_count = sum(p.numel() for p in model.parameters())
        if trial == 1:
            report['parameter_count'] = parameter_count
            log(f'Parameters (inclusief alle BatchNorm-banken): {parameter_count}')
        initial_hash = state_digest(model.state_dict())
        dataset = torch.utils.data.TensorDataset(data['inputs'], data['targets'])
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, generator=torch.Generator().manual_seed(args.seed + 2))
        train_records: list[dict[str, Any]] = []
        for step, (x, y) in enumerate(loader, start=1):
            train_records.extend(train_step(model, optimizer, x.to(device), y.to(device),
                                            soft_loss, accuracy_fn, step, trial, log))
        bn_result = bn_check(model, args.steps)
        model_state = cpu_copy(model.state_dict())
        opt_state = cpu_copy(optimizer.state_dict())
        outputs, eval_records = evaluate(model, data['eval_inputs'].to(device), data['eval_targets'].to(device), accuracy_fn)
        unchanged = not differences(model_state, cpu_copy(model.state_dict()))
        require(unchanged, 'Evaluatie wijzigde opgeslagen modeltoestand.')
        trial_info = {'trial': trial, 'initial_state_sha256': initial_hash,
                      'final_state_sha256': state_digest(model_state), 'train': train_records,
                      'eval': eval_records, 'batchnorm': bn_result, 'eval_state_unchanged': unchanged}
        report['trials'].append(trial_info)
        log(f'Herhaling {trial}: training en evaluatie op alle precisies OK; BatchNorm-tellers OK.')
        if trial == 1:
            baseline = {'model': model_state, 'optimizer': opt_state, 'outputs': outputs,
                        'train': train_records, 'eval': eval_records, 'initial_hash': initial_hash}
            payload = {'purpose': 'synthetic smoke test, not research model',
                       'config': report['config'], 'state_dict': model_state,
                       'optimizer': opt_state, 'eval_logits': outputs, 'seed': args.seed,
                       'completed_steps': args.steps}
            with (out / 'smoke_checkpoint.pt').open('xb') as handle:
                torch.save(payload, handle)
        else:
            tests = {'same_initial_state': initial_hash == baseline['initial_hash'],
                     'same_final_model_state': not differences(baseline['model'], model_state),
                     'same_final_optimizer_state': not differences(baseline['optimizer'], opt_state),
                     'same_eval_logits': not differences(baseline['outputs'], outputs),
                     'same_train_records': not differences(baseline['train'], train_records),
                     'same_eval_records': not differences(baseline['eval'], eval_records)}
            report['repeatability'] = tests
            report['repeatability']['differing_model_keys'] = differences(baseline['model'], model_state)[:20]
        del model, optimizer, soft_loss, model_state, opt_state, outputs
        gc.collect()
    log('Checkpoint: opgeslagen bestand opnieuw laden in een nieuw model en nieuwe optimizer.')
    # Only reload a checkpoint created by this invocation; no external pickle input.
    loaded = torch.load(out / 'smoke_checkpoint.pt', map_location='cpu', weights_only=True)
    restored = make_model(BITS).to(device)
    restored_optimizer = get_optimizer(restored, 'adam', 0.001, 0.0)
    restored.load_state_dict(loaded['state_dict'], strict=True)
    restored_optimizer.load_state_dict(loaded['optimizer'])
    restored_outputs, _ = evaluate(restored, data['eval_inputs'].to(device), data['eval_targets'].to(device), accuracy_fn)
    report['checkpoint'] = {
        'model_state_identical': not differences(baseline['model'], cpu_copy(restored.state_dict())),
        'optimizer_state_identical': not differences(baseline['optimizer'], cpu_copy(restored_optimizer.state_dict())),
        'eval_logits_identical': not differences(baseline['outputs'], restored_outputs),
        'weights_only': True, 'complete_training_resume_tested': False,
        'note': 'State/optimizer/logits round-trip; geen extra trainingsstap na laden, geen scheduler of data-cursor herstel.'}
    report['checks'].update({'all_precisions_forward_backward': True, 'finite_values_and_gradients': True,
                             'master_weights_updated_each_step': True, 'eval_state_unchanged': True,
                             'batchnorm_counters_correct': True})
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        report['cuda_peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 1024**2
        report['cuda_peak_reserved_MiB'] = torch.cuda.max_memory_reserved(device) / 1024**2
    for field in ['model_state_identical', 'optimizer_state_identical', 'eval_logits_identical']:
        require(report['checkpoint'][field], f'Checkpointcontrole mislukt: {field}.')
    for field, ok in report['repeatability'].items():
        if field != 'differing_model_keys':
            require(ok, f'Herhaalbaarheidscontrole mislukt: {field}.')
    # Verify code was not modified by any imported code/run.
    require(all(sha256(source / name) == digest for name, digest in source_info['workcopy_files_sha256'].items()),
            'Bronwerkkopie is gewijzigd tijdens de test.')
    require(sha256(ROOT / 'upstream' / 'Any-Precision-DNNs-master.zip') == source_info['archive_sha256'],
            'Origineel bronarchief gewijzigd tijdens de test.')
    report['checks']['source_unchanged_during_execution'] = True
    log('Herhaalbaarheid: identieke modeltoestand, optimizertoestand, losses en evaluatie-uitvoer.')
    log('Checkpoint: identieke modeltoestand, optimizertoestand en evaluatie-uitvoer na laden.')
    report['status'] = 'PASS'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output-root', type=Path, default=Path('technical_checks') / 'anyprecision_smoke')
    parser.add_argument('--seed', type=int, default=SEED_DEFAULT)
    parser.add_argument('--batch-size', type=int, default=2, choices=range(2, 9))
    parser.add_argument('--steps', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--threads', type=int, default=2, choices=range(1, 9))
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 - 2:
        parser.error('--seed moet tussen 0 en 2**32-3 liggen.')
    out = args.output_root.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    log = Log(out / 'run.log')
    report: dict[str, Any] = {'schema_version': 1, 'status': 'RUNNING',
        'purpose': 'technical synthetic Any-Precision smoke test; no research accuracy',
        'started_at': datetime.now().astimezone().isoformat(), 'run_directory': str(out),
        'runner_sha256': sha256(Path(__file__)), 'argv': sys.argv,
        'limitations': ['No CIFAR-10 dataset/download/augmentation test.',
                        'Original train.py launcher not executed.',
                        'No MX, truncation-from-stored-codes, continual learning or hardware benefit test.',
                        'Repeatability checked within one process/device, not between platforms.',
                        'Checkpoint round-trip is not a complete training-resume validation.']}
    started = time.perf_counter()
    # Save the exact driver script as part of each immutable run record.
    (out / 'runner_used.py').write_bytes(Path(__file__).read_bytes())
    caught: list[Any] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            run_test(args, out, report, log)
    except (Exception, KeyboardInterrupt) as exc:
        report['status'] = 'FAIL'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        log('FOUT: ' + str(exc))
        log(report['error']['traceback'])
    finally:
        report['elapsed_seconds_diagnostic_only'] = time.perf_counter() - started
        report['warnings'] = [{'category': w.category.__name__, 'message': str(w.message),
                               'filename': w.filename, 'line': w.lineno} for w in caught]
        try:
            pip = subprocess.run([sys.executable, '-B', '-m', 'pip', 'freeze', '--all'],
                                 capture_output=True, text=True, timeout=60, check=False)
            (out / 'pip_freeze.txt').write_text(pip.stdout, encoding='utf-8')
            report['pip_freeze'] = {'returncode': pip.returncode, 'stderr': pip.stderr}
        except Exception as exc:
            report['pip_freeze'] = {'error': str(exc)}
        report['ended_at'] = datetime.now().astimezone().isoformat()
        json_write(out / 'smoke_report.json', report)
        summary = '\n'.join([
            'ANY-PRECISION TECHNISCHE UITVOERTEST', 'STATUS: ' + report['status'],
            f'Apparaat: {args.device}', f'Batchgrootte: {args.batch_size}',
            f'Batches per herhaling: {args.steps}; herhalingen: 2',
            f'Waarschuwingen: {len(report["warnings"])}',
            'Geen datasetaccuracy, referentietraining, MX of hardwaremeting.',
            'Herhaalbaarheid: binnen dezelfde proces-, apparaat- en softwareconfiguratie.',
            f'Rapport: {out / "smoke_report.json"}',
            f'Fout: {report.get("error", {}).get("message", "geen")}', ''])
        (out / 'summary.txt').write_text(summary, encoding='utf-8')
        log(f'Waarschuwingen: {len(report["warnings"])} (details in smoke_report.json).')
        log('STATUS: ' + report['status'])
        log('Rapportmap: ' + str(out))
        log('Geen CIFAR-10 gedownload. Geen pakketten geinstalleerd. Geen MX toegevoegd.')
        log.handle.close()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
