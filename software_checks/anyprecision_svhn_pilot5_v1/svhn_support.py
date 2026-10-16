"""Source-faithful SVHN adapter and read-only instrumentation.

The upstream forward() function is parsed from the supplied train.py, not
rewritten. Its AST and all model/dataset/optimizer files remain unchanged.
Only utils.py receives the pre-existing compatibility patch (see manifest).
"""
from __future__ import annotations

import ast
import csv
from datetime import datetime
import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen
import uuid

import smoke_support as legacy

ROOT = Path(__file__).resolve().parent
BITS = [1, 2, 4, 8, 32]
require = legacy.require
write_json = legacy.json_write


def now() -> str:
    return datetime.now().astimezone().isoformat()


def hashes(path: Path) -> dict[str, Any]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            md5.update(chunk)
            sha.update(chunk)
    return {'md5': md5.hexdigest(), 'sha256': sha.hexdigest(), 'bytes': path.stat().st_size}


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), 'Geen rijen om op te slaan: ' + str(path))
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_package() -> dict:
    manifest = json.loads((ROOT / 'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        require(legacy.sha256(ROOT / name) == expected, 'Pakketbestand gewijzigd: ' + name)
    return manifest


def import_source(source: Path):
    sys.path.insert(0, str(source))
    modules = [importlib.import_module(n) for n in
               ['models', 'models.losses', 'optimizer', 'datasets.data', 'utils']]
    for module in modules:
        require(Path(module.__file__).resolve().is_relative_to(source.resolve()),
                'Onjuiste bronmodule: ' + module.__name__)
    return SimpleNamespace(models=modules[0], losses=modules[1], optimizer=modules[2],
                           data=modules[3], utils=modules[4])


def runtime(seed: int, device_name: str, strict: bool = True):
    import torch
    import numpy as np
    import scipy.io
    import torchvision
    # Import resume_core only after successful dependency checks.
    import resume_core as core
    if strict:
        require(sys.version_info[:2] == (3, 13), 'Gebruik de bestaande Python 3.13 .venv-train.')
        require(str(torch.__version__).split('+')[0] == '2.10.0', 'Verwacht PyTorch 2.10.0; niets upgraden.')
        require(str(torchvision.__version__).split('+')[0] == '0.25.0', 'Verwacht torchvision 0.25.0.')
        require(device_name.startswith('cuda'), 'Productierun vereist CUDA; geen stille CPU-fallback.')
        require(torch.cuda.is_available(), 'CUDA ontbreekt. Controleer .venv-train; niets upgraden.')
    device = torch.device("cuda:0" if device_name == "cuda" else device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    # Match the supplied train.py. Do NOT call legacy.configure(): that would
    # switch to deterministic kernels and disable the original benchmark policy.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    settings = {
        'seed': seed, 'seed_source': 'explicit local seed, not reported by authors',
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'torch_num_threads': torch.get_num_threads(),
        'matmul_fp32_precision': str(torch.backends.cuda.matmul.fp32_precision),
        'cudnn_conv_fp32_precision': str(torch.backends.cudnn.conv.fp32_precision),
        'precision_backend_policy': 'installed PyTorch defaults; no TF32 override',
        'AMP': False, 'torch_compile': False,
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'exact_cuda_resume_guaranteed': False,
    }
    environment = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
        'numpy': str(np.__version__), 'scipy': metadata.version('scipy'),
        'pillow': metadata.version('Pillow'), 'cuda_build': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
    }
    legacy.torch, legacy.np = torch, np
    return SimpleNamespace(torch=torch, np=np, core=core, device=device,
                           settings=settings, environment=environment)


def ensure_dataset(cache: Path, dataset_class, download: bool, log) -> dict:
    """Never overwrite raw files. Downloads are staged and MD5-verified first.

    Try the archived official URL, then HTTPS on the same host when HTTP fails.
    No alternate dataset, mirror, skipped examples or certificate bypass.
    """
    cache.mkdir(parents=True, exist_ok=True)
    result = {'root': str(cache), 'files': {}, 'download_events': []}
    for split in ['train', 'extra', 'test']:
        url, filename, expected = dataset_class.split_list[split]
        target = cache / filename
        if target.exists():
            info = hashes(target)
            require(info['md5'] == expected,
                    'Bestaand databestand faalt MD5 en blijft ongewijzigd: ' + str(target))
            log('DATA bestaand en correct: ' + filename)
        else:
            require(download, 'Datasetbestand ontbreekt: ' + str(target) + '. Gebruik --download.')
            stage = cache / ('.download_' + uuid.uuid4().hex)
            stage.mkdir()
            candidates = [url]
            if url.startswith('http://'):
                candidates.append('https://' + url[len('http://'):])
            success = None
            for attempt, candidate in enumerate(candidates, 1):
                partial = stage / (filename + f'.attempt{attempt}.part')
                event = {'split': split, 'requested_url': candidate, 'started_at': now(),
                         'partial_file': str(partial)}
                try:
                    log('DOWNLOAD ' + candidate)
                    request = Request(candidate, headers={'User-Agent': 'AnyPrecision-SVHN-research/1.0'})
                    with urlopen(request, timeout=90) as response, partial.open('xb') as f:
                        event['response_url'] = response.geturl()
                        count, last_message = 0, time.perf_counter()
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            f.write(block)
                            count += len(block)
                            if time.perf_counter() - last_message > 15:
                                log(f'  {filename}: {count / 1024**2:.1f} MiB ontvangen')
                                last_message = time.perf_counter()
                        f.flush()
                        os.fsync(f.fileno())
                    info = hashes(partial)
                    require(info['md5'] == expected, 'Download faalt de MD5 uit de auteurscode.')
                    # Exclusive target creation prevents overwriting even a file
                    # created by another process while this download was active.
                    with partial.open('rb') as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                        dst.flush()
                        os.fsync(dst.fileno())
                    require(hashes(target) == info, 'Kopie van databestand wijkt af.')
                    partial.unlink()  # Delete only our verified staging copy.
                    event.update(status='PASS', ended_at=now(), hashes=info)
                    success = info
                    result['download_events'].append(event)
                    break
                except Exception as exc:
                    event.update(status='FAIL', ended_at=now(), error=repr(exc))
                    result['download_events'].append(event)
                    log('DOWNLOAD-WAARSCHUWING: ' + repr(exc))
                    # Never retry into or overwrite a partially published target.
                    if target.exists():
                        write_json(stage / 'download_report.json', result['download_events'])
                        raise
            write_json(stage / 'download_report.json', result['download_events'])
            require(success is not None,
                    'Download niet voltooid; geen data vervangen. Zie ' + str(stage))
            info = success
        result['files'][filename] = {'split': split, **info}
    return result


def validate_dataset(data, expected: int, name: str) -> dict:
    import numpy as np
    require(len(data) == expected, f'{name}: {len(data)} voorbeelden, verwacht {expected}. Geen subset toegestaan.')
    require(data.data.shape == (expected, 3, 32, 32), name + ': onverwachte ruwe beeldvorm.')
    require(data.data.dtype == np.uint8, name + ': ruwe beelden zijn niet uint8.')
    require(data.labels.shape == (expected,), name + ': onjuiste labels.')
    require(data.labels.dtype == np.int64, name + ': labels zijn niet int64.')
    require(bool(np.isin(data.labels, np.arange(10)).all()), name + ': ongeldige klassen.')
    return {'n': expected, 'labels_histogram': np.bincount(data.labels, minlength=10).tolist(),
            'label_sha256': hashlib.sha256(data.labels.tobytes()).hexdigest()}


class IndexedDataset:
    """Adds an index for coverage logging; image/label/transform are unchanged."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index


class TrackedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.order = []
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        self.order = []
        for x, y, indices in self.loader:
            require(tuple(x.shape[1:]) == (3, 40, 40), 'Verwacht invoer 3x40x40.')
            self.order.extend(indices.tolist())
            yield x, y
    def validate(self, output: Path) -> dict:
        import numpy as np
        order = np.asarray(self.order, dtype=np.int32)
        n = len(self.loader.dataset)
        require(len(order) == n and np.array_equal(np.sort(order), np.arange(n)),
                'Epoch bevat niet exact eenmaal alle voorbeelden.')
        with output.open('xb') as f:
            np.save(f, order, allow_pickle=False)
        return {'n': n, 'order_sha256': hashlib.sha256(order.tobytes()).hexdigest(),
                'order_file': str(output), 'complete_once': True}


class MeterRecorder:
    """Read-only subclass of upstream AverageMeter, preserving its exact averages.

    Records loss and top-1/top-5 per minibatch at all five precisions. It does not
    recompute a loss, call the model, draw random values, or alter gradients.
    """
    def __init__(self, base_class, epoch: int, phase: str, lr: float, path: Path):
        self.base_class, self.epoch, self.phase, self.lr = base_class, epoch, phase, lr
        self.file = path.open('x', encoding='utf-8')
        self.meters, self.pending, self.batch = [], {}, 0
        self.correct1, self.correct5, self.n = [0] * 5, [0] * 5, 0
        recorder = self
        class RecordedMeter(base_class):
            def __init__(self):
                super().__init__()
                self.position = len(recorder.meters)
                recorder.meters.append(self)
            def update(self, val, n=1):
                require(math.isfinite(float(val)), 'Niet-eindige metric; run wordt gestopt.')
                super().update(val, n)
                recorder.updated(self.position, float(val), int(n))
        self.factory = RecordedMeter
    def updated(self, position: int, val: float, n: int):
        require(position not in self.pending, 'Metric twee keer in dezelfde minibatch.')
        self.pending[position] = (val, n)
        if len(self.pending) != 15:
            return
        require(len({v[1] for v in self.pending.values()}) == 1, 'Batchgroottes van metrics verschillen.')
        self.batch += 1
        self.n += n
        records = []
        for i, bit in enumerate(BITS):
            loss, top1, top5 = (self.pending[i + offset][0] for offset in [0, 5, 10])
            c1, c5 = round(top1 * n / 100), round(top5 * n / 100)
            require(abs(top1 * n / 100 - c1) < .001, 'Top-1-score kan niet naar aantal worden herleid.')
            self.correct1[i] += c1
            self.correct5[i] += c5
            records.append({'bits_w_and_a': bit, 'objective_loss': loss,
                            'top1_percent': top1, 'top5_percent': top5,
                            'correct1': c1, 'correct5': c5})
        self.file.write(json.dumps({'epoch': self.epoch, 'phase': self.phase,
                                   'batch': self.batch, 'n': n, 'lr': self.lr,
                                   'precisions': records}, allow_nan=False) + '\n')
        if self.batch % 20 == 0:
            self.file.flush()
        self.pending.clear()
    def rows(self, values) -> list[dict]:
        require(not self.pending and len(self.meters) == 15 and self.batch > 0, 'Onvolledige metrics.')
        losses, top1, top5 = values
        return [
            {'epoch': self.epoch, 'phase': self.phase, 'bits_w_and_a': bit, 'n': self.n,
             'objective_type': ('recursive_soft_cross_entropy' if self.phase == 'train' and bit != 32
                                else 'hard_cross_entropy'),
             'objective_loss': float(losses[i]), 'top1_percent': float(top1[i]),
             'top5_percent': float(top5[i]), 'correct1': self.correct1[i], 'correct5': self.correct5[i],
             'count_top1_percent': 100 * self.correct1[i] / self.n, 'lr': self.lr}
            for i, bit in enumerate(BITS)]
    def close(self):
        self.file.flush()
        self.file.close()


class UpstreamLog:
    def __init__(self, log):
        self.log = log
    def info(self, text, *args):
        self.log(('UPSTREAM ' + text) % args if args else 'UPSTREAM ' + text)


def load_forward(source: Path, utils, log, meter_class=None):
    """Load just the unmodified original function (no upstream CLI main/imports)."""
    import torch
    filename = source / 'train.py'
    tree = ast.parse(filename.read_text(encoding='utf-8'), filename=str(filename))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'forward']
    require(len(functions) == 1, 'Precies een upstream forward() verwacht.')
    node = functions[0]
    namespace = {'torch': torch, 'args': SimpleNamespace(bit_width_list='1,2,4,8,32', print_freq=20),
                 'AverageMeter': meter_class or utils.AverageMeter, 'accuracy': utils.accuracy,
                 'logging': UpstreamLog(log)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(filename), 'exec'), namespace)
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    return namespace['forward'], digest


def phase(rt, source, modules, model, optimizer, loader, epoch, phase_name, lr, out, log):
    torch, device = rt.torch, rt.device
    train = phase_name == 'train'
    model.train(train)
    recorder = MeterRecorder(modules.utils.AverageMeter, epoch, phase_name, lr,
                             out / f'{phase_name}_batches_epoch_{epoch:05d}.jsonl')
    forward, ast_hash = load_forward(source, modules.utils, log, recorder.factory)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        values = forward(loader, model, torch.nn.CrossEntropyLoss().to(device),
                         modules.losses.CrossEntropyLossSoft().to(device), epoch - 1,
                         training=train, optimizer=optimizer if train else None, sum_writer=None)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        rows = recorder.rows(values)
    finally:
        recorder.close()
    return rows, elapsed, ast_hash


def best_record(records: list[dict]) -> dict:
    require(bool(records), 'Geen afgeronde epochs voor selectie.')
    best = None
    for record in records:
        score = record['test'][-1]['top1_percent']
        require(record['test'][-1]['bits_w_and_a'] == 32, 'W32/A32 staat niet op laatste positie.')
        # Exactly upstream: first epoch wins; replace only on STRICT improvement.
        if best is None or score > best['test'][-1]['top1_percent']:
            best = record
    return best


def estimate_time(records: list[dict], total_epochs: int = 100) -> dict:
    measured = [r for r in records if 'total_epoch_seconds' in r]
    steady = [r['total_epoch_seconds'] for r in measured if not r['first_epoch_in_process']]
    if not steady:
        return {'available': False, 'reason': 'Nog geen volledig gemeten warme epoch.',
                'excludes': ['download', 'dependency installation', 'dataset loading', 'final report export']}
    cold = next((r['total_epoch_seconds'] for r in measured if r['epoch'] == 1), statistics.median(steady))
    return {'available': True, 'planned_epochs': total_epochs,
            'steady_epoch_seconds': steady, 'steady_epochs': [r['epoch'] for r in measured if not r['first_epoch_in_process']],
            'median_epoch_seconds': statistics.median(steady), 'first_epoch_seconds': cold,
            'estimated_100_epoch_seconds': cold + (total_epochs - 1) * statistics.median(steady),
            'observed_range_projection_seconds': [cold + (total_epochs - 1) * min(steady),
                                                   cold + (total_epochs - 1) * max(steady)],
            'range_is_confidence_interval': False,
            'method': 'first measured epoch + 99 * median of non-first-process epochs; includes train, test and checkpoint',
            'excludes': ['download', 'dependency installation', 'dataset loading', 'final report export'],
            'limitations': ['same laptop/backend/data loading conditions required',
                           'five epochs do not measure the LR transitions at 50/75/90',
                           'thermal and background-load drift over a long run not established']}
