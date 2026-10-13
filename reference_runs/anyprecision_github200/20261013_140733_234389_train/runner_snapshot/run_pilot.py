"""Small, exploratory CIFAR-10 training pilot using the supplied Any-Precision code.

Default protocol: 5,000 train / 1,000 validation images, three epochs, batch 32.
Official test set is downloaded/checksummed but never used for model evaluation.
No MX, stored-code truncation, pretrained input, resume, or package installation.
See PROTOCOL.md and CHANGELOG.md before interpreting any result.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
import warnings
from datetime import datetime
from typing import Any, Callable

sys.dont_write_bytecode = True
import smoke_support as support

ROOT = Path(__file__).resolve().parent
BITS = [1, 2, 4, 8, 32]
SEED = 20261009
PROFILES = {
    'pilot': dict(train_per_class=500, val_per_class=100, epochs=3, batch_size=32),
    'local-check': dict(train_per_class=2, val_per_class=2, epochs=1, batch_size=2),
}
# CPU checks use the same model and training path, with a deliberately tiny dataset.
torch: Any = None
np: Any = None
require = support.require
sha256 = support.sha256
json_write = support.json_write


def hash_file(path: Path, algorithm: str = 'sha256') -> str:
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_source(out: Path) -> tuple[Path, dict[str, Any]]:
    """Reuse the verified smoke preparation, then make SVHN an optional import."""
    source, info = support.prepare_source(out)
    # Preserve the intermediate audit rather than leaving it as a final manifest.
    (out / 'source_manifest.json').rename(out / 'source_manifest_model_stage.json')
    (out / 'compatibility.patch').rename(out / 'compatibility_model.patch')
    target = source / 'datasets' / 'data.py'
    before = target.read_text(encoding='utf-8')
    old = "    elif name == 'svhn':\n        if split == 'val': split = 'test'"
    require(before.count('from .svhn import SVHNMIX\n') == 1, 'Onverwachte SVHN-import.')
    require(before.count(old) == 1, 'SVHN-patch past niet exact.')
    after = before.replace('from .svhn import SVHNMIX\n', '', 1).replace(
        old, "    elif name == 'svhn':\n        from .svhn import SVHNMIX  # Only needed for SVHN.\n"
             "        if split == 'val': split = 'test'", 1)
    target.write_text(after, encoding='utf-8', newline='\n')
    data_patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='upstream/datasets/data.py', tofile='source_workcopy/datasets/data.py'))
    (out / 'compatibility.patch').write_text(
        (out / 'compatibility_model.patch').read_text(encoding='utf-8') + data_patch,
        encoding='utf-8', newline='\n')
    info['workcopy_files_sha256'] = {n: sha256(source / n) for n in info['original_files_sha256']}
    info['changed_files'] = sorted(n for n in info['original_files_sha256']
        if info['original_files_sha256'][n] != info['workcopy_files_sha256'][n])
    require(info['changed_files'] == ['datasets/data.py', 'utils.py'], 'Onbedoelde bronwijziging.')
    json_write(out / 'source_manifest.json', info)
    return source, info


def ensure_dataset(cache: Path, dataset_class: Any, download: bool,
                   log: Callable[[str], None]) -> dict[str, Any]:
    """Verify before reuse. Download/extract into a NEW staging directory only.

    Never redownload over existing raw data. Only the six named regular batch
    files are extracted, without calling tar.extractall() or the old downloader.
    Pickle decoding is left to the unchanged upstream CIFAR10 class, AFTER checks.
    """
    entries = dataset_class.train_list + dataset_class.test_list
    target = cache / dataset_class.base_folder

    def verified(folder: Path) -> dict[str, Any]:
        result = {}
        for name, expected in entries:
            f = folder / name
            require(f.is_file() and not f.is_symlink(), f'Ontbrekend/onveilig databestand: {f}')
            actual = hash_file(f, 'md5')
            require(actual == expected, f'Checksumfout: {f}; niets wordt overschreven.')
            result[name] = {'md5': actual, 'sha256': hash_file(f), 'bytes': f.stat().st_size}
        return result

    if target.exists():
        info = verified(target)
        log('CIFAR-10: bestaande batchbestanden gecontroleerd; geen download.')
        return {'root': str(cache), 'new_download': False, 'batch_files': info}
    require(download, 'CIFAR-10 ontbreekt. Gebruik --download om de officiele dataset op te halen.')
    cache.mkdir(parents=True, exist_ok=True)
    staging = cache / ('download_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    staging.mkdir(exist_ok=False)
    archive = staging / dataset_class.filename
    # Use the exact official URL from the supplied CIFAR10 class. TLS stays enabled.
    request = urllib.request.Request(dataset_class.url, headers={'User-Agent': 'Python-AnyPrecision-Pilot/1.0'})
    log('CIFAR-10: downloaden van de officiele bron; ongeveer 163 MiB.')
    downloaded = 0
    next_progress = 25 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=120) as response, archive.open('xb') as f:
        final_url = response.geturl()
        require(final_url.startswith('https://'), 'Download omgeleid naar niet-HTTPS; gestopt.')
        while block := response.read(1024 * 1024):
            downloaded += len(block)
            require(downloaded <= 200 * 1024 * 1024, 'Onverwacht groot datasetarchief.')
            f.write(block)
            if downloaded >= next_progress:
                log(f'  ontvangen: {downloaded / 1024**2:.0f} MiB')
                next_progress += 25 * 1024 * 1024
    require(hash_file(archive, 'md5') == dataset_class.tgz_md5,
            f'Archiefchecksum wijkt af. Bestand bewaard in {staging}; niets overschreven.')
    extracted = staging / dataset_class.base_folder
    extracted.mkdir()
    with tarfile.open(archive, 'r:gz') as tar:
        for name, _ in entries:
            member_name = dataset_class.base_folder + '/' + name
            members = [m for m in tar.getmembers() if m.name == member_name]
            require(len(members) == 1, f'Ontbrekende/dubbele archiefentry: {member_name}')
            member = members[0]
            require(member.isfile() and 0 < member.size < 40 * 1024**2,
                    f'Geen normaal CIFAR-batchbestand: {member_name}')
            inp = tar.extractfile(member)
            require(inp is not None, f'Kan archiefentry niet lezen: {member_name}')
            with inp, (extracted / name).open('xb') as f:
                shutil.copyfileobj(inp, f)
    info = verified(extracted)
    require(not target.exists(), 'Doeldata verschenen tijdens download; niets wordt overschreven.')
    extracted.rename(target)
    log('CIFAR-10: archief en alle zes batchbestanden gecontroleerd.')
    return {'root': str(cache), 'new_download': True, 'requested_url': dataset_class.url,
            'final_url': final_url, 'archive_path': str(archive),
            'archive_md5': dataset_class.tgz_md5, 'archive_sha256': hash_file(archive),
            'batch_files': info}


def make_split(labels: list[int], train_pc: int, val_pc: int) -> dict[str, Any]:
    """Permanent 45k/5k stratified split, with nested, disjoint pilot subsets."""
    labels_np = np.asarray(labels, dtype=np.int64)
    require(labels_np.shape == (50000,), 'Verwacht 50.000 originele trainingslabels.')
    require(np.array_equal(np.bincount(labels_np, minlength=10), np.full(10, 5000)),
            'Onverwachte CIFAR-10-klassenverdeling.')
    require(1 <= train_pc <= 4500 and 1 <= val_pc <= 500, 'Ongeldige subsetgrootte.')
    rng = np.random.default_rng(SEED)
    train_pool, val_pool, train_idx, val_idx = [], [], [], []
    for label in range(10):
        perm = rng.permutation(np.flatnonzero(labels_np == label)).tolist()
        val_pool.extend(perm[:500])
        train_pool.extend(perm[500:])
        val_idx.extend(perm[:val_pc])
        train_idx.extend(perm[500:500 + train_pc])
    require(not (set(train_pool) & set(val_pool)), 'Overlap tussen train- en validatiepool.')
    require(len(set(train_pool) | set(val_pool)) == 50000, 'Onvolledige splitsing.')
    require(set(train_idx) <= set(train_pool) and set(val_idx) <= set(val_pool), 'Subsetfout.')
    return {'source_split': 'official CIFAR-10 training set, original concatenation order',
            'seed': SEED, 'rng': 'numpy.random.default_rng(PCG64)',
            'full_train_pool': sorted(train_pool), 'full_validation_pool': sorted(val_pool),
            'pilot_train_indices': sorted(train_idx), 'pilot_validation_indices': sorted(val_idx),
            'unused_train_pool_in_this_run': 45000 - len(train_idx),
            'unused_validation_pool_in_this_run': 5000 - len(val_idx),
            'official_test_examples_used': 0,
            'train_class_counts': np.bincount(labels_np[train_idx], minlength=10).tolist(),
            'validation_class_counts': np.bincount(labels_np[val_idx], minlength=10).tolist()}


class IndexedSubset:
    """Keep original full-training-set indices alongside upstream image/label pairs."""
    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self.dataset, self.indices = dataset, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[Any, Any, int]:
        original_index = self.indices[i]
        x, y = self.dataset[original_index]
        return x, y, original_index


def train_batch(model: Any, optimizer: Any, x: Any, y: Any, soft_ce: Any) -> list[dict[str, Any]]:
    """Same recursive target order and summed gradients as upstream train.py."""
    model.train()
    optimizer.zero_grad(set_to_none=False)
    watched = model.layers[0].conv0.weight
    before = watched.detach().clone()
    teacher = None
    records = []
    for bit in reversed(BITS):
        support.select_precision(model, bit)
        output = model(x)
        require(tuple(output.shape) == (len(y), 10), 'Onverwachte outputvorm.')
        require(bool(torch.isfinite(output).all().item()), f'Niet-eindige output: W{bit}/A{bit}.')
        hard = torch.nn.functional.cross_entropy(output, y)
        loss = hard if teacher is None else soft_ce(output, teacher)
        require(bool(torch.isfinite(loss).item()), 'Niet-eindige trainingsloss.')
        loss.backward()
        support.finite_gradients(model, require_all=bit == 1)
        teacher = torch.nn.functional.softmax(output.detach(), dim=1)
        top5 = output.detach().topk(5, dim=1).indices
        records.append({'bits_w_and_a': bit,
            'objective_type': 'hard_cross_entropy' if bit == 32 else 'recursive_soft_cross_entropy',
            'objective_loss': float(loss.item()), 'hard_ce': float(hard.detach().item()),
            'correct1': int((top5[:, 0] == y).sum().item()),
            'correct5': int((top5 == y[:, None]).any(dim=1).sum().item()), 'n': len(y)})
        # At lower precisions, hard is a diagnostic only; it is NOT backpropagated.
    optimizer.step()
    support.finite_state(model)
    delta = float((watched.detach() - before).abs().max().item())
    require(math.isfinite(delta), 'Niet-eindige gewichtsupdate.')
    # A zero update is logged, not forced to be a training error.
    records[-1]['watched_weight_max_abs_update'] = delta
    return records


def summarize(acc: dict[int, list[dict[str, Any]]], epoch: int, phase: str,
              lr: float) -> list[dict[str, Any]]:
    rows = []
    for bit in BITS:
        records = acc[bit]
        n = sum(r['n'] for r in records)
        require(n > 0, 'Lege dataset in metrics.')
        rows.append({'epoch': epoch, 'phase': phase, 'bits_w_and_a': bit, 'n': n,
            'objective_type': records[0]['objective_type'],
            'objective_loss': sum(r['objective_loss'] * r['n'] for r in records) / n,
            'hard_ce': sum(r['hard_ce'] * r['n'] for r in records) / n,
            'top1_percent': 100.0 * sum(r['correct1'] for r in records) / n,
            'top5_percent': 100.0 * sum(r['correct5'] for r in records) / n, 'lr': lr})
    return rows


def evaluate(model: Any, loader: Any, device: Any, epoch: int, lr: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state_before = support.state_digest(model.state_dict())
    model.eval()
    acc: dict[int, list[dict[str, Any]]] = {b: [] for b in BITS}
    outputs: dict[str, list[Any]] = {str(b): [] for b in BITS}
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            for bit in BITS:
                support.select_precision(model, bit)
                output = model(x)
                require(bool(torch.isfinite(output).all().item()), 'Niet-eindige validatie-output.')
                loss = float(torch.nn.functional.cross_entropy(output, y).item())
                require(math.isfinite(loss), 'Niet-eindige validatieloss.')
                top5 = output.topk(5, dim=1).indices
                acc[bit].append({'n': len(y), 'objective_type': 'hard_cross_entropy',
                    'objective_loss': loss, 'hard_ce': loss,
                    'correct1': int((top5[:, 0] == y).sum().item()),
                    'correct5': int((top5 == y[:, None]).any(dim=1).sum().item())})
                outputs[str(bit)].append(output.cpu())
    require(state_before == support.state_digest(model.state_dict()), 'Validatie veranderde modeltoestand.')
    return summarize(acc, epoch, 'validation', lr), {b: torch.cat(xs) for b, xs in outputs.items()}


def rng_snapshot() -> dict[str, Any]:
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def run(args: Any, out: Path, report: dict[str, Any], log: Any) -> None:
    global torch, np
    profile = PROFILES[args.profile]
    report['config'] = dict(profile, profile=args.profile, seed=SEED, bits=BITS,
        model='resnet20q', expand=5, optimizer='Adam', lr=0.001, weight_decay=0.0,
        scheduler_milestones=[100, 150, 180], scheduler_gamma=0.1,
        workers=0, drop_last=False, repeats=1)
    log('ANY-PRECISION CIFAR-10 PILOT - geen referentietraining, geen MX')
    source, info = prepare_source(out)
    report['source'] = info
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch as t
    import numpy as n
    torch, np = t, n
    support.torch, support.np = t, n
    require(str(torch.__version__).split('+')[0] == '2.10.0', 'Dit pakket is getest voor PyTorch 2.10.0.')
    require(importlib.metadata.version('torchvision').split('+')[0] == '0.25.0',
            'Dit pakket is getest voor torchvision 0.25.0.')
    device = torch.device('cuda:0' if args.device == 'cuda' else 'cpu')
    report['execution_settings'] = support.configure(SEED, device, 2)
    report['environment'] = {'python': sys.version, 'platform': platform.platform(),
        'interpreter': sys.executable, 'torch': str(torch.__version__),
        'torchvision': importlib.metadata.version('torchvision'), 'numpy': str(np.__version__),
        'pillow': importlib.metadata.version('Pillow'), 'cuda_build': torch.version.cuda,
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}
    log('Apparaat: ' + report['environment']['device_name'])
    make_model, SoftCE, get_optimizer, _ = support.import_source(source)
    data_module = importlib.import_module('datasets.data')
    require(Path(data_module.__file__).resolve().is_relative_to(source), 'Verkeerde datasetimport.')
    report['dataset_files'] = ensure_dataset(args.data_root.resolve(), data_module.CIFAR10, args.download, log)
    data_module.data_paths['cifar10'] = str(args.data_root.resolve())
    train_tf = data_module.get_transform('cifar10', 'train')
    val_tf = data_module.get_transform('cifar10', 'val')
    # Both instances load the OFFICIAL TRAIN split; the separate validation indices do the split.
    train_data = data_module.get_dataset('cifar10', 'train', train_tf, download=False)
    val_data = data_module.get_dataset('cifar10', 'train', val_tf, download=False)
    require(train_data.train_labels == val_data.train_labels, 'Verschillende bronlabels.')
    split = make_split(train_data.train_labels, profile['train_per_class'], profile['val_per_class'])
    json_write(out / 'split_indices.json', split)
    report['split'] = {k: v for k, v in split.items() if k not in
        ['full_train_pool', 'full_validation_pool', 'pilot_train_indices', 'pilot_validation_indices']}
    report['split']['sha256'] = sha256(out / 'split_indices.json')
    report['transforms'] = {'train': repr(train_tf), 'validation': repr(val_tf),
        'implementation': 'get_transform from supplied datasets/data.py',
        'historical_library_bitwise_equivalence_tested': False}
    train_subset = IndexedSubset(train_data, split['pilot_train_indices'])
    val_subset = IndexedSubset(val_data, split['pilot_validation_indices'])
    log(f'Data: {len(train_subset)} train / {len(val_subset)} validatie; officiele testset niet gebruikt.')
    log(f'Protocol: {profile["epochs"]} epochs, batch {profile["batch_size"]}; W32/A32 -> W8/A8 -> W4/A4 -> W2/A2 -> W1/A1.')
    support.reseed(SEED)
    model = make_model(BITS).to(device)
    require(model.expand == 5, 'Modelbreedte gewijzigd.')
    report['parameter_count'] = sum(p.numel() for p in model.parameters())
    require(report['parameter_count'] == 6786330, 'Onverwacht aantal parameters.')
    optimizer = get_optimizer(model, 'adam', 0.001, 0.0)
    scheduler = importlib.import_module('optimizer').get_lr_scheduler('adam', optimizer, [100, 150, 180])
    soft_ce = SoftCE()
    report['initial_model_sha256'] = support.state_digest(model.state_dict())
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=profile['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False, generator=torch.Generator().manual_seed(SEED + 3))
    protocol = {'config': report['config'], 'execution': report['execution_settings'],
        'transforms': report['transforms'], 'source_archive_sha256': info['archive_sha256'],
        'runner_sha256': report['runner_sha256'], 'split_sha256': report['split']['sha256'],
        'interpretation': 'Exploratory small-data/short-budget pilot, not paper reproduction or test accuracy.',
        'test_examples_evaluated': 0, 'checkpoint_selection': 'last epoch, no best-validation selection'}
    json_write(out / 'protocol.json', protocol)
    columns = ['epoch', 'phase', 'bits_w_and_a', 'n', 'objective_type', 'objective_loss',
               'hard_ce', 'top1_percent', 'top5_percent', 'lr']
    report['epochs'] = []
    total_steps, zero_updates = 0, 0
    with (out / 'metrics.csv').open('x', encoding='utf-8', newline='') as metric_f, \
         (out / 'batch_metrics.jsonl').open('x', encoding='utf-8', newline='\n') as batch_f:
        writer = csv.DictWriter(metric_f, fieldnames=columns)
        writer.writeheader()
        lr = float(optimizer.param_groups[0]['lr'])
        initial_metrics, _ = evaluate(model, val_loader, device, 0, lr)
        report['initial_validation'] = initial_metrics
        writer.writerows(initial_metrics)
        metric_f.flush()
        log('Validatie voor training (epoch 0) opgeslagen; geen accuracy-drempel voor succes.')
        for epoch in range(1, profile['epochs'] + 1):
            start = time.perf_counter()
            # Explicit per-epoch RNGs; loading/evaluation/checkpointing cannot shift next epoch's stream.
            epoch_seed = SEED + 1000 + epoch
            support.reseed(epoch_seed)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=profile['batch_size'],
                shuffle=True, num_workers=0, drop_last=False, pin_memory=False,
                generator=torch.Generator().manual_seed(epoch_seed + 100000))
            epoch_order = []
            acc: dict[int, list[dict[str, Any]]] = {b: [] for b in BITS}
            lr = float(optimizer.param_groups[0]['lr'])
            for batch, (x, y, index) in enumerate(train_loader, 1):
                epoch_order.extend(index.tolist())
                tensor_hash = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
                records = train_batch(model, optimizer, x.to(device), y.to(device), soft_ce)
                delta = records[-1]['watched_weight_max_abs_update']
                zero_updates += int(delta == 0.0)
                total_steps += 1
                for record in records:
                    acc[record['bits_w_and_a']].append(record)
                batch_f.write(json.dumps({'epoch': epoch, 'batch': batch, 'original_indices': index.tolist(),
                    'transformed_inputs_sha256': tensor_hash, 'records': records}, allow_nan=False) + '\n')
                batch_f.flush()
                if batch == 1 or batch % 25 == 0 or batch == len(train_loader):
                    log(f'  epoch {epoch}/{profile["epochs"]}, batch {batch}/{len(train_loader)}: '
                        f'W32/A32 CE={records[0]["hard_ce"]:.4f}; gradients eindig; update={delta:.6g}')
            require(sorted(epoch_order) == sorted(split['pilot_train_indices']),
                    'Niet alle trainingsvoorbeelden exact eenmaal verwerkt.')
            json_write(out / f'train_order_epoch_{epoch:03d}.json', epoch_order)
            training = summarize(acc, epoch, 'train', lr)
            validation, final_outputs = evaluate(model, val_loader, device, epoch, lr)
            support.bn_check(model, total_steps)
            scheduler.step()
            report['epochs'].append({'epoch': epoch, 'train': training, 'validation': validation,
                'total_optimizer_steps': total_steps, 'seconds_diagnostic_only': time.perf_counter() - start})
            writer.writerows(training + validation)
            metric_f.flush()
            log('  validatie top-1: ' + ' | '.join(
                f'W{r["bits_w_and_a"]}/A{r["bits_w_and_a"]}={r["top1_percent"]:.2f}%' for r in validation))
        final_model = support.cpu_copy(model.state_dict())
        final_optimizer = support.cpu_copy(optimizer.state_dict())
        final_scheduler = scheduler.state_dict()
        payload = {'purpose': 'exploratory CIFAR-10 pilot, not reference/test accuracy',
            'config': report['config'], 'state_dict': final_model, 'optimizer': final_optimizer,
            'scheduler': final_scheduler, 'rng': rng_snapshot(), 'split_sha256': report['split']['sha256'],
            'completed_epochs': profile['epochs'], 'completed_optimizer_steps': total_steps,
            'eval_logits': final_outputs, 'official_test_evaluated': False,
            'resume_implemented': False}
        with (out / 'pilot_checkpoint.pt').open('xb') as f:
            torch.save(payload, f)
        report['final_model_sha256'] = support.state_digest(final_model)
        require(report['final_model_sha256'] != report['initial_model_sha256'], 'Modeltoestand onveranderd.')
        del model, optimizer, scheduler
        loaded = torch.load(out / 'pilot_checkpoint.pt', map_location='cpu', weights_only=True)
        restored = make_model(BITS).to(device)
        restored_opt = get_optimizer(restored, 'adam', 0.001, 0.0)
        restored_scheduler = importlib.import_module('optimizer').get_lr_scheduler('adam', restored_opt, [100, 150, 180])
        restored.load_state_dict(loaded['state_dict'], strict=True)
        restored_opt.load_state_dict(loaded['optimizer'])
        restored_scheduler.load_state_dict(loaded['scheduler'])
        restored_rows, restored_outputs = evaluate(restored, val_loader, device, profile['epochs'], lr)
        checkpoint_checks = {
            'model_identical': not support.differences(final_model, support.cpu_copy(restored.state_dict())),
            'optimizer_identical': not support.differences(final_optimizer, support.cpu_copy(restored_opt.state_dict())),
            'scheduler_identical': not support.differences(final_scheduler, restored_scheduler.state_dict()),
            'validation_logits_identical': not support.differences(final_outputs, restored_outputs),
            'validation_metrics_identical': not support.differences(validation, restored_rows)}
        report['checkpoint'] = {'checks': checkpoint_checks, 'weights_only': True,
            'complete_training_resume_tested': False, 'sha256': hash_file(out / 'pilot_checkpoint.pt')}
        require(all(checkpoint_checks.values()), 'Checkpoint-round-trip verschilt; zie pilot_report.json.')
    require(all(sha256(source / name) == h for name, h in info['workcopy_files_sha256'].items()),
            'Bronwerkkopie gewijzigd tijdens training.')
    require(sha256(ROOT / 'upstream' / 'Any-Precision-DNNs-master.zip') == info['archive_sha256'],
            'Originele bron-ZIP gewijzigd.')
    report['checks'] = {'finite_gradients_and_states': True, 'split_disjoint': True,
        'all_samples_used_once_per_epoch': True, 'batchnorm_counters_correct': True,
        'validation_state_unchanged': True, 'source_unchanged': True}
    report['completed_optimizer_steps'] = total_steps
    report['zero_updates_of_watched_weight'] = zero_updates
    report['repeatability'] = {'fresh_repetitions_this_run': 1, 'cross_process_repeatability_tested_here': False,
        'note': 'Seeds, split, order and transformed-batch hashes saved; full pilot repetition not automatic.'}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        report['cuda_peak_allocated_MiB'] = torch.cuda.max_memory_allocated(device) / 1024**2
        report['cuda_peak_reserved_MiB'] = torch.cuda.max_memory_reserved(device) / 1024**2
    report['status'] = 'PASS'
    log('Checkpoint opnieuw geladen: model, optimizer, scheduler en validatie-uitvoer identiek.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--data-root', type=Path, default=Path('data') / 'cifar10')
    parser.add_argument('--output-root', type=Path, default=Path('pilot_runs') / 'anyprecision_cifar10')
    parser.add_argument('--profile', choices=list(PROFILES), default='pilot')
    parser.add_argument('--download', action='store_true', help='Allow official data download when absent.')
    args = parser.parse_args()
    out = args.output_root.resolve() / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    log = support.Log(out / 'run.log')
    report: dict[str, Any] = {'schema_version': 1, 'status': 'RUNNING',
        'purpose': 'small exploratory CIFAR-10 Any-Precision pilot',
        'started_at': datetime.now().astimezone().isoformat(), 'run_directory': str(out),
        'runner_sha256': sha256(Path(__file__)), 'argv': sys.argv,
        'limitations': ['Original train.py launcher not executed; original model/loss/optimizer/data functions reused.',
            'Pilot batch, subsets and training budget differ from original 200-epoch training recipe.',
            'No official test accuracy, MX, stored-code truncation, continual learning or hardware benefit.',
            'No paper-accuracy reproduction claim, no automatic multi-seed comparison.',
            'Checkpoint state round-trip, not a complete training-resume test.']}
    (out / 'runner_used.py').write_bytes(Path(__file__).read_bytes())
    (out / 'support_used.py').write_bytes((ROOT / 'smoke_support.py').read_bytes())
    report['support_sha256'] = sha256(ROOT / 'smoke_support.py')
    for name in ['PROTOCOL.md', 'CHANGELOG.md', 'README.md']:
        if (ROOT / name).is_file():
            (out / name).write_bytes((ROOT / name).read_bytes())
    started = time.perf_counter()
    counts: dict[tuple[str, str, str, int], int] = {}
    def record_warning(message: Any, category: Any, filename: str, lineno: int, file: Any = None, line: Any = None) -> None:
        key = (category.__name__, str(message), filename, lineno)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == 1:
            log(f'WAARSCHUWING: {category.__name__}: {message}')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.showwarning = record_warning
            run(args, out, report, log)
    except (Exception, KeyboardInterrupt) as exc:
        report['status'] = 'FAIL'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        log('FOUT: ' + str(exc))
        log(report['error']['traceback'])
    finally:
        report['elapsed_seconds_diagnostic_only'] = time.perf_counter() - started
        report['warnings'] = [{'category': k[0], 'message': k[1], 'filename': k[2], 'line': k[3], 'count': v}
                              for k, v in counts.items()]
        try:
            pip = subprocess.run([sys.executable, '-B', '-m', 'pip', 'freeze', '--all'],
                capture_output=True, text=True, timeout=60, check=False)
            (out / 'pip_freeze.txt').write_text(pip.stdout, encoding='utf-8')
            report['pip_freeze'] = {'returncode': pip.returncode, 'stderr': pip.stderr}
        except Exception as exc:
            report['pip_freeze'] = {'error': str(exc)}
        report['ended_at'] = datetime.now().astimezone().isoformat()
        json_write(out / 'pilot_report.json', report)
        log(f'Waarschuwingen: {sum(counts.values())} ({len(counts)} unieke meldingen).')
        log('STATUS: ' + report['status'])
        log('Rapportmap: ' + str(out))
        log('Geen pakketten geinstalleerd. Geen MX. Geen officiele testset-evaluatie.')
        log.handle.close()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
