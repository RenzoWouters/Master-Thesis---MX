"""Epoch-boundary checkpoint primitives for the technical resume check.

No model/quantizer changes. Checkpoints consist only of tensors and plain Python
containers, loaded with weights_only=True. Same-device/environment restore only.
This module is intended to be reused by the future reference runner, not copied
and independently reimplemented. No mid-epoch or cross-machine resume claim.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

SCHEMA = 'anyprecision-epoch-checkpoint-v1'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                   separators=(',', ':')).encode('utf-8')).hexdigest()


def cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    return value


def differences(a: Any, b: Any, name: str = '') -> list[str]:
    """Exact equality, not approximate accuracy/loss equality. Paths aid diagnosis."""
    if isinstance(a, torch.Tensor):
        same = (isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
        return [] if same else [name]
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [name + ':keys']
        return [p for k in a for p in differences(a[k], b[k], name + '/' + str(k))]
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            return [name + ':container']
        return [p for i, (x, y) in enumerate(zip(a, b))
                for p in differences(x, y, name + '/' + str(i))]
    return [] if type(a) is type(b) and a == b else [name]


def capture_rng(generators: dict[str, torch.Generator], device: torch.device) -> dict[str, Any]:
    ns = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [ns[0], ns[1].tolist(), int(ns[2]), int(ns[3]), float(ns[4])],
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
                      if device.type == 'cuda' else [],
        'loader_generators': {k: g.get_state().clone() for k, g in generators.items()},
    }


def restore_rng(state: dict[str, Any], generators: dict[str, torch.Generator],
                device: torch.device) -> None:
    require(set(state['loader_generators']) == set(generators), 'DataLoader-generators verschillen.')
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if device.type == 'cuda':
        require(len(state['torch_cuda']) == torch.cuda.device_count(), 'Aantal CUDA-RNGs verschilt.')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])
    else:
        require(not state['torch_cuda'], 'Dit CPU-pad laadt geen CUDA-run.')
    for k, g in generators.items():
        g.set_state(state['loader_generators'][k].cpu())


def rng_probe(generators: dict[str, torch.Generator], device: torch.device) -> dict[str, Any]:
    """Draw diagnostic random numbers, then restore state: no effect on training."""
    before = capture_rng(generators, device)
    try:
        return {
            'python': [random.random() for _ in range(4)],
            'numpy': np.random.random(4).tolist(),
            'torch_cpu': torch.rand(4).tolist(),
            'torch_cuda': torch.rand(4, device=device).cpu().tolist()
                          if device.type == 'cuda' else [],
            'loader_generators': {k: torch.rand(4, generator=g).tolist() for k, g in generators.items()},
        }
    finally:
        restore_rng(before, generators, device)


def atomic_save(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Publish a newly named checkpoint only after the full file has been written.

    Never overwrite an existing checkpoint. This protects an earlier completed
    checkpoint from a new write failure; power-loss durability is not guaranteed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.pt')
    marker = path.with_suffix(path.suffix + '.sha256')
    require(not path.exists() and not pending.exists() and not marker.exists(),
            'Checkpointpad bestaat al; niets wordt overschreven: ' + str(path))
    with pending.open('xb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    require(not path.exists(), 'Checkpoint is tijdens opslaan verschenen.')
    # In this runner there is a single writer per new output directory.
    os.replace(pending, path)
    digest = file_sha256(path)
    with marker.open('x', encoding='ascii') as f:
        f.write(digest + '\n')
        f.flush()
        os.fsync(f.fileno())
    return {'file': path.name, 'sha256': digest, 'bytes': path.stat().st_size}


def load_verified(path: Path) -> dict[str, Any]:
    marker = path.with_suffix(path.suffix + '.sha256')
    require(path.is_file() and marker.is_file(), 'Checkpoint of checksum ontbreekt: ' + str(path))
    require(file_sha256(path) == marker.read_text(encoding='ascii').strip(),
            'Checkpointchecksum wijkt af; bestand wordt niet geladen.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and state.get('schema') == SCHEMA, 'Onbekend checkpointformaat.')
    require(state.get('boundary') == 'epoch_complete_after_validation_and_scheduler',
            'Geen volledig afgerond epoch-checkpoint.')
    return state


def capture_checkpoint(model: Any, optimizer: Any, scheduler: Any,
                       generators: dict[str, torch.Generator], device: torch.device,
                       completed_epoch: int, optimizer_steps: int,
                       contract: dict[str, Any], trace: dict[str, Any],
                       logits: dict[str, torch.Tensor]) -> dict[str, Any]:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return {
        'schema': SCHEMA,
        'boundary': 'epoch_complete_after_validation_and_scheduler',
        'contract': contract,
        'completed_epoch': completed_epoch,
        'next_epoch': completed_epoch + 1,
        'optimizer_steps': optimizer_steps,
        'model': cpu_copy(model.state_dict()),
        'optimizer': cpu_copy(optimizer.state_dict()),
        'scheduler': cpu_copy(scheduler.state_dict()),
        'rng': capture_rng(generators, device),
        'trace': cpu_copy(trace),
        'validation_logits': cpu_copy(logits),
    }


def restore_checkpoint(state: dict[str, Any], model: Any, optimizer: Any, scheduler: Any,
                       generators: dict[str, torch.Generator], device: torch.device,
                       expected_contract: dict[str, Any]) -> dict[str, bool]:
    """Construct model, optimizer, scheduler and loaders BEFORE calling this.

    Restore RNG last, after any construction/load operations that might use it.
    The next training batch calls train() and selects each precision explicitly.
    No gradients need saving at the end of an epoch: the next batch zeroes them.
    """
    require(state['schema'] == SCHEMA, 'Onbekend schema.')
    mismatch = differences(state['contract'], expected_contract)
    require(not mismatch, 'Hervatprotocol/omgeving/bron verschilt: ' + ', '.join(mismatch[:8]))
    require(state['next_epoch'] == state['completed_epoch'] + 1, 'Ongeldige epochteller.')
    require(state['scheduler']['last_epoch'] == state['completed_epoch'], 'Scheduler/epoch mismatch.')
    model.load_state_dict(state['model'], strict=True)
    # Scheduler is already constructed; follow the documented restoration order.
    scheduler.load_state_dict(state['scheduler'])
    optimizer.load_state_dict(state['optimizer'])
    model.eval()
    restore_rng(state['rng'], generators, device)
    checks = {
        'model_restored': not differences(cpu_copy(model.state_dict()), state['model']),
        'optimizer_restored': not differences(cpu_copy(optimizer.state_dict()), state['optimizer']),
        'scheduler_restored': not differences(cpu_copy(scheduler.state_dict()), state['scheduler']),
        'rng_restored': not differences(capture_rng(generators, device), state['rng']),
        'lr_matches_scheduler': [float(p['lr']) for p in optimizer.param_groups]
                                == [float(x) for x in scheduler.get_last_lr()],
    }
    require(all(checks.values()), 'Onvolledig herstel: ' + str(checks))
    return checks
