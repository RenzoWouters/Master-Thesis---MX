"""Frozen two-mode inference from a single high MX source; no master or quantizer.

The established decoders, bit-shift and model loader are reused byte-for-byte.
Only the new evaluation orchestration and diagnostic records live here.
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch
import hashlib
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import trunc_adapter as a
import resume_core as core
import svhn_support as s

BITS = (4, 6)  # Same order as evaluation during training.
require = s.require


@contextmanager
def inference_guards(source_file: Path):
    """Instrument torch.load and prohibit weight encoding/updates.

    This is a fail-fast correctness guard, not an OS security sandbox. The
    subprocess receives only the staged high file, test location and JSON facts.
    """
    allowed = source_file.resolve()
    opened = []
    original_load, original_train = torch.load, torch.nn.Module.train
    def load(path, *args, **kwargs):
        require(isinstance(path, (str, os.PathLike)) and Path(path).resolve() == allowed,
                'Verboden modelbestand: uitsluitend de staged E2M3-bron mag worden geladen.')
        require(kwargs.get('weights_only') is True, 'weights_only=True verplicht.')
        opened.append(str(allowed))
        return original_load(path, *args, **kwargs)
    def no_call(*args, **kwargs):
        raise RuntimeError('Verboden: encoder, gewichtsvoorbewerking, optimizer of backward in bron-only evaluatie.')
    def eval_only(module, mode=True):
        require(mode is False, 'Training/BatchNorm-kalibratie is verboden in deze evaluatie.')
        return original_train(module, mode)
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch, 'load', load))
        stack.enter_context(patch.object(torch.nn.Module, 'train', eval_only))
        stack.enter_context(patch.object(torch.optim.Optimizer, '__init__', no_call))
        stack.enter_context(patch.object(torch.autograd, 'backward', no_call))
        for module in (a.high, a.low):
            for name in ('encode_tensor', 'encode_elements', 'fake_quantize_mx'):
                stack.enter_context(patch.object(module, name, no_call))
        for module in (a, a.base, a.base.original_adapter):
            stack.enter_context(patch.object(module, 'effective_reference_weight', no_call))
        yield opened


def freeze_digest(model, bit: int) -> str:
    require(all(not m.training for m in model.modules()), 'Model/BN niet in eval().')
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()), 'Gradient actief.')
    require(all(m.abit == 32 for m in model.modules() if hasattr(m, 'abit')), 'A32 gewijzigd.')
    require(model.weight_mode == bit and model._frozen_weight_bit == bit, 'Verkeerde gewichtsmodus.')
    for name in a.BN_LAYERS:
        bank = model.get_submodule(name)
        require(bank.weight_mode == bit and a.BANK[bit] == ('32' if bit == 6 else '4'),
                'Verkeerde BN-bank.')
    forbidden = {n + '.weight' for n in a.TARGET_LAYERS}
    require(not forbidden.intersection(dict(model.named_parameters())), 'FP32-master in uitvoermodel.')
    for name in a.TARGET_LAYERS:
        require(isinstance(model.get_submodule(name), a.base.original_adapter.FrozenMXConv),
                'Geen vaste code-decoderconvolutie.')
    return s.legacy.state_digest(model.state_dict())


def build_models(models_module, source: dict, device):
    """Both models use the very same high payload. The low file is unnecessary."""
    a.validate_source(source)
    before = core.cpu_copy(source)
    models = {b: a.load_exported_model(models_module, source, device, bit=b) for b in BITS}
    rows = []
    for name, tensor in source['auxiliary_fp32_state'].items():
        require(isinstance(tensor, torch.Tensor), 'Hulpstaat bevat geen tensor: ' + name)
        for model in models.values():
            target = model.state_dict()[name]
            require(target.dtype == tensor.dtype and target.shape == tensor.shape,
                    'Stilzwijgende dtype/vormconversie in hulpstaat: ' + name)
            require(torch.equal(target.cpu(), tensor.cpu()), 'Hulpstaat niet exact hersteld.')
        require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()), 'Niet-eindige hulpstaat.')
    for name in a.TARGET_LAYERS:
        h = a.high.MXTensor.from_payload(source['mx_layers'][name])
        low = a.truncate_codes(h)
        require(torch.equal(low.codes, h.codes >> 2), 'Lage codes zijn niet de bitshift van de hoge codes.')
        require(torch.equal(low.scale_codes, h.scale_codes), 'Schaal gewijzigd.')
        for bit, mx in ((6, h), (4, low)):
            decoded = models[bit].get_submodule(name).decoded_weight.cpu()
            expected = mx.decode().cpu()
            require(torch.equal(decoded, expected) and torch.equal(torch.signbit(decoded), torch.signbit(expected)),
                    'Reconstructie niet exact op ' + name)
        rows.append({'layer': name, 'actual_weights': int(np.prod(h.shape)),
                     'padded_codes': h.codes.numel(), 'blocks': h.scale_codes.numel(),
                     'high_code_bytes': source['mx_layers'][name]['packed_codes'].numel(),
                     'low_code_bytes_if_materialized': low.payload()['packed_codes'].numel(),
                     'scale_bytes': h.scale_codes.numel(), 'low_equals_high_shift_2': True,
                     'scales_identical': True, 'sign_and_exponent_preserved': True,
                     'decoded_tensors_exact': True})
    require(not core.differences(before, source), 'Hoge bron werd gewijzigd.')
    states = {str(b): freeze_digest(m, b) for b, m in models.items()}
    return models, {'high_source_unchanged_during_reconstruction': True,
                    'six_fp32_masters_absent': True, 'auxiliary_state_exact': True,
                    'both_modes_from_same_high_payload': True, 'no_encoder_or_G_W_called': True,
                    'bn_bank_mapping': {'6': '32', '4': '4'}, 'state_sha256_before': states}, rows


def evaluate_models(models, loader, n: int, device, out: Path, role: str, epoch: int, log):
    """Full ordered data pass, both modes per batch, retained raw per-image outputs."""
    require(tuple(models) == BITS and n > 0, 'Onjuiste evaluatiemodi of sampleaantal.')
    before = {b: freeze_digest(models[b], b) for b in BITS}
    records = {b: {'logits': [], 'predictions': [], 'per_image_ce': [], 'correct5': 0} for b in BITS}
    indices, labels = [], []
    t = time.perf_counter()
    with torch.no_grad():
        for i, (x, y, idx) in enumerate(loader):
            require(tuple(x.shape[1:]) == (3, 40, 40) and x.dtype == torch.float32
                    and bool(torch.isfinite(x).all()), 'Afwijkende invoer.')
            indices.append(idx.cpu().numpy().astype(np.int64, copy=False))
            labels.append(y.cpu().numpy().astype(np.int64, copy=False))
            x, y = x.to(device), y.to(device)
            for b, model in models.items():
                logits = model(x)
                require(tuple(logits.shape) == (len(y), 10) and bool(torch.isfinite(logits).all()), 'Ongeldige logits.')
                # Keep the upstream top-k convention, including its tie handling.
                top = logits.topk(5, dim=1, largest=True, sorted=True).indices
                d = records[b]
                d['logits'].append(logits.cpu().numpy())
                d['predictions'].append(top[:, 0].cpu().numpy())
                d['per_image_ce'].append(F.cross_entropy(logits, y, reduction='none').cpu().numpy())
                d['correct5'] += int((top == y[:, None]).any(dim=1).sum().item())
            if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(loader):
                log(f'{role}: batch {i+1}/{len(loader)}; E2M1 uit bron + E2M3; geen updates.')
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t
    ids, yy = np.concatenate(indices), np.concatenate(labels)
    require(len(ids) == n and np.array_equal(ids, np.arange(n)), 'Testset niet precies eenmaal in vaste volgorde.')
    require(np.isin(yy, np.arange(10)).all(), 'Ongeldige labels.')
    npz = {'indices': ids, 'labels': yy}
    metrics = []
    for b, d in records.items():
        for field in ('logits', 'predictions', 'per_image_ce'):
            d[field] = np.concatenate(d[field])
            require(np.isfinite(d[field]).all(), 'Niet-eindige opgeslagen gegevens.')
            npz[f'E2M{3 if b == 6 else 1}__{field}'] = d[field]
        correct = int((d['predictions'] == yy).sum())
        metrics.append({'role': role, 'epoch': epoch, 'weight_bits': b, 'weight_format': a.FORMATS[b],
                        'activation_bits': 32, 'bn_bank_key': a.BANK[b], 'n': n, 'correct1': correct,
                        'correct5': d['correct5'], 'top1_percent': 100 * correct / n,
                        'hard_cross_entropy': float(d['per_image_ce'].astype(np.float64).mean()),
                        'source_only': True, 'truncated_from_high': b == 4, 'optimizer_steps': 0})
    with (out / 'per_image_outputs.npz').open('xb') as f: np.savez_compressed(f, **npz)
    for b, model in models.items():
        require(freeze_digest(model, b) == before[b], 'Model- of BatchNorm-staat gewijzigd tijdens evaluatie.')
    hp, lp = records[6]['predictions'], records[4]['predictions']
    hc, lc = hp == yy, lp == yy
    pair = {'role': role, 'epoch': epoch, 'n': n, 'comparison': 'E2M1_from_source_minus_E2M3',
            'prediction_changes': int((hp != lp).sum()),
            'correct_to_wrong': int((hc & ~lc).sum()), 'wrong_to_correct': int((~hc & lc).sum()),
            'wrong_to_other_wrong': int((~hc & ~lc & (hp != lp)).sum()),
            'net_correct_change': int(lc.sum()) - int(hc.sum()),
            'accuracy_difference_pp': 100 * (int(lc.sum()) - int(hc.sum())) / n}
    return metrics, pair, {'label_sha256': hashlib.sha256(yy.tobytes()).hexdigest(),
        'sample_order_sha256': hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest(),
        'all_states_unchanged': True, 'no_gradients': True, 'n': n,
        'paired_evaluation_seconds_not_benchmark': elapsed,
        'per_image_sha256': core.file_sha256(out/'per_image_outputs.npz'),
        'training_logits_available_for_comparison': False}
