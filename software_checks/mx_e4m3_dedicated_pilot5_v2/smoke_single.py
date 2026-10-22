"""Two-update, dataset-free integration check for dedicated E4M3/A32 QAT.

Uses the actual archived CNN-8. This is a correctness check, not an SVHN
experiment or timing estimate. The production backend policy is not changed.
"""
from __future__ import annotations

import io
import math
import time

import torch
import torch.nn.functional as F

import mx_adapter as a
import mx_e4m3 as f
import single_support as t
import svhn_support as s


@torch.no_grad()
def _check_quantized_weights(model) -> list[dict]:
    rows = []
    for name in a.TARGET_LAYERS:
        layer = model.get_submodule(name)
        effective = a.effective_reference_weight(layer.weight)
        encoded = f.encode_tensor(effective)
        actual = layer.quantize_fn(layer.weight)
        expected = encoded.decode().to(actual.device)
        s.require(torch.equal(actual, expected), 'Forward verschilt van E4M3-decode: ' + name)
        s.require(actual.dtype == torch.float32, 'Uitvoeringsgewicht is niet FP32: ' + name)
        s.require(encoded.payload()['format'] == 'MXFP8_E4M3',
                  'Verkeerd elementformaat: ' + name)
        rows.append({'layer': name, 'shape': list(layer.weight.shape),
                     'quantizer_matches_decode_exactly': True})
    return rows


def _check_weight_gradient(model) -> dict:
    # Compare the complete adapter derivative against the preserved G(W)
    # derivative. Identical upstream derivatives must reach the FP32 masters.
    weight = model.get_submodule(a.TARGET_LAYERS[0]).weight
    probe = weight.detach().clone().requires_grad_(True)
    reference = weight.detach().clone().requires_grad_(True)
    quantizer = a.MXWeightQuantizer(enabled=True)
    actual = torch.autograd.grad(quantizer(probe).sum(), probe)[0]
    expected = torch.autograd.grad(a.effective_reference_weight(reference).sum(), reference)[0]
    s.require(torch.equal(actual, expected), 'MX-STE verandert de gradient door G(W).')
    s.require(actual.dtype == torch.float32 and bool(torch.isfinite(actual).all()),
              'STE geeft geen eindige FP32-mastergradient.')
    return {'mx_identity_ste_with_effective_weight_derivative_exact': True,
            'master_gradient_dtype': str(actual.dtype)}


def check(modules, rt) -> dict:
    """Run exactly two synthetic minibatch updates and restore caller RNG state."""
    device, core = rt.device, rt.core
    rng_before = core.capture_rng({}, device)
    started = time.perf_counter()
    try:
        model = t.build_model(modules, 'mx', device)
        t.assert_policy(model, 'mx')
        s.require(model.expand == 2 and sum(p.numel() for p in model.parameters()) == 132954,
                  'Niet de verwachte CNN-8 expand=2.')
        initial = t.layer_snapshot(model, 'mx')
        parameter_ids = {name: id(p) for name, p in model.named_parameters()}
        t.validate_bn(model, 0)
        before_rows = _check_quantized_weights(model)
        ste = _check_weight_gradient(model)

        # Fixed values and labels: no network, dataset, sampler or download.
        x = torch.linspace(-1.0, 1.0, 2 * 3 * 40 * 40,
                           dtype=torch.float32, device=device).reshape(2, 3, 40, 40)
        y = torch.tensor([0, 7], dtype=torch.long, device=device)
        optimizer = modules.optimizer.get_optimizer_config(model, 'adam', .001, 0.)
        loss_rows = []
        training_forwards = 0

        def count_training_forward(module, inputs):
            nonlocal training_forwards
            if module.training:
                training_forwards += 1

        handle = model.register_forward_pre_hook(count_training_forward)
        try:
            model.train()
            for step in range(1, 3):
                optimizer.zero_grad()
                logits = model(x)
                s.require(tuple(logits.shape) == (2, 10) and bool(torch.isfinite(logits).all()),
                          'Niet-eindige of onverwachte synthetische netwerkuitvoer.')
                loss = F.cross_entropy(logits, y)
                s.require(bool(torch.isfinite(loss)), 'Niet-eindige synthetische hard-label loss.')
                loss.backward()
                gradient = t.validate_gradients(model)
                s.require(gradient['l2_norm'] > 0, 'Alle actieve gradients zijn nul.')
                optimizer.step()
                adam = t.validate_adam_state(model, optimizer, step)
                bn = t.validate_bn(model, step)
                t.assert_policy(model, 'mx')
                s.legacy.finite_state(model)
                loss_rows.append({'step': step, 'loss': float(loss.detach()),
                                  'gradient': gradient})
        finally:
            handle.remove()
        s.require(training_forwards == 2, 'Dedicated proef vereist een forward per update.')
        s.require(parameter_ids == {name: id(p) for name, p in model.named_parameters()},
                  'FP32-masterparameter vervangen tijdens de twee updates.')
        updates = t.layer_updates(initial, t.layer_snapshot(model, 'mx'), 'mx', epoch=0)
        s.require(all(row['master_changed_values'] > 0 for row in updates),
                  'Niet elke MX-doellaag ontving een masterupdate.')
        after_rows = _check_quantized_weights(model)

        # The export contains only element/scale codes for the six MX layers;
        # non-MX layers and all inherited BN banks remain auxiliary model state.
        model.eval()
        state_before_eval = s.legacy.state_digest(model.state_dict())
        with torch.no_grad():
            expected_logits = model(x)
            payload, distortion = a.export_payload(model, {
                'scope': 'synthetic_integration_check', 'synthetic_fixture_only': True,
                'optimizer_steps': 2, 'weight_format': 'MXFP8_E4M3',
            })
        s.require(payload['schema'] == 'mx-weights-e4m3-source-v1', 'Verkeerd exportschema.')
        s.require(not {name + '.weight' for name in a.TARGET_LAYERS}.intersection(
                  payload['auxiliary_fp32_state']), 'FP32-master in de E4M3-export.')
        # Also exercise serialization and weights_only loading, without files.
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        restored_payload = torch.load(stream, map_location='cpu', weights_only=True)
        frozen = a.load_exported_model(modules.models, restored_payload, device)
        frozen_parameters = dict(frozen.named_parameters())
        export_rows = []
        for name in a.TARGET_LAYERS:
            s.require(name + '.weight' not in frozen_parameters,
                      'Gereconstrueerd model bevat MX-masterparameter: ' + name)
            decoded = frozen.get_submodule(name).decoded_weight
            with torch.no_grad():
                actual = model.get_submodule(name).quantize_fn(model.get_submodule(name).weight)
            s.require(torch.equal(actual, decoded), 'Exportgewicht niet exact gelijk: ' + name)
            export_rows.append({'layer': name, 'export_weight_matches_exactly': True,
                                'decoded_execution_buffer_dtype': str(decoded.dtype)})
        with torch.no_grad():
            actual_logits = frozen(x)
        s.require(bool(torch.isfinite(actual_logits).all()), 'Niet-eindige exportlogits.')
        atol = 1e-5 if device.type == 'cuda' else 0.0
        rtol = 1e-5 if device.type == 'cuda' else 0.0
        s.require(torch.allclose(actual_logits, expected_logits, atol=atol, rtol=rtol),
                  'Exportlogits overschrijden de vooraf vastgelegde integratietolerantie.')
        max_difference = float((actual_logits - expected_logits).abs().max())
        s.require(s.legacy.state_digest(model.state_dict()) == state_before_eval,
                  'Evaluatie/export heeft de originele modelstaat of BN gewijzigd.')
        t.validate_bn(model, 2)
        t.validate_adam_state(model, optimizer, 2)
        elapsed = time.perf_counter() - started
        s.require(math.isfinite(elapsed), 'Niet-eindige controletijd.')
        return {
            'status': 'PASS', 'scope': 'SYNTHETIC_CNN8_ONLY', 'device': str(device),
            'real_svhn_training': False, 'dataset_downloaded': False,
            'model': 'svhnq', 'expand': 2, 'synthetic_batch_shape': list(x.shape),
            'weight_format': 'MXFP8_E4M3', 'activation_bits': 32,
            'master_dtype': 'float32', 'gradient_dtype': 'float32',
            'optimizer_steps': 2, 'training_forward_passes': training_forwards,
            'distillation_used': False, 'fp32_teacher_forward_used': False,
            'steps': loss_rows, 'ste': ste, 'adam_checks': adam, 'bn_checks': bn,
            'quantizer_checks_before_updates': before_rows,
            'quantizer_checks_after_updates': after_rows,
            'weight_updates': updates, 'export_weights': export_rows,
            'export_contains_no_mx_master_parameters': True,
            'export_logits_max_abs_difference': max_difference,
            'export_logit_tolerance': {'atol': atol, 'rtol': rtol},
            'export_synthetic_layer_distortion': distortion,
            'evaluation_preserved_model_and_bn': True,
            'elapsed_seconds': elapsed,
            'note': 'Alleen technische controle; geen SVHN-accuracy of trainingsduurvoorspelling.',
        }
    finally:
        core.restore_rng(rng_before, {}, device)
