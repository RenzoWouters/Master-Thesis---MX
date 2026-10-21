"""Very small synthetic gradient/update check. Not an accuracy experiment."""
from __future__ import annotations
import torch
import mx_adapter as a
import single_support as t
import svhn_support as s


def check(modules, rt):
    result = {}
    for mode in ('control', 'mx'):
        torch.manual_seed(83411)
        model = t.build_model(modules, mode, rt.device)
        initial = t.layer_snapshot(model, mode)
        optimizer = modules.optimizer.get_optimizer_config(model, 'adam', .001, 0.)
        x = torch.linspace(-1, 1, 3*3*40*40, device=rt.device).reshape(3, 3, 40, 40)
        labels = torch.tensor([1, 4, 9], device=rt.device)
        losses = []
        model.train()
        for _ in range(2):
            optimizer.zero_grad()
            prediction = model(x)
            loss = torch.nn.functional.cross_entropy(prediction, labels)
            s.require(bool(torch.isfinite(loss)), 'Niet-eindige synthetische loss.')
            loss.backward()
            t.validate_gradients(model)
            optimizer.step()
            losses.append(float(loss.detach()))
        s.legacy.finite_state(model)
        t.validate_bn(model, 2)
        t.validate_adam_state(model, optimizer, 2)
        updates = t.layer_updates(initial, t.layer_snapshot(model, mode), mode, 0)
        result[mode] = {'synthetic_only': True, 'optimizer_steps': 2,
                        'losses_not_research_results': losses, 'updates': updates,
                        'fp32_masters_gradients_and_moments': True, 'only_bn32_updates': True}
    return result
