"""Small real-Adam freeze checks, using synthetic gradients and no SVHN data."""
from __future__ import annotations
from runtime_bridge import s, t, a, core, ref, torch
from freeze_core import freeze_optimizer, FrozenOffsetGuard


def run_tests(modules, rt):
    model = t.build_model(modules, 'joint', rt.device)
    optimizer = t.build_optimizer(modules, model, ref.config())
    scheduler = modules.optimizer.get_lr_scheduler('adam', optimizer, [50, 75, 90])
    checks = {}
    def check(ok, name):
        s.require(ok, 'Freezecheck faalt: ' + name)
        checks[name] = True
    # Populate real Adam state and reach the actual first LR transition.
    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            if not a.inactive_bn(name):
                parameter.grad = torch.full_like(parameter, .01)
        optimizer.step()
        scheduler.step()
    with torch.no_grad():
        for parameter in a.scale_parameters(model):
            pattern = torch.tensor([5.0, 5.499999523162842, 5.5, 5.500000476837158, 6.0],
                                   dtype=torch.float32, device=rt.device)
            parameter.copy_(pattern.repeat((parameter.numel() + 4) // 5)[:parameter.numel()].reshape(parameter.shape))
    anchor = t.scale_state(model)
    weight_before = core.cpu_copy(a.base_state_dict(model))
    theta = list(a.scale_parameters(model))[0]
    moment_before = optimizer.state[theta]['exp_avg'].detach().clone()
    rng_before = core.capture_rng({}, rt.device)
    intervention = freeze_optimizer(model, optimizer, scheduler)
    check(not core.differences(core.capture_rng({}, rt.device), rng_before), 'freeze_consumes_no_rng')
    check(optimizer.param_groups[0]['lr'] == .0001 and optimizer.param_groups[1]['lr'] == 0,
          'weight_lr_preserved_theta_lr_zero')
    check(scheduler.last_epoch == 50 and scheduler.get_last_lr() == [.0001, 0.0],
          'absolute_scheduler_position50_preserved')
    guard = FrozenOffsetGuard(model, optimizer, anchor)
    try:
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            for name, parameter in model.named_parameters():
                if not a.inactive_bn(name):
                    parameter.grad = torch.full_like(parameter, -.03 if name.endswith('.theta') else .02)
            optimizer.step()
            scheduler.step()
            guard.assert_unchanged()
        check(guard.steps == 3, 'all_three_optimizer_updates_guarded')
        check(not core.differences(t.scale_state(model), anchor), 'all_theta_exact_at_bounds_and_threshold')
        check(bool(core.differences(a.base_state_dict(model), weight_before)), 'weights_still_update')
        check(not torch.equal(optimizer.state[theta]['exp_avg'], moment_before), 'theta_moments_still_diagnostic_updates')
        check(int(optimizer.state[theta]['step']) == 53 and scheduler.last_epoch == 53,
              'absolute_adam_scheduler_steps53')
        check(scheduler.get_last_lr() == [.0001, 0.0], 'zero_theta_lr_survives_scheduler_steps')
        # Frozen offset does not freeze the E4 source scale: change a synthetic
        # master tensor and re-encode using exactly the inherited source route.
        layer = model.get_submodule(a.TARGET_LAYERS[0])
        with torch.no_grad():
            high_before = a.codec(8).encode_tensor(a.effective_reference_weight(layer.weight))
            layer.weight.mul_(4)
            high_after = a.codec(8).encode_tensor(a.effective_reference_weight(layer.weight))
            offsets = a.hard_offsets(layer.quantize_fn.theta)
            chain = a.encode_chain(a.effective_reference_weight(layer.weight), offsets)
        check(not torch.equal(high_before.scale_codes, high_after.scale_codes), 'E4_source_scale_remains_dynamic')
        check(torch.equal(chain[6].scale_codes.to(torch.int32), high_after.scale_codes.to(torch.int32) + offsets)
              and torch.equal(chain[4].scale_codes, chain[6].scale_codes), 'both_lower_scales_follow_current_source_plus_fixed_offset')
        guard.assert_unchanged()
        with torch.no_grad():
            original = theta.reshape(-1)[0].clone()
            old_hard = a.hard_offsets(theta).clone()
            theta.reshape(-1)[0] = torch.nextafter(original, torch.tensor(6.0, device=rt.device))
            check(torch.equal(a.hard_offsets(theta), old_hard), 'negative_theta_drift_stays_in_same_hard_bin')
        rejected = False
        try:
            guard.assert_unchanged()
        except RuntimeError:
            rejected = True
        finally:
            with torch.no_grad(): theta.reshape(-1)[0].copy_(original)
        check(rejected, 'one_ulp_theta_drift_rejected_even_without_choice_change')
        optimizer.param_groups[1]['lr'] = .0001
        rejected = False
        try:
            guard.assert_unchanged()
        except RuntimeError:
            rejected = True
        finally:
            optimizer.param_groups[1]['lr'] = 0.0
        check(rejected, 'nonzero_theta_lr_rejected')
    finally:
        guard.close()
    return dict(status='PASS', synthetic_checks_only=True, SVHN_training_performed=False,
                checks=checks, intervention=intervention)

