"""MSB training from fresh initialization with the original dynamic x64 route.

The unchanged historical fixed64 author flow supplies the three passes,
recursive distillation, shared code cache and Adam sequencing. Only the final
E2M3-to-E2M1 mantissa mapping is replaced. There is no parent checkpoint and
there are no learned or frozen absolute scale parameters.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "vendor" / "legacy"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

import torch
import resume_core as core
import svhn_support as support
import trunc_adapter as legacy_adapter
import trunc_support as legacy_training
import msb64_adapter as adapter
import mantissa_rules as rules

for _module in (core, support, legacy_adapter, legacy_training):
    if not Path(_module.__file__).resolve().is_relative_to(LEGACY.resolve()):
        raise RuntimeError("Importconflict voor " + _module.__name__ + "; start een nieuw Pythonproces.")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _active_named_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if not legacy_adapter.inactive_bn(name)]


def _optimizer_steps(state):
    if not state.optimizer.state:
        return 0
    for name, parameter in _active_named_parameters(state.model):
        saved = state.optimizer.state.get(parameter)
        _require(saved is not None and "step" in saved, "Ontbrekende Adamstaat: " + name)
        step = saved["step"]
        return int(step.item() if torch.is_tensor(step) else step)
    raise RuntimeError("Geen actieve modelparameters.")


def build_state(rt, modules, config, expected_initial_hash=None):
    """Construct the original fresh model; never reseed or load a checkpoint.

    The caller applies the original runtime seed before model construction.
    Install and optimizer/scheduler construction must preserve the RNG state
    reached by that exact original construction path.
    """
    _require(float(config["lr"]) == 0.001 and float(config["weight_decay"]) == 0.0,
             "Behoud de historische Adam-LR 0,001 en weight decay 0.")
    _require(list(config["scheduler_milestones"]) == [50, 75, 90]
             and float(config.get("scheduler_gamma", 0.1)) == 0.1,
             "Behoud het oorspronkelijke 100-epoch LR-schema.")
    model = legacy_training.build_model(modules, "joint", rt.device)
    initial_hash = support.legacy.state_digest(model.state_dict())
    if expected_initial_hash is not None:
        _require(initial_hash == expected_initial_hash,
                 "Nieuwe modelinitialisatie verschilt van de vastgelegde fixed64-referentie: "
                 f"actual={initial_hash}; expected={expected_initial_hash}.")
    construction_rng = core.capture_rng({}, rt.device)
    installation = adapter.install(model)
    _require(support.legacy.state_digest(model.state_dict()) == initial_hash,
             "MSB-installatie veranderde de nieuwe modelwaarden.")
    _require(not core.differences(construction_rng, core.capture_rng({}, rt.device)),
             "MSB-installatie verbruikte RNG.")
    optimizer = modules.optimizer.get_optimizer_config(model, "adam", float(config["lr"]),
                                                        float(config["weight_decay"]))
    scheduler = modules.optimizer.get_lr_scheduler("adam", optimizer, list(config["scheduler_milestones"]))
    _require(isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
             and not optimizer.state, "Verwacht één originele, verse Adam-groep.")
    _require(scheduler.last_epoch == 0, "Nieuwe scheduler start niet bij epoch0.")
    initial_rng = core.capture_rng({}, rt.device)
    _require(not core.differences(construction_rng, initial_rng),
             "Optimizer-/schedulerconstructie verbruikte RNG.")
    state = SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler,
        method="msb_bestexp", initial_model_sha256=initial_hash, initial_rng=initial_rng,
        installation_audit={**installation, "model_values_preserved": True,
            "installation_preserves_rng": True, "optimizer_scheduler_preserve_rng": True,
            "expected_initial_hash_checked": expected_initial_hash is not None,
            "expected_initial_model_sha256": expected_initial_hash,
            "historical_initial_rng_available": False},
        initial_lr=float(config["lr"]), weight_decay=float(config["weight_decay"]),
        milestones=tuple(config["scheduler_milestones"]), gamma=float(config.get("scheduler_gamma", 0.1)),
        inactive_bn_state={name: value.detach().clone() for name, value in model.state_dict().items()
                           if legacy_adapter.inactive_bn(name)})
    verify_state(state, 0)
    _require(not core.differences(initial_rng, core.capture_rng({}, rt.device)),
             "Initialisatiecontrole verbruikte RNG.")
    return state


def verify_state(state, steps):
    """Check ordinary Adam/BN counters and absence of any scale parameters."""
    _require(type(steps) is int and steps >= 0, "Ongeldige Adamstapteller.")
    model, optimizer = state.model, state.optimizer
    _require(state.method == "msb_bestexp" and getattr(model, "_mantissa_method", None) == state.method,
             "Onjuiste MSB-trainingsmethode.")
    legacy_training.assert_policy(model, "joint")
    _require(sum(parameter.numel() for parameter in model.parameters()) == 132954,
             "Onverwacht parameteraantal; er mogen geen schaalparameters bijkomen.")
    _require(not any("phi" in name.split(".") for name, _ in model.named_parameters()),
             "Deze fixed64-proef mag geen phi bevatten.")
    for name in legacy_adapter.TARGET_LAYERS:
        quantizer = model.get_submodule(name).quantize_fn
        _require(quantizer.method == state.method and not list(quantizer.parameters())
                 and not quantizer.state_dict(), "Quantizer bevat verkeerde methode of blijvende schaalstaat.")
    all_parameters = list(model.parameters())
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    _require(len(optimizer_parameters) == len({id(parameter) for parameter in optimizer_parameters})
             and {id(parameter) for parameter in optimizer_parameters} == {id(p) for p in all_parameters},
             "Adam-parametergroep verschilt van de oorspronkelijke modelparameters.")
    _require(all(parameter.dtype == torch.float32 and parameter.requires_grad
                 and bool(torch.isfinite(parameter).all()) for parameter in all_parameters),
             "Niet-eindige, bevroren of niet-FP32 modelparameter.")
    active = dict(_active_named_parameters(model))
    _require(all(id(parameter) in {id(p) for p in active.values()} for parameter in optimizer.state),
             "Ongebruikte BN-bank of vreemde parameter heeft Adamstaat.")
    if steps == 0:
        _require(not optimizer.state, "Verse Adam bevat reeds geschiedenis.")
        adam = {"all_active_adam_steps_correct": True, "moments_fp32_finite": True,
                "unused_bn_has_no_optimizer_state": True}
    else:
        _require(all(parameter in optimizer.state for parameter in active.values()),
                 "Ontbrekende actieve Adamstaat.")
        adam = legacy_training.validate_adam_state(model, optimizer, steps)
    bn = legacy_training.validate_bn(model, steps)
    current_state = model.state_dict()
    _require(not core.differences({name: current_state[name] for name in state.inactive_bn_state},
                                  state.inactive_bn_state), "Ongebruikte BN-bank gewijzigd.")
    _require(all(parameter.grad is None for name, parameter in model.named_parameters()
                 if legacy_adapter.inactive_bn(name)), "Gradient in ongebruikte BN-bank.")
    _require(all(bool(torch.isfinite(value).all()) for value in current_state.values()),
             "Niet-eindige modelbuffer.")
    expected_milestones = {epoch: state.milestones.count(epoch) for epoch in state.milestones}
    _require(dict(state.scheduler.milestones) == expected_milestones
             and state.scheduler.gamma == state.gamma
             and state.scheduler.base_lrs == [state.initial_lr], "Het oorspronkelijke LR-schema is gewijzigd.")
    expected_lr = state.initial_lr
    for milestone in state.milestones:
        if milestone <= state.scheduler.last_epoch:
            expected_lr *= state.gamma
    _require(state.scheduler.get_last_lr() == [expected_lr]
             and all(float(group["lr"]) == expected_lr and float(group["weight_decay"]) == state.weight_decay
                     for group in optimizer.param_groups), "LR of weight decay wijkt af van het oorspronkelijke schema.")
    return {"optimizer_steps": steps, "Adam_checks": adam, "BN_checks": bn,
            "BN_counts": bn["counts"], "inactive_BN_unchanged": True,
            "no_phi_or_scale_parameters": True, "model_parameter_count": 132954,
            "fixed_scale_factor": 64, "scale_exponent_offset": 6,
            "absolute_scales_are_dynamic": True, "learning_rate": expected_lr,
            "original_100_epoch_scheduler": True}


def _gradient_audit(state):
    result = legacy_training.validate_gradients(state.model)
    return result["active_gradient_tensors"], result["l2_norm"]


def run_phase(ctx, state, loader, epoch, train, out, log):
    """Execute one author train/eval phase; scheduler stepping stays with caller.

    ``out`` must already be a new phase directory. Evaluation predictions are
    indexed by the supplied tracked indices, or by evaluation order when the
    loader supplies only input/label pairs. No additional model pass is made.
    """
    _require(type(train) is bool and type(epoch) is int
             and epoch >= (1 if train else 0), "Ongeldige lokale epoch/fase.")
    out = Path(out)
    _require(out.is_dir(), "De uitvoermap voor deze fase ontbreekt.")
    _require(len(loader) > 0, "Lege fase is niet toegestaan.")
    model, optimizer, rt = state.model, state.optimizer, ctx.rt
    lr = float(optimizer.param_groups[0]["lr"])
    start_steps = _optimizer_steps(state)
    verify_state(state, start_steps)
    legacy_training.assert_policy(model, "joint")
    _require(legacy_adapter.cache_is_empty(model), "Oude broncache vóór fase.")
    before_model = support.legacy.state_digest(model.state_dict())
    before_optimizer = core.cpu_copy(optimizer.state_dict()) if not train else None
    phase_name = "train" if train else "test"
    recorder = legacy_training.JointMeterRecorder(ctx.modules.utils.AverageMeter, epoch,
        phase_name, lr, out / "batches.jsonl", "joint")
    forward, ast_hash = legacy_training.load_joint_forward(ctx.source, ctx.modules.utils, log, recorder.factory)
    soft_loss = legacy_training.CheckedSoftLoss(ctx.modules.losses.CrossEntropyLossSoft())
    lut = torch.tensor(rules.lookup(state.method), dtype=torch.uint8, device=rt.device)
    pending_modes, predictions = [], []
    current_batch = {}
    audit = {"method": state.method, "phase": phase_name, "epoch": epoch,
             "optimizer_steps_in_phase": 0, "finite_gradient_checks": 0,
             "gradient_l2_min": None, "gradient_l2_max": None,
             "mapping_checks": 0, "actual_quantizer_output_checks": 0,
             "forward_calls": {str(bit): 0 for bit in legacy_adapter.MODES}}
    observed = {bit: {"n": 0, "correct1": 0, "correct5": 0} for bit in legacy_adapter.MODES}

    class ObservedLoader:
        def __len__(self):
            return len(loader)

        def __iter__(self):
            offset = 0
            for batch in loader:
                _require(len(batch) in (2, 3), "Verwacht invoer, labels en eventueel indices.")
                inputs, labels = batch[:2]
                _require(inputs.dtype == torch.float32 and tuple(inputs.shape[1:]) == (3, 40, 40)
                         and labels.dtype == torch.int64 and labels.ndim == 1
                         and len(inputs) == len(labels), "Ongeldige batchvorm of dtype.")
                cpu_labels = labels.detach().cpu()
                _require(bool(((cpu_labels >= 0) & (cpu_labels < 10)).all()), "Ongeldige labels.")
                count = len(labels)
                if len(batch) == 3:
                    indices = batch[2].tolist() if hasattr(batch[2], "tolist") else list(batch[2])
                elif hasattr(loader, "order"):
                    indices = loader.order[offset:offset + count]
                else:
                    indices = list(range(offset, offset + count))
                _require(len(indices) == count and all(type(index) is int for index in indices),
                         "Onvolledige batchindices.")
                current_batch.clear()
                current_batch.update(labels=cpu_labels, indices=indices, predictions={})
                yield inputs, labels
                _require(set(current_batch["predictions"]) == set(legacy_adapter.MODES),
                         "Niet alle drie modi hebben deze minibatch verwerkt.")
                if not train:
                    for position, label in enumerate(cpu_labels.tolist()):
                        predictions.append({"index": indices[position], "label": label,
                            **{"w" + str(bit): current_batch["predictions"][bit][position]
                               for bit in legacy_adapter.MODES}})
                offset += count

    def model_pre(module, inputs):
        bit = module.weight_mode
        _require(bit in legacy_adapter.MODES, "Onbekende forwardmodus.")
        pending_modes.append(bit)
        audit["forward_calls"][str(bit)] += 1
        if not train and len(pending_modes) == 3:
            _require(pending_modes == [4, 6, 8], "Onjuiste evaluatievolgorde.")
            pending_modes.clear()

    def model_post(module, inputs, output):
        labels = current_batch["labels"]
        _require(tuple(output.shape) == (len(labels), 10) and bool(torch.isfinite(output).all()),
                 "Niet-eindige of onjuiste modeluitvoer.")
        top5 = output.detach().topk(5, dim=1, largest=True, sorted=True).indices.cpu()
        bit = module.weight_mode
        _require(bit not in current_batch["predictions"], "Dubbele modusvoorspelling.")
        current_batch["predictions"][bit] = top5[:, 0].tolist()
        values = observed[bit]
        values["n"] += len(labels)
        values["correct1"] += int((top5[:, 0] == labels).sum())
        values["correct5"] += int((top5 == labels[:, None]).any(dim=1).sum())

    def quantizer_post(quantizer, inputs, output):
        _require(quantizer.method == state.method and quantizer._chain is not None,
                 "Verkeerde mantissamethode of ontbrekende gedeelde bron.")
        chain, bit = quantizer._chain, quantizer.weight_mode
        _require(torch.equal(output.detach(), chain[bit].decode()),
                 "Werkelijke quantizeruitvoer verschilt van de gecontroleerde bronketen.")
        audit["actual_quantizer_output_checks"] += 1
        if bit == 4:
            _require(torch.equal(chain[6].codes, legacy_adapter.exponent_truncate(chain[8]).codes),
                     "E4→E2M3-broncodes gewijzigd.")
            _require(torch.equal(chain[4].codes, lut[chain[6].codes.long()]),
                     "Werkelijke E2M1-mapping verschilt van de exacte methodereferentie.")
            _require(torch.equal(chain[6].scale_codes.to(torch.int32), chain[8].scale_codes.to(torch.int32) + 6)
                     and torch.equal(chain[4].scale_codes, chain[6].scale_codes),
                     "De dynamische E4-schaal +6 of de gedeelde E2-schaal wijkt af.")
            audit["mapping_checks"] += 1

    def pre_step(opt, args, kwargs):
        _require(pending_modes == [8, 6, 4], "Adam vereist precies de drie trainingpasses.")
        pending_modes.clear()
        _require(legacy_adapter.cache_is_empty(model), "Broncache niet leeg vóór Adam.")
        count, norm = _gradient_audit(state)
        audit["finite_gradient_checks"] += 1
        audit["active_gradient_tensors"] = count
        audit["gradient_l2_min"] = norm if audit["gradient_l2_min"] is None else min(norm, audit["gradient_l2_min"])
        audit["gradient_l2_max"] = norm if audit["gradient_l2_max"] is None else max(norm, audit["gradient_l2_max"])

    def post_step(opt, args, kwargs):
        audit["optimizer_steps_in_phase"] += 1
        report = getattr(ctx, "report", None)
        if report is not None and report.get("command") in ("run", "resume"):
            report["training_performed"] = True
            report["actual_new_optimizer_steps"] = report.get("actual_new_optimizer_steps", 0) + 1

    handles = [model.register_forward_pre_hook(model_pre), model.register_forward_hook(model_post)]
    handles.extend(model.get_submodule(name).quantize_fn.register_forward_hook(quantizer_post)
                   for name in legacy_adapter.TARGET_LAYERS)
    if train:
        handles.extend([optimizer.register_step_pre_hook(pre_step), optimizer.register_step_post_hook(post_step)])
    if rt.device.type == "cuda":
        torch.cuda.synchronize(rt.device)
    started = time.perf_counter()
    try:
        values = forward(ObservedLoader(), model, torch.nn.CrossEntropyLoss().to(rt.device), soft_loss,
                         max(epoch - 1, 0), training=train,
                         optimizer=optimizer if train else None, sum_writer=None)
        if rt.device.type == "cuda":
            torch.cuda.synchronize(rt.device)
        seconds = time.perf_counter() - started
        rows = recorder.rows(values)
        expected_steps = len(loader) if train else 0
        _require(audit["optimizer_steps_in_phase"] == audit["finite_gradient_checks"] == expected_steps,
                 "Onjuist aantal Adamupdates/gradientcontroles.")
        _require(not pending_modes and soft_loss.calls == 2 * expected_steps,
                 "Onvolledige passes of recursieve distillatie.")
        _require(all(value == len(loader) for value in audit["forward_calls"].values()),
                 "Onjuist aantal forwards per modus.")
        layer_count = len(legacy_adapter.TARGET_LAYERS)
        _require(audit["mapping_checks"] == layer_count * len(loader)
                 and audit["actual_quantizer_output_checks"] == 3 * layer_count * len(loader),
                 "Niet alle werkelijke bronketens zijn gecontroleerd.")
        source_audit = model._last_source_audit
        _require(source_audit["source_triplets"] == len(loader)
                 and source_audit["high_encodings"] == layer_count * len(loader)
                 and source_audit["exponent_derivations"] == layer_count * len(loader)
                 and source_audit["mantissa_derivations"] == layer_count * len(loader)
                 and source_audit["scale_offsets_plus6"] == 2 * layer_count * len(loader)
                 and source_audit["scale_factor"] == 64 and source_audit["scale_exponent_offset"] == 6
                 and source_audit["source_cache_cleared"] and legacy_adapter.cache_is_empty(model),
                 "Gedeelde broncyclus, dynamische factor64 of cache-opruiming mislukt.")
        for row in rows:
            for key in ("n", "correct1", "correct5"):
                _require(row[key] == observed[row["weight_bits"]][key],
                         "Auteurmetrics en werkelijke voorspellingen verschillen: " + key)
            row["mantissa_method"] = state.method
        final_check = verify_state(state, start_steps + expected_steps)
        if not train:
            _require(support.legacy.state_digest(model.state_dict()) == before_model,
                     "Evaluatie veranderde modelwaarden of BN-statistieken.")
            _require(not core.differences(before_optimizer, core.cpu_copy(optimizer.state_dict())),
                     "Evaluatie veranderde Adamstaat.")
            _require(len(predictions) == recorder.n, "Onvolledige evaluatievoorspellingen.")
        audit.update(detached_teacher_checks=soft_loss.calls, no_extra_FP32_teacher=True,
                     shared_source_checks=source_audit, cache_empty_before_all_optimizer_steps=True,
                     actual_method_mapping_verified=True, actual_quantizer_outputs_verified=True,
                     dynamic_source_plus6_verified=True, no_phi_or_scale_parameters=True,
                     new_optimizer_steps_before=start_steps,
                     new_optimizer_steps_after=start_steps + expected_steps,
                     state_checks=final_check, original_forward_ast_sha256=ast_hash,
                     evaluation_model_and_BN_unchanged=True if not train else None,
                     evaluation_optimizer_unchanged=True if not train else None)
        return {"rows": rows, "audit": audit, "seconds": seconds, "predictions": predictions}
    finally:
        for handle in handles:
            handle.remove()
        recorder.close()
