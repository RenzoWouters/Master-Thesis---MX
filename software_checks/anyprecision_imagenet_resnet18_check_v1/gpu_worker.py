"""Synthetic ResNet-18 updates; the CPU parent owns the 300-second deadline."""
from __future__ import annotations

import math
import statistics
import time
import traceback

import support as s


def model_and_optimizer(sources, device):
    import torch
    from resnet18_model import build_model, build_soft_loss
    model = build_model(sources["anyprecision"], sources["truncquant"], num_classes=1000).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=s.CONFIG["lr"],
                                momentum=s.CONFIG["momentum"], weight_decay=s.CONFIG["weight_decay"])
    soft = build_soft_loss(sources["anyprecision"]).to(device)
    return model, optimizer, soft


def check_metrics(metrics):
    s.require(metrics["samples"] == 256 and metrics["optimizer_updates"] == 1,
              "Een controle-update moet precies 256 beelden en één SGD-update gebruiken.")
    s.require(metrics["mode"] == "serial_dp4" and metrics["chunk_sizes"] == [64] * 4 and
              metrics["forwards"] == 20 and metrics["replicas_used"] == 4,
              "De vier BN-groepen van 64 of vijf precisiepasses zijn gewijzigd.")
    s.require(set(metrics["by_bit"]) == {str(bit) for bit in s.BITS}, "Niet alle vijf precisies zijn gemeten.")
    for value in metrics["by_bit"].values():
        s.require(math.isfinite(value["loss"]) and value["samples"] == 256,
                  "Niet-eindige loss of onjuist aantal samples.")
        s.require(0 <= value["top1_correct"] <= value["top5_correct"] <= 256,
                  "Ongeldige synthetische top-k-tellingen.")


def bn_counter_snapshot(model):
    import torch
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    names = [name for name, _ in modules]
    values = torch.stack([module.num_batches_tracked.detach() for _, module in modules]).cpu().tolist()
    return dict(zip(names, map(int, values)))


def check_bn_updates(before, after):
    s.require(before.keys() == after.keys() and before, "De BatchNorm-structuur is gewijzigd.")
    shared, switched = 0, 0
    for name, old in before.items():
        expected = 1 if ".bn_dict." in name else 5
        s.require(after[name] - old == expected,
                  f"Onjuist aantal BN-updates voor {name}: {after[name] - old}, verwacht {expected}.")
        if expected == 1:
            switched += 1
        else:
            shared += 1
    s.require(shared == 4 and switched == 80, "Onverwachte ResNet-18 BatchNorm-bankindeling.")
    return {"status": "PASS", "shared_bn_modules": shared, "switchable_bn_bank_modules": switched,
            "updates_per_shared_bn": 5, "updates_per_precision_bn_bank": 1,
            "scope": "One retained replica-zero forward per precision; BN64 is unchanged."}


def assert_finite_state(model, optimizer):
    import torch
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point():
            s.require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                      "Niet-eindige of niet-FP32 modelwaarde: " + name)
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                s.require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                          "Niet-eindige of niet-FP32 SGD-state: " + name)


def run_worker(args):
    import torch
    import engine
    from resnet18_model import model_report

    out = args.worker_out.resolve()
    out.mkdir(exist_ok=False)
    log = s.Log(out / "worker.log")
    began = time.monotonic()
    result = {
        "model": "ResNet-18 reconstructed", "mode": "serial_dp4", "status": "RUNNING",
        "eligible_as_pilot_check": False, "exact_original_paper_reproduction_claim": False,
        "input_shape": [256, 3, 224, 224], "global_batch": 256,
        "precision_bits": list(s.BITS), "batchnorm_shard_sizes": [64] * 4,
        "warmup_updates": 1, "measured_updates": 2,
        "optimizer": {"name": "SGD", "lr": s.CONFIG["lr"], "momentum": s.CONFIG["momentum"],
                      "weight_decay": s.CONFIG["weight_decay"]},
        "synthetic_data": True, "training_accuracy_reported": False,
        "optimizer_updates_completed": 0, "step_seconds": [], "memory_updates": [], "bn_update_checks": [],
        "timing_scope": "One global256 update across five precisions and four BN64 shards, including backward/SGD. "
                        "Inputs already on GPU; excludes data loading, boundary telemetry and final validity checks.",
        "memory_counter_scope": "PyTorch allocated/reserved and CUDA mem_get_info, not physical VRAM residency "
                                "or Windows shared-memory usage. A killed update has no final peak observation.",
        "comparison_note": "Previous R50 checks provide hardware context only: architecture and LR differ. "
                           "CPU equivalence is not a claim of bit-exact CUDA execution.",
    }
    cumulative = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    current_update = None

    def collect_peaks():
        peaks = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                 "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        for name, value in peaks.items():
            cumulative[name] = value if cumulative[name] is None else max(cumulative[name], value)
        result.update(cumulative)
        return peaks

    def memory_state():
        free, total = torch.cuda.mem_get_info()
        return {"allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "free_bytes": int(free), "device_total_bytes": int(total),
                "elapsed_seconds": time.monotonic() - began}

    def begin_update(kind, index):
        collect_peaks()
        before = memory_state()
        torch.cuda.reset_peak_memory_stats()
        record = {"kind": kind, "index": index, "status": "RUNNING", "before": before}
        result["memory_updates"].append(record)
        s.json_write(out / "progress.json", result)
        return record

    def finish_update(record, elapsed, bn_check):
        record.update(status="PASS", seconds=float(elapsed), after=memory_state(), **collect_peaks())
        result["optimizer_updates_completed"] += 1
        result["bn_update_checks"].append(bn_check)
        s.json_write(out / "progress.json", result)

    def progress(event):
        if event["stage"].endswith("_done") and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        event = dict(event, elapsed_seconds=time.monotonic() - began)
        result["last_progress"] = event
        s.json_write(out / "progress.json", result)
        details = ""
        if event.get("bit") is not None:
            details += f"; W{event['bit']}/A{event['bit']}"
        if event.get("shard") is not None:
            details += f"; groep {event['shard']}/{event['shards']}"
        log(f"ResNet-18: {event['stage']}{details}; {event['elapsed_seconds']:.1f} s")

    code = 1
    try:
        sources = s.verify_source_copy(args.source_root)
        progress({"stage": "cuda_initialization_start"})
        result["runtime"] = s.configure(s.CONFIG["seed"], threads=args.cpu_threads)
        result["device"] = result["runtime"]["device"]
        log("GPU: " + result["device"]["name"])
        log("RESNET-18: FP32; W/A 1,2,4,8,32; batch256 als vier BN64-groepen; SGD lr=0.3.")
        free, total = torch.cuda.mem_get_info()
        result["free_memory_before_model_bytes"] = int(free)
        result["device_memory_bytes"] = int(total)
        torch.cuda.reset_peak_memory_stats()
        progress({"stage": "model_allocation_start"})
        model, optimizer, soft = model_and_optimizer(sources, "cuda:0")
        result["architecture"] = model_report(model)
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        x = torch.randn(256, 3, 224, 224, device="cuda", dtype=torch.float32)
        y = torch.randint(0, 1000, (256,), device="cuda")
        progress({"stage": "model_and_inputs_done"})
        log("Opwarming: één volledige update met alle vijf precisies, backward en SGD.")
        before_bn = bn_counter_snapshot(model)
        current_update = begin_update("warmup", 0)
        start = time.perf_counter()
        metrics = engine.train_step(model, optimizer, x, y, soft, mode="serial_dp4", validate=True, progress=progress)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        check_metrics(metrics)
        bn_check = check_bn_updates(before_bn, bn_counter_snapshot(model))
        finish_update(current_update, elapsed, bn_check)
        result["warmup_seconds"] = elapsed
        current_update = None

        for step in range(2):
            before_bn = bn_counter_snapshot(model)
            current_update = begin_update("measured", step + 1)
            progress({"stage": "measured_update_start", "update": step + 1})
            log(f"Gemeten volledige update {step + 1}/2; geen extra synchronisatie binnen de engine.")
            torch.cuda.synchronize()
            start = time.perf_counter()
            metrics = engine.train_step(model, optimizer, x, y, soft, mode="serial_dp4", validate=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            check_metrics(metrics)
            bn_check = check_bn_updates(before_bn, bn_counter_snapshot(model))
            result["step_seconds"].append(elapsed)
            finish_update(current_update, elapsed, bn_check)
            current_update = None
            progress({"stage": "measured_update_done", "update": step + 1})
            log(f"Stapduur: {elapsed:.3f} s")

        progress({"stage": "final_finite_checks_start"})
        assert_finite_state(model, optimizer)
        progress({"stage": "final_finite_checks_done"})
        median = statistics.median(result["step_seconds"])
        s.require(median > 0, "Ongeldige gemeten stapduur.")
        result.update(status="PASS", median_step_seconds=median,
                      synthetic_images_per_second=256 / median,
                      logical_precision_passes=15, model_forward_backward_pairs=60,
                      finite_losses_and_gradients=True,
                      finite_check_scope="Warmup losses/gradients; measured losses; final model/BN/SGD state.",
                      interpretation="PASS means the short synthetic check succeeded, not that long training is feasible.")
        code = 0
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="OOM", error=str(exc), traceback=traceback.format_exc())
        log("GPU-geheugen onvoldoende voor de afgesproken configuratie.")
        code = 3
    except KeyboardInterrupt:
        result.update(status="FAIL", error="De synthetische GPU-controle werd onderbroken.")
        log(result["error"])
    except Exception as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log("WORKER FAIL: " + str(exc))
        log(result["traceback"])
    finally:
        if current_update is not None:
            current_update.update(status=result["status"], completed=False)
        if torch.cuda.is_initialized():
            try:
                peaks = collect_peaks()
                result["final_memory"] = memory_state()
                if current_update is not None:
                    current_update.update(peaks)
            except Exception as exc:
                result["memory_report_error"] = str(exc)
        else:
            result.update(cumulative)
        result["ended_utc"] = s.utc_now()
        s.json_write(out / "result.json", result)
        log("WORKER STATUS: " + result["status"])
        log.close()
    return code
