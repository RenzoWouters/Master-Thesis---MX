"""One bounded-parent GPU experiment: cuDNN heuristic selection, no training run.

The existing v1_1 engine/model/optimizer are supplied through ``reference``.
Only cudnn.benchmark changes; memory observations sit outside measured regions.
"""
from __future__ import annotations

import statistics
import time
import traceback


def run_worker(args, reference):
    """Write a check-only result in args.worker_out and return 0, 3 or 1.

    The parent owns the wall timeout. This function never starts ImageNet,
    saves model tensors, changes the source package, or grants a pilot PASS.
    """
    import torch
    import engine

    c = reference.c
    out = args.worker_out.resolve()
    out.mkdir(exist_ok=False)
    log = c.Log(out / "worker.log")
    began = time.monotonic()
    result = {
        "mode": "serial_dp4", "candidate": "cudnn_benchmark_false", "status": "RUNNING",
        "eligible_as_pilot_check": False,
        "backend_change": {"cudnn_benchmark": {"baseline": True, "candidate": False}},
        "input_shape": [256, 3, 224, 224], "global_batch": 256,
        "precision_bits": list(c.BITS), "batchnorm_shard_sizes": [64, 64, 64, 64],
        "reference_bn_policy": True, "warmup_updates": 1, "measured_updates": 2,
        "synthetic_data": True, "training_accuracy_reported": False,
        "optimizer_updates_completed": 0, "step_seconds": [], "memory_updates": [],
        "timing_scope": "Complete five-precision forward/backward/SGD step; inputs already on GPU; "
                        "excludes ImageNet I/O, progress writes and boundary memory observations.",
        "memory_counter_scope": "PyTorch allocated/reserved counters and CUDA mem_get_info; "
                                "not a measurement of physical VRAM residency or Windows shared memory.",
        "comparison_note": "Same instrumented warmup and uninstrumented timed engine calls as v1_1; "
                           "cuDNN kernel choices may differ numerically, so no bit-exact CUDA claim.",
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
        # Called only at an already synchronized boundary, never inside timing.
        free, total = torch.cuda.mem_get_info()
        return {"allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "free_bytes": int(free), "device_total_bytes": int(total),
                "elapsed_seconds": time.monotonic() - began}

    def begin_update(kind, index):
        # Preserve the peak since the previous boundary before resetting the
        # counters for one update. reset_peak_memory_stats does not free memory.
        collect_peaks()
        before = memory_state()
        torch.cuda.reset_peak_memory_stats()
        record = {"kind": kind, "index": index, "status": "RUNNING", "before": before}
        result["memory_updates"].append(record)
        c.json_write(out / "progress.json", result)
        return record

    def finish_update(record, elapsed):
        record.update(status="PASS", seconds=float(elapsed), after=memory_state(), **collect_peaks())
        result["optimizer_updates_completed"] += 1
        # Save completion before another update can start or time out.
        c.json_write(out / "progress.json", result)

    def progress(event):
        # Match v1_1 warmup instrumentation. A *_done message follows completion
        # on the GPU. The two measured engine calls do not use this callback.
        if event["stage"].endswith("_done") and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        event = dict(event, elapsed_seconds=time.monotonic() - began)
        result["last_progress"] = event
        c.json_write(out / "progress.json", result)
        details = ""
        if event.get("bit") is not None:
            details += f"; W{event['bit']}/A{event['bit']}"
        if event.get("shard") is not None:
            details += f"; groep {event['shard']}/{event['shards']}"
        log(f"serial_dp4: {event['stage']}{details}; {event['elapsed_seconds']:.1f} s")

    exit_code = 1
    try:
        progress({"stage": "cuda_initialization_start"})
        result["runtime"] = c.configure(c.CONFIG["seed"], threads=args.cpu_threads)
        torch.backends.cudnn.benchmark = False
        c.require(torch.backends.cudnn.benchmark is False, "cuDNN benchmark=False kon niet worden ingesteld.")
        result["runtime"]["cudnn_benchmark"] = False
        result["device"] = result["runtime"]["device"]
        log("GPU: " + result["device"]["name"])
        log("BACKENDPROEF: cudnn.benchmark=False; batch 256, BN-groepen 64 en FP32 blijven behouden.")
        free, total = torch.cuda.mem_get_info()
        result["free_memory_before_model_bytes"] = int(free)
        result["device_memory_bytes"] = int(total)
        torch.cuda.reset_peak_memory_stats()
        progress({"stage": "model_allocation_start"})
        model, optimizer, soft = reference.model_and_optimizer(args.source, "cuda:0")
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        x = torch.randn(256, 3, 224, 224, device="cuda", dtype=torch.float32)
        y = torch.randint(0, 1000, (256,), device="cuda")
        progress({"stage": "model_and_inputs_done"})
        log("serial_dp4: opwarming met alle vijf precisies, backward en SGD; effectieve batch 256.")
        current_update = begin_update("warmup", 0)
        start = time.perf_counter()
        engine.train_step(model, optimizer, x, y, soft, mode="serial_dp4", validate=True,
                          progress=progress)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        finish_update(current_update, elapsed)
        result["warmup_seconds"] = elapsed
        current_update = None

        for step in range(2):
            current_update = begin_update("measured", step + 1)
            progress({"stage": "measured_update_start", "update": step + 1})
            log(f"serial_dp4: gemeten volledige stap {step + 1}/2.")
            torch.cuda.synchronize()
            start = time.perf_counter()
            metrics = engine.train_step(model, optimizer, x, y, soft, mode="serial_dp4", validate=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            # Boundary queries and validity checks are deliberately not timed.
            reference.add_totals(reference.fresh_totals(), metrics)
            result["step_seconds"].append(elapsed)
            finish_update(current_update, elapsed)
            current_update = None
            progress({"stage": "measured_update_done", "update": step + 1})
            log(f"Stapduur: {elapsed:.3f} s")

        progress({"stage": "final_finite_checks_start"})
        reference.assert_finite_checkpoint_state(model, optimizer)
        progress({"stage": "final_finite_checks_done"})
        median = statistics.median(result["step_seconds"])
        c.require(median > 0, "Ongeldige gemeten stapduur.")
        result.update(status="PASS", median_step_seconds=median,
                      synthetic_images_per_second=256 / median,
                      optimizer_updates=3, logical_precision_passes=15,
                      physical_forward_backward_passes=60,
                      finite_losses_and_gradients=True,
                      finite_check_scope="Warmup losses/gradients; measured losses; final model/BN/SGD state.")
        exit_code = 0
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="OOM", error=str(exc), traceback=traceback.format_exc())
        log("GPU-GEHEUGEN ONVOLDOENDE. Geen kleinere batch, AMP of automatisch alternatief toegepast.")
        exit_code = 3
    except KeyboardInterrupt:
        result.update(status="FAIL", error="De synthetische GPU-controle werd onderbroken.")
        log(result["error"])
    except Exception as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log("WORKER FAIL: " + str(exc))
        log(traceback.format_exc())
    finally:
        if current_update is not None:
            current_update["status"] = result["status"]
            current_update["completed"] = False
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
        result["ended_utc"] = c.utc_now()
        c.json_write(out / "result.json", result)
        log("WORKER STATUS: " + result["status"])
        log.close()
    return exit_code
