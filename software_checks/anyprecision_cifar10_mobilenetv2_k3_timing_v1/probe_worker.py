"""Bounded CUDA work. The parent owns deadlines and process-tree cleanup."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import traceback

import support as s


def validate_metrics(result, batch_size, training):
    s.require(result["batch_size"] == batch_size, "Onjuiste minibatchgrootte in metrics.")
    s.require(result["optimizer_steps"] == int(training), "Onjuist aantal optimizerupdates.")
    s.require(set(result["metrics"]) == {str(bit) for bit in s.BITS}, "Niet alle vijf precisies uitgevoerd.")
    for values in result["metrics"].values():
        s.require(math.isfinite(float(values["loss"])), "Niet-eindige loss.")
        s.require(0 <= values["top1"] <= values["top5"] <= 100, "Ongeldige accuracymetriek.")


def aggregate(rows):
    n = sum(row["batch_size"] for row in rows)
    if not n:
        return {}
    return {str(bit): {**{field: sum(row["metrics"][str(bit)][field] * row["batch_size"]
                                      for row in rows) / n for field in ("loss", "top1", "top5")},
                       "samples": n}
            for bit in s.BITS}


def approved_runtime_matches(approved, runtime):
    old = approved["worker_result"]["runtime"]
    for key in ("torch", "torchvision", "cuda_build", "cudnn_version", "precision_api",
                "cudnn_benchmark", "cudnn_deterministic", "cpu_threads", "amp"):
        s.require(old.get(key) == runtime.get(key),
                  f"Runtime gewijzigd sinds check ({key}); voer eerst opnieuw check uit.")
    for key in ("name", "total_memory_bytes", "capability"):
        s.require(old["device"].get(key) == runtime["device"].get(key),
                  "Andere GPU dan in de geslaagde check; voer opnieuw check uit.")


def save_checkpoint(out, model, optimizer, result):
    import torch
    checkpoint_path = out / "diagnostic_checkpoint.pt"
    temp = checkpoint_path.with_suffix(".pt.tmp")
    torch.cuda.synchronize()
    began = time.perf_counter()
    payload = {"schema_version": 1, "experiment": s.CONFIG["experiment"],
               "protocol": s.CONFIG, "partial_epoch": True, "epoch_completed": 0,
               "optimizer_updates": result["completed_updates"],
               "training_samples_seen": result["completed_updates"] * s.CONFIG["batch_size"],
               "resume_supported": False, "full_training_must_restart_from_zero": True,
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "rng_torch_cpu": torch.get_rng_state(), "rng_torch_cuda": torch.cuda.get_rng_state_all(),
               "package_fingerprint": s.package_fingerprint()["sha256"]}
    torch.save(payload, temp)
    temp.replace(checkpoint_path)
    seconds = time.perf_counter() - began
    return {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size,
            "seconds": seconds, "sha256": s.sha256(checkpoint_path),
            "partial_epoch": True, "in_feedback_bundle": False, "resume_supported": False}


def run(args):
    import torch
    import engine
    from mobilenet_model import build_model, model_summary
    from timing_analysis import summarize_timing

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = s.Logger(out / "worker.log")
    result = {"schema_version": 1, "experiment": s.CONFIG["experiment"],
              "status": "RUNNING", "kind": args.kind, "eligible_check": False,
              "protocol": s.CONFIG, "completed_updates": 0,
              "train_rows": [], "validation_rows": [], "synthetic_rows": [], "memory_rows": [],
              "timing_scope": "Each step contains all five precision forwards; training also contains five backwards and one Adam update. No extra factor five.",
              "memory_scope": "PyTorch caching-allocator allocated/reserved/peak counters and CUDA device-free bytes; not physical residency or Windows shared GPU memory.",
              "long_training_started": False, "exact_paper_reproduction": False}
    began = time.monotonic()

    def persist():
        result["elapsed_seconds"] = time.monotonic() - began
        s.json_write(out / "worker_result.json", result)

    def stage(message):
        result["last_progress"] = {"message": str(message), "elapsed_seconds": time.monotonic() - began}
        log(message)
        persist()

    exit_code = 1
    try:
        s.verify_upstream()
        stage("CUDA en FP32-instellingen controleren.")
        result["runtime"] = s.configure_cuda()
        device = result["runtime"]["device"]
        log("GPU: " + device["name"])
        log(f"CUDA-allocatorbudget: {device['allocator_budget_bytes'] / 1024**3:.2f} GiB; "
            f"fysieke capaciteit {device['total_memory_bytes'] / 1024**3:.2f} GiB.")
        if args.kind == "real":
            s.require(bool(args.approved_report), "Een geslaagde check ontbreekt.")
            approved_runtime_matches(s.json_read(args.approved_report), result["runtime"])
            stage("Bestaande CIFAR-10-bestanden controleren; alleen met --download wordt zo nodig gedownload.")
            from data_support import make_loaders
            train_loader, val_loader, data_info = make_loaders(
                args.data_root, seed=s.CONFIG["seed"], workers=s.CONFIG["runtime"]["data_workers"],
                download=args.download, train_steps=s.CONFIG["real_data_probe"]["train_steps"],
                validation_examples=s.CONFIG["real_data_probe"]["validation_examples"])
            result["dataset"] = data_info
            log("CIFAR-10: 50.000 train / 10.000 test; proef gebruikt 12.800 trainbeelden en 1.000 testbeelden.")
        stage("Nieuw MobileNetV2-model initialiseren; K=3, native batch128, vijf precisies.")
        model = build_model().to(device="cuda", dtype=torch.float32)
        optimizer = engine.create_optimizer(model)
        result["model_summary"] = model_summary(model)
        log(f"Model: {sum(parameter.numel() for parameter in model.parameters()):,} FP32-parameters.")
        before_bn = s.bn_snapshot(model)
        torch.cuda.reset_peak_memory_stats()
        result["memory_rows"].append(s.memory_snapshot("initial", 0))
        if args.kind == "synthetic":
            x = torch.randn(s.CONFIG["batch_size"], *s.CONFIG["image_shape"], device="cuda")
            y = torch.randint(0, 10, (s.CONFIG["batch_size"],), device="cuda")
            config = s.CONFIG["synthetic_check"]
            count = config["warmup_steps"] + config["measured_steps"]
            for index in range(count):
                warmup = index < config["warmup_steps"]
                stage(f"{'Opwarming' if warmup else 'Meting'} {index + 1}/{count}: "
                      "32 -> 8 -> 4 -> 2 -> 1, backward en Adam.")
                torch.cuda.synchronize()
                start = time.perf_counter()
                metrics = engine.train_step(model, optimizer, x, y, validate=warmup,
                                            progress=stage if warmup else None)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                validate_metrics(metrics, s.CONFIG["batch_size"], True)
                result["completed_updates"] += 1
                result["synthetic_rows"].append({"step": index + 1, "batch_size": len(y),
                    "warmup": warmup, "data_wait_s": 0., "h2d_s": 0., "compute_s": elapsed,
                    "wall_s": elapsed, "loss_by_bit": {bit: values["loss"] for bit, values in metrics["metrics"].items()}})
                result["memory_rows"].append(s.memory_snapshot("synthetic", index + 1))
                log(f"Volledige update: {elapsed:.3f} s; piek gereserveerd "
                    f"{result['memory_rows'][-1]['peak_reserved_bytes'] / 1024**3:.2f} GiB.")
                persist()
            result["bn_training_check"] = s.check_bn_updates(before_bn, s.bn_snapshot(model), count)
            result["timing_summary"] = summarize_timing(result["synthetic_rows"],
                warmup_train_steps=config["warmup_steps"], synthetic=True)
            result["accuracy_interpretation"] = "Synthetic inputs: no accuracy result; no CIFAR-10 training."
            result["eligible_check"] = True
        else:
            config = s.CONFIG["real_data_probe"]
            result["accuracy_interpretation"] = "Diagnostic partial-epoch training; 1,000-example evaluation only. Not a trained-model or paper accuracy comparison."
            iterator_start = time.perf_counter()
            train_iterator = iter(train_loader)
            result["training_iterator_startup_seconds"] = time.perf_counter() - iterator_start
            loop_start = time.perf_counter()
            for index in range(config["train_steps"]):
                step_start = time.perf_counter()
                x, y = next(train_iterator)
                data_wait = time.perf_counter() - step_start
                h2d_start = time.perf_counter()
                x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                h2d = time.perf_counter() - h2d_start
                compute_start = time.perf_counter()
                metrics = engine.train_step(model, optimizer, x, y, validate=False,
                                            progress=log if index == 0 else None)
                torch.cuda.synchronize()
                compute = time.perf_counter() - compute_start
                elapsed = time.perf_counter() - step_start
                validate_metrics(metrics, s.CONFIG["batch_size"], True)
                result["completed_updates"] += 1
                row = {"step": index + 1, "batch_size": len(y), "data_wait_s": data_wait,
                       "h2d_s": h2d, "compute_s": compute, "wall_s": elapsed, "metrics": metrics["metrics"]}
                result["train_rows"].append(row)
                result["memory_rows"].append(s.memory_snapshot("train", index + 1))
                if index < 3 or (index + 1) % 10 == 0:
                    log(f"Train {index + 1}/{config['train_steps']}: {elapsed:.3f} s/update; "
                        f"W32 loss={metrics['metrics']['32']['loss']:.4f}; alle vijf precisies voltooid.")
                    persist()
            result["train_loop_wall_seconds"] = time.perf_counter() - loop_start
            result["train_rows_wall_seconds"] = sum(row["wall_s"] for row in result["train_rows"])
            result["train_instrumentation_overhead_seconds"] = max(0., result["train_loop_wall_seconds"] - result["train_rows_wall_seconds"])
            result["bn_training_check"] = s.check_bn_updates(before_bn, s.bn_snapshot(model), config["train_steps"])
            result["finite_check"] = s.finite_state(model, optimizer)
            stage("Diagnosecheckpoint opslaan; toekomstige volledige training begint opnieuw vanaf nul.")
            result["checkpoint"] = save_checkpoint(out, model, optimizer, result)
            before_eval = s.bn_snapshot(model)
            stage("Tijdmeting evaluatie: 1.000 beelden, alle vijf precisies, geen updates.")
            val_iterator = iter(val_loader)
            val_loop_start = time.perf_counter()
            for index in range(len(val_loader)):
                step_start = time.perf_counter()
                x, y = next(val_iterator)
                data_wait = time.perf_counter() - step_start
                h2d_start = time.perf_counter()
                x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                h2d = time.perf_counter() - h2d_start
                compute_start = time.perf_counter()
                metrics = engine.validation_step(model, x, y)
                torch.cuda.synchronize()
                compute = time.perf_counter() - compute_start
                elapsed = time.perf_counter() - step_start
                validate_metrics(metrics, len(y), False)
                result["validation_rows"].append({"step": index + 1, "batch_size": len(y),
                    "data_wait_s": data_wait, "h2d_s": h2d, "compute_s": compute,
                    "wall_s": elapsed, "metrics": metrics["metrics"]})
                result["memory_rows"].append(s.memory_snapshot("validation", index + 1))
                log(f"Validatie {index + 1}/{len(val_loader)}: {elapsed:.3f} s, {len(y)} beelden, vijf precisies.")
                persist()
            result["validation_loop_wall_seconds"] = time.perf_counter() - val_loop_start
            result["bn_evaluation_check"] = s.check_bn_frozen(before_eval, s.bn_snapshot(model))
            result["metrics_summary"] = {"training": aggregate(result["train_rows"]),
                                         "validation": aggregate(result["validation_rows"])}
            result["timing_summary"] = summarize_timing(result["train_rows"], result["validation_rows"],
                warmup_train_steps=config["warmup_train_steps"],
                warmup_validation_steps=config["warmup_validation_steps"],
                checkpoint_seconds=result["checkpoint"]["seconds"], synthetic=False)
            estimate = result["timing_summary"]["projections"]
            for epochs in (5, 100, 400):
                value = estimate[str(epochs)]["total_hours"]
                log(f"SCHATTING {epochs} epochs: {value:.2f} uur inclusief vijf-precisievalidatie en checkpoint per epoch.")
        result["finite_check"] = s.finite_state(model, optimizer)
        result["memory_rows"].append(s.memory_snapshot("final", result["completed_updates"]))
        result["status"] = "PASS"
        log("PASS: de begrensde proef is voltooid. Geen volledige epoch- of meerdaagse training gestart.")
        exit_code = 0
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="OOM", error=str(exc), traceback=traceback.format_exc(), eligible_check=False)
        log("OOM binnen het ingestelde CUDA-allocatorbudget; batchgrootte is niet automatisch gewijzigd.")
        exit_code = 3
    except KeyboardInterrupt:
        result.update(status="INTERRUPTED", error="Gebruiker onderbrak de proef.", eligible_check=False)
        exit_code = 130
    except Exception as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc(), eligible_check=False)
        log(result["traceback"])
    finally:
        # A hung CUDA operation is handled by the parent's absolute deadline.
        # Do not introduce a fresh synchronize() here after a CUDA failure.
        persist()
        for key, filename in (("train_rows", "train_steps.csv"), ("validation_rows", "validation_steps.csv"),
                              ("synthetic_rows", "synthetic_steps.csv"), ("memory_rows", "memory.csv")):
            s.csv_write(out / filename, result[key])
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("synthetic", "real"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--approved-report")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
