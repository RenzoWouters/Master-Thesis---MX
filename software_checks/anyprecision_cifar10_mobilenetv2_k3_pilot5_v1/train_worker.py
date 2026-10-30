"""GPU worker for the five-epoch pilot; all long-lived work is parent-bounded."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time
import traceback

import support as s


def approved_runtime_matches(approved, runtime):
    old = approved["worker_result"]["runtime"]
    for key in ("torch", "torchvision", "cuda_build", "cudnn_version", "precision_api",
                "cudnn_benchmark", "cudnn_deterministic", "cpu_threads", "amp"):
        s.require(old.get(key) == runtime.get(key), f"Runtime gewijzigd sinds check ({key}); herhaal check.")
    for key in ("name", "total_memory_bytes", "capability"):
        s.require(old["device"].get(key) == runtime["device"].get(key), "Andere GPU dan in check; herhaal check.")


def validate_metrics(result, batch_size, training):
    s.require(result["batch_size"] == batch_size, "Onjuiste minibatchgrootte in metrics.")
    s.require(result["optimizer_steps"] == int(training), "Onjuist aantal optimizerupdates.")
    s.require(set(result["metrics"]) == {str(bit) for bit in s.BITS}, "Niet alle vijf precisies uitgevoerd.")
    for values in result["metrics"].values():
        s.require(math.isfinite(float(values["loss"])), "Niet-eindige loss.")
        s.require(0 <= values["top1"] <= values["top5"] <= 100, "Ongeldige accuracymetriek.")


class MetricAccumulator:
    def __init__(self):
        self.samples = 0
        self.sums = {str(bit): {key: 0.0 for key in ("loss", "top1", "top5")} for bit in s.BITS}

    def add(self, result, batch_size, training):
        validate_metrics(result, batch_size, training)
        self.samples += batch_size
        for bit, values in self.sums.items():
            for key in values:
                values[key] += float(result["metrics"][bit][key]) * batch_size

    def finish(self):
        s.require(self.samples > 0, "Lege epoch in metrics.")
        return {bit: {**{key: value / self.samples for key, value in values.items()}, "samples": self.samples}
                for bit, values in self.sums.items()}


def checkpoint_selection(history):
    if not history:
        return {}
    last = max(history, key=lambda row: row["epoch"])
    best = max(history, key=lambda row: (row["validation"]["32"]["top1"], -row["epoch"]))
    def entry(row):
        return {"epoch": row["epoch"], "metrics": copy.deepcopy(row["validation"]),
                "checkpoint": copy.deepcopy(row.get("checkpoint", {}))}
    return {"rule": "Highest W32/A32 full-test top1; earliest epoch on an exact tie; all precisions at that same checkpoint.",
            "last": entry(last), "best_w32": entry(best)}


def hydrate_resume_history(payload, checkpoint_path):
    """The checkpoint cannot include its own final write time/hash; use verified sidecar if present."""
    history = copy.deepcopy(payload["history"])
    path = Path(checkpoint_path)
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        row = s.json_read(sidecar)
        s.require(row["epoch"] == payload["epoch_completed"], "Checkpoint-sidecar heeft een ander epochnummer.")
        s.require(row.get("checkpoint", {}).get("sha256") == s.sha256(path), "Checkpoint-sidecar hoort niet bij deze gewichten.")
        for key in ("epoch", "global_updates", "training", "validation", "lr"):
            s.require(row.get(key) == history[-1].get(key), "Checkpoint-sidecar wijzigt historische metrics: " + key)
        row["checkpoint"]["path"] = str(path.resolve())
        history[-1] = row
    else:
        history[-1]["checkpoint"] = {"path": str(path.resolve()), "sha256": s.sha256(path), "bytes": path.stat().st_size}
        history[-1].setdefault("timing", {})["checkpoint_seconds"] = None
        history[-1]["timing"]["total_seconds"] = None
    return history


def history_csv_rows(history):
    rows = []
    for epoch in history:
        for phase in ("training", "validation"):
            for bit in s.BITS:
                rows.append({"epoch": epoch["epoch"], "phase": phase, "bits": bit, "lr": epoch["lr"],
                             **epoch[phase][str(bit)]})
    return rows


def iterate_loader(model, optimizer, loader, *, training, epoch, device, log, progress=None):
    import torch
    import engine
    start = time.perf_counter()
    iterator = iter(loader)
    accumulator = MetricAccumulator()
    rows = []
    for index in range(len(loader)):
        step_start = time.perf_counter()
        inputs, targets = next(iterator)
        data_wait = time.perf_counter() - step_start
        transfer_start = time.perf_counter()
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            torch.cuda.synchronize()
        transfer_time = time.perf_counter() - transfer_start
        compute_start = time.perf_counter()
        if training:
            metrics = engine.train_step(model, optimizer, inputs, targets,
                                        progress=log if epoch == 1 and index == 0 else None)
        else:
            metrics = engine.validation_step(model, inputs, targets)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_time = time.perf_counter() - compute_start
        elapsed = time.perf_counter() - step_start
        count = len(targets)
        accumulator.add(metrics, count, training)
        rows.append({"epoch": epoch, "phase": "training" if training else "validation",
                     "step": index + 1, "batch_size": count, "data_wait_s": data_wait,
                     "h2d_s": transfer_time, "compute_s": compute_time, "wall_s": elapsed})
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(loader):
            phase = "Train" if training else "Test"
            log(f"Epoch {epoch}/5 {phase} {index + 1}/{len(loader)}: "
                f"{elapsed:.3f} s; W32 loss={metrics['metrics']['32']['loss']:.4f}; vijf precisies.")
            if progress:
                progress({"epoch": epoch, "phase": phase, "batch": index + 1,
                          "total_batches": len(loader), "samples_processed": accumulator.samples})
    elapsed = time.perf_counter() - start
    return accumulator.finish(), rows, elapsed


def run_check(model, optimizer, result, out, stage, log):
    import torch
    import engine
    from checkpoint_support import run_resume_diagnostic
    config = s.CONFIG["synthetic_check"]
    before_bn = s.bn_snapshot(model)
    inputs = torch.randn(128, 3, 32, 32, device="cuda")
    targets = torch.randint(0, 10, (128,), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    rows = []
    count = config["warmup_steps"] + config["measured_steps"]
    for index in range(count):
        stage(f"CUDA-update {index + 1}/{count}: native batch128, vijf precisies en één Adam-update.")
        torch.cuda.synchronize()
        start = time.perf_counter()
        metrics = engine.train_step(model, optimizer, inputs, targets, validate=index == 0,
                                    progress=stage if index == 0 else None)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        validate_metrics(metrics, 128, True)
        rows.append({"step": index + 1, "batch_size": 128, "seconds": elapsed, "warmup": index == 0})
        log(f"Volledige CUDA-update: {elapsed:.3f} s.")
    stage("Kleinere eindbatches controleren: train80 en test16; alle vijf precisies.")
    validate_metrics(engine.train_step(model, optimizer, inputs[:80], targets[:80], validate=True), 80, True)
    result["bn_training_check"] = s.check_bn_updates(before_bn, s.bn_snapshot(model), count + 1)
    before_eval = s.bn_snapshot(model)
    validate_metrics(engine.validation_step(model, inputs[:16], targets[:16], validate=True), 16, False)
    result["bn_evaluation_check"] = s.check_bn_frozen(before_eval, s.bn_snapshot(model))
    result["finite_check"] = s.finite_state(model, optimizer)
    result["memory"] = s.memory_snapshot("check", count + 1)
    result["synthetic_updates"] = rows
    stage("CUDA-checkpoint/hervatcontrole op kleine synthetische batches.")
    result["resume_diagnostic"] = run_resume_diagnostic(device="cuda")
    s.require(result["resume_diagnostic"]["status"] == "PASS", "CUDA-hervatcontrole niet geslaagd.")
    result["accuracy_interpretation"] = "Synthetic check only; no CIFAR-10 accuracy or model training result."
    result["eligible_check"] = True
    result["status"] = "PASS"


def run_pilot(model, optimizer, result, out, args, stage, log, persist):
    import torch
    from checkpoint_support import apply_learning_rate, load_checkpoint, restore_checkpoint, save_checkpoint
    from data_support import make_full_loaders
    stage("Volledige CIFAR-10-data controleren: 50.000 train / 10.000 test.")
    train_loader, val_loader, info, train_generator, val_generator = make_full_loaders(
        args.data_root, seed=s.CONFIG["seed"], workers=0, download=args.download)
    result["dataset"] = info
    s.require(len(train_loader) == 391 and len(val_loader) == 79, "Onvolledige epochloaders.")
    if args.checkpoint:
        stage("Volledige epochcheckpoint laden; eerdere resultaten blijven behouden.")
        payload = load_checkpoint(args.checkpoint, expected_experiment=s.CONFIG["experiment"],
                                  expected_fingerprint=s.package_fingerprint()["sha256"])
        s.require(payload["protocol"] == s.CONFIG, "Checkpoint gebruikt een ander proefprotocol.")
        s.require(payload["epoch_completed"] < args.stop_after_epoch,
                  "Stop-epoch moet groter zijn dan de opgeslagen volledige epoch.")
        result["resume"] = restore_checkpoint(payload, model, optimizer, train_generator, val_generator,
            device="cuda", expected_runtime=result["runtime"], expected_dataset_identity=info["identity"])
        result["resume"]["resumed"] = True
        result["resume"]["source_checkpoint"] = str(Path(args.checkpoint).resolve())
        result["resume"]["source_sha256"] = s.sha256(args.checkpoint)
        result["history"] = hydrate_resume_history(payload, args.checkpoint)
        result["completed_epochs"] = payload["epoch_completed"]
        result["global_updates"] = payload["global_updates"]
    else:
        result["resume"] = {"resumed": False, "fresh_initialization": True,
                            "diagnostic_timing_weights_used": False}
    result["initial_completed_epochs"] = result["completed_epochs"]
    result["stop_after_epoch"] = args.stop_after_epoch
    result["accuracy_interpretation"] = "Five-epoch pilot on full CIFAR-10; early learning curves, not final paper reproduction."
    persist()
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    for epoch in range(result["completed_epochs"] + 1, args.stop_after_epoch + 1):
        epoch_start = time.perf_counter()
        lr = apply_learning_rate(optimizer, epoch - 1)
        stage(f"Start epoch {epoch}/5, learning rate {lr:g}; volledige train- en testset.")
        before_bn = s.bn_snapshot(model)
        torch.cuda.reset_peak_memory_stats()

        def progress(value):
            result["active_progress"] = value
            persist()

        training, train_steps, train_seconds = iterate_loader(
            model, optimizer, train_loader, training=True, epoch=epoch, device=device, log=log, progress=progress)
        s.require(all(row["samples"] == 50_000 for row in training.values()), "Niet alle 50.000 trainingsbeelden gebruikt.")
        bn_training = s.check_bn_updates(before_bn, s.bn_snapshot(model), 391)
        before_eval = s.bn_snapshot(model)
        validation, validation_steps, validation_seconds = iterate_loader(
            model, optimizer, val_loader, training=False, epoch=epoch, device=device, log=log, progress=progress)
        s.require(all(row["samples"] == 10_000 for row in validation.values()), "Niet alle 10.000 testbeelden gebruikt.")
        bn_evaluation = s.check_bn_frozen(before_eval, s.bn_snapshot(model))
        finite = s.finite_state(model, optimizer)
        result["global_updates"] += 391
        path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        row = {"epoch": epoch, "lr": lr, "global_updates": result["global_updates"],
               "training": training, "validation": validation,
               "timing": {"train_seconds": train_seconds, "validation_seconds": validation_seconds,
                          "checkpoint_seconds": None, "total_seconds": None},
               "memory": s.memory_snapshot("epoch", epoch), "checkpoint": {"path": str(path)},
               "bn_training_check": bn_training, "bn_evaluation_check": bn_evaluation, "finite_check": finite}
        # The checkpoint holds all metrics and RNG at the completed epoch; its
        # own final write time/hash is appended only in the verified sidecar.
        prospective_history = result["history"] + [row]
        selection = checkpoint_selection(prospective_history)
        best = selection["best_w32"]
        stage(f"Epoch {epoch}: checkpoint opslaan; beste W32-checkpoint tot nu toe epoch {best['epoch']}.")
        saved = save_checkpoint(path, model, optimizer, epoch_completed=epoch,
            global_updates=result["global_updates"], history=prospective_history,
            train_generator=train_generator, val_generator=val_generator, protocol=s.CONFIG,
            package_fingerprint=s.package_fingerprint()["sha256"], runtime=result["runtime"],
            dataset_identity=info["identity"], best_epoch=best["epoch"], best_top1=best["metrics"]["32"]["top1"])
        row["checkpoint"] = saved
        row["timing"]["checkpoint_seconds"] = saved["seconds"]
        row["timing"]["total_seconds"] = time.perf_counter() - epoch_start
        s.json_write(path.with_suffix(".json"), row)
        result["history"].append(row)
        result["completed_epochs"] = epoch
        result["last_checkpoint"] = saved
        result["checkpoint_selection"] = checkpoint_selection(result["history"])
        result["best_epoch"] = result["checkpoint_selection"]["best_w32"]["epoch"]
        result["best_top1"] = result["checkpoint_selection"]["best_w32"]["metrics"]["32"]["top1"]
        result["active_progress"] = {"epoch": epoch, "phase": "epoch_complete"}
        s.csv_write(out / f"epoch_{epoch:04d}_steps.csv", train_steps + validation_steps)
        s.csv_write(out / "epoch_metrics.csv", history_csv_rows(result["history"]))
        s.json_write(out / "history.json", result["history"])
        persist()
        scores = "; ".join(f"W{bit}/A{bit}={validation[str(bit)]['top1']:.2f}%" for bit in (32, 8, 4, 2, 1))
        log(f"EPOCH {epoch}/5 VOLTOOID: {scores}; {row['timing']['total_seconds']:.1f} s.")

    result["status"] = "PASS" if result["completed_epochs"] == 5 else "PAUSED"
    result["pilot_complete"] = result["completed_epochs"] == 5
    result["full_reference_must_start_from_zero"] = True
    result["checkpoint_selection"] = checkpoint_selection(result["history"])
    log("Pilot van vijf epochs voltooid." if result["pilot_complete"] else
        f"Geplande pauze na epoch {result['completed_epochs']}; hervatten kan binnen deze pilot tot epoch 5.")


def run(args):
    import torch
    import engine
    from mobilenet_model import build_model, model_summary
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = s.Logger(out / "worker.log")
    result = {"schema_version": 1, "experiment": s.CONFIG["experiment"], "status": "RUNNING",
              "kind": args.kind, "eligible_check": False, "protocol": s.CONFIG,
              "history": [], "completed_epochs": 0, "global_updates": 0,
              "long_training_started": False, "exact_paper_reproduction": False,
              "memory_scope": "Per-epoch PyTorch allocated/reserved peak counters; not Windows physical residency.",
              "cuda_resume_policy": "Full state restored; cudnn.benchmark=True, no bitexact CUDA guarantee."}
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
        s.require(args.kind == "check" or 1 <= args.stop_after_epoch <= 5, "Maximaal vijf pilotepochs toegestaan.")
        s.verify_upstream()
        stage("CUDA en FP32-instellingen controleren.")
        result["runtime"] = s.configure_cuda()
        d = result["runtime"]["device"]
        log(f"GPU: {d['name']}; allocatorbudget {d['allocator_budget_bytes'] / 1024**3:.2f} GiB.")
        if args.kind == "pilot":
            s.require(bool(args.approved_report), "Een geslaagde check ontbreekt.")
            approved_runtime_matches(s.json_read(args.approved_report), result["runtime"])
        stage("Nieuw MobileNetV2-model initialiseren; K=3, vijf BN-banken per locatie.")
        model = build_model().to(device="cuda", dtype=torch.float32)
        optimizer = engine.create_optimizer(model)
        result["model_summary"] = model_summary(model)
        if args.kind == "check":
            run_check(model, optimizer, result, out, stage, log)
        else:
            run_pilot(model, optimizer, result, out, args, stage, log, persist)
        exit_code = 0
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="OOM", error=str(exc), traceback=traceback.format_exc())
        log("CUDA-OOM binnen het vastgelegde budget; de batchgrootte is niet gewijzigd.")
        log(result["traceback"])
    except KeyboardInterrupt:
        result.update(status="INTERRUPTED", error="Onderbroken; alleen volledig opgeslagen epochs zijn hervatbaar.")
        exit_code = 130
        log(result["error"])
    except Exception as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log(result["traceback"])
    finally:
        persist()
        log("WORKERSTATUS: " + result["status"])
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("check", "pilot"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--approved-report")
    parser.add_argument("--data-root")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--stop-after-epoch", type=int, choices=range(1, 6), default=5)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
