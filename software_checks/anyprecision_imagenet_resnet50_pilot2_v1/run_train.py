"""ResNet-50 / ImageNet: isolated GPU checks and a strictly two-epoch pilot.

No MX, no full reference training, no dataset downloads, no overwriting old runs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import common as c

CSV_FIELDS = ("epoch", "split", "bit", "loss", "top1", "top5", "samples", "seconds")


def parser():
    p = argparse.ArgumentParser(description="GitHub ResNet-50: GPU-controle en ImageNet-pilot van twee epochs.")
    p.add_argument("command", choices=["check", "data", "run", "resume", "selftest", "_memory_worker"])
    p.add_argument("--project-root", type=Path, default=c.default_project_root())
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--imagenet-root", type=Path, default=None,
                   help="Map met train/<wnid>/ en val/<wnid>/. Geen downloads.")
    p.add_argument("--check-report", type=Path, default=None,
                   help="Een geslaagd run_report.json van check; anders nieuwste passende check.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Uitsluitend een eigen checkpoint na epoch 1 van dit pilotpakket.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--mode", choices=["native256", "serial_dp4"], help=argparse.SUPPRESS)
    p.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    return p


def verify_fixed_contract():
    cfg = c.CONFIG
    c.require(cfg["global_batch"] == 256 and cfg["reference_replicas"] == 4,
              "Dit pakket vereist globale batch 256 en vier referentiegroepen.")
    c.require(cfg["pilot_epochs"] == 2 and cfg["reference_epochs"] == 80,
              "Dit pakket mag uitsluitend twee pilotepochs uitvoeren.")
    c.require(cfg["bits"] == [1, 2, 4, 8, 32] and cfg["model"] == "resnet50q",
              "Onverwacht model of precisielijst.")
    c.require(cfg["lr"] == 0.5 and cfg["momentum"] == 0.9 and cfg["weight_decay"] == 1e-5
              and cfg["lr_milestones"] == [45, 60, 70] and cfg["lr_gamma"] == 0.1,
              "De vaste GitHub-hyperparameters zijn gewijzigd.")
    c.require(cfg["execution"] == "serial_dp4" and not cfg["amp"],
              "Alleen de gecontroleerde FP32-uitvoering is toegestaan.")
    c.require(cfg["optimizer"] == "SGD" and cfg["fp32_precision"] == "ieee"
              and cfg["bn_policy"] == "retain_first_shard_buffers_once_per_precision"
              and cfg["cudnn_benchmark"] is True,
              "De optimizer-, BN- of backendconfiguratie is gewijzigd.")


def model_and_optimizer(source, device):
    factory, soft_factory, optimizer_factory = c.import_source(source)
    model = factory(list(c.BITS), num_classes=1000).to(device)
    optimizer = optimizer_factory(model, "sgd", c.CONFIG["lr"], c.CONFIG["weight_decay"])
    c.require(all(p.dtype.is_floating_point for p in model.parameters()), "Onverwachte modelparameter.")
    return model, optimizer, soft_factory()


def cpu_selftests(source, log):
    import torch
    import test_engine
    import test_runner
    c.require(__debug__, "Voer de controle zonder Python -O of PYTHONOPTIMIZE uit; de zelftests gebruiken assertions.")
    torch.set_num_threads(2)
    tests = test_engine.run_selftests()
    runner_tests = test_runner.run_selftests()
    factory, _, _ = c.import_source(source)
    c.reseed(c.CONFIG["seed"])
    model = factory(list(c.BITS), num_classes=1000).eval()
    parameter_count = sum(p.numel() for p in model.parameters())
    c.require(all(p.dtype == torch.float32 for p in model.parameters()), "FP32-modelcontrole mislukt.")
    banks = [m for m in model.modules() if hasattr(m, "bn_dict")]
    c.require(bool(banks) and all(set(m.bn_dict) == {str(b) for b in c.BITS} for m in banks),
              "Het model heeft niet de verwachte vijf BN-banken.")
    x = torch.randn(2, 3, 64, 64)
    shapes = {}
    with torch.no_grad():
        for bit in reversed(c.BITS):
            model.apply(lambda m: setattr(m, "wbit", bit))
            model.apply(lambda m: setattr(m, "abit", bit))
            y = model(x)
            c.require(tuple(y.shape) == (2, 1000) and bool(torch.isfinite(y).all()),
                      "Auteursmodel levert geen eindige 1000-klassenuitvoer.")
            shapes[str(bit)] = list(y.shape)
    del model, x, y
    gc.collect()
    log("CPU-zelftests: onafhankelijke replica-controle en auteursmodel met vijf precisies geslaagd.")
    return {"engine": tests, "runner": runner_tests, "upstream_model": {"parameter_count": parameter_count,
            "switchable_bn_modules": len(banks), "output_shapes": shapes,
            "input_shape": [2, 3, 64, 64],
            "purpose": "Kleine functionele controle; geen geheugenschatting en geen datasetaccuracy."}}


def memory_worker(args):
    """Fresh process for every candidate, so a native256 OOM does not taint the pilot path."""
    import torch
    import engine
    out = args.worker_out.resolve()
    out.mkdir(exist_ok=False)
    log = c.Log(out / "worker.log")
    result = {"mode": args.mode, "status": "RUNNING", "input_shape": [256, 3, 224, 224],
              "global_batch": 256, "precision_bits": list(c.BITS),
              "batchnorm_shard_sizes": [256] if args.mode == "native256" else [64] * 4,
              "warmup_updates": 1, "measured_updates": 2,
              "synthetic_data": True, "training_accuracy_reported": False,
              "timing_scope": "Complete five-precision forward/backward/SGD step; tensors already on GPU; excludes ImageNet I/O.",
              "reference_bn_policy": args.mode == "serial_dp4"}
    exit_code = 1
    try:
        result["runtime"] = c.configure(c.CONFIG["seed"], threads=args.cpu_threads)
        result["device"] = result["runtime"]["device"]
        log("GPU: " + result["device"]["name"])
        free, total = torch.cuda.mem_get_info()
        result["free_memory_before_model_bytes"] = free
        result["device_memory_bytes"] = total
        torch.cuda.reset_peak_memory_stats()
        model, optimizer, soft = model_and_optimizer(args.source, "cuda:0")
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        x = torch.randn(256, 3, 224, 224, device="cuda", dtype=torch.float32)
        y = torch.randint(0, 1000, (256,), device="cuda")
        log(f"{args.mode}: opwarming met alle vijf precisies, backward en SGD; effectieve batch 256.")
        engine.train_step(model, optimizer, x, y, soft, mode=args.mode, validate=True)
        torch.cuda.synchronize()
        times = []
        for step in range(2):
            log(f"{args.mode}: gemeten volledige stap {step + 1}/2.")
            torch.cuda.synchronize()
            start = time.perf_counter()
            metrics = engine.train_step(model, optimizer, x, y, soft, mode=args.mode, validate=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            log(f"Stapduur: {elapsed:.3f} s")
        tensors = [v for v in model.state_dict().values() if v.is_floating_point()]
        c.require(bool(torch.stack([torch.isfinite(v).all() for v in tensors]).all().item()),
                  "Niet-eindige model- of BatchNorm-waarden na de GPU-test.")
        result.update(status="PASS", step_seconds=times,
                      median_step_seconds=statistics.median(times),
                      synthetic_images_per_second=256 / statistics.median(times),
                      optimizer_updates=3, logical_precision_passes=15,
                      physical_forward_backward_passes=15 if args.mode == "native256" else 60)
        # Synthetic loss is not a research result; only keep the finite-check outcome.
        result["finite_losses_and_gradients"] = True
        exit_code = 0
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="OOM", error=str(exc), traceback=traceback.format_exc())
        log("GPU-GEHEUGEN ONVOLDOENDE voor deze uitvoering. Geen kleinere batch of AMP toegepast.")
        exit_code = 3
    except Exception as exc:
        result.update(status="FAIL", error=str(exc), traceback=traceback.format_exc())
        log("WORKER FAIL: " + str(exc))
        log(traceback.format_exc())
    finally:
        if torch.cuda.is_initialized():
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        else:
            result["peak_allocated_bytes"] = None
            result["peak_reserved_bytes"] = None
        result["ended_utc"] = c.utc_now()
        c.json_write(out / "result.json", result)
        log("WORKER STATUS: " + result["status"])
        log.close()
    return exit_code


def execute_memory_test(mode, source, out, args, log):
    target = out / ("memory_" + mode)
    command = [sys.executable, "-u", str(c.ROOT / "run_train.py"), "_memory_worker",
               "--mode", mode, "--source", str(source), "--worker-out", str(target),
               "--cpu-threads", str(args.cpu_threads)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    log("Start afzonderlijk proces: " + mode)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", env=env)
    try:
        for line in process.stdout:
            log(line.rstrip())
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    result_path = target / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {"mode": mode, "status": "FAIL", "error": "Het GPU-testproces eindigde zonder rapport."}
    result["process_returncode"] = code
    c.require((result["status"] == "PASS") == (code == 0),
              "De exitcode en het GPU-testrapport spreken elkaar tegen.")
    return result


def command_check(args, out, report, log, source):
    report["environment"] = c.check_environment(log)
    report["selftests"] = cpu_selftests(source, log)
    report["memory_tests"] = {}
    log("native256 is een geheugendiagnose. Een OOM daar is toegestaan als serial_dp4 daarna slaagt.")
    for mode in ("native256", "serial_dp4"):
        report["memory_tests"][mode] = execute_memory_test(mode, source, out, args, log)
        c.json_write(out / "run_report.json", report)
    selected = report["memory_tests"]["serial_dp4"]
    report["selected_execution"] = "serial_dp4"
    report["device"] = selected.get("device")
    if selected["status"] == "OOM":
        report["status"] = "BLOCKED_GPU_MEMORY"
        log("De referentiegroep van 64 past niet. Deel de feedbackbundel; run wordt geblokkeerd.")
        return
    c.require(selected["status"] == "PASS", "De gekozen serial_dp4-uitvoering is niet geslaagd.")
    # A native256 OOM is an informative diagnostic, not a failure of the selected path.
    c.require(report["memory_tests"]["native256"]["status"] in ("PASS", "OOM"),
              "De native256-diagnose gaf een fout die geen geheugenlimiet is; eerst onderzoeken.")
    report["status"] = "PASS"
    log("CHECK PASS: batch 256; vier groepen van 64; vijf precisies; één SGD-update; BN-replicabeleid gecontroleerd.")
    log("Dit is een synthetische controle. Nog geen ImageNet-training uitgevoerd.")


def find_check(args, report):
    if args.check_report:
        candidates = [args.check_report]
    else:
        base = Path(args.output_root) if args.output_root else (
            args.project_root / "mx_runs" / "anyprecision_imagenet_resnet50_pilot2")
        candidates = sorted(base.glob("*_check/run_report.json"), reverse=True)
    for candidate in candidates:
        try:
            proof = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (proof.get("command") == "check" and proof.get("status") == "PASS" and
                proof.get("package_signature", {}).get("sha256") == report["package_signature"]["sha256"] and
                proof.get("memory_tests", {}).get("serial_dp4", {}).get("status") == "PASS"):
            return candidate.resolve(), proof
    raise RuntimeError("Geen passende geslaagde check gevonden. Voer eerst run_train.py check uit met ditzelfde pakket.")


def fresh_totals():
    return {str(b): {"loss_sum": 0., "top1_correct": 0, "top5_correct": 0, "samples": 0} for b in c.BITS}


def add_totals(totals, metrics):
    for bit, m in metrics["by_bit"].items():
        n = int(m["samples"])
        c.require(n > 0 and math.isfinite(float(m["loss"])),
                  f"Niet-eindige loss of lege batch bij W{bit}/A{bit}; training gestopt.")
        c.require(0 <= int(m["top1_correct"]) <= int(m["top5_correct"]) <= n,
                  f"Ongeldige accuracy-tellers bij W{bit}/A{bit}.")
        totals[str(bit)]["loss_sum"] += float(m["loss"]) * n
        totals[str(bit)]["top1_correct"] += int(m["top1_correct"])
        totals[str(bit)]["top5_correct"] += int(m["top5_correct"])
        totals[str(bit)]["samples"] += n


def metric_rows(totals, epoch, split, seconds):
    rows = []
    for bit in c.BITS:
        m = totals[str(bit)]
        n = m["samples"]
        c.require(n > 0, "Geen voorbeelden verwerkt.")
        rows.append(dict(epoch=epoch, split=split, bit=bit, loss=m["loss_sum"] / n,
                         top1=100 * m["top1_correct"] / n, top5=100 * m["top5_correct"] / n,
                         samples=n, seconds=seconds))
    return rows


def write_metrics(out, rows):
    path = out / "metrics.csv"
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


class StopAfterEpoch:
    def __init__(self, log):
        self.requested = False
        self.log = log
        self.previous = None

    def handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt()
        self.requested = True
        self.log("Stop aangevraagd: lopende epoch, validatie en checkpoint afwerken. Nogmaals Ctrl+C stopt direct.")

    def __enter__(self):
        self.previous = signal.signal(signal.SIGINT, self.handler)
        return self

    def __exit__(self, *unused):
        signal.signal(signal.SIGINT, self.previous)


def expected_lr(completed_epochs):
    return c.CONFIG["lr"] * c.CONFIG["lr_gamma"] ** sum(completed_epochs >= x for x in c.CONFIG["lr_milestones"])


def run_epoch(model, optimizer, soft, loader, epoch, training, out, report, args, log):
    import torch
    import engine
    totals = fresh_totals()
    split = "train" if training else "val"
    model.train(training)
    torch.cuda.synchronize()
    started = time.perf_counter()
    updates = 0
    physical = 0
    for index, (x, y) in enumerate(loader):
        x = x.to("cuda:0", non_blocking=True)
        y = y.to("cuda:0", non_blocking=True)
        if training:
            stats = engine.train_step(model, optimizer, x, y, soft, mode="serial_dp4",
                                      validate=(index == 0))
            updates += 1
            physical += int(stats["forwards"])
        else:
            stats = engine.eval_batch(model, x, y)
        add_totals(totals, stats)
        if index == 0 or (index + 1) % args.log_every == 0 or index + 1 == len(loader):
            seen = totals["32"]["samples"]
            elapsed = time.perf_counter() - started
            top1 = 100 * totals["32"]["top1_correct"] / seen
            log(f"Epoch {epoch}/2 {split}: batch {index + 1}/{len(loader)}; "
                f"W32/A32 top1={top1:.3f}%; {seen} beelden; {elapsed:.1f} s")
            report["progress"] = {"epoch": epoch, "split": split, "completed_batches": index + 1,
                                  "total_batches": len(loader), "samples": seen, "seconds": elapsed}
            c.json_write(out / "run_report.json", report)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    c.require(totals["32"]["samples"] == len(loader.dataset), "Niet de volledige datasetsplit verwerkt.")
    return metric_rows(totals, epoch, split, elapsed), elapsed, updates, physical


def load_resume(args, report, dataset_manifest):
    import torch
    c.require(args.checkpoint is not None and args.checkpoint.is_file(), "Geef een bestaand pilotcheckpoint met --checkpoint.")
    # This package encodes RNG metadata using safe primitive values/tensors.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    c.require(saved.get("package_id") == c.CONFIG["package_id"] and saved.get("stage") == "pilot2",
              "Dit checkpoint is geen checkpoint van dit pilotpakket.")
    c.require(saved.get("package_signature") == report["package_signature"]["sha256"],
              "De pakketcode verschilt van de opgeslagen pilot.")
    c.require(saved.get("config") == c.CONFIG, "Het trainingsrecept van het checkpoint verschilt.")
    c.require(saved.get("completed_epoch") == 1, "Alleen een onvoltooide pilot na epoch 1 kan worden hervat.")
    c.require(saved.get("loader_workers") == args.workers,
              "Hervatten vereist hetzelfde aantal dataloader-workers als de eerste pilotepoch.")
    c.require(saved.get("optimizer_updates") == 5005 and len(saved.get("metrics", [])) == 10,
              "Het checkpoint bevat niet één volledige pilotepoch met alle vijf precisies.")
    old = saved["dataset_identity"]
    now = dataset_identity(dataset_manifest)
    c.require(old == now, "De datasetstructuur of bestandenlijst verschilt van de pilot.")
    timing_path = args.checkpoint.with_name(args.checkpoint.stem + "_metadata.json")
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if timing.get("epoch") == 1:
            for key in ("checkpoint_seconds", "epoch_seconds"):
                value = timing.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                    saved["epochs"][0][key] = value
    return saved


def dataset_identity(manifest):
    return {"class_to_idx_sha256": manifest["class_to_idx_sha256"],
            "train_count": manifest["train_count"], "val_count": manifest["val_count"],
            "train_filelist_sha256": manifest["splits"]["train"]["filelist_sha256"],
            "val_filelist_sha256": manifest["splits"]["val"]["filelist_sha256"]}


def assert_finite_checkpoint_state(model, optimizer):
    import torch
    values = list(model.state_dict().values())
    values.extend(value for state in optimizer.state.values() for value in state.values()
                  if isinstance(value, torch.Tensor))
    for value in values:
        if value.is_floating_point():
            c.require(value.dtype == torch.float32 and bool(torch.isfinite(value).all().item()),
                      "Niet-eindige of niet-FP32-model/optimizerstaat; geen geslaagd checkpoint opgeslagen.")


def command_run(args, out, report, log, source):
    import torch
    from data_support import audit_imagenet, build_loaders
    report["environment"] = c.check_environment(log)
    check_path, check = find_check(args, report)
    for name in ("torch", "torchvision", "numpy", "Pillow"):
        c.require(check["environment"]["packages"][name] == report["environment"]["packages"][name],
                  f"Versie van {name} verschilt van de geslaagde check; voer check opnieuw uit.")
    c.require(check["environment"].get("torch_cuda_build") == report["environment"].get("torch_cuda_build"),
              "De CUDA-build verschilt van de geslaagde check; voer check opnieuw uit.")
    report["check_report"] = str(check_path)
    report["memory_tests"] = check["memory_tests"]
    report["selftests"] = check["selftests"]
    root = (args.imagenet_root or args.project_root / "data" / "imagenet").resolve()
    log("ImageNet controleren: " + str(root))
    dataset = audit_imagenet(root, strict=True, progress=log)
    report["dataset"] = dataset
    c.json_write(out / "dataset_manifest.json", dataset)
    report["runtime"] = c.configure(c.CONFIG["seed"], threads=args.cpu_threads)
    report["device"] = report["runtime"]["device"]
    expected_device = check["memory_tests"]["serial_dp4"]["device"]
    for field in ("name", "total_memory_bytes", "compute_capability"):
        c.require(report["device"][field] == expected_device[field],
                  "De GPU verschilt van de geslaagde check; voer de check opnieuw uit.")
    c.require(shutil.disk_usage(out).free >= 2 * 1024 ** 3,
              "Minder dan 2 GiB vrije schijfruimte voor pilotcheckpoints en resultaten.")
    saved = load_resume(args, report, dataset) if args.command == "resume" else None
    train_loader, val_loader, generator = build_loaders(root, batch_size=256,
                    workers=args.workers, seed=c.CONFIG["seed"], val_batch_size=64)
    # Dataset checks and CPU selftests must never consume the pilot's initialization RNG.
    c.reseed(c.CONFIG["seed"])
    model, optimizer, soft = model_and_optimizer(source, "cuda:0")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 60, 70], gamma=0.1)
    start_epoch = 0
    history = []
    report["epochs"] = []
    report["optimizer_updates"] = 0
    report["physical_forward_backward_passes"] = 0
    report["best_epoch_by_highest_precision"] = None
    report["best_top1"] = None
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = saved["completed_epoch"]
        history = saved["metrics"]
        report["epochs"] = saved["epochs"]
        report["optimizer_updates"] = saved["optimizer_updates"]
        report["physical_forward_backward_passes"] = saved["physical_forward_backward_passes"]
        report["best_epoch_by_highest_precision"] = saved["best_epoch_by_highest_precision"]
        report["best_top1"] = saved["best_top1"]
        report["best_checkpoint"] = str(args.checkpoint.resolve())
        c.restore_rng(saved["rng"], generator)
        report["resumed_from"] = str(args.checkpoint.resolve())
    report["latest_completed_epoch"] = start_epoch
    report["parameter_count"] = sum(p.numel() for p in model.parameters())
    report["dataloader"] = {"workers": args.workers, "training_batch": 256,
                            "validation_batch": 64, "drop_last": False,
                            "train_batches_per_epoch": len(train_loader),
                            "val_batches_per_epoch": len(val_loader)}
    c.require(len(train_loader) == 5005, "Onverwacht aantal ImageNet-trainingsbatches.")
    c.require(math.isclose(optimizer.param_groups[0]["lr"], expected_lr(start_epoch)),
              "Learning rate komt niet overeen met de volledige 80-epochplanning.")
    write_metrics(out, history)
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    log("PILOT: twee epochs vanaf nul (of hervatten eigen epoch 1); volledige 80-epoch-LR-planning blijft behouden.")
    log("Batch 256 = vier groepen van 64 op één GPU; vijf precisies; FP32; één SGD-update per minibatch.")
    log("Ctrl+C eenmaal: stoppen na de epoch. Nogmaals: direct stoppen.")
    torch.cuda.reset_peak_memory_stats()
    with StopAfterEpoch(log) as stopper:
        for completed in range(start_epoch, 2):
            epoch = completed + 1
            epoch_start = time.perf_counter()
            lr_used = float(optimizer.param_groups[0]["lr"])
            train_rows, train_seconds, updates, physical = run_epoch(
                model, optimizer, soft, train_loader, epoch, True, out, report, args, log)
            val_rows, val_seconds, _, _ = run_epoch(
                model, optimizer, soft, val_loader, epoch, False, out, report, args, log)
            scheduler.step()
            lr_next = float(optimizer.param_groups[0]["lr"])
            c.require(math.isclose(lr_next, expected_lr(epoch)), "LR-planning wijkt af.")
            history.extend(train_rows + val_rows)
            score = next(r["top1"] for r in val_rows if r["bit"] == 32)
            is_best = report["best_top1"] is None or score > report["best_top1"]
            if is_best:
                report["best_top1"] = score
                report["best_epoch_by_highest_precision"] = epoch
            report["optimizer_updates"] += updates
            report["logical_precision_passes"] = report["optimizer_updates"] * 5
            report["physical_forward_backward_passes"] += physical
            assert_finite_checkpoint_state(model, optimizer)
            path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            timing = {"epoch": epoch, "train_seconds": train_seconds, "val_seconds": val_seconds,
                      "checkpoint_seconds": None, "epoch_seconds": None,
                      "lr_used": lr_used, "lr_next": lr_next, "checkpoint_file": str(path),
                      "samples_train": len(train_loader.dataset), "samples_val": len(val_loader.dataset)}
            report["epochs"].append(timing)
            payload = {"schema_version": 1, "package_id": c.CONFIG["package_id"], "stage": "pilot2",
                       "package_signature": report["package_signature"]["sha256"], "config": c.CONFIG,
                       "completed_epoch": epoch, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "rng": c.rng_state(generator), "metrics": history,
                       "epochs": report["epochs"], "dataset_identity": dataset_identity(dataset),
                       "optimizer_updates": report["optimizer_updates"],
                       "physical_forward_backward_passes": report["physical_forward_backward_passes"],
                       "loader_workers": args.workers,
                       "best_epoch_by_highest_precision": report["best_epoch_by_highest_precision"],
                       "best_top1": report["best_top1"]}
            save_start = time.perf_counter()
            c.save_checkpoint(path, payload)
            report["latest_completed_epoch"] = epoch
            timing["checkpoint_seconds"] = time.perf_counter() - save_start
            timing["epoch_seconds"] = time.perf_counter() - epoch_start
            c.json_write(checkpoint_dir / f"epoch_{epoch:03d}_metadata.json", timing)
            report["last_checkpoint"] = str(path)
            if is_best:
                report["best_checkpoint"] = str(path)
            c.json_write(out / "checkpoint_selection.json", {
                "last_completed_epoch": epoch, "last_checkpoint": str(path),
                "best_epoch_by_highest_precision": report["best_epoch_by_highest_precision"],
                "best_checkpoint": report["best_checkpoint"],
                "selection_bit": 32, "selection_metric": "ImageNet validation top1",
                "note": "Alle precisies vergelijken op hetzelfde gekozen checkpoint. Geen epoch-80-resultaat in deze pilot."})
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            write_metrics(out, history)
            c.json_write(out / "run_report.json", report)
            log("Epoch " + str(epoch) + " klaar: " + "; ".join(
                f"W{r['bit']}/A{r['bit']}={r['top1']:.3f}%" for r in reversed(val_rows)))
            log(f"Epochduur incl. validatie en checkpoint: {timing['epoch_seconds'] / 60:.2f} min.")
            if epoch == 2:
                # Use warmed epoch 2 for the repeated cost, while keeping measured epoch 1.
                first = report["epochs"][0].get("epoch_seconds")
                second = timing["epoch_seconds"]
                report["projection80_seconds"] = (first if first is not None else second) + 79 * second
                report["projection_note"] = "Verse 80-epoch-run: gemeten epoch 1 + 79 maal epoch 2; inclusief validatie/checkpoint, exclusief data-audit; extrapolatie, geen garantie."
                log(f"Indicatie verse 80 epochs: {report['projection80_seconds'] / 3600:.2f} uur bij dezelfde laptopcondities.")
            if stopper.requested or (out / "STOP_AFTER_EPOCH").exists():
                report["status"] = "PASS" if epoch == 2 else "PAUSED"
                break
        else:
            report["status"] = "PASS"
    if report["latest_completed_epoch"] == 2:
        report["status"] = "PASS"
        log("PILOT PASS: twee volledige epochs. Geen automatische voortzetting naar 80 epochs.")


def main(argv=None):
    args = parser().parse_args(argv)
    c.require(args.workers >= 0 and args.cpu_threads > 0 and args.log_every > 0,
              "Ongeldige worker-, thread- of loginstelling.")
    verify_fixed_contract()
    if args.command == "_memory_worker":
        return memory_worker(args)
    if args.command not in ("resume",) and args.checkpoint is not None:
        raise SystemExit("--checkpoint mag alleen bij resume; een verse pilot laadt nooit een warmstart.")
    out = c.new_output(args.project_root, args.command, args.output_root)
    log = c.Log(out / "console.log")
    report = {"schema_version": 1, "package_id": c.CONFIG["package_id"], "command": args.command,
              "status": "RUNNING", "started_utc": c.utc_now(), "output_directory": str(out),
              "config": c.CONFIG, "package_signature": c.package_signature(),
              "epochs": [], "latest_completed_epoch": 0,
              "authorized_scope": "synthetic checks and two real ImageNet pilot epochs only"}
    began = time.perf_counter()
    log("RESULTATENMAP: " + str(out))
    log("RESNET-50 / IMAGENET: GitHub any-precision; W/A 1,2,4,8,32; batch 256; FP32; geen MX.")
    try:
        c.copy_notebook(out)
        source, report["source"] = c.prepare_source(out)
        if args.command == "check":
            command_check(args, out, report, log, source)
        elif args.command == "data":
            from data_support import audit_imagenet
            root = args.imagenet_root or args.project_root / "data" / "imagenet"
            report["dataset"] = audit_imagenet(root, strict=True, progress=log)
            c.json_write(out / "dataset_manifest.json", report["dataset"])
            report["status"] = "PASS"
            log("DATA PASS: structuur en aantallen passen bij volledige ImageNet-1k. Geen training.")
        elif args.command == "selftest":
            report["environment"] = c.check_environment(log, pip_check=False)
            report["selftests"] = cpu_selftests(source, log)
            report["status"] = "PASS"
        else:
            command_run(args, out, report, log, source)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "Direct gestopt; alleen voltooide epochs hebben een hervatbaar checkpoint."
        log(report["error"])
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        log("FAIL: " + str(exc))
        log(report["traceback"])
    finally:
        report["ended_utc"] = c.utc_now()
        report["wall_seconds"] = time.perf_counter() - began
        c.json_write(out / "run_report.json", report)
        log("EINDSTATUS: " + report["status"])
        log("Rapport: " + str(out / "run_report.json"))
        log("Deel feedback_bundle.zip uit deze map. Checkpoints en dataset blijven lokaal.")
        log.close()
        c.feedback_bundle(out)
    return 0 if report["status"] in ("PASS", "PAUSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
