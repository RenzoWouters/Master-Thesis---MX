"""Read-only tables and plots for the five-epoch MobileNetV2 pilot.

Only recorded complete epochs are shown. This module never trains, evaluates
checkpoints, reads model pickle files or modifies experiment results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = "anyprecision_cifar10_mobilenetv2_k3_pilot5"
BITS = (32, 8, 4, 2, 1)
COLORS = {32: "#355F82", 8: "#40877B", 4: "#B77C34", 2: "#89709E", 1: "#AE6555"}
INK = "#263D4B"
PHASE_COLORS = {"train_seconds": "#355F82", "validation_seconds": "#40877B",
                "checkpoint_seconds": "#B77C34", "other_seconds": "#ABB6C1"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Verwacht een JSON-object in {path}.")
    return value


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _matches(report: dict[str, Any]) -> bool:
    return report.get("experiment") == EXPERIMENT


def discover_reports(project_root: str | Path | None = None) -> list[Path]:
    """Search bounded known locations, not the whole user's disk."""
    roots = []
    if project_root is not None:
        roots.append(Path(project_root).expanduser())
    for origin in (Path.cwd(), Path(__file__).resolve().parent):
        roots.extend([origin, *list(origin.parents)[:5]])
    candidates = set()
    for root in dict.fromkeys(roots):
        for path in (root / "run_report.json", root / "feedback_bundle" / "run_report.json"):
            if path.is_file():
                candidates.add(path.resolve())
        parent = root / "mx_runs" / EXPERIMENT
        if parent.is_dir():
            candidates.update(p.resolve() for p in parent.glob("*/run_report.json"))
        if root.name.startswith("feedback") or root.name == EXPERIMENT:
            candidates.update(p.resolve() for p in root.glob("*/run_report.json"))
    accepted = []
    for path in candidates:
        try:
            if _matches(_read_json(path)):
                accepted.append(path)
        except (OSError, ValueError):
            continue
    return sorted(accepted, key=lambda p: (p.parent.name, p.stat().st_mtime_ns))


def load_run(run_dir: str | Path | None = None,
             project_root: str | Path | None = None) -> dict[str, Any]:
    """Load the latest report including FAIL/PAUSED, or one explicit run."""
    if run_dir is None:
        candidates = discover_reports(project_root)
        if not candidates:
            raise FileNotFoundError("Geen pilotrapport gevonden. Vul RUN_DIR in met een resultaatmap of uitgepakte feedbackbundel.")
        report_path = candidates[-1]
    else:
        path = Path(run_dir).expanduser()
        report_path = path if path.is_file() else path / "run_report.json"
        if not report_path.is_file() and path.is_dir():
            candidates = sorted(path.glob("*/run_report.json"))
            if len(candidates) == 1:
                report_path = candidates[0]
            elif len(candidates) > 1:
                raise ValueError("Meerdere runs in deze map: kies één concrete resultaatmap.")
        if not report_path.is_file():
            raise FileNotFoundError(f"Geen run_report.json in {path}.")
    report_path = report_path.resolve()
    report = _read_json(report_path)
    if not _matches(report):
        raise ValueError("Dit rapport hoort niet bij de MobileNetV2-K3-vijf-epochpilot.")
    worker = report.get("worker_result") or {}
    if not isinstance(worker, dict):
        worker = {}
    relative = report.get("worker_result_file", "worker/worker_result.json")
    worker_path = report_path.parent / relative
    if worker_path.is_file():
        sidecar = _read_json(worker_path)
        if sidecar.get("experiment", EXPERIMENT) != EXPERIMENT:
            raise ValueError("Workerbestand hoort bij een andere proef.")
        # During a running process this file can be newer than the parent.
        if not worker or worker_path.stat().st_mtime_ns >= report_path.stat().st_mtime_ns:
            worker = sidecar
    return {"report": report, "worker": worker, "report_path": report_path,
            "run_dir": report_path.parent}


def completed_history(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = context["worker"].get("history", [])
    if not isinstance(rows, list):
        raise ValueError("Epochgeschiedenis moet een lijst zijn.")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Ongeldige rij in de epochgeschiedenis.")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= 5:
            raise ValueError("Pilotgeschiedenis moet epochs 1 t/m 5 bevatten.")
        result.append(row)
    result.sort(key=lambda row: row["epoch"])
    if len({row["epoch"] for row in result}) != len(result):
        raise ValueError("Dubbele epoch in het rapport; niet stilzwijgend samengevoegd.")
    return result


def _is_resumed(context: dict[str, Any]) -> bool:
    value = context["worker"].get("resume")
    if not isinstance(value, dict) or value.get("resumed") is False:
        return False
    return value.get("resumed") is True or bool(value.get("source_checkpoint"))


def overview_table(context: dict[str, Any]) -> pd.DataFrame:
    report, worker = context["report"], context["worker"]
    history = completed_history(context)
    runtime, model = worker.get("runtime", {}), worker.get("model_summary", {})
    device = runtime.get("device", {})
    if not isinstance(device, dict):
        device = {"name": device}
    resume = _is_resumed(context)
    values = [
        ("Resultaatmap", str(context["run_dir"])),
        ("Opdracht", report.get("command", "onbekend")),
        ("Status", report.get("status", "onbekend")),
        ("Laatste gerapporteerde complete epoch", history[-1]["epoch"] if history else "geen"),
        ("Pilotlimiet", "5 epochs"),
        ("Hervatting", "ja; eerdere epochgeschiedenis inbegrepen" if resume else "nee / niet gemeld"),
        ("GPU", device.get("name", "niet gemeten")),
        ("Aantal parameters incl. BN-banken", model.get("parameter_count", "niet gemeld")),
        ("Aparte BatchNorm-banken", model.get("bn_bank_count", "niet gemeld")),
        ("Dataset per complete epoch", "50.000 train / 10.000 test"),
    ]
    return pd.DataFrame(values, columns=["Onderdeel", "Waarde"])


def status_notes(context: dict[str, Any]) -> list[str]:
    report, worker = context["report"], context["worker"]
    history = completed_history(context)
    notes = []
    status = report.get("status", "onbekend")
    if status not in ("PASS", "PAUSED"):
        notes.append(f"Status {status}: bekijk de fout- of voortgangsmelding; eerder voltooide epochs kunnen wel zichtbaar zijn.")
    if status == "PAUSED":
        notes.append("De pilot is bewust gepauzeerd; er zijn nog geen vijf volledige epochs afgerond.")
    error = report.get("error") or worker.get("error")
    if error:
        notes.append("Rapportmelding: " + (json.dumps(error, ensure_ascii=False) if isinstance(error, dict) else str(error)))
    if not history:
        notes.append("Dit rapport bevat geen volledige CIFAR-10-epochs. Er worden geen accuracies of trainingscurves ingevuld.")
    else:
        notes.append(f"Getoond: {len(history)} complete epochs. Elke testscore gebruikt de officiële 10.000 testbeelden.")
        if history[-1]["epoch"] < 5:
            notes.append("De laatste getoonde epoch is nog niet epoch 5; ontbrekende epochs worden niet aangevuld.")
        notes.append("Vijf epochs beoordelen vroege leerprogressie; zij bewijzen geen convergentie of paper-eindaccuracy.")
    if _is_resumed(context):
        notes.append("De grafieken bevatten de geërfde complete epochs en de epochs van deze hervatting.")
    notes.append("Checkpointselectie gebruikt uitsluitend W32/A32; alle andere modi komen uit hetzelfde gekozen checkpoint.")
    return notes


def epoch_metrics_table(context: dict[str, Any], phase="validation") -> pd.DataFrame:
    if phase not in ("training", "validation"):
        raise ValueError("phase must be training or validation")
    rows = []
    for epoch in completed_history(context):
        values = epoch.get(phase) or {}
        for bit in BITS:
            metric = values.get(str(bit), values.get(bit, {}))
            if not isinstance(metric, dict) or not metric:
                continue
            rows.append({"Epoch": epoch["epoch"], "Precisie": bit, "Modus": f"W{bit}/A{bit}",
                         "Top-1 [%]": _number(metric.get("top1")), "Top-5 [%]": _number(metric.get("top5")),
                         "Loss": _number(metric.get("loss")), "Voorbeelden": metric.get("samples"),
                         "Learning rate": _number(epoch.get("lr"))})
    return pd.DataFrame(rows, columns=["Epoch", "Precisie", "Modus", "Top-1 [%]", "Top-5 [%]",
                                      "Loss", "Voorbeelden", "Learning rate"])


def _selections(context):
    history = completed_history(context)
    if not history:
        return []
    candidates = [row for row in history
                  if _number(row.get("validation", {}).get("32", {}).get("top1")) is not None]
    choices = [("Laatste complete epoch", history[-1])]
    if candidates:
        best = min(candidates, key=lambda row: (-float(row["validation"]["32"]["top1"]), row["epoch"]))
        choices.append(("Beste W32-checkpoint", best))
    return choices


def selection_table(context: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, epoch in _selections(context):
        for bit in BITS:
            metric = (epoch.get("validation") or {}).get(str(bit), {})
            if metric:
                rows.append({"Selectie": label, "Epoch": epoch["epoch"], "Modus": f"W{bit}/A{bit}",
                             "Precisie": bit, "Test top-1 [%]": _number(metric.get("top1")),
                             "Test top-5 [%]": _number(metric.get("top5")),
                             "Checkpoint": (epoch.get("checkpoint") or {}).get("path", "niet gemeld")})
    return pd.DataFrame(rows, columns=["Selectie", "Epoch", "Modus", "Precisie", "Test top-1 [%]",
                                      "Test top-5 [%]", "Checkpoint"])


def timing_table(context: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for epoch in completed_history(context):
        t = epoch.get("timing") or {}
        values = {key: _number(t.get(key)) for key in
                  ("train_seconds", "validation_seconds", "checkpoint_seconds", "total_seconds")}
        complete = all(value is not None and value >= 0 for value in values.values())
        other = max(0., values["total_seconds"] - sum(values[key] for key in
                    ("train_seconds", "validation_seconds", "checkpoint_seconds"))) if complete else None
        rows.append({"Epoch": epoch["epoch"], "Training [s]": values["train_seconds"],
                     "Evaluatie [s]": values["validation_seconds"], "Checkpoint [s]": values["checkpoint_seconds"],
                     "Overige epochkosten [s]": other, "Totaal [s]": values["total_seconds"],
                     "Complete tijdmeting": complete})
    return pd.DataFrame(rows, columns=["Epoch", "Training [s]", "Evaluatie [s]", "Checkpoint [s]",
                                      "Overige epochkosten [s]", "Totaal [s]", "Complete tijdmeting"])


def _forecast_basis(context):
    frame = timing_table(context)
    if frame.empty:
        return frame
    eligible = frame[frame["Complete tijdmeting"]].copy()
    # Drop epoch 1 only if another fully measured epoch exists. Missing timing
    # on a resumed checkpoint is never reconstructed from guessed averages.
    if len(eligible) > 1:
        eligible = eligible[eligible["Epoch"] != 1]
    return eligible


def forecast_table(context: dict[str, Any]) -> pd.DataFrame:
    basis = _forecast_basis(context)
    columns = ["Epochs (raming)", "Training [uur]", "Evaluatie [uur]", "Checkpoint [uur]",
               "Overig [uur]", "Totaal [uur]", "Gebaseerd op epochs", "Gemeten epochs gebruikt"]
    if basis.empty:
        return pd.DataFrame(columns=columns)
    mean = basis[["Training [s]", "Evaluatie [s]", "Checkpoint [s]", "Overige epochkosten [s]", "Totaal [s]"]].mean()
    rows = []
    for epochs in (5, 50, 100, 400):
        rows.append({"Epochs (raming)": epochs,
                     "Training [uur]": mean["Training [s]"] * epochs / 3600,
                     "Evaluatie [uur]": mean["Evaluatie [s]"] * epochs / 3600,
                     "Checkpoint [uur]": mean["Checkpoint [s]"] * epochs / 3600,
                     "Overig [uur]": mean["Overige epochkosten [s]"] * epochs / 3600,
                     "Totaal [uur]": mean["Totaal [s]"] * epochs / 3600,
                     "Gebaseerd op epochs": ", ".join(str(value) for value in basis["Epoch"]),
                     "Gemeten epochs gebruikt": len(basis)})
    return pd.DataFrame(rows, columns=columns)


def memory_table(context: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for epoch in completed_history(context):
        memory = epoch.get("memory") or {}
        allocated, reserved = (_number(memory.get(key)) for key in ("peak_allocated_bytes", "peak_reserved_bytes"))
        if allocated is None and reserved is None:
            continue
        rows.append({"Epoch": epoch["epoch"],
                     "Piek gealloceerd [GiB]": allocated / 1024 ** 3 if allocated is not None else None,
                     "Piek gereserveerd [GiB]": reserved / 1024 ** 3 if reserved is not None else None})
    return pd.DataFrame(rows, columns=["Epoch", "Piek gealloceerd [GiB]", "Piek gereserveerd [GiB]"])


def _style(ax, *, percent=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#B9C4CC")
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="#E5EAF0", linewidth=.8)
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(0, 100)


def plot_accuracy(context: dict[str, Any]):
    frame = epoch_metrics_table(context)
    if frame.empty or frame["Top-1 [%]"].notna().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(11.7, 5.1), layout="constrained")
    for bit in BITS:
        part = frame[frame["Precisie"] == bit]
        if not part.empty:
            ax.plot(part["Epoch"], part["Top-1 [%]"], marker="o", ms=5, lw=2,
                    label=f"W{bit}/A{bit}", color=COLORS[bit])
    choices = _selections(context)
    if len(choices) > 1:
        best_epoch = choices[1][1]["epoch"]
        ax.axvline(best_epoch, color="#7C8996", ls="--", lw=1,
                   label=f"Gekozen op W32: epoch {best_epoch}")
    ax.set(title="Testaccuracy · volledige CIFAR-10-testset · vijf-epochpilot",
           xlabel="Voltooide epoch", ylabel="Top-1 [%]")
    ax.set_xticks(sorted(frame["Epoch"].unique()))
    ax.legend(loc="upper left", fontsize=9, ncols=3)
    _style(ax, percent=True)
    return fig


def plot_train_validation(context: dict[str, Any]):
    train, test = epoch_metrics_table(context, "training"), epoch_metrics_table(context)
    if train.empty and test.empty:
        return None
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharex=True, sharey=True, layout="constrained")
    epochs = sorted({row["epoch"] for row in completed_history(context)})
    for ax, bit in zip(axes, BITS):
        for frame, label, style in ((train, "Train", "--"), (test, "Test", "-")):
            part = frame[frame["Precisie"] == bit]
            if not part.empty:
                ax.plot(part["Epoch"], part["Top-1 [%]"], linestyle=style, marker="o", ms=3.5,
                        color=COLORS[bit], lw=1.7, label=label)
        ax.set(title=f"W{bit}/A{bit}", xlabel="Epoch")
        ax.set_xticks(epochs)
        _style(ax, percent=True)
    axes[0].set_ylabel("Top-1 [%]")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Training en test per precisie · trainingsscore is over de epoch gemiddeld", fontsize=12)
    return fig


def plot_losses(context: dict[str, Any]):
    train, test = epoch_metrics_table(context, "training"), epoch_metrics_table(context)
    if train.empty and test.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, frame, title in zip(axes, (train, test),
                               ("Train: W32 harde CE; lagere modi distillatie", "Test: harde CE bij alle modi")):
        for bit in BITS:
            part = frame[frame["Precisie"] == bit]
            if not part.empty:
                ax.plot(part["Epoch"], part["Loss"], marker="o", ms=4, color=COLORS[bit], label=f"W{bit}/A{bit}")
        ax.set(title=title, xlabel="Voltooide epoch", ylabel="Gemiddelde loss")
        ax.set_xticks(sorted(frame["Epoch"].unique()))
        ax.set_ylim(bottom=0)
        _style(ax)
    axes[1].legend(fontsize=8, loc="best")
    return fig


def plot_selection(context: dict[str, Any]):
    frame = selection_table(context)
    if frame.empty or frame["Test top-1 [%]"].notna().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(11.7, 5), layout="constrained")
    positions = np.arange(len(BITS))
    choices = list(dict.fromkeys(frame["Selectie"]))
    width = .34
    for index, choice in enumerate(choices):
        part = frame[frame["Selectie"] == choice].set_index("Precisie").reindex(BITS)
        offset = (index - (len(choices) - 1) / 2) * width
        epoch_values = part["Epoch"].dropna()
        epoch = int(epoch_values.iloc[0]) if not epoch_values.empty else "?"
        bars = ax.bar(positions + offset, part["Test top-1 [%]"], width=width,
                      color=("#355F82", "#B77C34")[index], label=f"{choice}: {epoch}")
        for bar, value in zip(bars, part["Test top-1 [%]"]):
            if pd.notna(value):
                ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width()/2, value),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(positions, [f"W{bit}/A{bit}" for bit in BITS])
    ax.set(title="Laatste epoch en checkpoint gekozen op W32/A32", ylabel="Test top-1 [%]", xlabel="Uitvoeringsmodus")
    ax.legend(loc="upper left", fontsize=9)
    _style(ax, percent=True)
    return fig


def plot_epoch_times(context: dict[str, Any]):
    frame = timing_table(context)
    if frame.empty:
        return None
    frame = frame[frame["Complete tijdmeting"]]
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(11.7, 4.7), layout="constrained")
    bottom = np.zeros(len(frame))
    for key, caption, color in (("Training [s]", "Training (vijf modi)", "#355F82"),
                               ("Evaluatie [s]", "Volledige testset", "#40877B"),
                               ("Checkpoint [s]", "Checkpoint", "#B77C34"),
                               ("Overige epochkosten [s]", "Overige epochkosten", "#ABB6C1")):
        values = frame[key].to_numpy(dtype=float)
        ax.bar(frame["Epoch"], values, bottom=bottom, width=.64, color=color, label=caption)
        bottom += values
    for epoch, value in zip(frame["Epoch"], bottom):
        ax.annotate(f"{value:.1f} s", (epoch, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set(title="Gemeten epochduur · uitsluitend complete tijdmetingen", xlabel="Voltooide epoch", ylabel="Tijd [s]")
    ax.set_xticks(frame["Epoch"])
    ax.set_ylim(0, max(1., bottom.max() * 1.24))
    ax.legend(loc="upper left", ncols=2, fontsize=8.5)
    _style(ax)
    return fig


def plot_memory(context: dict[str, Any]):
    frame = memory_table(context)
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(11.7, 4.7), layout="constrained")
    ax.plot(frame["Epoch"], frame["Piek gealloceerd [GiB]"], marker="o", color="#355F82", lw=2, label="Piek gealloceerd")
    ax.plot(frame["Epoch"], frame["Piek gereserveerd [GiB]"], marker="s", color="#89709E", lw=1.8, label="Piek gereserveerd")
    device = context["worker"].get("runtime", {}).get("device", {})
    if isinstance(device, dict):
        total = _number(device.get("total_memory_bytes"))
        budget = _number(device.get("allocator_budget_bytes"))
        if total is not None:
            ax.axhline(total/1024**3, color="#7C8996", ls="--", lw=1, label="Fysieke capaciteit")
        if budget is not None:
            ax.axhline(budget/1024**3, color="#B77C34", ls=":", lw=1.2, label="Allocatorbudget bij start")
    ax.set(title="GPU-geheugen · piek opnieuw gemeten per epoch", xlabel="Voltooide epoch", ylabel="Geheugen [GiB]")
    ax.set_xticks(frame["Epoch"])
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9, ncols=2)
    _style(ax)
    return fig


def plot_forecast(context: dict[str, Any]):
    frame = forecast_table(context)
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(11.7, 4.7), layout="constrained")
    positions, bottom = np.arange(len(frame)), np.zeros(len(frame))
    for key, caption, color in (("Training [uur]", "Training", "#355F82"),
                               ("Evaluatie [uur]", "Evaluatie", "#40877B"),
                               ("Checkpoint [uur]", "Checkpoints", "#B77C34"),
                               ("Overig [uur]", "Overige epochkosten", "#ABB6C1")):
        values = frame[key].to_numpy(dtype=float)
        ax.bar(positions, values, bottom=bottom, width=.6, color=color, label=caption)
        bottom += values
    for position, value in zip(positions, bottom):
        ax.annotate(f"{value:.2f} u", (position, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(positions, frame["Epochs (raming)"].astype(str))
    ax.set(title=f"Looptijdraming · gebaseerd op epochs {frame.iloc[0]['Gebaseerd op epochs']}",
           xlabel="Aantal epochs (raming, geen uitgevoerde training)", ylabel="Geschatte duur [uur]")
    ax.set_ylim(0, max(.01, bottom.max() * 1.2))
    ax.legend(loc="upper left", fontsize=9, ncols=2)
    _style(ax)
    return fig
