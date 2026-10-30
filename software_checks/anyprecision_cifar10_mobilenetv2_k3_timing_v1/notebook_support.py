"""Read-only views of MobileNetV2 timing reports; never starts a training run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT = "anyprecision_cifar10_mobilenetv2_k3_timing"
BITS = (32, 8, 4, 2, 1)
COLORS = {"train": "#3E6383", "validation": "#4B897E", "checkpoint": "#B47D37",
          "reserved": "#8A739B", "text": "#263D4B", "shade": "#E9EEF3"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Verwacht een JSON-object in {path}.")
    return result


def _matches(report: dict[str, Any], path: Path) -> bool:
    name = report.get("experiment", "")
    if isinstance(name, dict):
        name = name.get("id", name.get("name", ""))
    return str(name).startswith(EXPERIMENT) or (not name and EXPERIMENT in str(path))


def discover_reports(project_root: str | Path | None = None) -> list[Path]:
    """Check bounded, known locations; do not recursively search a user's disk."""
    roots: list[Path] = []
    for origin in [Path.cwd(), Path(__file__).resolve().parent]:
        roots.extend([origin, *list(origin.parents)[:5]])
    if project_root is not None:
        roots.insert(0, Path(project_root).expanduser())
    candidates: set[Path] = set()
    for root in dict.fromkeys(roots):
        for candidate in [root / "run_report.json", root / "feedback_bundle" / "run_report.json"]:
            if candidate.is_file():
                candidates.add(candidate.resolve())
        parent = root / "mx_runs" / EXPERIMENT
        if parent.is_dir():
            candidates.update(p.resolve() for p in parent.glob("*/run_report.json"))
        # A feedback bundle can be extracted as one timestamped directory.
        if root.name.startswith("feedback") or root.name == EXPERIMENT:
            candidates.update(p.resolve() for p in root.glob("*/run_report.json"))
    accepted = []
    for path in candidates:
        try:
            if _matches(_read_json(path), path):
                accepted.append(path)
        except (OSError, ValueError):
            continue
    return sorted(accepted, key=lambda p: (p.parent.name, p.stat().st_mtime_ns))


def load_run(run_dir: str | Path | None = None,
             project_root: str | Path | None = None) -> dict[str, Any]:
    """Load the newest report, including FAIL reports; a PASS is never assumed."""
    if run_dir is None:
        candidates = discover_reports(project_root)
        if not candidates:
            raise FileNotFoundError(
                "Geen resultaat gevonden voor deze proef. Voer eerst check/run uit, "
                "of vul RUN_DIR in met de resultaatmap of uitgepakte feedbackbundel.")
        report_path = candidates[-1]
    else:
        path = Path(run_dir).expanduser()
        report_path = path if path.is_file() else path / "run_report.json"
        if not report_path.is_file() and path.is_dir():
            candidates = sorted(path.glob("*/run_report.json"))
            if len(candidates) == 1:
                report_path = candidates[0]
            elif len(candidates) > 1:
                raise ValueError("Deze map bevat meerdere runs. Kies één concrete resultaatmap.")
        if not report_path.is_file():
            raise FileNotFoundError(f"Geen run_report.json in {path}.")
    report_path = report_path.resolve()
    report = _read_json(report_path)
    if not _matches(report, report_path):
        raise ValueError("Dit rapport behoort niet tot de MobileNetV2-K3-tijdproef.")
    worker = report.get("worker_result") or {}
    if not isinstance(worker, dict):
        worker = {}
    if not worker:
        relative = report.get("worker_result_file", "worker/worker_result.json")
        worker_path = report_path.parent / relative
        if worker_path.is_file():
            worker = _read_json(worker_path)
    return {"report": report, "worker": worker, "report_path": report_path,
            "run_dir": report_path.parent}


def _real(context: dict[str, Any]) -> bool:
    return context["worker"].get("kind") == "real"


def overview_table(context: dict[str, Any]) -> pd.DataFrame:
    report, worker = context["report"], context["worker"]
    runtime, model = worker.get("runtime", {}), worker.get("model_summary", {})
    device = runtime.get("device", {})
    if not isinstance(device, dict):
        device = {"name": device}
    values = [
        ("Resultaatmap", str(context["run_dir"])),
        ("Opdracht", report.get("command", "onbekend")),
        ("Eindstatus", report.get("status", "onbekend")),
        ("Metingen", "CIFAR-10-data" if _real(context) else "Synthetisch / nog geen dataresultaat"),
        ("GPU", device.get("name", "niet gemeten")),
        ("Voltooide optimizerupdates", worker.get("completed_updates", "niet gemeten")),
        ("Parametergetal incl. BN-banken", model.get("parameter_count", "niet gemeten")),
        ("Groepsconvoluties", model.get("grouped_spatial_conv_count", "niet gemeten")),
        ("Aparte BatchNorm-banken", model.get("bn_bank_count", "niet gemeten")),
    ]
    return pd.DataFrame(values, columns=["Onderdeel", "Waarde"])


def status_notes(context: dict[str, Any]) -> list[str]:
    report, worker = context["report"], context["worker"]
    notes = []
    if report.get("status") != "PASS":
        notes.append("Deze opdracht is niet met PASS afgerond. Beoordeel de foutmelding vóór de ramingen.")
    error = report.get("error") or worker.get("error")
    if error:
        notes.append("Rapportmelding: " + (json.dumps(error, ensure_ascii=False) if isinstance(error, dict) else str(error)))
    if not _real(context):
        notes.append("Dit is geen CIFAR-10-trainingsresultaat. Eventuele tijden zijn synthetische rekentijden; accuracy heeft hier geen betekenis.")
    else:
        notes.append("Dit is een gedeeltelijke trainingsdiagnose vanaf nul, geen voltooide epoch of paperreproductie.")
        notes.append("De evaluatie gebruikt 1.000 testbeelden. De weergegeven accuracy is geen eindaccuracy van de volledige training.")
        timing = worker.get("timing_summary") or {}
        if timing and not timing.get("estimate_complete", False):
            notes.append("De tijdsraming is onvolledig: bruikbare trainings- en validatietijden zijn beide nodig voor een totaal.")
    notes.append("Eén trainingsstap omvat vijf forward/backward-passes en één Adam-update.")
    return notes


def group_table(context: dict[str, Any]) -> pd.DataFrame:
    rows = context["worker"].get("model_summary", {}).get("spatial_groups", [])
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    columns = [c for c in ["block", "hidden_channels", "groups", "channels_per_group", "stride", "residual"] if c in frame]
    return frame[columns].rename(columns={"block": "Blok (0-gebaseerd)",
        "hidden_channels": "Ingangskanalen 3×3", "groups": "Groepen",
        "channels_per_group": "Kanalen per groep", "stride": "Stride", "residual": "Residual"})


def step_table(context: dict[str, Any], phase: str = "training") -> pd.DataFrame:
    worker = context["worker"]
    if phase == "validation":
        rows = worker.get("validation_rows", [])
    elif _real(context):
        rows = worker.get("train_rows", [])
    else:
        rows = worker.get("synthetic_rows") or worker.get("train_rows", [])
    return pd.DataFrame([{k: v for k, v in row.items() if k != "metrics"} for row in rows])


def timing_table(context: dict[str, Any]) -> pd.DataFrame:
    summary = context["worker"].get("timing_summary", {})
    rows = []
    for phase, title in [("training", "Training"), ("validation", "Validatie")]:
        values = summary.get(phase) or {}
        if not values:
            continue
        stats = values.get("stats", values)
        for component, caption in [("wall_s", "Totale stap"), ("data_wait_s", "Wachten op data"),
                                   ("h2d_s", "Overdracht naar GPU"), ("compute_s", "Berekening")]:
            data = stats.get(component) or {}
            if not isinstance(data, dict) or "mean" not in data:
                continue
            rows.append({"Fase": title, "Component": caption,
                         "Gemiddelde [s]": data.get("mean"), "Mediaan [s]": data.get("median"),
                         "P10 [s]": data.get("p10"), "P90 [s]": data.get("p90"),
                         "Gebruikte volle batches": values.get("eligible_rows"),
                         "Opwarming verwijderd": values.get("warmup_discarded_rows")})
    return pd.DataFrame(rows)


def precision_table(context: dict[str, Any], phase: str = "validation") -> pd.DataFrame:
    if not _real(context):
        return pd.DataFrame()
    metrics = context["worker"].get("metrics_summary", {}).get(phase, {})
    rows = []
    for bit in BITS:
        item = metrics.get(str(bit), metrics.get(bit))
        if item:
            rows.append({"Modus": f"W{bit}/A{bit}", "Loss": item.get("loss"),
                         "Top-1 [%]": item.get("top1"), "Top-5 [%]": item.get("top5"),
                         "Voorbeelden": item.get("samples")})
    return pd.DataFrame(rows)


def forecast_table(context: dict[str, Any]) -> pd.DataFrame:
    projections = context["worker"].get("timing_summary", {}).get("projections", {})
    rows = []
    for epochs in [5, 100, 400]:
        item = projections.get(str(epochs), projections.get(epochs))
        if not item:
            continue
        rows.append({"Epochs (raming)": epochs, "Training [uur]": item.get("training_hours"),
                     "Validatie [uur]": item.get("validation_hours"),
                     "Checkpoints [uur]": item.get("checkpoint_hours"),
                     "Totaal [uur]": item.get("total_hours"),
                     "Laag scenario [uur]": item.get("scenario_low_hours"),
                     "Hoog scenario [uur]": item.get("scenario_high_hours")})
    return pd.DataFrame(rows)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#B9C4CC")
    ax.tick_params(colors=COLORS["text"])
    ax.grid(axis="y", color="#E5EAF0", linewidth=.8)
    ax.set_axisbelow(True)


def plot_step_times(context: dict[str, Any]):
    frame = step_table(context)
    if frame.empty or "wall_s" not in frame:
        return None
    summary = context["worker"].get("timing_summary", {}).get("training", {})
    warmup = int(summary.get("warmup_discarded_rows", 10 if _real(context) else 1))
    steps = frame["step"].to_numpy() if "step" in frame else np.arange(1, len(frame) + 1)
    fig, ax = plt.subplots(figsize=(11.5, 4.6), layout="constrained")
    ax.plot(steps, frame["wall_s"], color=COLORS["train"], lw=1.7, label="Volledige stapwandtijd")
    if "compute_s" in frame:
        ax.plot(steps, frame["compute_s"], color=COLORS["validation"], lw=1.3, alpha=.9,
                label="Berekening incl. vijf precisies + Adam")
    if warmup > 0:
        end = steps[min(warmup, len(steps)) - 1] + .5
        ax.axvspan(steps[0] - .5, end, color=COLORS["shade"], label="Opwarming; buiten raming")
    label = "CIFAR-10" if _real(context) else "synthetische beelden"
    ax.set(title=f"Stapduur · {label} · batch 128", xlabel="Volledige optimizerupdate", ylabel="Tijd [s]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9)
    _style(ax)
    return fig


def plot_memory(context: dict[str, Any]):
    rows = context["worker"].get("memory_rows", [])
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    gib = 1024 ** 3
    fig, ax = plt.subplots(figsize=(11.5, 4.5), layout="constrained")
    x = np.arange(1, len(frame) + 1)
    for key, caption, color in [("allocated_bytes", "Gealloceerd door PyTorch", COLORS["train"]),
                                ("reserved_bytes", "Gereserveerd door PyTorch", COLORS["reserved"])]:
        if key in frame:
            ax.plot(x, pd.to_numeric(frame[key], errors="coerce") / gib, marker=".", ms=4,
                    lw=1.5, color=color, label=caption)
    if "total_bytes" in frame and frame["total_bytes"].notna().any():
        total = pd.to_numeric(frame["total_bytes"], errors="coerce").max() / gib
        ax.axhline(total, color="#707F8C", ls="--", lw=1.2, label="Fysiek GPU-geheugen")
    ax.set(title="GPU-geheugen · allocatorwaarden en fysiek totaal",
           xlabel="Geheugenmeetpunt (chronologisch)", ylabel="Geheugen [GiB]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9)
    _style(ax)
    return fig


def memory_table(context: dict[str, Any]) -> pd.DataFrame:
    rows = context["worker"].get("memory_rows", [])
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    gib, result = 1024 ** 3, []
    for key, caption, mode in [
        ("peak_allocated_bytes", "Hoogste gemelde piekallocatie", "max"),
        ("peak_reserved_bytes", "Hoogste gemelde piekreservering", "max"),
        ("total_bytes", "Fysiek GPU-geheugen", "max"),
        ("free_bytes", "Laagste vrije GPU-geheugenmeting", "min")]:
        if key in frame:
            values = pd.to_numeric(frame[key], errors="coerce").dropna()
            if not values.empty:
                result.append({"Meting": caption, "GiB": getattr(values, mode)() / gib})
    return pd.DataFrame(result)


def plot_forecast(context: dict[str, Any]):
    frame = forecast_table(context)
    if frame.empty:
        return None
    synthetic = not _real(context)
    if not synthetic and not context["worker"].get("timing_summary", {}).get("estimate_complete", False):
        return None
    components = [("Training [uur]", "Rekencomponent" if synthetic else "Training", COLORS["train"])]
    if not synthetic:
        components.extend([("Validatie [uur]", "Validatie", COLORS["validation"]),
                           ("Checkpoints [uur]", "Checkpoints", COLORS["checkpoint"])])
    fig, ax = plt.subplots(figsize=(10.5, 4.4), layout="constrained")
    positions, bottom = np.arange(len(frame)), np.zeros(len(frame))
    for key, caption, color in components:
        values = pd.to_numeric(frame[key], errors="coerce").fillna(0).to_numpy()
        ax.bar(positions, values, bottom=bottom, color=color, label=caption, width=.6)
        bottom += values
    for i, total in enumerate(bottom):
        ax.annotate(f"{total:.2f} u", (i, total), xytext=(0, 6), textcoords="offset points",
                    ha="center", color=COLORS["text"], fontsize=10)
    title = "Raming van uitsluitend rekenwerk · synthetische meting" if synthetic else "Geprojecteerde trainingsduur · geen uitgevoerde epochs"
    ax.set(title=title, ylabel="Geschatte duur [uur]", xlabel="Aantal epochs")
    ax.set_xticks(positions, frame["Epochs (raming)"].astype(str))
    ax.set_ylim(0, max(1e-8, bottom.max() * 1.18))
    ax.legend(loc="upper left", fontsize=9)
    _style(ax)
    return fig
