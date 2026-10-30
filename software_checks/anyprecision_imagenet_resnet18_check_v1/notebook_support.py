"""Presentation of actual ResNet-50 checks and a bounded ResNet-18 probe.

Missing values remain missing. A completed update from an interrupted check is
shown as partial evidence, never silently discarded or labelled a full PASS.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


COLORS = ("#54738c", "#8aa2b5", "#b66f54", "#468878")


def load_run(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    baseline_path = root / "baseline_results.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("baseline_results.json ontbreekt in de gekozen resultatenmap: " + str(root))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report_path = root / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "status": "NIET_UITGEVOERD"}
    return {"root": root, "baseline": baseline, "report": report}


def _number(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def _gib(value: Any) -> float:
    return _number(value) / 2**30


def _status(item):
    return str(item.get("status", "ONBEKEND")).upper()


def records(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    ordinary_index = 0
    for item in run["baseline"].get("records", []):
        benchmark = item.get("runtime", {}).get("cudnn_benchmark", item.get("benchmark", True))
        if benchmark is False:
            label = "ResNet-50 · backendproef"
        else:
            ordinary_index += 1
            label = f"ResNet-50 · controle {ordinary_index}"
        rows.append({**item, "label": item.get("label", label), "benchmark": benchmark,
                     "model": "ResNet-50", "lr": item.get("lr", 0.5)})
    candidate = run["report"].get("gpu_check")
    result = candidate if isinstance(candidate, dict) and candidate else {"status": "GEEN_GPU_METING"}
    rows.append({**result, "label": "ResNet-18 · nieuwe controle", "benchmark": True,
                 "model": "ResNet-18", "lr": run["report"].get("config", {}).get("lr", 0.3), "run_name": run["root"].name,
                 "status": result.get("status", "GEEN_GPU_METING")})
    return rows


def _scope(item):
    status = _status(item)
    if status == "PASS":
        return "Volledige synthetische proef"
    if status in ("GEEN_GPU_METING", "NIET_UITGEVOERD"):
        return "Geen GPU-meting"
    return "Gedeeltelijk; piek afgebroken update onbekend"


def comparison_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in records(run):
        steps = item.get("step_seconds") or []
        complete = _status(item) == "PASS" and len(steps) >= 2
        rows.append({
            "Proef": item["label"], "Run": item.get("run_name", ""),
            "Model": item["model"], "LR": item["lr"], "benchmark": item["benchmark"],
            "Status": _status(item), "Meetbereik": _scope(item),
            "Voltooide gemeten stappen": len(steps),
            "Stap 1 [s]": _number(steps[0]) if len(steps) >= 1 else np.nan,
            "Stap 2 [s]": _number(steps[1]) if len(steps) >= 2 else np.nan,
            "Mediaan [s]": _number(item.get("median_step_seconds")) if complete else np.nan,
            "Vastgelegde piek allocated [GiB]": _gib(item.get("peak_allocated_bytes")),
            "Vastgelegde piek reserved [GiB]": _gib(item.get("peak_reserved_bytes")),
            "Vrij vóór model [GiB]": _gib(item.get("free_memory_before_model_bytes")),
            "GPU totaal [GiB]": _gib(item.get("device_memory_bytes")),
        })
    return pd.DataFrame(rows)


def update_memory_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in records(run):
        for update in item.get("memory_updates", []):
            before, after = update.get("before", {}), update.get("after", {})
            status = str(update.get("status", "ONBEKEND"))
            if status == "RUNNING" and _status(item) in ("TIMEOUT", "FAIL", "OOM", "INTERRUPTED"):
                status = "AFGEBROKEN / laatste workerstatus RUNNING"
            rows.append({
                "Proef": item["label"], "Update": update.get("kind", ""),
                "Nr.": update.get("index"), "Status": status,
                "Duur [s]": _number(update.get("seconds")),
                "Allocated vóór [GiB]": _gib(before.get("allocated_bytes")),
                "Allocated na [GiB]": _gib(after.get("allocated_bytes")),
                "Reserved na [GiB]": _gib(after.get("reserved_bytes")),
                "Vrij na [GiB]": _gib(after.get("free_bytes")),
                "Piek allocated [GiB]": _gib(update.get("peak_allocated_bytes")),
                "Piek reserved [GiB]": _gib(update.get("peak_reserved_bytes")),
            })
    return pd.DataFrame(rows)


def correctness_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    proof = run["report"].get("selftests")
    if not proof:
        return pd.DataFrame([{"Controle": "CPU-zelftests", "Status": "GEEN BEWIJS IN RAPPORT"}])

    def visit(value, path):
        if isinstance(value, dict):
            if "status" in value:
                rows.append({"Controle": path, "Status": value["status"]})
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    visit(item, path + "." + str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + f"[{index}]")

    visit(proof, "CPU")
    return pd.DataFrame(rows or [{"Controle": "CPU-zelftests", "Status": "Zie ruwe bewijsvelden"}])


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce3e9", linewidth=.7)
    ax.set_axisbelow(True)


def _label(item):
    label = item["label"] + f"\nbenchmark={item['benchmark']}"
    status = _status(item)
    return label + (f"\n{status}" if status != "PASS" else "")


def plot_step_times(run: dict[str, Any]):
    items = records(run)
    fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    _style(ax)
    observed = []
    for index, item in enumerate(items):
        status = _status(item)
        steps = [_number(value) for value in (item.get("step_seconds") or [])[:2]]
        valid = [(j, value) for j, value in enumerate(steps) if math.isfinite(value)]
        color = COLORS[index % len(COLORS)]
        partial = status != "PASS"
        if len(valid) == 2:
            ax.plot([index - .14, index + .14], [v for _, v in valid], color=color,
                    linestyle="--" if partial else "-", linewidth=1.5, zorder=2)
        for j, value in valid:
            offset = -.14 if j == 0 else .14
            observed.append(value)
            ax.scatter(index + offset, value, marker="o" if j == 0 else "s", s=72,
                       facecolors="white" if partial else color, edgecolors=color,
                       linewidth=1.6, zorder=3)
            ax.annotate(f"{value:.3f}", (index + offset, value), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(range(len(items)), [_label(item) for item in items])
    ax.set_xlim(-.5, len(items) - .5)
    ax.set_ylim(0, max(observed) * 1.28 if observed else 1)
    ax.set_ylabel("Volledige update [s] · alle vijf precisies")
    ax.set_title("Gemeten staptijden · ResNet-18 naast eerdere ResNet-50-controles", loc="left", weight="bold")
    if observed:
        ax.legend(handles=[
            Line2D([], [], marker="o", color="#56616a", linestyle="None", label="Gemeten stap 1"),
            Line2D([], [], marker="s", color="#56616a", linestyle="None", label="Gemeten stap 2"),
            Line2D([], [], marker="o", markerfacecolor="white", color="#a05f35", linestyle="None",
                   label="Voltooide stap uit onvoltooide proef")], loc="upper left", frameon=False, fontsize=9)
    else:
        ax.set_yticks([])
    fig.text(.01, -.025, "Synthetische input · 256 = 4 × 64 · FP32. Model en learning rate verschillen tussen R18 en R50. "
             "Open punten zijn voltooide stappen uit een onvoltooide proef.", fontsize=9, color="#56616a")
    return fig


def plot_memory(run: dict[str, Any]):
    items = records(run)
    fig, ax = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    _style(ax)
    observed, totals = [], []
    any_partial = False
    for index, item in enumerate(items):
        status = _status(item)
        partial = status != "PASS"
        total = _gib(item.get("device_memory_bytes"))
        if math.isfinite(total):
            totals.append(total)
        for offset, key, color in [(-.18, "peak_allocated_bytes", "#54738c"),
                                   (.18, "peak_reserved_bytes", "#a2b8c9")]:
            value = _gib(item.get(key))
            if math.isfinite(value):
                observed.append(value)
                any_partial = any_partial or partial
                ax.bar(index + offset, value, width=.32, color=color,
                       edgecolor="#40505c" if partial else color, hatch="///" if partial else None)
                ax.text(index + offset, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(items)), [_label(item) for item in items])
    ax.set_xlim(-.5, len(items) - .5)
    ax.set_ylim(0, max(observed + totals) * 1.4 if observed or totals else 1)
    ax.set_ylabel("Vastgelegde PyTorch-geheugenteller [GiB]")
    ax.set_title("Vastgelegde geheugenpieken · inclusief opwarming", loc="left", weight="bold")
    handles = [Patch(color="#54738c", label="Allocated: tensors"),
               Patch(color="#a2b8c9", label="Reserved: allocator")]
    if any_partial:
        handles.append(Patch(facecolor="white", edgecolor="#40505c", hatch="///",
                             label="Deelmeting: piek afgebroken update onbekend"))
    if totals and max(totals) - min(totals) < .001:
        handles.append(ax.axhline(totals[0], color="#a86835", linestyle="--", linewidth=1.3,
                                 label=f"Gerapporteerde GPU-capaciteit: {totals[0]:.3f} GiB"))
    if observed:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.01), frameon=True,
                  fontsize=8.5, ncol=2)
    else:
        ax.set_yticks([])
    fig.text(.01, -.025, "Allocated/reserved meten geen fysieke VRAM-residentie. Een gearceerde staaf is "
             "uitsluitend de vastgelegde deelmeting, geen volledig gemeten proefpiek.", fontsize=9, color="#56616a")
    return fig


def outcome_text(run: dict[str, Any]) -> str:
    report = run["report"]
    result = report.get("gpu_check")
    if not isinstance(result, dict) or not result:
        if report.get("requested_command", report.get("command")) == "selftest":
            return ("CPU-zelftest: " + str(report.get("status", "ONBEKEND")) +
                    ". Er is geen nieuwe GPU-meting; dit is geen GPU-PASS.")
        return ("Er is geen nieuwe GPU-meting in dit rapport. Totale rapportstatus: " +
                str(report.get("status", "ONBEKEND")) + ".")
    status = _status(result)
    if status == "PASS":
        return ("De synthetische ResNet-18-proef is PASS. Beoordeel daarnaast tijd en geheugen; "
                "dit is geen ImageNet-accuracyresultaat en geeft de pilot niet automatisch vrij.")
    if status == "TIMEOUT":
        return ("De proef is bij de tijdslimiet gestopt. TIMEOUT is geen bewijs van OOM. "
                "Voltooide updates blijven als deelmeting zichtbaar; de afgebroken update blijft onbekend.")
    if status == "OOM":
        return "De proef eindigde met een geheugenfout. Voltooide deelmetingen blijven zichtbaar."
    return "Nieuwe proefstatus: " + status + ". Raadpleeg foutmelding en log voordat een vervolg wordt gekozen."


def projection_table(run: dict[str, Any]) -> pd.DataFrame:
    """Arithmetic scenarios from completed R18 steps; not epoch measurements.

    A TIMEOUT's completed step can be inspected in other tables, but it does
    not qualify as a completed feasibility run for a duration projection.
    """
    result = run["report"].get("gpu_check") or {}
    steps = [_number(value) for value in (result.get("step_seconds") or [])]
    steps = [value for value in steps if math.isfinite(value) and value > 0]
    columns = ["Scenario", "Stapduur [s]", "1 epoch [uur]", "2 epochs [uur]", "80 epochs [dagen]"]
    if _status(result) != "PASS" or len(steps) < 2:
        return pd.DataFrame(columns=columns)
    # Standard ILSVRC2012 training-set count and the agreed global batch.
    updates = math.ceil(1_281_167 / 256)
    scenarios = [("Snelste van de twee metingen", min(steps)),
                 ("Mediaan van de twee metingen", float(np.median(steps))),
                 ("Langzaamste van de twee metingen", max(steps))]
    return pd.DataFrame([{
        "Scenario": name, "Stapduur [s]": seconds,
        "1 epoch [uur]": seconds * updates / 3600,
        "2 epochs [uur]": seconds * updates * 2 / 3600,
        "80 epochs [dagen]": seconds * updates * 80 / 86400,
    } for name, seconds in scenarios], columns=columns)


def comparison_context_table(run: dict[str, Any]) -> pd.DataFrame:
    report = run["report"]
    rows = []
    current_packages = report.get("environment", {}).get("packages", {})
    current_gpu = (report.get("gpu_check") or {}).get("device")
    for item in records(run)[:-1]:
        prior_packages = item.get("environment", {}).get("packages", {})
        package_names = ("torch", "torchvision", "numpy", "Pillow")
        known_versions = all(name in current_packages and name in prior_packages for name in package_names)
        prior_gpu = item.get("device") or item.get("runtime", {}).get("device")
        rows.append({
            "Eerdere proef": item["label"],
            "Zelfde versies Torch/vision/NumPy/Pillow":
                all(current_packages[name] == prior_packages[name] for name in package_names)
                if known_versions else "Onbekend",
            "Zelfde GPU-rapport": current_gpu == prior_gpu if current_gpu and prior_gpu else "Onbekend",
            "Vergelijkingsbereik": "Hardwarecontext; ander model en LR, geen gecontroleerde architectuurisolatie",
        })
    return pd.DataFrame(rows)
