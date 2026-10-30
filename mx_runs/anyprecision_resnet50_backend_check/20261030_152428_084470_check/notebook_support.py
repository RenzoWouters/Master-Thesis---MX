"""Compact presentation of real baseline checks and a bounded backend diagnosis."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


COLORS = ("#54738c", "#8aa2b5", "#c18638")


def load_run(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    baseline_path = root / "baseline_results.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("baseline_results.json ontbreekt in de gekozen resultatenmap: " + str(root))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report_path = root / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "status": "NIET_UITGEVOERD", "backend_test": {"status": "NIET_UITGEVOERD"}}
    return {"root": root, "baseline": baseline, "report": report}


def _number(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def _gib(value: Any) -> float:
    return _number(value) / 2**30


def records(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(run["baseline"].get("records", []), start=1):
        rows.append({**item, "label": f"Bestaande controle {index}", "benchmark": True})
    backend = run["report"].get("backend_test")
    result = backend if isinstance(backend, dict) and backend else {"status": "GEEN_GPU_METING"}
    nested = result.get("result")
    if isinstance(nested, dict):
        result = {**result, **nested}
    rows.append({**result, "label": "Nieuwe diagnose", "benchmark": False,
                 "run_name": run["root"].name,
                 "status": result.get("status", "GEEN_GPU_METING")})
    return rows


def comparison_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in records(run):
        steps = item.get("step_seconds", [])
        status = str(item.get("status", "ONBEKEND")).upper()
        if status == "PASS":
            scope = "Volledige synthetische proef"
        elif status in ("GEEN_GPU_METING", "NIET_UITGEVOERD"):
            scope = "Geen GPU-meting"
        else:
            scope = "Gedeeltelijk; piek afgebroken update onbekend"
        rows.append({
            "Proef": item["label"], "Run": item.get("run_name", ""),
            "benchmark": item["benchmark"], "Status": item.get("status", "ONBEKEND"),
            "Meetbereik": scope,
            "Stap 1 [s]": _number(steps[0]) if len(steps) >= 1 else np.nan,
            "Stap 2 [s]": _number(steps[1]) if len(steps) >= 2 else np.nan,
            "Mediaan [s]": _number(item.get("median_step_seconds")),
            "Piek allocated [GiB]": _gib(item.get("peak_allocated_bytes")),
            "Piek reserved [GiB]": _gib(item.get("peak_reserved_bytes")),
            "Vrij vóór model [GiB]": _gib(item.get("free_memory_before_model_bytes")),
            "GPU totaal [GiB]": _gib(item.get("device_memory_bytes")),
        })
    return pd.DataFrame(rows)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce3e9", linewidth=.7)
    ax.set_axisbelow(True)


def plot_step_times(run: dict[str, Any]):
    items = records(run)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    _style(ax)
    labels = []
    has_measurements = False
    observed = []
    for index, item in enumerate(items):
        status = str(item.get("status", "ONBEKEND")).upper()
        labels.append(item["label"] + ("\nbenchmark=True" if item["benchmark"] else "\nbenchmark=False"))
        steps = item.get("step_seconds", [])
        if status == "PASS" and len(steps) >= 2:
            values = [_number(step) for step in steps[:2]]
            if all(math.isfinite(value) for value in values):
                has_measurements = True
                observed.extend(values)
                ax.plot([index - .14, index + .14], values, color=COLORS[index % len(COLORS)],
                        linewidth=2, zorder=2)
                for offset, marker, value in zip([-.14, .14], ["o", "s"], values):
                    ax.scatter(index + offset, value, marker=marker, color=COLORS[index % len(COLORS)],
                               s=65, zorder=3)
                    ax.annotate(f"{value:.3f}", (index + offset, value), xytext=(0, 9),
                                textcoords="offset points", ha="center", fontsize=10)
                continue
        ax.text(index, .07, status, transform=ax.get_xaxis_transform(), ha="center",
                color="#a86835" if status == "TIMEOUT" else "#6c7680", fontweight="bold")
    ax.set_xticks(range(len(items)), labels)
    ax.set_xlim(-.5, len(items) - .5)
    ax.set_ylim(0, max(observed) * 1.25 if observed else 1)
    ax.set_ylabel("Volledige update [s] · alle vijf precisies")
    ax.set_title("Gemeten staptijden na één opwarmupdate", loc="left", weight="bold")
    if has_measurements:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", color="#56616a", linestyle="None", label="Gemeten stap 1"),
                           Line2D([], [], marker="s", color="#56616a", linestyle="None", label="Gemeten stap 2")],
                  loc="upper left", frameon=False)
    else:
        ax.set_yticks([])
    fig.text(.01, -.035, "Synthetische input · 256 = 4 × 64 · FP32 · geen dataladen of validatie. "
             "Een verbindingslijn koppelt de twee metingen van dezelfde proef.", fontsize=9, color="#56616a")
    return fig


def plot_memory(run: dict[str, Any]):
    items = records(run)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    _style(ax)
    labels = []
    measured = False
    totals = []
    observed = []
    for index, item in enumerate(items):
        status = str(item.get("status", "ONBEKEND")).upper()
        labels.append(item["label"] + ("\nbenchmark=True" if item["benchmark"] else "\nbenchmark=False"))
        total = _gib(item.get("device_memory_bytes"))
        if math.isfinite(total):
            totals.append(total)
        if status == "PASS":
            for offset, key, color in [(-.18, "peak_allocated_bytes", "#54738c"),
                                       (.18, "peak_reserved_bytes", "#a2b8c9")]:
                value = _gib(item.get(key))
                if math.isfinite(value):
                    measured = True
                    observed.append(value)
                    ax.bar(index + offset, value, width=.32, color=color)
                    ax.text(index + offset, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
        else:
            ax.text(index, .07, status, transform=ax.get_xaxis_transform(), ha="center",
                    color="#a86835" if status == "TIMEOUT" else "#6c7680", fontweight="bold")
    ax.set_xticks(range(len(items)), labels)
    ax.set_xlim(-.5, len(items) - .5)
    ax.set_ylim(0, max(observed + totals) * 1.2 if observed or totals else 1)
    ax.set_ylabel("PyTorch-geheugenteller [GiB]")
    ax.set_title("Piekgeheugen tijdens de volledige GPU-proef", loc="left", weight="bold")
    if measured:
        handles = [Patch(color="#54738c", label="Allocated: tensors"), Patch(color="#a2b8c9", label="Reserved: allocator")]
        if totals and max(totals) - min(totals) < .001:
            line = ax.axhline(totals[0], color="#a86835", linestyle="--", linewidth=1.3,
                              label=f"Gerapporteerde GPU-capaciteit: {totals[0]:.3f} GiB")
            handles.append(line)
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    else:
        ax.set_yticks([])
    fig.text(.01, -.035, "Allocated/reserved zijn geen metingen van fysieke VRAM-residentie. "
             "De capaciteitslijn is context, geen bewijs van paging of de oorzaak van vertraging.",
             fontsize=9, color="#56616a")
    return fig


def outcome_text(run: dict[str, Any]) -> str:
    report = run["report"]
    result = report.get("backend_test")
    if not isinstance(result, dict) or not result:
        if report.get("command") == "selftest":
            return ("CPU-zelftest: " + str(report.get("status", "ONBEKEND")) +
                    ". Er is geen nieuwe GPU-meting; dit is geen GPU-PASS.")
        return ("Er is geen nieuwe GPU-meting in dit rapport. De totale rapportstatus is " +
                str(report.get("status", "ONBEKEND")) + ".")
    status = str(result.get("status", "GEEN_GPU_METING"))
    if status == "PASS":
        return ("De nieuwe synthetische proef is PASS. Vergelijk beide staptijden en het geheugen; "
                "dit is geen ImageNet-accuracyresultaat en geen automatische vrijgave voor de pilot.")
    if status == "TIMEOUT":
        return "De nieuwe proef overschreed de tijdslimiet. TIMEOUT is geen bewijs van geheugen-OOM."
    if status == "OOM":
        return "De nieuwe proef eindigde met een geheugenfout. Er is geen volledige nieuwe snelheidsmeting."
    if status == "NIET_UITGEVOERD":
        return "Er is nog geen nieuw run_report.json geladen; alleen de twee bestaande controles zijn beschikbaar."
    return "Nieuwe proefstatus: " + status + ". Raadpleeg de foutmelding en log voordat een vervolg wordt gekozen."
