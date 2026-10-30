"""Read-only presentation of the 200-batch ResNet-18 feasibility run.

Only recorded observations are shown. Missing data remain missing, partial
runs retain completed observations, and no paper-accuracy claim is made.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BITS = (1, 2, 4, 8, 32)
BIT_COLORS = {1: "#b66f54", 2: "#b58b39", 4: "#468878", 8: "#5b8da8", 32: "#5d578b"}
TIMING_COLUMNS = ("data_wait_seconds", "transfer_seconds", "compute_seconds", "total_seconds")


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def _status(value: dict) -> str:
    return str(value.get("status", "ONBEKEND")).upper()


def _gib(value: Any) -> float:
    return _number(value) / 2**30


def load_run(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("De gekozen resultatenmap bestaat niet: " + str(root))
    report_path = root / "run_report.json"
    baseline_path = root / "baseline_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "status": "NIET_UITGEVOERD"}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {"records": []}
    return {"root": root, "report": report, "baseline": baseline}


def pilot_result(run: dict[str, Any]) -> dict:
    result = run["report"].get("pilot")
    return result if isinstance(result, dict) else {}


def _baseline(run: dict[str, Any]) -> dict:
    records = run.get("baseline", {}).get("records", [])
    return records[0] if records and isinstance(records[0], dict) else {}


def outcome_text(run: dict[str, Any]) -> str:
    report, pilot = run["report"], pilot_result(run)
    status = _status(report)
    if not pilot:
        command = report.get("requested_command", report.get("command"))
        if command == "data":
            return f"**Datasetcontrole: {status}.** Er is geen nieuwe GPU-training of accuracy gemeten."
        if command == "selftest":
            return f"**CPU-zelftests: {status}.** Dit is geen GPU- of ImageNet-trainingsresultaat."
        return f"**Rapportstatus: {status}.** Er is geen GPU-proef in dit rapport."
    if status == "PASS" and _status(pilot) == "PASS":
        return ("**De korte proef is PASS.** De resultaten beschrijven de eerste 200 updates en een "
                "validatieselectie van 5.000 beelden. Ze zijn geen volledige epoch of paperaccuracy.")
    if status == "TIMEOUT" or _status(pilot) == "TIMEOUT":
        return ("**De proef is bij de tijdslimiet gestopt.** TIMEOUT bewijst geen OOM. Voltooide updates "
                "blijven zichtbaar; de afgebroken update en haar definitieve geheugenpiek zijn onbekend.")
    if status == "OOM" or _status(pilot) == "OOM":
        return "**De proef meldde een geheugenallocatiefout.** Reeds vastgelegde deelmetingen blijven zichtbaar."
    return (f"**Rapportstatus: {status}; proefstatus: {_status(pilot)}.** Bekijk foutmelding en log. "
            "Onvoltooide gegevens worden niet tot een volledige training aangevuld.")


def step_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in pilot_result(run).get("train_steps", []):
        row = {"step": _number(item.get("step")), "samples": _number(item.get("samples"))}
        row.update({key: _number(item.get(key)) for key in TIMING_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=("step", "samples", *TIMING_COLUMNS)).sort_values("step", kind="stable")


def summary_table(run: dict[str, Any]) -> pd.DataFrame:
    report, pilot = run["report"], pilot_result(run)
    steps = step_table(run)
    validation, checkpoint = pilot.get("validation", {}), pilot.get("checkpoint", {})
    baseline = _baseline(run)
    measured_total = steps["total_seconds"].sum(min_count=1)
    loop_seconds = _number(pilot.get("train_loop_seconds"))
    loop_overhead = loop_seconds - measured_total if math.isfinite(loop_seconds) and math.isfinite(measured_total) else np.nan
    rows = [
        ("Rapportstatus", _status(report)), ("Proefstatus", _status(pilot) if pilot else "GEEN_GPU_PROEF"),
        ("Voltooide globale updates", pilot.get("optimizer_updates_completed", np.nan)),
        ("Vastgelegde staptijden", len(steps)), ("Trainingsbeelden", pilot.get("train_samples", np.nan)),
        ("Validatiebeelden", validation.get("samples", np.nan)),
        ("Validatiestatus", _status(validation) if validation else "NIET_UITGEVOERD"),
        ("Volledige trainingslus [s]", loop_seconds),
        ("Som gemeten batchintervallen [s]", measured_total),
        ("Overige tijd binnen trainingslus [s]", loop_overhead),
        ("Initialisatie [s]", _number(pilot.get("initialization_seconds"))),
        ("Dataladers opbouwen [s]", _number(pilot.get("data_loader_build_seconds"))),
        ("Trainingsiterator starten [s]", _number(pilot.get("loader_start_seconds"))),
        ("Validatiefase [s]", _number(validation.get("seconds"))),
        ("Checkpoint opslaan [s]", _number(checkpoint.get("seconds"))),
        ("Checkpoint", checkpoint.get("filename", "Niet vastgelegd")),
        ("Gereserveerde geheugenpiek [GiB]", _gib(pilot.get("peak_reserved_bytes"))),
        ("Actieve tensorgeheugenpiek [GiB]", _gib(pilot.get("peak_allocated_bytes"))),
        ("Hele opdracht [s]", _number(report.get("wall_seconds"))),
        ("Eerdere synthetische update, mediaan [s]", _number(baseline.get("median_step_seconds"))),
    ]
    return pd.DataFrame(rows, columns=("Grootheid", "Waarde"))


def block_table(run: dict[str, Any], block_size: int = 20) -> pd.DataFrame:
    steps = step_table(run)
    columns = ["first_step", "last_step", "recorded_steps", *TIMING_COLUMNS]
    if steps.empty:
        return pd.DataFrame(columns=columns)
    steps = steps[steps["step"].notna()].copy()
    steps["block"] = ((steps["step"] - 1) // block_size).astype(int)
    rows = []
    for _, group in steps.groupby("block", sort=True):
        rows.append({"first_step": int(group["step"].min()), "last_step": int(group["step"].max()),
                     "recorded_steps": len(group),
                     **{key: group[key].mean() for key in TIMING_COLUMNS}})
    return pd.DataFrame(rows, columns=columns)


def validation_table(run: dict[str, Any]) -> pd.DataFrame:
    validation = pilot_result(run).get("validation", {})
    rows = []
    for bit in BITS:
        item = validation.get("by_bit", {}).get(str(bit))
        if not isinstance(item, dict):
            continue
        rows.append({"Precisie": f"W{bit}/A{bit}", "Status": _status(validation),
                     "Beelden": _number(item.get("samples")), "Top-1 [%]": _number(item.get("top1")),
                     "Top-5 [%]": _number(item.get("top5")), "Hard CE": _number(item.get("loss"))})
    return pd.DataFrame(rows, columns=("Precisie", "Status", "Beelden", "Top-1 [%]", "Top-5 [%]", "Hard CE"))


def memory_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in pilot_result(run).get("memory_snapshots", []):
        rows.append({"Fase": item.get("phase", ""), "Stap": item.get("step"),
                     "Allocated [GiB]": _gib(item.get("allocated_bytes")),
                     "Reserved [GiB]": _gib(item.get("reserved_bytes")),
                     "Vrij [GiB]": _gib(item.get("free_bytes")),
                     "Vastgelegde piek allocated [GiB]": _gib(item.get("peak_allocated_bytes")),
                     "Vastgelegde piek reserved [GiB]": _gib(item.get("peak_reserved_bytes"))})
    return pd.DataFrame(rows, columns=("Fase", "Stap", "Allocated [GiB]", "Reserved [GiB]", "Vrij [GiB]",
                                      "Vastgelegde piek allocated [GiB]", "Vastgelegde piek reserved [GiB]"))


def correctness_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = []

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

    visit(run["report"].get("selftests", {}), "CPU")
    for key in ("bn_training_check", "bn_validation_unchanged"):
        value = pilot_result(run).get(key)
        if isinstance(value, dict):
            visit(value, "GPU." + key)
        elif value is not None:
            rows.append({"Controle": "GPU." + key, "Status": value})
    return pd.DataFrame(rows or [{"Controle": "Geen bewijsvelden aanwezig", "Status": "ONBEKEND"}])


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce3e9", linewidth=.7)
    ax.set_axisbelow(True)


def _empty(ax, text):
    ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes, color="#56616a")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_times(run: dict[str, Any]):
    steps, blocks = step_table(run), block_table(run)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2), constrained_layout=True)
    for ax in axes:
        _style(ax)
    axes[0].set_title("Training met echte beelden · tijd per volledige vijf-precisie-update", loc="left", weight="bold")
    axes[1].set_title("Gemiddelden per groep van twintig updates", loc="left", weight="bold")
    if steps.empty:
        for ax in axes:
            _empty(ax, "Geen voltooide trainingsupdates vastgelegd")
        return fig
    axes[0].plot(steps["step"], steps["total_seconds"], color="#468878", label="Datawacht + overdracht + compute", linewidth=1.3)
    axes[0].plot(steps["step"], steps["compute_seconds"], color="#54738c", label="Compute: forward + backward + SGD", linewidth=1.3)
    baseline = _number(_baseline(run).get("median_step_seconds"))
    if math.isfinite(baseline):
        axes[0].axhline(baseline, linestyle="--", color="#b66f54", linewidth=1.2,
                       label=f"Eerdere synthetische compute: {baseline:.3f} s")
    axes[0].axvspan(.5, 20.5, color="#e9edf1", alpha=.65, zorder=0)
    axes[0].set_xlim(.5, max(200.5, steps["step"].max() + .5))
    finite_times = steps[["total_seconds", "compute_seconds"]].to_numpy(dtype=float)
    finite_times = finite_times[np.isfinite(finite_times)]
    axes[0].set_ylim(0, max(float(finite_times.max()) * 1.25, baseline * 1.25 if math.isfinite(baseline) else 0)
                     if finite_times.size else 1)
    axes[0].set_xlabel("Voltooide globale update · eerste twintig apart van tijdprojectie")
    axes[0].set_ylabel("Tijd [s]")
    axes[0].legend(loc="upper right", fontsize=9, frameon=True)
    positions = np.arange(len(blocks))
    bottom = np.zeros(len(blocks))
    for key, label, color in [("compute_seconds", "Compute", "#54738c"),
                              ("transfer_seconds", "Overdracht", "#b58b39"),
                              ("data_wait_seconds", "Zichtbare datawachttijd", "#468878")]:
        values = blocks[key].to_numpy(dtype=float)
        axes[1].bar(positions, values, bottom=bottom, width=.68, color=color, label=label)
        bottom += values
    labels = [f"{int(row.first_step)}–{int(row.last_step)}" for row in blocks.itertuples()]
    axes[1].set_xticks(positions, labels)
    axes[1].set_xlabel("Updates in de groep · een laatste onvolledige groep blijft herkenbaar")
    axes[1].set_ylabel("Gemiddelde tijd [s]")
    axes[1].legend(fontsize=9, ncol=3, loc="upper right")
    finite_bottom = bottom[np.isfinite(bottom)]
    axes[1].set_ylim(0, float(finite_bottom.max()) * 1.25 if finite_bottom.size and finite_bottom.max() > 0 else 1)
    fig.text(.01, -.025, "De batchintervallen sluiten grenscontroles en rapportage uit. Totale lus- en opdrachttijden staan in de tabel. "
             "Prefetching kan CPU-dataverwerking met GPU-compute laten overlappen.", fontsize=9, color="#56616a")
    return fig


def plot_memory(run: dict[str, Any]):
    table, pilot = memory_table(run), pilot_result(run)
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    _style(ax)
    ax.set_title("Vastgelegde GPU-geheugenstanden en allocatorpiek", loc="left", weight="bold")
    if table.empty:
        _empty(ax, "Geen geheugensnapshots vastgelegd")
        return fig
    x = np.arange(len(table))
    for key, label, color, style in [
        ("Allocated [GiB]", "Allocated: actieve tensors", "#54738c", "-"),
        ("Reserved [GiB]", "Reserved: allocator", "#468878", "-"),
        ("Vastgelegde piek reserved [GiB]", "Vastgelegde reserved-piek", "#468878", "--"),
        ("Vrij [GiB]", "Vrij volgens CUDA", "#b58b39", "-")]:
        ax.plot(x, table[key], marker="o", markersize=3, color=color, linestyle=style, label=label)
    total = _gib(pilot.get("device_memory_bytes"))
    if math.isfinite(total):
        ax.axhline(total, color="#b66f54", linestyle=":", linewidth=1.2,
                   label=f"Gerapporteerde GPU-capaciteit: {total:.3f} GiB")
    chosen = sorted(set([0, len(table) - 1] + list(range(0, len(table), max(1, len(table) // 9)))))
    labels = [str(table.iloc[i]["Fase"]) + (" " + str(int(table.iloc[i]["Stap"]))
              if math.isfinite(_number(table.iloc[i]["Stap"])) else "") for i in chosen]
    ax.set_xticks(chosen, labels, rotation=25, ha="right")
    observed = table[["Allocated [GiB]", "Reserved [GiB]", "Vrij [GiB]",
                      "Vastgelegde piek reserved [GiB]"]].to_numpy(dtype=float)
    observed = observed[np.isfinite(observed)]
    maximum = max(float(observed.max()) if observed.size else 0,
                  total if math.isfinite(total) else 0)
    ax.set_ylim(0, maximum * 1.25 if maximum > 0 else 1)
    ax.set_ylabel("Geheugenteller [GiB]")
    ax.set_xlabel("Chronologische meetgrens")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    qualifier = "Onvoltooide proef: de definitieve piek van een afgebroken update kan ontbreken. " if _status(pilot) != "PASS" else ""
    fig.text(.01, -.025, qualifier + "Allocated/reserved meten geen fysieke VRAM-residentie of Windows-gedeeld geheugen.",
             fontsize=9, color="#56616a")
    return fig


def _accuracy_blocks(run: dict[str, Any]) -> pd.DataFrame:
    buckets = {}
    for step in pilot_result(run).get("train_steps", []):
        number = _number(step.get("step"))
        if not math.isfinite(number) or number < 1:
            continue
        group = int((number - 1) // 20)
        for bit in BITS:
            item = step.get("by_bit", {}).get(str(bit), {})
            count, correct = _number(item.get("samples")), _number(item.get("top1_correct"))
            if not math.isfinite(count) or count <= 0 or not math.isfinite(correct):
                continue
            bucket = buckets.setdefault((group, bit), {"samples": 0., "correct": 0., "last_step": number})
            bucket["samples"] += count
            bucket["correct"] += correct
            bucket["last_step"] = max(bucket["last_step"], number)
    return pd.DataFrame([{"block": group, "bit": bit, "last_step": item["last_step"],
                          "top1": 100 * item["correct"] / item["samples"]}
                         for (group, bit), item in sorted(buckets.items())],
                        columns=("block", "bit", "last_step", "top1"))


def plot_accuracy(run: dict[str, Any]):
    train, validation = _accuracy_blocks(run), validation_table(run)
    pilot = pilot_result(run)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax in axes:
        _style(ax)
        ax.set_ylabel("Top-1 accuracy [%]")
    axes[0].set_title("Vroege training · groepen van twintig updates", loc="left", weight="bold", fontsize=11)
    axes[1].set_title("Korte validatie · vijf beelden per klasse", loc="left", weight="bold", fontsize=11)
    if train.empty:
        _empty(axes[0], "Geen trainaccuracy vastgelegd")
    else:
        for bit in BITS:
            part = train[train["bit"] == bit]
            if not part.empty:
                axes[0].plot(part["last_step"], part["top1"], marker="o", markersize=3,
                             color=BIT_COLORS[bit], label=f"W{bit}/A{bit}")
        axes[0].set_xlim(0, 205)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlabel("Laatste globale update van de groep")
        axes[0].legend(fontsize=8.5, ncol=2)
    if validation.empty:
        _empty(axes[1], "Geen validatieaccuracy vastgelegd")
    else:
        colors = [BIT_COLORS[int(label.split("/")[0][1:])] for label in validation["Precisie"]]
        positions = np.arange(len(validation))
        axes[1].bar(positions, validation["Top-1 [%]"], color=colors, width=.65)
        axes[1].set_xticks(positions, validation["Precisie"])
        values = [value for value in validation["Top-1 [%]"] if math.isfinite(value)]
        if values:
            axes[1].set_ylim(0, max(max(values) * 1.3, .1))
        for index, value in enumerate(validation["Top-1 [%]"]):
            if math.isfinite(value):
                axes[1].text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        state = pilot.get("validation", {})
        axes[1].set_xlabel(f"{state.get('samples', '?')} validatiebeelden · {_status(state)}")
    fig.text(.01, -.025, "Diagnostiek na een gedeeltelijke epoch: train- en validatiesamples verschillen. "
             "Deze scores zijn geen papervergelijking of oordeel over eindaccuracy.", fontsize=9, color="#56616a")
    return fig


def projection_table(run: dict[str, Any]) -> pd.DataFrame:
    columns = ("Scenario", "Update [s]", "Training 1 epoch [uur]", "Val 50k, schatting [min]",
               "Checkpoint [s]", "Epoch incl. val/opslag [uur]", "80 epochs incl. val/opslag [dagen]")
    pilot = pilot_result(run)
    validation, checkpoint = pilot.get("validation", {}), pilot.get("checkpoint", {})
    steps = step_table(run)
    complete = (_status(run["report"]) == "PASS" and _status(pilot) == "PASS" and
                pilot.get("optimizer_updates_completed") == 200 and len(steps) == 200 and
                list(steps["step"]) == list(range(1, 201)) and
                _status(validation) == "PASS" and validation.get("samples") == 5000 and
                bool(checkpoint.get("filename")))
    if not complete:
        return pd.DataFrame(columns=columns)
    tail = steps[steps["step"] > 20]
    values = tail["total_seconds"].to_numpy(dtype=float)
    val_seconds, checkpoint_seconds = _number(validation.get("seconds")), _number(checkpoint.get("seconds"))
    if (len(values) != 180 or not np.all(np.isfinite(values)) or not np.all(values > 0) or
            not math.isfinite(val_seconds) or val_seconds <= 0 or
            not math.isfinite(checkpoint_seconds) or checkpoint_seconds < 0):
        return pd.DataFrame(columns=columns)
    block_means = values.reshape(9, 20).mean(axis=1)
    full_val_seconds = val_seconds * 10
    rows = []
    for label, seconds in (("Snelste groep van 20 (updates 21–200)", float(block_means.min())),
                           ("Gemiddelde van updates 21–200", float(values.mean())),
                           ("Langzaamste groep van 20 (updates 21–200)", float(block_means.max()))):
        train_seconds = seconds * math.ceil(1_281_167 / 256)
        epoch_seconds = train_seconds + full_val_seconds + checkpoint_seconds
        rows.append({"Scenario": label, "Update [s]": seconds, "Training 1 epoch [uur]": train_seconds / 3600,
                     "Val 50k, schatting [min]": full_val_seconds / 60,
                     "Checkpoint [s]": checkpoint_seconds, "Epoch incl. val/opslag [uur]": epoch_seconds / 3600,
                     "80 epochs incl. val/opslag [dagen]": epoch_seconds * 80 / 86400})
    return pd.DataFrame(rows, columns=columns)
