"""Read-only analysis helpers for the ResNet-50 technical check and pilot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BITS = [32, 8, 4, 2, 1]
COLORS = {32: "#315d82", 8: "#408373", 4: "#b98136", 2: "#ae5d53", 1: "#8c73a8"}
REFERENCE = {32: 74.63, 8: 74.07, 4: 73.84, 2: 71.66, 1: 58.77}
REFERENCE_URL = "https://github.com/SHI-Labs/Any-Precision-DNNs#resnet-50-models-on-imagenet"
METRIC_COLUMNS = ["epoch", "split", "bit", "loss", "top1", "top5", "samples", "seconds"]


def label(bit: int) -> str:
    return f"W{bit}/A{bit}"


def load_run(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve()
    report_path = directory / "run_report.json"
    report = (json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file()
              else {"status": "NIET_UITGEVOERD", "note": "Geen run_report.json in deze map."})
    config = report.get("config", report.get("experiment_config", {}))
    if not config and (directory / "experiment_config.json").is_file():
        config = json.loads((directory / "experiment_config.json").read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.csv"
    if metrics_path.is_file() and metrics_path.stat().st_size:
        try:
            metrics = pd.read_csv(metrics_path)
        except pd.errors.EmptyDataError:
            metrics = pd.DataFrame(columns=METRIC_COLUMNS)
    else:
        metrics = pd.DataFrame(columns=METRIC_COLUMNS)
    if not metrics.empty:
        aliases = {"bits": "bit", "precision": "bit", "top1_percent": "top1", "top5_percent": "top5"}
        metrics = metrics.rename(columns={k: v for k, v in aliases.items() if k in metrics and v not in metrics})
        missing = set(METRIC_COLUMNS[:7]) - set(metrics.columns)
        if missing:
            raise ValueError(f"metrics.csv mist kolommen: {sorted(missing)}")
        for column in ["epoch", "bit", "loss", "top1", "top5", "samples"]:
            metrics[column] = pd.to_numeric(metrics[column], errors="raise")
        metrics["epoch"] = metrics["epoch"].astype(int)
        metrics["bit"] = metrics["bit"].astype(int)
        if metrics.duplicated(["epoch", "split", "bit"]).any():
            raise ValueError("metrics.csv bevat dubbele epoch/split/precisie-rijen.")
        if not metrics["bit"].isin(BITS).all():
            raise ValueError("metrics.csv bevat een onverwachte precisie.")
        if not metrics[["top1", "top5"]].apply(lambda s: s.between(0, 100)).all().all():
            raise ValueError("Accuracy moet een percentage tussen 0 en 100 zijn.")
        metrics = metrics.sort_values(["epoch", "split", "bit"]).reset_index(drop=True)
    return {"directory": directory, "report": report, "config": config, "metrics": metrics}


def run_summary(run: dict[str, Any]) -> pd.DataFrame:
    report, config = run["report"], run["config"]
    records = [
        ("Resultatenmap", str(run["directory"])),
        ("Status", report.get("status", report.get("end_status", "onbekend"))),
        ("Opdracht", report.get("command", "onbekend")),
        ("Model", config.get("model", "resnet50q")),
        ("Pilot / volledig recept", "2 / 80 epochs"),
        ("Afgeronde epochs", report.get("latest_completed_epoch", len(complete_epochs(run)))),
        ("Trainbeelden per gewichtsupdate", config.get("global_batch", 256)),
        ("Uitvoering", config.get("execution", "serial_dp4")),
        ("BN-beleid", config.get("bn_policy", "eerste shard bewaart running-statistieken")),
        ("Seed van onze proef", config.get("seed", "zie run_report.json")),
        ("Mixed precision / AMP", config.get("amp", False)),
    ]
    return pd.DataFrame(records, columns=["Onderdeel", "Waarde"])


def memory_table(run: dict[str, Any]) -> pd.DataFrame:
    tests = run["report"].get("memory_tests", {})
    if isinstance(tests, list):
        tests = {str(item.get("mode", i)): item for i, item in enumerate(tests)}
    rows = []
    for mode in ["native256", "serial_dp4"]:
        record = tests.get(mode)
        if not isinstance(record, dict):
            continue
        nested = record.get("result")
        if isinstance(nested, dict):
            record = {**record, **nested}
        rows.append({
            "mode": mode,
            "Uitvoering": "256 tegelijk · BN256" if mode == "native256" else "4 × 64 · BN zoals 4 GPU’s",
            "Status": str(record.get("status", "onbekend")),
            "Piek allocated [GiB]": _gib(record.get("peak_allocated_bytes")),
            "Piek reserved [GiB]": _gib(record.get("peak_reserved_bytes")),
            "Mediane stap [s]": record.get("median_step_seconds", np.nan),
            "Metingen": len(record.get("step_seconds", [])),
            "BN-shards": str(record.get("batchnorm_shard_sizes", "onbekend")),
        })
    return pd.DataFrame(rows)


def _gib(value: Any) -> float:
    return float(value) / 2**30 if value is not None else np.nan


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dfe5eb", linewidth=0.7)
    ax.set_axisbelow(True)


def plot_memory(run: dict[str, Any]):
    table = memory_table(run)
    if table.empty:
        print("Geen GPU-proefmetingen in dit rapport. Bij een pilot staan ze in de afzonderlijke check-map.")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(table))
    names = ["256 tegelijk\nBN256", "4 × 64\nBN64 + replica-0-statistieken"]
    names = [names[0 if mode == "native256" else 1] for mode in table["mode"]]
    for ax in axes:
        _style(ax)
        ax.set_xticks(x, names)
    axes[0].set(title="GPU-geheugen · volledige trainingsstap", ylabel="Geheugen [GiB]")
    axes[1].set(title="Rekentijd na opwarming", ylabel="Mediane stapduur [s]")
    for index, row in table.iterrows():
        passed = row["Status"].upper() == "PASS"
        if passed:
            for offset, column, color in [(-0.18, "Piek allocated [GiB]", "#315d82"),
                                          (0.18, "Piek reserved [GiB]", "#9fb6c9")]:
                value = row[column]
                if pd.notna(value):
                    axes[0].bar(index + offset, value, width=0.32, color=color,
                                label=column.split(" [")[0] if index == 0 else None)
                    axes[0].text(index + offset, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
            value = row["Mediane stap [s]"]
            if pd.notna(value):
                axes[1].bar(index, value, width=0.55, color="#408373")
                axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
        else:
            for ax in axes:
                ax.text(index, 0.08, row["Status"], ha="center", va="bottom", color="#ae5d53",
                        transform=ax.get_xaxis_transform(), fontweight="bold")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#315d82", label="Allocated"),
                            Patch(color="#9fb6c9", label="Reserved")], loc="upper left")
    for ax in axes:
        ax.set_ylim(bottom=0)
        # Failed/OOM trials have no bar, so matplotlib would otherwise shrink
        # the x-range to the successful trial and hide the missing category.
        ax.set_xlim(-0.6, len(table) - 0.4)
        ax.margins(y=0.18)
    fig.suptitle("Synthetische input · één GPU · batch 256 · alle vijf precisies", fontsize=13)
    return fig


def complete_epochs(run: dict[str, Any]) -> list[int]:
    metrics, report = run["metrics"], run["report"]
    if metrics.empty:
        return []
    valid = metrics[metrics["split"].astype(str).str.lower().isin(["val", "validation", "test"])]
    limit = report.get("latest_completed_epoch")
    if limit is not None:
        valid = valid[valid["epoch"] <= int(limit)]
    return sorted(int(epoch) for epoch, frame in valid.groupby("epoch")
                  if set(frame["bit"]) == set(BITS))


def validation_metrics(run: dict[str, Any]) -> pd.DataFrame:
    metrics = run["metrics"]
    if metrics.empty:
        return metrics.copy()
    return metrics[metrics["split"].astype(str).str.lower().isin(["val", "validation", "test"])
                   & metrics["epoch"].isin(complete_epochs(run))].copy()


def plot_accuracy_curves(run: dict[str, Any], metric: str = "top1"):
    if metric not in ("top1", "top5"):
        raise ValueError("Alleen top1 of top5 is toegestaan.")
    valid = validation_metrics(run)
    if valid.empty:
        print("Nog geen volledige ImageNet-validatie beschikbaar; er wordt geen accuracy ingevuld.")
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    _style(ax)
    for bit in BITS:
        frame = valid[valid["bit"] == bit].sort_values("epoch")
        ax.plot(frame["epoch"], frame[metric], marker="o", linewidth=2,
                color=COLORS[bit], label=label(bit))
    ax.set(xticks=complete_epochs(run), xlabel="Afgeronde epoch",
           ylabel=f"Validatie-{metric.replace('top', 'top-')} [%]", ylim=(0, 100),
           title=f"ResNet-50 / ImageNet · volledige {metric.replace('top', 'top-')}-historie van deze pilot")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    return fig


def checkpoint_table(run: dict[str, Any]) -> pd.DataFrame:
    valid = validation_metrics(run)
    if valid.empty:
        return pd.DataFrame(columns=["Selectie", "Epoch", "Precisie", "Top-1 [%]", "Top-5 [%]"])
    final_epoch = int(valid["epoch"].max())
    best = valid[valid["bit"] == 32].sort_values(["top1", "epoch"], ascending=[False, True]).iloc[0]
    best_epoch = int(best["epoch"])
    reported = run["report"].get("best_epoch_by_highest_precision")
    if reported is not None and int(reported) != best_epoch:
        raise ValueError("Het beste W32/A32-checkpoint in run_report.json stemt niet overeen met metrics.csv.")
    rows = []
    for title, epoch in [("Laatste afgeronde epoch", final_epoch),
                         ("Beste W32/A32-checkpoint", best_epoch)]:
        at_epoch = valid[valid["epoch"] == epoch].set_index("bit")
        for bit in BITS:
            row = at_epoch.loc[bit]
            rows.append({"Selectie": title, "Epoch": epoch, "Precisie": label(bit),
                         "Top-1 [%]": float(row["top1"]), "Top-5 [%]": float(row["top5"])})
    return pd.DataFrame(rows)


def plot_checkpoints(run: dict[str, Any]):
    table = checkpoint_table(run)
    if table.empty:
        print("Er zijn nog geen vergelijkbare checkpoints met alle vijf validatiemodi.")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, (selection, frame) in zip(axes, table.groupby("Selectie", sort=False)):
        _style(ax)
        epoch = int(frame["Epoch"].iloc[0])
        values = frame["Top-1 [%]"].to_numpy()
        ax.bar(np.arange(5), values, color=[COLORS[b] for b in BITS], width=0.66)
        for x, value in enumerate(values):
            ax.text(x, value + 1, f"{value:.2f}", ha="center", fontsize=9)
        ax.set(xticks=np.arange(5), xticklabels=[label(b) for b in BITS],
               ylim=(0, 105), ylabel="Validatie-top-1 [%]", title=f"{selection}\nepoch {epoch}")
        ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Alle precisies komen telkens uit hetzelfde checkpoint", fontsize=13)
    return fig


def reference_table() -> pd.DataFrame:
    return pd.DataFrame({"Precisie": [label(b) for b in BITS],
                         "GitHub top-1 [%]": [REFERENCE[b] for b in BITS],
                         "Referentie": ["Gezamenlijk model · recursive KD · 80-epochrecept"] * 5})


def epoch_timing_table(run: dict[str, Any]) -> pd.DataFrame:
    rows = run["report"].get("epochs", [])
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    columns = [c for c in ["epoch", "train_seconds", "val_seconds", "checkpoint_seconds",
                            "epoch_seconds", "lr_used", "lr_next"] if c in frame]
    frame = frame[columns].copy()
    return frame.rename(columns={"epoch": "Epoch", "train_seconds": "Training [s]",
                                 "val_seconds": "Validatie [s]", "checkpoint_seconds": "Checkpoint [s]",
                                 "epoch_seconds": "Hele epoch [s]", "lr_used": "Gebruikte LR",
                                 "lr_next": "Volgende LR"})


def runtime_projection(run: dict[str, Any]) -> pd.DataFrame:
    frame = epoch_timing_table(run)
    first = None
    later = pd.Series(dtype=float)
    if not frame.empty and "Hele epoch [s]" in frame:
        frame = frame.sort_values("Epoch").copy()
        frame["Hele epoch [s]"] = pd.to_numeric(frame["Hele epoch [s]"], errors="coerce")
        valid_time = np.isfinite(frame["Hele epoch [s]"]) & (frame["Hele epoch [s]"] > 0)
        first_values = frame.loc[valid_time & (frame["Epoch"] == 1), "Hele epoch [s]"]
        if not first_values.empty:
            first = float(first_values.iloc[0])
        later = frame.loc[valid_time & (frame["Epoch"] >= 2), "Hele epoch [s]"]

    explicit = run["report"].get("projection80_seconds")
    try:
        explicit = float(explicit) if explicit is not None and not isinstance(explicit, bool) else None
    except (TypeError, ValueError):
        explicit = None
    if explicit is not None and np.isfinite(explicit) and explicit > 0:
        projection = explicit
        method = "80-epochprojectie geregistreerd in run_report.json"
        if first is None:
            method += "; epoch-1-duur ontbreekt, eerste-epochoverhead is niet afzonderlijk vastgesteld"
    elif first is not None and not later.empty:
        projection = first + 79 * float(later.median())
        method = "eerste epoch + 79 × mediane volgende epoch"
    elif not later.empty:
        projection = 80 * float(later.median())
        method = ("80 × mediane gemeten epoch vanaf epoch 2; epoch-1-duur ontbreekt "
                  "en eerste-epochoverhead is niet afzonderlijk vastgesteld")
    elif first is not None:
        projection = 80 * first
        method = "80 × eerste epoch; voorlopige schatting zonder normale tweede epoch"
    else:
        return pd.DataFrame()
    return pd.DataFrame([{"Run": "Verse volledige training van 80 epochs",
                          "Geschatte uren": projection / 3600,
                          "Berekening": method,
                          "Beperking": "Zelfde hardware/data-opslag; temperatuur en I/O kunnen variëren. "
                                         "Eenmalige voorbereiding buiten epochs niet inbegrepen."}])


def plot_epoch_times(run: dict[str, Any]):
    frame = epoch_timing_table(run)
    if frame.empty:
        print("Geen echte epochduren beschikbaar. De synthetische check meet alleen de trainingsstap.")
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    _style(ax)
    x = np.arange(len(frame))
    bottom = np.zeros(len(frame))
    for column, color in [("Training [s]", "#315d82"), ("Validatie [s]", "#408373"),
                           ("Checkpoint [s]", "#b98136")]:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy() / 60
            ax.bar(x, values, bottom=bottom, label=column.replace(" [s]", ""), color=color, width=0.6)
            bottom += values
    ax.set(xticks=x, xticklabels=[f"Epoch {int(e)}" for e in frame["Epoch"]],
           ylabel="Tijd [minuten]", title="Gemeten echte ImageNet-pilot · tijd per onderdeel")
    ax.legend()
    return fig
