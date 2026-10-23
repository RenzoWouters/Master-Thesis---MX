"""Read-only analysis of saved dedicated E4M3/A32 results.

No model, checkpoint, torch, training or dataset import is performed here.
Only an explicit call to save_figures writes files, in a fresh analysis folder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_FORMAT = "MXFP8_E4M3"
BLUE, GREEN, AMBER, RED = "#245A81", "#33846D", "#C98220", "#B44040"
STYLE = {"axes.spines.top": False, "axes.spines.right": False,
         "axes.titleweight": "bold", "axes.labelsize": 10,
         "font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
         "savefig.facecolor": "white", "axes.grid": True, "grid.alpha": .16}


@dataclass
class RunAnalysis:
    path: Path
    report: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected: pd.DataFrame = field(default_factory=pd.DataFrame)
    times: pd.DataFrame = field(default_factory=pd.DataFrame)
    updates: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list[dict] = field(default_factory=list)
    input_files: list[Path] = field(default_factory=list)
    metrics_safe: bool = True

    @property
    def config(self):
        return self.report.get("config", {})

    @property
    def synthetic(self):
        return any(value is True for value in (
            self.report.get("synthetic_fixture_only"),
            self.report.get("synthetic_checks_only"),
            self.summary.get("synthetic_fixture_only")))

    @property
    def provenance_confirmed(self):
        return not self.synthetic and any(value is False for value in (
            self.report.get("synthetic_fixture_only"),
            self.summary.get("synthetic_fixture_only")))

    @property
    def test(self):
        if not self.metrics_safe or self.metrics.empty or "phase" not in self.metrics:
            return pd.DataFrame()
        return self.metrics.loc[self.metrics["phase"] == "test"].sort_values("epoch")

    @property
    def last_epoch(self):
        return int(self.test["epoch"].max()) if not self.test.empty else None

    @property
    def observed_best_epoch(self):
        # idxmax returns the FIRST maximum, matching strict-improvement selection.
        return int(self.test.loc[self.test["top1_percent"].idxmax(), "epoch"]) if not self.test.empty else None

    def metric(self, epoch):
        if epoch is None or self.test.empty:
            return None
        rows = self.test.loc[self.test["epoch"] == epoch]
        return rows.iloc[0] if len(rows) == 1 else None

    def issue(self, level, text):
        self.issues.append({"Niveau": level, "Melding": text})


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def load_run(run_dir) -> RunAnalysis:
    """Load JSON/CSV; report missing, invalid and inconsistent data explicitly."""
    run = RunAnalysis(Path(run_dir).expanduser().resolve())
    json_files = {"run_report.json": "report", "history.json": "history",
                  "training_summary.json": "summary", "checkpoint_selection.json": "selection",
                  "timing_estimate.json": "timing"}
    csv_files = {"epoch_metrics.csv": "metrics", "selected_metrics.csv": "selected",
                 "epoch_times.csv": "times", "weight_updates.csv": "updates"}
    for filename, attr in json_files.items():
        path = run.path / filename
        if not path.is_file():
            run.issue("ONTBREEKT", filename)
            continue
        run.input_files.append(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("Verwacht een JSON-object.")
            setattr(run, attr, data)
        except (OSError, ValueError) as exc:
            run.issue("FOUT", f"{filename}: {exc}")
    for filename, attr in csv_files.items():
        path = run.path / filename
        if not path.is_file():
            run.issue("ONTBREEKT", filename)
            continue
        run.input_files.append(path)
        try:
            setattr(run, attr, pd.read_csv(path))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            run.issue("FOUT", f"{filename}: {exc}")

    history_rows = []
    for record in run.history.get("epochs", []):
        if not isinstance(record, dict):
            run.issue("FOUT", "history.json bevat een ongeldige epochrij.")
            continue
        for phase in ("train", "test"):
            for row in record.get(phase, []):
                history_rows.append({"epoch": record.get("epoch"), "phase": phase, **row})
    if run.metrics.empty and history_rows:
        run.metrics = pd.DataFrame(history_rows)
        run.issue("LET OP", "Epochgrafieken gebruiken history.json; epoch_metrics.csv ontbreekt of is leeg.")
    if run.times.empty:
        fields = ("epoch", "train_seconds", "test_seconds", "checkpoint_seconds", "other_seconds",
                  "total_epoch_seconds", "first_epoch_in_process", "peak_gpu_allocated_bytes",
                  "peak_gpu_reserved_bytes")
        rows = [{key: record.get(key) for key in fields} for record in run.history.get("epochs", [])
                if isinstance(record, dict)]
        if rows:
            run.times = pd.DataFrame(rows)
            run.issue("LET OP", "Tijd/geheugengrafieken gebruiken history.json; epoch_times.csv ontbreekt of is leeg.")

    if not run.metrics.empty:
        required = {"epoch", "phase", "top1_percent", "objective_loss"}
        missing = required - set(run.metrics.columns)
        if missing:
            run.issue("FOUT", "Ontbrekende metricvelden: " + ", ".join(sorted(missing)))
            run.metrics_safe = False
        else:
            for column in ("epoch", "top1_percent", "objective_loss", "n", "correct1", "count_top1_percent"):
                if column in run.metrics:
                    run.metrics[column] = pd.to_numeric(run.metrics[column], errors="coerce")
            essential = run.metrics[["epoch", "top1_percent", "objective_loss"]].to_numpy(dtype=float)
            if not np.isfinite(essential).all():
                run.issue("FOUT", "Niet-eindige/ontbrekende epoch, accuracy of loss. Metricgrafieken geblokkeerd.")
                run.metrics_safe = False
            elif not ((run.metrics["epoch"] >= 1) & (run.metrics["epoch"] % 1 == 0)).all():
                run.issue("FOUT", "Ongeldige epochnummers. Metricgrafieken geblokkeerd.")
                run.metrics_safe = False
            if run.metrics.duplicated(["epoch", "phase"]).any():
                run.issue("FOUT", "Meerdere metrics voor dezelfde epoch/fase: dit notebook vereist één E4M3-modus.")
                run.metrics_safe = False
            if not run.metrics["phase"].isin(["train", "test"]).all():
                run.issue("FOUT", "Onbekende fase in epochmetrics.")
                run.metrics_safe = False
            if not run.metrics["top1_percent"].between(0, 100).all():
                run.issue("FOUT", "Accuracy buiten [0, 100].")
                run.metrics_safe = False
            if "weight_format" in run.metrics and not run.metrics["weight_format"].eq(EXPECTED_FORMAT).all():
                run.issue("FOUT", "Deze analyse verwacht uitsluitend MXFP8_E4M3-metrics.")
                run.metrics_safe = False
            if "activation_bits" in run.metrics and not run.metrics["activation_bits"].eq(32).all():
                run.issue("FOUT", "Activatieprecisie wijkt af van A32.")
                run.metrics_safe = False
            if {"n", "correct1"} <= set(run.metrics):
                n, correct = run.metrics["n"], run.metrics["correct1"]
                valid_counts = (n > 0) & (correct >= 0) & (correct <= n) & (correct % 1 == 0) & (n % 1 == 0)
                if not valid_counts.all():
                    run.issue("FOUT", "Ontbrekende of ongeldige aantallen n/correct1.")
                else:
                    error = abs(run.metrics["top1_percent"] - 100 * correct / n)
                    if (error > 1e-4).any():
                        run.issue("FOUT", "Accuracy stemt niet overeen met correct1/n (tolerantie 0,0001 procentpunt).")
            if run.metrics_safe:
                epochs = sorted(run.test["epoch"].astype(int).tolist())
                if epochs and epochs != list(range(1, epochs[-1] + 1)):
                    run.issue("FOUT", "Testgeschiedenis bevat ontbrekende epochs.")
                if set(run.metrics.loc[run.metrics["phase"] == "train", "epoch"]) != set(epochs):
                    run.issue("FOUT", "Train- en testgeschiedenis bevatten verschillende epochs.")
                for phase, config_key in (("train", "train_examples"), ("test", "test_examples")):
                    expected = run.config.get(config_key)
                    rows = run.metrics.loc[run.metrics["phase"] == phase]
                    if expected is not None and "n" in rows and not rows["n"].eq(expected).all():
                        run.issue("FOUT", f"{phase}: n wijkt af van de configuratie ({expected}).")
                if history_rows:
                    for row in history_rows:
                        match = run.metrics.loc[(run.metrics["epoch"] == row.get("epoch")) &
                                                (run.metrics["phase"] == row.get("phase"))]
                        if len(match) != 1:
                            run.issue("FOUT", "history.json en epoch_metrics.csv bevatten verschillende rijen.")
                            break
                        if any(_number(row.get(key)) is None or
                               not math.isclose(float(match.iloc[0][key]), float(row[key]), rel_tol=1e-9, abs_tol=1e-7)
                               for key in ("top1_percent", "objective_loss")):
                            run.issue("FOUT", "history.json en epoch_metrics.csv bevatten verschillende scores.")
                            break

    if run.config.get("weight_format", EXPECTED_FORMAT) != EXPECTED_FORMAT:
        run.issue("FOUT", "Het runcontract noemt een ander gewichtsformaat dan MXFP8_E4M3.")
    if run.config.get("activation_bits", 32) != 32:
        run.issue("FOUT", "Het runcontract noemt een andere activatieprecisie dan A32.")
    if run.config.get("master_dtype", "float32") != "float32":
        run.issue("FOUT", "De updatemaster is volgens het contract geen FP32.")
    if run.test.empty:
        run.issue("INFO", "Geen bruikbare afgeronde SVHN-trainingsresultaten ingelezen.")
    else:
        if not run.synthetic and not run.provenance_confirmed:
            run.issue("LET OP", "Herkomst niet bevestigd: synthetic_fixture_only ontbreekt. Niet automatisch als echte SVHN-run beschouwen.")
        for role, expected_epoch in (("best", run.observed_best_epoch), ("last", run.last_epoch)):
            detail = run.selection.get(role, {})
            if detail and detail.get("epoch") != expected_epoch:
                run.issue("FOUT", f"Checkpointselectie {role}: opgeslagen epoch {detail.get('epoch')}, verwacht {expected_epoch}.")
            metric = run.metric(detail.get("epoch"))
            score = _number(detail.get("selection_score", detail.get("score")))
            if metric is not None and score is not None and abs(score - float(metric["top1_percent"])) > 1e-7:
                run.issue("FOUT", f"Checkpointselectie {role}: opgeslagen selectiescore verschilt van epochmetrics.")
        if not run.selected.empty:
            if not {"role", "epoch", "top1_percent"} <= set(run.selected.columns):
                run.issue("FOUT", "selected_metrics.csv mist role, epoch of top1_percent.")
            else:
                for role, selection_key in (("best_active_test", "best"), ("last_epoch", "last")):
                    selected_rows = run.selected.loc[run.selected["role"] == role]
                    if len(selected_rows) != 1:
                        run.issue("FOUT", f"selected_metrics.csv: verwacht precies één rij voor {role}.")
                        continue
                    row = selected_rows.iloc[0]
                    metric = run.metric(_number(row["epoch"]))
                    score = _number(row["top1_percent"])
                    selected_epoch = run.selection.get(selection_key, {}).get("epoch")
                    if metric is None or score is None or abs(score - float(metric["top1_percent"])) > 1e-7:
                        run.issue("FOUT", f"selected_metrics.csv: {role} wijkt af van de epochmetrics.")
                    if selected_epoch is not None and _number(row["epoch"]) != selected_epoch:
                        run.issue("FOUT", f"selected_metrics.csv: {role} verwijst naar een andere epoch dan checkpoint_selection.json.")
        if run.summary.get("completed_epochs", run.last_epoch) != run.last_epoch:
            run.issue("FOUT", "Aantal afgeronde epochs in training_summary.json verschilt van de metrics.")
        if run.summary.get("full_100_epoch_training_completed") is True and run.metric(100) is None:
            run.issue("FOUT", "Samenvatting claimt een volledige training, maar epoch100-metrics ontbreken.")
    for warning in run.report.get("warnings", []):
        text = warning.get("message", str(warning)) if isinstance(warning, dict) else str(warning)
        run.issue("RUNWAARSCHUWING", text)
    if run.report.get("error"):
        run.issue("RUNFOUT", str(run.report["error"]))
    return run


def status_markdown(run):
    if run.synthetic:
        heading = "SYNTHETISCHE CONTROLE — geen SVHN-onderzoeksresultaat"
    elif run.test.empty:
        heading = "GEEN TRAININGSRESULTATEN — leeg of onvolledig analysesjabloon"
    elif run.provenance_confirmed:
        heading = "Opgeslagen SVHN-run — herkomst volgens het runrapport"
    else:
        heading = "HERKOMST NIET BEVESTIGD — eerst runrapport controleren"
    status = str(run.report.get("status", "NIET BESCHIKBAAR"))
    errors = sum(item["Niveau"] in ("FOUT", "RUNFOUT") for item in run.issues)
    last = run.last_epoch
    progress = f"Laatste ingelezen testepoch: **{last}**." if last else "Er is nog geen bruikbare testepoch beschikbaar."
    if run.metric(100) is not None:
        endpoint = "**Epoch100-metrics zijn beschikbaar.** De figuren tonen de gemeten waarden."
    elif run.synthetic:
        endpoint = "**Geen echte epoch100 uitgevoerd; geen eindresultaat.** Synthetische cijfers dienen alleen voor softwarecontrole."
    elif last is not None and last <= 5 and run.summary.get("full_100_epoch_training_completed") is False:
        endpoint = "**Epoch100 niet uitgevoerd; geen eindresultaat.** Vijf epochs beoordelen techniek, eerste leergedrag en looptijd."
    else:
        endpoint = "**Epoch100 niet beschikbaar; geen eindresultaat.** Ontbrekende accuracies worden niet geëxtrapoleerd."
    stopped = ""
    if status not in ("PASS", "NIET BESCHIKBAAR"):
        stopped = "\n\n**De run is niet als PASS afgesloten.** Lees de meldingen hieronder vóór interpretatie."
    return (f"### {heading}\n\nRunstatus: **{status}** · Analysefouten: **{errors}**. {progress}\n\n"
            f"{endpoint}{stopped}\n\nDeze analyse leest uitsluitend opgeslagen JSON/CSV. "
            "Zij start geen training en laadt geen model of checkpoint.")


def protocol_table(run):
    c = run.config
    items = [("Gewichtsformaat", c.get("weight_format", run.summary.get("weight_format", "niet gerapporteerd"))),
             ("Activaties", f"A{c['activation_bits']}" if "activation_bits" in c else "niet gerapporteerd"),
             ("Updatemaster", c.get("master_dtype", "niet gerapporteerd")),
             ("Model", f"{c.get('model', '?')}, expand={c.get('expand', '?')}"),
             ("Data", f"{c.get('dataset', '?')}; train-split={c.get('train_split', '?')}; extra={c.get('extra_used', '?')}"),
             ("Train / test", f"{c.get('train_examples', '?')} / {c.get('test_examples', '?')}"),
             ("Groepsgrootte / schaal", f"{c.get('mx_block_size', '?')} / {c.get('mx_scale_format', '?')}"),
             ("Batch / seed", f"{c.get('batch_size', '?')} / {c.get('seed', '?')}"),
             ("Afgeronde epochs", run.last_epoch if run.last_epoch is not None else "geen"),
             ("Beste checkpoint", run.selection.get("policy", "selectiebestand ontbreekt"))]
    return pd.DataFrame(items, columns=["Onderdeel", "Opgeslagen configuratie"])


def checkpoint_table(run):
    rows = []
    for role, label, fallback in (("last", "Laatste epoch", run.last_epoch),
                                  ("best", "Beste E4M3-checkpoint", run.observed_best_epoch)):
        detail = run.selection.get(role, {})
        epoch = detail.get("epoch", fallback)
        metric = run.metric(epoch)
        rows.append({"Controlepunt": label, "Epoch": epoch,
                     "Testaccuracy [%]": float(metric["top1_percent"]) if metric is not None else np.nan,
                     "Correct / n": _counts(metric),
                     "CE-loss": float(metric["objective_loss"]) if metric is not None else np.nan,
                     "Selectiebron": "checkpoint_selection.json" if detail else "afgeleid uit metrics; selectie niet bevestigd",
                     "Checkpoint": detail.get("file", "niet gerapporteerd")})
    metric = run.metric(100)
    rows.append({"Controlepunt": "Vast eindpunt epoch100", "Epoch": 100,
                 "Testaccuracy [%]": float(metric["top1_percent"]) if metric is not None else np.nan,
                 "Correct / n": _counts(metric),
                 "CE-loss": float(metric["objective_loss"]) if metric is not None else np.nan,
                 "Selectiebron": "epoch_metrics/history" if metric is not None else "niet uitgevoerd / niet beschikbaar",
                 "Checkpoint": "vast epochnummer; geen nieuwe selectie"})
    return pd.DataFrame(rows)


def _counts(row):
    if row is None:
        return "—"
    correct, n = _number(row.get("correct1")), _number(row.get("n"))
    return f"{int(correct):,} / {int(n):,}".replace(",", ".") if correct is not None and n is not None else "niet gerapporteerd"


def diagnostics_table(run):
    return pd.DataFrame(run.issues, columns=["Niveau", "Melding"]) if run.issues else pd.DataFrame(
        [{"Niveau": "OK", "Melding": "Geen analyse-inconsistenties of runwaarschuwingen gevonden."}])


def audit_table(run):
    items = [("Gradientcontrole vóór iedere update", run.summary.get("gradient_checks_before_every_update")),
             ("FP32-masters en Adam-momenten geverifieerd", run.summary.get("fp32_masters_and_moments_verified")),
             ("Bronbestanden onveranderd", run.report.get("source_unchanged")),
             ("Ruwe datasetbestanden onveranderd", run.report.get("dataset_files_unchanged")),
             ("Optimizerupdates", run.summary.get("optimizer_steps", run.report.get("optimizer_steps"))),
             ("Selectie op officiële testset", run.selection.get("selected_on_official_test")),
             ("Onafhankelijke testschatting", run.selection.get("independent_test_estimate"))]
    return pd.DataFrame([(key, "niet gerapporteerd" if val is None else val) for key, val in items],
                        columns=["Opgeslagen controle", "Resultaat volgens rapport"])


def weight_update_table(run):
    if run.updates.empty:
        return pd.DataFrame({"Melding": ["Geen weight_updates.csv beschikbaar."]})
    columns = [key for key in ("epoch", "layer", "master_changed_values", "master_update_l2", "master_dtype",
                               "changed_element_codes_including_padding", "changed_scale_codes") if key in run.updates]
    return run.updates[columns].copy()


def _title(run, text):
    return ("SYNTHETISCHE CONTROLE · " if run.synthetic else "") + text


def _empty(ax, text):
    ax.set_axis_off()
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center", wrap=True,
            fontsize=11, color="#525D65", bbox=dict(boxstyle="round,pad=.9", facecolor="#F1F4F6", edgecolor="none"))


def plot_learning(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
        fig.suptitle(_title(run, "Dedicated E4M3/A32 — gemeten leercurves"), fontsize=14, fontweight="bold")
        if run.test.empty:
            for ax in axes:
                _empty(ax, "Geen bruikbare epochmetrics\nZie status en meldingen.")
            return fig
        for phase, label, color in (("train", "Train", BLUE), ("test", "Test", GREEN)):
            data = run.metrics.loc[run.metrics["phase"] == phase].sort_values("epoch")
            for ax, metric in zip(axes, ("top1_percent", "objective_loss")):
                ax.plot(data["epoch"], data[metric], marker="o", markersize=4, linewidth=1.8,
                        color=color, label=label)
        axes[0].set(title="Top-1-accuracy", ylabel="Accuracy [%]", ylim=(0, 100))
        axes[1].set(title="Harde cross-entropie", ylabel="Gemiddelde CE-loss")
        for ax in axes:
            ax.set_xlabel("Afgeronde epoch")
            ax.legend(frameon=False)
            if run.last_epoch <= 10:
                ax.set_xticks(sorted(run.test["epoch"].astype(int)))
        return fig


def plot_checkpoint_accuracy(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), width_ratios=[1.7, 1], constrained_layout=True)
        fig.suptitle(_title(run, "E4M3/A32 — laatste, beste en vast epoch100-eindpunt"), fontsize=14, fontweight="bold")
        if run.test.empty:
            _empty(axes[0], "Nog geen gemeten checkpointaccuracy")
        else:
            table = checkpoint_table(run).iloc[:2]
            for i, (_, row) in enumerate(table.iterrows()):
                score = _number(row["Testaccuracy [%]"])
                if score is None:
                    axes[0].text(i, 45, "metric\nontbreekt", ha="center")
                    continue
                axes[0].bar(i, score, width=.5, color=(BLUE, GREEN)[i])
                axes[0].text(i, min(score + 2.2, 102), f"{score:.4f}%", ha="center", va="bottom", fontsize=11)
            labels = [f"Laatste\n(epoch {table.iloc[0]['Epoch']})", f"Beste E4M3\n(epoch {table.iloc[1]['Epoch']})"]
            axes[0].set(xticks=[0, 1], xticklabels=labels, ylim=(0, 108), xlim=(-.65, 1.65),
                        ylabel="Testaccuracy [%]", title="Gemeten checkpointvergelijking")
            axes[0].set_yticks(range(0, 101, 20))
        metric = run.metric(100)
        if metric is None:
            _empty(axes[1], "Epoch100 niet uitgevoerd /\nniet beschikbaar\n\nGEEN EINDRESULTAAT\nGeen voorspelde accuracy")
        else:
            score = float(metric["top1_percent"])
            axes[1].bar([0], [score], width=.5, color=BLUE)
            axes[1].text(0, min(score + 2.2, 102), f"{score:.4f}%", ha="center", va="bottom", fontsize=11)
            axes[1].set(xticks=[0], xticklabels=["Epoch100"], ylim=(0, 108), xlim=(-.65, .65),
                        ylabel="Testaccuracy [%]", title="Vast eindpunt")
            axes[1].set_yticks(range(0, 101, 20))
        return fig


def plot_timing(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), width_ratios=[1.35, 1], constrained_layout=True)
        fig.suptitle(_title(run, "Gemeten epochtijden en voorzichtige looptijdschatting"), fontsize=14, fontweight="bold")
        data = run.times.copy()
        if data.empty or "epoch" not in data or "total_epoch_seconds" not in data:
            _empty(axes[0], "Geen opgeslagen epochtijden")
        else:
            data = data.sort_values("epoch")
            bottom = np.zeros(len(data))
            for key, label, color in (("train_seconds", "Train", BLUE), ("test_seconds", "Test", GREEN),
                                      ("checkpoint_seconds", "Checkpoint", AMBER), ("other_seconds", "Overig", "#A3ADB5")):
                if key in data:
                    values = pd.to_numeric(data[key], errors="coerce").to_numpy(dtype=float)
                    if np.isfinite(values).all() and (values >= 0).all():
                        axes[0].bar(data["epoch"], values, bottom=bottom, label=label, color=color, width=.68)
                        bottom += values
            total = pd.to_numeric(data["total_epoch_seconds"], errors="coerce")
            axes[0].plot(data["epoch"], total, "ko", markersize=3.5, label="Gemeten totaal")
            axes[0].set(title="Tijd per afgeronde epoch", xlabel="Epoch", ylabel="Seconden")
            if len(data) <= 10:
                axes[0].set_xticks(data["epoch"])
            axes[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.94, fontsize=8, ncol=2)
        timing = run.timing
        estimate = _number(timing.get("estimated_100_epoch_seconds"))
        if not timing.get("available") or estimate is None:
            _empty(axes[1], "Nog geen betrouwbare tijdsbasis\n" + str(timing.get("reason", "Tijdschatting ontbreekt.")))
        else:
            estimate /= 3600
            bounds = timing.get("observed_range_projection_seconds", [])
            low, high = [float(x) / 3600 for x in bounds] if len(bounds) == 2 and all(_number(x) is not None for x in bounds) else (estimate, estimate)
            if low <= estimate <= high:
                axes[1].errorbar([estimate], [0], xerr=[[estimate - low], [high - estimate]],
                                 fmt="o", markersize=9, color=AMBER, capsize=7, linewidth=2)
            else:
                axes[1].plot([estimate], [0], "o", color=AMBER, markersize=9)
            axes[1].text(estimate, .23, f"{estimate:.2f} uur", ha="center", fontsize=13, fontweight="bold", color=AMBER)
            axes[1].text(.5, .16, f"Geprojecteerd waargenomen bereik:\n{low:.2f}–{high:.2f} uur\nGeen betrouwbaarheidsinterval",
                         transform=axes[1].transAxes, ha="center", va="center", fontsize=10)
            axes[1].set(title="Schatting: nieuwe run van 100 epochs", xlabel="Looptijd [uur]",
                        yticks=[], ylim=(-.8, .7), xlim=(0, max(high, estimate, .01) * 1.3))
            axes[1].spines["left"].set_visible(False)
        return fig


def timing_markdown(run):
    t = run.timing
    if not t.get("available"):
        return "Er is nog geen opgeslagen looptijdschatting beschikbaar. Er wordt geen schatting uit accuracies afgeleid."
    steady = t.get("steady_epochs", [])
    median = _number(t.get("median_epoch_seconds"))
    warm = f"{median:.1f} s" if median is not None else "niet gerapporteerd"
    return (f"Warme epochs volgens het tijdsrapport: **{steady}**; mediaan: **{warm}**. "
            "De schatting geldt voor een **nieuwe run van 100 epochs vanaf initialisatie**, niet voor het hervatten van deze pilot. "
            "De eerste gemeten epoch wordt apart geteld; het waargenomen minimum/maximum van de warme epochs levert alleen een scenario-bereik. "
            "Dit is geen statistisch betrouwbaarheidsinterval. Download, installatie, data-inladen en eindrapportage zijn niet inbegrepen. "
            "Vijf epochs meten geen langdurige thermische vertraging of LR-overgangen op epochs50/75/90. "
            "Er wordt **geen accuracy voor epoch100 voorspeld**.")


def plot_memory(run):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 3.5), constrained_layout=True)
        fig.suptitle(_title(run, "Gemeten CUDA-geheugenpiek per epoch"), fontsize=14, fontweight="bold")
        found = False
        data = run.times
        for key, label, color in (("peak_gpu_allocated_bytes", "Toegewezen", BLUE),
                                  ("peak_gpu_reserved_bytes", "Gereserveerd", GREEN)):
            if not data.empty and key in data and "epoch" in data:
                values = pd.to_numeric(data[key], errors="coerce")
                valid = values.notna() & np.isfinite(values) & (values >= 0)
                if valid.any():
                    ax.plot(data.loc[valid, "epoch"], values.loc[valid] / (1024 ** 2), marker="o",
                            color=color, label=label)
                    found = True
        if found:
            ax.set(xlabel="Epoch", ylabel="MiB", ylim=(0, None))
            ax.legend(frameon=False)
            if len(data) <= 10:
                ax.set_xticks(data["epoch"])
        else:
            _empty(ax, "Geen CUDA-geheugenmetingen beschikbaar\nOntbrekende waarden worden niet als nul weergegeven.")
        return fig


def save_figures(figures: dict, run: RunAnalysis) -> Path:
    """Write PNG figures only to a new folder; retain every previous analysis."""
    if not run.path.is_dir():
        raise ValueError("RUN_DIR moet bestaan voordat figuren kunnen worden opgeslagen.")
    destination = run.path / ("analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    destination.mkdir(exist_ok=False)
    for name, figure in figures.items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Ongeldige figuurnaam.")
        figure.savefig(destination / (name + ".png"), dpi=160, bbox_inches="tight")
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "analysis_only": True, "training_started": False, "checkpoint_loaded": False,
                "synthetic_fixture_only": run.synthetic, "source_run": str(run.path),
                "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.input_files},
                "figures": [name + ".png" for name in figures], "issues": run.issues}
    (destination / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination
