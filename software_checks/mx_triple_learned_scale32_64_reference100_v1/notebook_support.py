"""Read-only analysis of saved learned per-block scale32/64 MX/A32 hundred-epoch results.

Reuses the established triple-direct report reader; reports the E4 source chain.

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


FORMATS = {4: "MXFP4_E2M1", 6: "MXFP6_E2M3", 8: "MXFP8_E4M3"}
BANKS = {4: 4, 6: 32, 8: 8}
DISPLAY_BITS = (8, 6, 4)
SELECTION_FORMAT = FORMATS[8]
LABELS = {8: "E4M3", 6: "E2M3", 4: "E2M1"}
BLUE, GREEN, AMBER, RED = "#245A81", "#33846D", "#C98220", "#B44040"
COLORS = {8: BLUE, 6: GREEN, 4: AMBER}
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
    selection_safe: bool = True
    prediction_safe: bool = True
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    scale_epochs: pd.DataFrame = field(default_factory=pd.DataFrame)
    scale_blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    scales_safe: bool = True

    @property
    def config(self):
        return self.report.get("config", {})

    @property
    def synthetic(self):
        return any(value is True for value in (
            self.report.get("synthetic_fixture_only"), self.report.get("synthetic_checks_only"),
            self.summary.get("synthetic_fixture_only")))

    @property
    def provenance_confirmed(self):
        return not self.synthetic and any(value is False for value in (
            self.report.get("synthetic_fixture_only"), self.summary.get("synthetic_fixture_only")))

    @property
    def test(self):
        if not self.metrics_safe or self.metrics.empty or "phase" not in self.metrics:
            return pd.DataFrame()
        return self.metrics.loc[self.metrics["phase"] == "test"].sort_values(["epoch", "weight_bits"])

    @property
    def last_epoch(self):
        return int(self.test["epoch"].max()) if not self.test.empty else None

    @property
    def observed_best_epoch(self):
        if self.test.empty:
            return None
        high = self.test.loc[self.test["weight_format"] == SELECTION_FORMAT].sort_values("epoch")
        # idxmax selects the FIRST maximum; never select a separate best per mode.
        return int(high.loc[high["top1_percent"].idxmax(), "epoch"]) if not high.empty else None

    @property
    def reference_complete(self):
        return (self.report.get("status") == "PASS"
                and self.report.get("full_training_completed") is True
                and self.summary.get("full_training_completed") is True
                and self.summary.get("full_100_epoch_training_completed") is True
                and self.last_epoch == 100 and self.metrics_safe and self.selection_safe and self.prediction_safe and self.scales_safe
                and not any(x['Niveau'] in ('FOUT', 'RUNFOUT') for x in self.issues)
                and all(self.test.loc[self.test["weight_format"] == fmt, "epoch"].astype(int).tolist()
                        == list(range(1, 101)) for fmt in FORMATS.values()))

    def metric(self, epoch, weight_format=SELECTION_FORMAT):
        if epoch is None or self.test.empty:
            return None
        rows = self.test.loc[(self.test["epoch"] == epoch) & (self.test["weight_format"] == weight_format)]
        return rows.iloc[0] if len(rows) == 1 else None

    def issue(self, level, text):
        self.issues.append({"Niveau": level, "Melding": text})



def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def expected_lr(config, epoch, after_step=False, scale=False):
    value = float(config.get('scale_lr' if scale else 'lr', .001))
    for milestone in config.get('scheduler_milestones', [50, 75, 90]):
        if epoch >= milestone if after_step else epoch > milestone:
            value *= float(config.get('scheduler_gamma', .1))
    return value


def epoch_ticks(last, zero=False):
    first = 0 if zero else 1
    if last is None:
        return [first]
    if last <= 10:
        return list(range(first, last + 1))
    step = 10 if last <= 50 else 20
    return sorted(set([first, last] + list(range(step, last + 1, step))))


def mark_lr_phases(ax, last):
    boundaries = [x + .5 for x in (50, 75, 90) if last is not None and last > x]
    if boundaries:
        ax.vlines(boundaries, 0, 1, transform=ax.get_xaxis_transform(),
                  color='#7B858E', linestyles=':', linewidth=.9, alpha=.65)


def _validate_contract(run):
    if not run.report:
        return
    failures = []
    if run.report.get('schema') != 'mx-triple-learned-scale32-64-reference100-report-v1':
        failures.append('Dit is niet het rapport van de geleerde schaal32/64-honderd-epochtraining.')
    c = run.config
    expected = {'schema': 'mx-triple-learned-scale32-64-reference100-protocol-v1',
                'activation_bits': 32, 'master_dtype': 'float32', 'mode': 'joint',
                'truncation': True, 'exponent_truncation': True, 'mantissa_truncation': True,
                'source_weight_bits': 8, 'shared_scale_compensation': True,
                'scale_exponent_offset_choices': [5, 6], 'scale_factor_choices': [32, 64],
                'direct_lower_quantization': False, 'exponent_bias_compensation': False,
                'learned_shared_scale': True, 'distillation': True,
                'extra_FP32_teacher_forward': False, 'reference_end_epoch': 100,
                'planned_epochs': 100, 'base_parameters': 132954,
                'learned_scale_parameters': 4080, 'parameters': 137034,
                'scale_proxy_dtype': 'float32', 'scale_proxy_init': 6.0,
                'scale_proxy_bounds': [5.0, 6.0], 'scale_threshold': 5.5,
                'scale_tie_break': 'offset6', 'scale_granularity': 'per_MX_block',
                'source_scale_learned': False, 'scale_gradient_normalization': 'none',
                'scale_projection': 'after_each_Adam_step', 'scale_lr': .001,
                'lr': .001, 'scheduler_milestones': [50, 75, 90], 'scheduler_gamma': .1}
    for key, value in expected.items():
        if key not in c or c[key] != value:
            failures.append(f'Runcontract: {key}={c.get(key)!r}; verwacht {value!r}.')
    if c.get('weight_formats') != {str(bit): fmt for bit, fmt in FORMATS.items()}:
        failures.append('Runcontract bevat niet precies E4M3, E2M3 en E2M1.')
    banks = c.get('bn_bank_mapping', {})
    if not isinstance(banks, dict) or {str(k): str(v) for k, v in banks.items()} != {str(k): str(v) for k, v in BANKS.items()}:
        failures.append('BN-bankmapping wijkt af van het drievoudige truncatiecontract.')
    if failures:
        run.metrics_safe = False
        for message in failures:
            run.issue('FOUT', message)



def _validate_predictions(run, records):
    """Validate aggregate test distributions; never claim per-image matching."""
    if run.test.empty:
        return
    rows = []
    fixed_labels = None
    for record in records:
        if not isinstance(record, dict):
            continue
        epoch = record.get('epoch')
        counts_by_bit = record.get('test_prediction_counts')
        labels = record.get('test_label_counts')
        n = run.config.get('test_examples')
        def valid_counts(values):
            return (isinstance(values, list) and len(values) == 10
                    and all(type(x) is int and x >= 0 for x in values)
                    and sum(values) == n)
        if not valid_counts(labels) or not isinstance(counts_by_bit, dict) or set(counts_by_bit) != {'8', '6', '4'}:
            run.prediction_safe = False
            run.issue('FOUT', f'Epoch {epoch}: volledige label- en voorspellingstellingen per testmodus ontbreken.')
            continue
        if fixed_labels is not None and labels != fixed_labels:
            run.prediction_safe = False
            run.issue('FOUT', f'Epoch {epoch}: labelverdeling verschilt van eerdere volledige testepochs.')
        fixed_labels = labels
        for bit in DISPLAY_BITS:
            counts = counts_by_bit[str(bit)]
            if not valid_counts(counts):
                run.prediction_safe = False
                run.issue('FOUT', f'Epoch {epoch}/{LABELS[bit]}: verwacht tien niet-negatieve tellingen met som n.')
                continue
            metric = run.metric(epoch, FORMATS[bit])
            if metric is None:
                continue
            used = sum(x > 0 for x in counts)
            dominant = max(range(10), key=lambda i: counts[i])
            correct = int(metric['correct1'])
            if correct > sum(min(a, b) for a, b in zip(counts, labels)):
                run.prediction_safe = False
                run.issue('FOUT', f'Epoch {epoch}/{LABELS[bit]}: correct-aantal is onmogelijk bij de label-/voorspellingstellingen.')
            if used == 1 and correct != labels[dominant]:
                run.prediction_safe = False
                run.issue('FOUT', f'Epoch {epoch}/{LABELS[bit]}: constante voorspelling strookt niet met de accuracy.')
            if used == 1:
                run.issue('UITKOMST', f'Epoch {epoch}/{LABELS[bit]} voorspelt alle testbeelden als cijfer {dominant}; dit is een meetuitkomst, geen technische FAIL.')
            rows.append({'Epoch': epoch, 'Gewichtsformaat': FORMATS[bit], 'weight_bits': bit,
                         'Voorspelde klassen': used, 'Meest voorspelde cijfer': dominant,
                         'Aandeel dominant cijfer [%]': 100 * counts[dominant] / n,
                         'Aandeel dat cijfer in labels [%]': 100 * labels[dominant] / n,
                         'Testaccuracy [%]': float(metric['top1_percent']),
                         'Constante voorspelling': used == 1, 'Testbeelden': n,
                         'Tellingen voorspeld cijfer 0..9': counts,
                         'Tellingen echt cijfer 0..9': labels})
    expected = 3 * len(records)
    if len(rows) != expected:
        run.prediction_safe = False
    if run.prediction_safe:
        run.predictions = pd.DataFrame(rows)


def _validate_exports(run):
    exports = run.report.get('mx_exports')
    if not exports:
        if not run.test.empty:
            run.issue('FOUT', 'Geen exportcontroles bij de afgeronde trainingsmetrics.')
        return
    if not isinstance(exports, dict) or set(exports) != {'best_E4M3_test', 'last_epoch'}:
        run.issue('FOUT', 'Exportrollen verschillen van de twee vaste checkpointrollen.')
        return
    for role, modes in exports.items():
        if not isinstance(modes, dict) or set(modes) != {'8', '6', '4'}:
            run.issue('FOUT', f'{role}: niet precies één bronexport en twee afgeleide modi.')
            continue
        key = 'best' if role == 'best_E4M3_test' else 'last'
        expected_epoch = run.selection.get(key, {}).get('epoch')
        for bit in DISPLAY_BITS:
            detail = modes[str(bit)]
            expected = {'epoch': expected_epoch, 'weight_bits': bit,
                        'six_fp32_masters_absent': True, 'decoded_weights_exact': True,
                        'source_scale_bytes_copied': bit == 8,
                        'scale_exponent_offset': 0 if bit == 8 else None,
                        'scale_factor': 1 if bit == 8 else None,
                        'scale_exponent_offset_choices': [0] if bit == 8 else [5, 6],
                        'scale_factor_choices': [1] if bit == 8 else [32, 64],
                        'shared_scale_compensation': bit != 8,
                        'learned_shared_scale': bit != 8,
                        'fp32_scale_proxies_absent': True,
                        'packed_scale_choices_in_source': True,
                        'packed_scale_choices_bytes': 510,
                        'truncation': bit != 8, 'is_source_export': bit == 8,
                        'derived_from_saved_source': bit != 8,
                        'master_weights_read_to_derive': False, 'saved_and_reloaded': True}
            if any(k not in detail or detail[k] != v for k, v in expected.items()):
                different = [k for k, v in expected.items() if k not in detail or detail[k] != v]
                run.issue('FOUT', f'{role}/{bit}: exportcontract wijkt af: ' + ', '.join(different))
            if detail.get('source_file_sha256') != modes['8'].get('sha256'):
                run.issue('FOUT', f'{role}/{bit}: afleiding verwijst niet naar de opgeslagen E4M3-bronhash.')



def load_run(run_dir) -> RunAnalysis:
    """Read saved JSON/CSV only; do not infer missing mode scores or load checkpoints."""
    run = RunAnalysis(Path(run_dir).expanduser().resolve())
    files = {"run_report.json": "report", "history.json": "history",
             "training_summary.json": "summary", "checkpoint_selection.json": "selection",
             "timing_estimate.json": "timing", "epoch_metrics.csv": "metrics",
             "selected_metrics.csv": "selected", "epoch_times.csv": "times", "weight_updates.csv": "updates",
             "scale_epoch_metrics.csv": "scale_epochs", "scale_block_state.csv": "scale_blocks"}
    for name, attr in files.items():
        path = run.path / name
        if not path.is_file():
            run.issue("ONTBREEKT", name)
            continue
        run.input_files.append(path)
        try:
            if name.endswith(".json"):
                value = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(value, dict):
                    raise ValueError("Verwacht een JSON-object.")
            else:
                value = pd.read_csv(path, float_precision='round_trip')
            setattr(run, attr, value)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            run.issue("FOUT", f"{name}: {exc}")
    records = run.history.get("epochs", [])
    history_rows = []
    if not isinstance(records, list):
        run.issue("FOUT", "history.json: epochs is geen lijst.")
        records = []
    for record in records:
        if not isinstance(record, dict):
            run.issue("FOUT", "history.json bevat een ongeldige epochrij.")
            continue
        for phase in ("train", "test"):
            rows = record.get(phase, [])
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                run.issue("FOUT", f"history.json: ongeldige {phase}-rijen.")
                continue
            history_rows.extend({**row, "epoch": record.get("epoch"), "phase": phase} for row in rows)
    if run.metrics.empty and history_rows:
        run.metrics = pd.DataFrame(history_rows)
        run.issue("LET OP", "Epochgrafieken gebruiken history.json; epoch_metrics.csv ontbreekt of is leeg.")
    if run.times.empty and records:
        keys = ("epoch", "train_seconds", "test_seconds", "checkpoint_seconds", "other_seconds",
                "total_epoch_seconds", "first_epoch_in_process", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
        run.times = pd.DataFrame([{key: row.get(key) for key in keys} for row in records if isinstance(row, dict)])
        run.issue("LET OP", "Tijd/geheugen gebruiken history.json; epoch_times.csv ontbreekt of is leeg.")
    _validate_contract(run)
    _validate_metrics(run, history_rows)
    _validate_selection(run)
    _validate_audits(run, records)
    _validate_predictions(run, records)
    _validate_exports(run)
    _validate_scales(run, records)
    if run.test.empty:
        run.issue("INFO", "Geen bruikbare afgeronde trainingsresultaten met alle drie modi ingelezen.")
    else:
        if not run.synthetic and not run.provenance_confirmed:
            run.issue("LET OP", "Herkomst niet bevestigd: synthetic_fixture_only ontbreekt. Niet automatisch als echte SVHN-run beschouwen.")
        if run.summary.get("completed_epochs", run.last_epoch) != run.last_epoch:
            run.issue("FOUT", "Aantal afgeronde epochs verschilt tussen samenvatting en metrics.")
        if run.summary.get("last_epoch", run.last_epoch) != run.last_epoch:
            run.issue("FOUT", "Laatste epoch verschilt tussen samenvatting en metrics.")
        if run.summary.get("best_epoch_E4M3_A32", run.observed_best_epoch) != run.observed_best_epoch:
            run.issue("FOUT", "Beste E4M3-epoch in de samenvatting verschilt van het eerste strikte maximum.")
        if run.report.get("status") == "PASS" and not run.reference_complete:
            run.issue("FOUT", "Een PASS voor deze training vereist honderd complete, consistente epochs.")
        if any(source.get(key) is True for source in (run.report, run.summary, run.timing)
               for key in ("full_training_completed", "full_100_epoch_training_completed")) and not run.reference_complete:
            run.issue("FOUT", "Een geclaimde volledige honderd-epochtraining wordt niet door de resultaten bevestigd.")
        if _number(run.timing.get("measured_100_epoch_seconds")) is not None and not run.reference_complete:
            run.issue("FOUT", "Gemeten honderd-epochtotaal gerapporteerd zonder volledige referentie.")
        measured = measured_completed_seconds(run)
        for source, field_name in ((run.summary, "measured_completed_epoch_seconds"),
                                   (run.timing, "measured_completed_epoch_seconds"),
                                   (run.timing, "measured_100_epoch_seconds")):
            saved = _number(source.get(field_name))
            if measured is not None and saved is not None and not math.isclose(saved, measured, rel_tol=1e-9, abs_tol=1e-6):
                run.issue("FOUT", f"{field_name} verschilt van de som van gemeten epochtijden.")
    for warning in run.report.get("warnings", []):
        run.issue("RUNWAARSCHUWING", warning.get("message", str(warning)) if isinstance(warning, dict) else str(warning))
    if run.report.get("error"):
        run.issue("RUNFOUT", str(run.report["error"]))
    return run


def _validate_metrics(run, history_rows):
    if run.metrics.empty:
        return
    def fail(message):
        run.issue("FOUT", message)
        run.metrics_safe = False
    data = run.metrics
    required = {"epoch", "phase", "weight_bits", "weight_format", "activation_bits", "bn_bank_key",
                "n", "correct1", "top1_percent", "objective_type", "objective_loss"}
    if not required <= set(data):
        fail("Ontbrekende metricvelden: " + ", ".join(sorted(required - set(data))))
        return
    numeric = ("epoch", "weight_bits", "activation_bits", "bn_bank_key", "n", "correct1", "top1_percent", "objective_loss")
    for key in (*numeric, "count_top1_percent"):
        if key in data:
            data[key] = pd.to_numeric(data[key], errors="coerce")
    if not np.isfinite(data[list(numeric)].to_numpy(dtype=float)).all():
        fail("Niet-eindige of ontbrekende essentiële metrics. Metricgrafieken geblokkeerd.")
        return
    if not ((data["epoch"] >= 1) & (data["epoch"] <= 100) & (data["epoch"] % 1 == 0)).all():
        fail("Ongeldige epochnummers: deze training bevat uitsluitend epochs 1 t/m 100.")
    if data.duplicated(["epoch", "phase", "weight_format"]).any():
        fail("Dubbele metrics voor dezelfde epoch/fase/gewichtsmodus.")
    if not data["phase"].isin(["train", "test"]).all() or not data["weight_bits"].isin(FORMATS).all():
        fail("Onbekende fase of gewichtsmodus.")
    if not data["weight_bits"].map(FORMATS).eq(data["weight_format"]).all():
        fail("Gewichtsbits en formaatnaam komen niet overeen.")
    if not data["weight_bits"].map(BANKS).eq(data["bn_bank_key"]).all():
        fail("Metricrij bevat een verkeerde BN-bank.")
    if not data["activation_bits"].eq(32).all() or not data["top1_percent"].between(0, 100).all():
        fail("Activaties wijken af van A32 of accuracy ligt buiten [0, 100].")
    n, correct = data["n"], data["correct1"]
    if not ((n > 0) & (n % 1 == 0) & (correct >= 0) & (correct <= n) & (correct % 1 == 0)).all():
        fail("Ongeldige aantallen n/correct1.")
    elif (abs(data["top1_percent"] - 100 * correct / n) > 1e-4).any():
        fail("Accuracy verschilt van correct1/n (tolerantie 0,0001 procentpunt).")
    if "count_top1_percent" in data and (not np.isfinite(data["count_top1_percent"]).all()
            or (abs(data["count_top1_percent"] - 100 * correct / n) > 1e-7).any()):
        fail("count_top1_percent verschilt van correct1/n.")
    for (epoch, phase), rows in data.groupby(["epoch", "phase"]):
        if len(rows) != 3 or set(rows["weight_format"]) != set(FORMATS.values()):
            fail(f"Epoch {epoch:g}/{phase}: niet precies drie verschillende modi.")
        expected_n = run.config.get("train_examples" if phase == "train" else "test_examples")
        if rows["n"].nunique() != 1 or (expected_n is not None and not rows["n"].eq(expected_n).all()):
            fail(f"Epoch {epoch:g}/{phase}: datadekking verschilt tussen modi of van de configuratie.")
    expected_objective = np.where((data["phase"] == "train") & (data["weight_bits"] != 8),
                                  "recursive_soft_cross_entropy", "hard_cross_entropy")
    if not data["objective_type"].eq(expected_objective).all():
        fail("Losslabels wijken af: train-E4M3 hard CE; train-E2M3/E2M1 soft CE; test alle modi hard CE.")
    if "teacher_weight_format" in data:
        for _, row in data.iterrows():
            expected_teacher = FORMATS[8 if row["weight_bits"] == 6 else 6] if row["phase"] == "train" and row["weight_bits"] != 8 else None
            actual = row["teacher_weight_format"]
            if (expected_teacher is None and pd.notna(actual)) or (expected_teacher is not None and actual != expected_teacher):
                fail("Teacherlabel wijkt af van de recursieve E4M3 → E2M3 → E2M1-supervisie.")
                break
    if not run.metrics_safe:
        return
    epochs = sorted(data["epoch"].astype(int).unique())
    expected_epochs = list(range(1, epochs[-1] + 1))
    if epochs != expected_epochs or any(sorted(data.loc[data["phase"] == phase, "epoch"].astype(int).unique()) != epochs for phase in ("train", "test")):
        fail("Ontbrekende epochs of verschillende train-/testgeschiedenis.")
    if history_rows:
        history = pd.DataFrame(history_rows)
        keys = ["epoch", "phase", "weight_format"]
        if not required <= set(history) or len(history) != len(data) or history.duplicated(keys).any():
            fail("history.json en epoch_metrics.csv bevatten verschillende of onvolledige rijen.")
        else:
            for row in history_rows:
                found = data.loc[(data["epoch"] == row.get("epoch")) & (data["phase"] == row.get("phase")) & (data["weight_format"] == row.get("weight_format"))]
                if len(found) != 1 or any(not _same_number(found.iloc[0][key], row.get(key)) for key in numeric) or found.iloc[0]["objective_type"] != row.get("objective_type"):
                    fail("history.json en epoch_metrics.csv verschillen in scores, tellingen of leerdoelen.")
                    break


def _same_number(left, right, tolerance=1e-7):
    a, b = _number(left), _number(right)
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=tolerance)


def _validate_selection(run):
    if run.test.empty:
        return
    def fail(message):
        run.issue("FOUT", message)
        run.selection_safe = False
    selection = run.selection
    if selection:
        if selection.get("selection_mode") != SELECTION_FORMAT + "/A32":
            fail("Checkpointselectie is niet uitsluitend op E4M3/A32 gebaseerd.")
        if selection.get("all_modes_from_same_selected_checkpoint") is not True:
            fail("Gedeeld geselecteerd checkpoint voor alle drie modi niet bevestigd.")
        for role, expected in (("best", run.observed_best_epoch), ("last", run.last_epoch)):
            detail = selection.get(role, {})
            if detail.get("epoch") != expected:
                fail(f"Selectie {role}: epoch {detail.get('epoch')}; verwacht {expected}.")
            row = run.metric(detail.get("epoch"))
            if row is None or not _same_number(detail.get("selection_score_E4M3"), row["top1_percent"]):
                fail(f"Selectiescore {role} verschilt van de E4M3-epochmetric.")
        same = selection.get("best", {}).get("epoch") == selection.get("last", {}).get("epoch")
        if "same_checkpoint" in selection and selection["same_checkpoint"] != same:
            fail("same_checkpoint strookt niet met de geselecteerde epochs.")
        best_hash, last_hash = (selection.get(role, {}).get("sha256") for role in ("best", "last"))
        if same and best_hash and last_hash and best_hash != last_hash:
            fail("Dezelfde best-/last-epoch heeft verschillende checkpointhashes.")
    if run.selected.empty:
        return
    data = run.selected
    required = {"role", "epoch", "phase", "weight_format", "top1_percent", "objective_loss", "n", "correct1"}
    if not required <= set(data):
        fail("selected_metrics.csv mist verplichte checkpointmetrics.")
        return
    if set(data["role"]) != {"best_E4M3_test", "last_epoch"}:
        fail("selected_metrics.csv bevat andere rollen dan best_E4M3_test en last_epoch.")
    for role, key, fallback in (("best_E4M3_test", "best", run.observed_best_epoch), ("last_epoch", "last", run.last_epoch)):
        rows = data.loc[data["role"] == role]
        expected = selection.get(key, {}).get("epoch", fallback)
        if len(rows) != 3 or set(rows["weight_format"]) != set(FORMATS.values()) or rows["epoch"].nunique() != 1:
            fail(f"{role}: verwacht drie modi van precies één checkpoint; geen mix van modusspecifieke beste epochs.")
            continue
        for _, row in rows.iterrows():
            actual = run.metric(_number(row["epoch"]), row["weight_format"])
            if row["phase"] != "test" or _number(row["epoch"]) != expected or actual is None or any(not _same_number(row[field], actual[field]) for field in ("top1_percent", "objective_loss", "n", "correct1")):
                fail(f"{role}/{row['weight_format']}: geselecteerde metrics verschillen van de gedeelde checkpointepoch.")


def _validate_audits(run, records):
    c = run.config
    train_n, batch = _number(c.get("train_examples")), _number(c.get("batch_size"))
    expected = math.ceil(train_n / batch) if train_n and batch and batch > 0 else None
    test_n = _number(c.get('test_examples'))
    expected_test = math.ceil(test_n / batch) if test_n and batch and batch > 0 else None
    for record in records:
        if not isinstance(record, dict):
            continue
        epoch = record.get("epoch")
        if isinstance(epoch, int):
            expected_used, expected_next = expected_lr(c, epoch), expected_lr(c, epoch, after_step=True)
            for key, expected_rate in (("lr_used", expected_used), ("lr_next", expected_next),
                                       ("scale_lr_used", expected_used), ("scale_lr_next", expected_next)):
                value = _number(record.get(key))
                if value is None or not math.isclose(value, expected_rate, rel_tol=1e-9, abs_tol=1e-14):
                    run.issue("FOUT", f"Epoch {epoch}: {key} wijkt af van de gezamenlijke LR-fasen 50/75/90.")
        audit = record.get("gradient_checks", {})
        if audit and expected is not None:
            checks = {"optimizer_steps_in_phase": expected, "finite_gradient_checks": expected,
                      "detached_teacher_checks": 2 * expected,
                      "forward_calls_E4M3": expected, "forward_calls_E2M3": expected, "forward_calls_E2M1": expected}
            if any(audit.get(key) != value for key, value in checks.items()):
                run.issue("FOUT", f"Epoch {epoch}: gradient-, teacher- of forwardtellingen passen niet bij drie passes en één gezamenlijke update.")
            if audit.get("no_extra_FP32_teacher") is not True:
                run.issue("FOUT", f"Epoch {epoch}: geen-extra-FP32-teacher niet bevestigd.")
        for phase, source, batches, order in (
                ('train', audit.get('shared_source_checks'), expected, [8, 6, 4]),
                ('test', record.get('test_source_checks'), expected_test, [4, 6, 8])):
            if batches is None:
                continue
            source_checks = {'source_triplets': batches, 'high_encodings': 6 * batches,
                             'exponent_derivations': 6 * batches, 'mantissa_derivations': 6 * batches,
                             'scale_offsets_learned': 12 * batches,
                             'mantissa_scale_copies_unchanged': 6 * batches,
                             'offset_choices': [5,6], 'scale_factors': [32,64], 'learned_shared_scale': True,
                             'source_weight_bits': 8, 'training_order': order,
                             'source_cache_cleared': True, 'cache_cleared_before_adam': True,
                             'no_direct_lower_quantizer': True}
            if not isinstance(source, dict) or any(source.get(k) != v for k, v in source_checks.items()):
                run.issue('FOUT', f'Epoch {epoch}/{phase}: één bron, geleerde lagere schaal, mantissaschaalbehoud of cachecontrole niet bevestigd.')
            if isinstance(source, dict) and 'scale_copies_unchanged' in source:
                run.issue('FOUT', f'Epoch {epoch}/{phase}: verouderde claim van ongewijzigde E4-schalen bij lagere modi.')
        if record.get("batchnorm_counters_correct") is False:
            run.issue("FOUT", f"Epoch {epoch}: BatchNorm-tellercontrole mislukt.")
        if record.get("test_optimizer_steps", 0) != 0:
            run.issue("FOUT", f"Epoch {epoch}: evaluatie heeft optimizerupdates uitgevoerd.")
        adam = record.get("adam_checks", {})
        if adam and any(adam.get(key) is not True for key in ("all_active_adam_steps_correct", "moments_fp32_finite", "unused_bn_has_no_optimizer_state")):
            run.issue("FOUT", f"Epoch {epoch}: Adam-/FP32-controle niet bevestigd.")
    steps, passes = _number(run.summary.get("optimizer_steps")), _number(run.summary.get("training_forward_passes"))
    if steps is not None and passes is not None and passes != 3 * steps:
        run.issue("FOUT", "Totaal aantal trainpasses is niet driemaal het aantal optimizerupdates.")
    if steps is not None and expected is not None and run.last_epoch is not None and steps != expected * run.last_epoch:
        run.issue("FOUT", "Aantal optimizerupdates strookt niet met de volledige trainset en afgeronde epochs.")
    if run.summary:
        expected_summary = {'schema': 'mx-triple-learned-scale32-64-reference100-completion-v1',
                            'shared_scale_compensation': True, 'learned_shared_scale': True,
                            'scale_exponent_offset_choices': [5,6], 'scale_factor_choices': [32,64],
                            'all_lower_scale_bytes_equal_current_E4_source': False,
                            'all_mantissa_scale_bytes_equal_E2M3': True, 'reference_end_epoch': 100}
        if any(run.summary.get(k) != v for k, v in expected_summary.items()):
            run.issue('FOUT', 'Samenvatting bevestigt het geleerde schaal32/64-beleid niet correct.')



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
    last = f"Laatste ingelezen testepoch: **{run.last_epoch}**." if run.last_epoch else "Nog geen bruikbare testepoch beschikbaar."
    if run.reference_complete:
        endpoint = "**Alle honderd epochs voor alle drie modi aanwezig; volledige referentie volgens het runrapport.** Epoch100 staat naast het door E4M3 gekozen beste checkpoint."
    else:
        endpoint = "**Epoch100 niet gemeten als bevestigd eindresultaat.** De laatste beschikbare epoch en het door E4M3 gekozen beste checkpoint blijven zichtbaar. Er wordt geen ontbrekende accuracy voorspeld."
    if run.synthetic:
        endpoint += " **Alle getoonde fixturecijfers zijn uitsluitend synthetische softwarecontroles.**"
    if status == "STOPPED_AT_EPOCH_BOUNDARY":
        endpoint += " **Gecontroleerd gestopt op een epochgrens; alleen afgeronde epochs zijn beschikbaar.**"
    elif status not in ("PASS", "NIET BESCHIKBAAR"):
        endpoint += " **De run is niet als PASS afgesloten; lees de meldingen.**"
    return (f"### {heading}\n\nRunstatus: **{status}** · Analysefouten: **{errors}**. {last}\n\n"
            f"{endpoint}\n\nDeze analyse leest uitsluitend opgeslagen JSON/CSV. Zij start geen training en laadt geen model of checkpoint.")



def protocol_table(run):
    c, backend = run.config, run.report.get('execution_settings', {})
    items = [('Gewichtsmodi', ', '.join(FORMATS[b] for b in DISPLAY_BITS)),
             ('Gewichtsroute', 'Actuele E4M3-bron → exponent-MSB-truncatie naar E2M3 → mantissatruncatie naar E2M1.'),
             ('Geleerde correctie', 'Eén FP32 θ per MX-blok; θ∈[5,6], begin6; θ<5,5 kiest+5/×32, anders+6/×64.'),
             ('Gedeelde lagere schaal', 'E2M3 en E2M1 delen dezelfde blokkeuze. De E4M3-bronschaal zelf wordt niet geleerd.'),
             ('Mastergradient', 'Identity-STE naar G(W), zonder extra factor32/64; bestaande G(W)-afgeleide behouden.'),
             ('Schaalgradient', 'ln(2) × som(gradient_q × werkelijk gebruikt q) per blok, opgeteld over E2M3 en E2M1.'),
             ('Gradientnormalisatie', 'Geen extra LSQ-normalisatie; geen gemiddelde in plaats van de blocksom.'),
             ('Schaalgrenzen', 'θ na Adam projecteren op [5,6]; E8M0-code in breder integer-type; geen codeclipping of wrap.'),
             ('Trainbare parameters', f"{c.get('base_parameters', '?')} basis + {c.get('learned_scale_parameters', '?')} schaalproxies = {c.get('parameters', '?')}"),
             ('Activaties / masters / proxies', 'A32 / FP32 / FP32'),
             ('Optimizer', 'Eén Adam-update na drie passes; afzonderlijke parametergroep voor schaalproxies.'),
             ('Distillatie', c.get('loss', 'E4M3 hard CE; E2M3 detached E4M3; E2M1 detached E2M3.')),
             ('BN-banken', str(c.get('bn_bank_mapping', 'niet gerapporteerd'))),
             ('Backend', f"cuDNN benchmark={backend.get('cudnn_benchmark', '?')}; deterministisch={backend.get('deterministic_algorithms', '?')}"),
             ('Data', f"{c.get('dataset', '?')}; train={c.get('train_examples', '?')}; test={c.get('test_examples', '?')}; extra={c.get('extra_used', '?')}"),
             ('Groep / schaalformaat', f"{c.get('mx_block_size', '?')} / {c.get('mx_scale_format', '?')}"),
             ('Batch / seed', f"{c.get('batch_size', '?')} / {c.get('seed', '?')}"),
             ('Eindepoch / gepland LR-schema', f"{c.get('reference_end_epoch', '?')} / {c.get('planned_epochs', '?')} epochs"),
             ('Beide Adam-groepen', 'lr 0,001 → 0,0001 → 0,00001 → 0,000001; daling na epochs 50/75/90.'),
             ('Checkpointselectie', run.selection.get('policy', 'selectiebestand ontbreekt'))]
    return pd.DataFrame(items, columns=['Onderdeel', 'Opgeslagen configuratie'])




def checkpoint_table(run):
    rows = []
    roles = (("last_epoch", "Epoch 100" if run.reference_complete else "Laatste afgeronde epoch (training onvoltooid)", "last", run.last_epoch),
             ("best_E4M3_test", "Beste checkpoint gekozen door E4M3 binnen de gemeten training", "best", run.observed_best_epoch))
    for role, label, selection_key, fallback in roles:
        detail = run.selection.get(selection_key, {}) if selection_key else {}
        epoch = detail.get("epoch", fallback)
        if role == "last_epoch" and not run.selection_safe:
            epoch = fallback
        usable = run.reference_complete if role == "epoch100" else (role == "last_epoch" or run.selection_safe)
        for bit in DISPLAY_BITS:
            metric = run.metric(epoch, FORMATS[bit]) if usable else None
            source = "checkpoint_selection.json" if detail else "afgeleid uit metrics; selectie niet bevestigd"
            if role == "epoch100":
                source = "alle drie modi; volledige referentie bevestigd" if run.reference_complete else "epoch100 niet gemeten als bevestigd eindresultaat"
            elif not run.selection_safe and role == "best_E4M3_test":
                source = "inconsistente checkpointselectie; scores geblokkeerd"
            elif not run.selection_safe and role == "last_epoch":
                source = "laatste volledige epoch uit metrics; checkpointselectie inconsistent"
            rows.append({"Rol": role, "Controlepunt": label, "Epoch": epoch, "Gewichtsformaat": FORMATS[bit],
                         "Testaccuracy [%]": float(metric["top1_percent"]) if metric is not None else np.nan,
                         "Correct / n": _counts(metric),
                         "Harde test-CE": float(metric["objective_loss"]) if metric is not None else np.nan,
                         "Selectiebron": source, "Checkpoint": detail.get("file", "niet gerapporteerd")})
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
    records = [x for x in run.history.get('epochs', []) if isinstance(x, dict)]
    items = [('Gradientcontrole vóór iedere gezamenlijke update', run.summary.get('gradient_checks_before_every_update')),
             ('FP32-masters en Adam-momenten gecontroleerd', run.summary.get('fp32_masters_and_moments_verified')),
             ('Geleerde schaalcorrectie ingeschakeld', run.summary.get('learned_shared_scale')),
             ('Schaalkeuzes en θ-staten onafhankelijk consistent', run.scales_safe if records else None),
             ('E2M1 behoudt de E2M3-schaal', run.summary.get('all_mantissa_scale_bytes_equal_E2M3')),
             ('Eén E4-bron per laag en minibatch', run.summary.get('one_E4_source_per_layer_per_minibatch_verified')),
             ('Broncache leeg vóór iedere optimizerupdate', run.summary.get('source_cache_cleared_before_every_optimizer_update')),
             ('BN-tellers correct in elke opgeslagen epoch', all(x.get('batchnorm_counters_correct') is True for x in records) if records else None),
             ('Evaluatie zonder optimizerupdates', all(x.get('test_optimizer_steps') == 0 for x in records) if records else None),
             ('Bronbestanden ongewijzigd', run.report.get('source_unchanged')),
             ('Datasetbestanden ongewijzigd', run.report.get('dataset_files_unchanged')),
             ('Optimizerupdates', run.summary.get('optimizer_steps')),
             ('Trainforwardpasses', run.summary.get('training_forward_passes')),
             ('E4M3 bepaalt gezamenlijk checkpoint', run.selection.get('all_modes_from_same_selected_checkpoint')),
             ('Selectie op officiële testset', run.selection.get('selected_on_official_test')),
             ('Onafhankelijke testschatting na selectie', run.selection.get('independent_test_estimate'))]
    return pd.DataFrame([(k, 'niet gerapporteerd' if v is None else v) for k, v in items],
                        columns=['Opgeslagen controle', 'Resultaat volgens rapport'])



def export_table(run):
    rows = []
    for role, modes in run.report.get('mx_exports', {}).items():
        for mode, detail in modes.items():
            bit = _number(detail.get('weight_bits', mode))
            rows.append({'Rol': role, 'Epoch': detail.get('epoch'), 'Gewichtsformaat': FORMATS.get(bit, str(mode)),
                         'Bestand': detail.get('file'), 'Bestandsbytes': detail.get('bytes'),
                         'Gewichten exact gereconstrueerd': detail.get('decoded_weights_exact'),
                         'Zes MX-masters afwezig': detail.get('six_fp32_masters_absent'),
                         'FP32-schaalproxies afwezig': detail.get('fp32_scale_proxies_absent'),
                         'Gepakte keuze-informatie [bytes]': detail.get('packed_scale_choices_bytes'),
                         'Eigen schaaloffsets': detail.get('scale_exponent_offset_choices'),
                         'E4-bronschaal behouden': detail.get('source_scale_bytes_copied'),
                         'Afgeleid uit opgeslagen bron': detail.get('derived_from_saved_source'),
                         'Masters gelezen voor afleiding': detail.get('master_weights_read_to_derive'),
                         'Volledige testset opnieuw uitgevoerd': detail.get('full_test_forward_rerun'),
                         'Bron-SHA256': detail.get('source_file_sha256')})
    return pd.DataFrame(rows) if rows else pd.DataFrame({'Melding': ['Geen exportcontroles gerapporteerd.']})




def weight_update_table(run):
    if run.updates.empty:
        return pd.DataFrame({"Melding": ["Geen weight_updates.csv beschikbaar."]})
    data = run.updates.copy()
    if "weight_format" not in data and "weight_bits" in data:
        data["weight_format"] = pd.to_numeric(data["weight_bits"], errors="coerce").map(FORMATS)
    keys = ("epoch", "weight_format", "layer", "master_changed_values", "master_update_l2", "master_dtype",
            "changed_element_codes_including_padding", "changed_scale_codes")
    return data[[key for key in keys if key in data]]


def prediction_table(run, selected_only=False):
    if not run.prediction_safe or run.predictions.empty:
        return pd.DataFrame({'Melding': ['Geen bevestigde volledige voorspellingstellingen; geen collapse-diagnose afgeleid.']})
    data = run.predictions.copy()
    if selected_only:
        selected = []
        for role, epoch in [('last_epoch', run.last_epoch), ('best_E4M3_test', run.observed_best_epoch if run.selection_safe else None)]:
            if epoch is not None:
                group = data.loc[data['Epoch'] == epoch].copy()
                group.insert(0, 'Rol', role)
                selected.append(group)
        data = pd.concat(selected, ignore_index=True) if selected else data.iloc[0:0]
    return data.drop(columns=['weight_bits'])


def plot_prediction_diversity(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
        fig.suptitle(_title(run, 'Voorspellingen op de testset — diversiteit en concentratie'), fontsize=14, fontweight='bold')
        if not run.prediction_safe or run.predictions.empty:
            for ax in axes:
                _empty(ax, 'Geen bevestigde volledige voorspellingstellingen')
            return fig
        styles = {8: '-', 6: '--', 4: ':'}
        markers = {8: 'o', 6: 's', 4: '^'}
        for bit in DISPLAY_BITS:
            data = run.predictions.loc[run.predictions['weight_bits'] == bit].sort_values('Epoch')
            kwargs = dict(color=COLORS[bit], label=LABELS[bit] + '/A32',
                          linestyle=styles[bit], marker=markers[bit], markersize=5)
            axes[0].plot(data['Epoch'], data['Voorspelde klassen'], **kwargs)
            axes[1].plot(data['Epoch'], data['Aandeel dominant cijfer [%]'], **kwargs)
        axes[0].set(title='Hoeveel verschillende cijfers worden voorspeld?',
                    xlabel='Afgeronde epoch', ylabel='Aantal voorspelde klassen', ylim=(.5, 10.5), yticks=range(1, 11))
        axes[1].set(title='Aandeel van het meest voorspelde cijfer',
                    xlabel='Afgeronde epoch', ylabel='Aandeel testbeelden [%]', ylim=(0, 105))
        for ax in axes:
            ax.set_xticks(epoch_ticks(run.last_epoch))
            ax.legend(frameon=False)
        fig.supxlabel('Eén voorspelde klasse en 100% concentratie: constante voorspelling. Lage accuracy blijft een geldige meetuitkomst.',
                      fontsize=9, color='#525D65')
        return fig



def _title(run, text):
    return ("SYNTHETISCHE CONTROLE · " if run.synthetic else "") + text


def _empty(ax, text):
    ax.set_axis_off()
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center", wrap=True,
            fontsize=11, color="#525D65", bbox=dict(boxstyle="round,pad=.9", facecolor="#F1F4F6", edgecolor="none"))


def plot_learning(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
        fig.suptitle(_title(run, "Truncatie met geleerde schaal ×32/×64 / A32 — accuracy per modus"), fontsize=14, fontweight="bold")
        for ax, phase, title in zip(axes, ("train", "test"), ("Trainaccuracy tijdens updates", "Testaccuracy na de epoch")):
            if run.test.empty:
                _empty(ax, "Geen bruikbare metrics voor alle drie modi")
                continue
            for bit in DISPLAY_BITS:
                data = run.metrics.loc[(run.metrics["phase"] == phase) & (run.metrics["weight_bits"] == bit)].sort_values("epoch")
                ax.plot(data["epoch"], data["top1_percent"], marker="o", markersize=4,
                        color=COLORS[bit], label=LABELS[bit] + "/A32")
            ax.set(title=title, xlabel="Afgeronde epoch", ylabel="Accuracy [%]", ylim=(0, 100))
            ax.legend(frameon=False)
            mark_lr_phases(ax, run.last_epoch)
            if run.last_epoch <= 10:
                ax.set_xticks(epoch_ticks(run.last_epoch))
        return fig


def plot_objectives(run):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(11.5, 7.3), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        test_ax = fig.add_subplot(grid[1, :])
        fig.suptitle(_title(run, "Trainobjectieven en harde test-CE"), fontsize=14, fontweight="bold")
        if run.test.empty:
            for ax in [*axes, test_ax]:
                _empty(ax, "Geen bruikbare lossmetrics")
            return fig
        titles = {8: "E4M3 · harde CE van labels", 6: "E2M3 · soft CE van E4M3", 4: "E2M1 · soft CE van E2M3"}
        for ax, bit in zip(axes, DISPLAY_BITS):
            data = run.metrics.loc[(run.metrics["phase"] == "train") & (run.metrics["weight_bits"] == bit)].sort_values("epoch")
            ax.plot(data["epoch"], data["objective_loss"], marker="o", markersize=4, color=COLORS[bit])
            ax.set(title=titles[bit], xlabel="Epoch", ylabel="Gemiddeld trainobjectief", ylim=(0, None))
        for bit in DISPLAY_BITS:
            data = run.test.loc[run.test["weight_bits"] == bit]
            test_ax.plot(data["epoch"], data["objective_loss"], marker="o", markersize=4,
                         color=COLORS[bit], label=LABELS[bit] + "/A32")
        test_ax.set(title="Vergelijkbare testloss: harde CE van labels voor alle modi",
                    xlabel="Afgeronde epoch", ylabel="Gemiddelde harde CE", ylim=(0, None))
        test_ax.legend(frameon=False)
        if run.last_epoch <= 10:
            for ax in [*axes, test_ax]:
                ax.set_xticks(epoch_ticks(run.last_epoch))
        return fig



def plot_checkpoint_accuracy(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
        fig.suptitle(_title(run, "Truncatie met geleerde schaal ×32/×64 — hetzelfde checkpoint per groep"), fontsize=14, fontweight="bold")
        table = checkpoint_table(run)
        for ax, role, title in zip(axes, ("last_epoch", "best_E4M3_test"),
                                   ("Epoch 100" if run.reference_complete else "Laatste afgeronde epoch", "Beste E4M3-checkpoint")):
            rows = table.loc[table["Rol"] == role]
            epoch = rows.iloc[0]["Epoch"] if not rows.empty else None
            if run.test.empty:
                _empty(ax, "Nog geen gemeten checkpointaccuracies")
                continue
            for x, (_, row) in enumerate(rows.iterrows()):
                score = _number(row["Testaccuracy [%]"])
                if score is None:
                    ax.text(x, 45, "niet bevestigd", rotation=90, ha="center", va="center", color="#525D65")
                    continue
                ax.bar(x, score, width=.62, color=COLORS[DISPLAY_BITS[x]])
                ax.text(x, min(score + 2.2, 102), f"{score:.4f}%", ha="center", va="bottom", fontsize=10)
            ax.set(title=f"{title} (epoch {epoch:g})" if _number(epoch) is not None else title,
                   xticks=range(3), xticklabels=[LABELS[bit] + "/A32" for bit in DISPLAY_BITS],
                   ylabel="Testaccuracy [%]", ylim=(0, 108), xlim=(-.6, 2.6))
            ax.set_yticks(range(0, 101, 20))
        endpoint = ("Alle drie modi gebruiken per groep hetzelfde checkpoint."
                    + (" Honderd epochs volledig gemeten." if run.reference_complete else " Epoch100 is nog geen bevestigd eindresultaat."))
        fig.supxlabel(endpoint, fontsize=10, color="#525D65")
        return fig



def plot_timing(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), width_ratios=[1.35, 1], constrained_layout=True)
        title = "Gemeten epochtijden en totale duur van 100 epochs" if run.reference_complete else "Gemeten epochtijden en voorlopige looptijdschatting"
        fig.suptitle(_title(run, title), fontsize=14, fontweight="bold")
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
            axes[0].set(title=f"Tijd per epoch · gemeten som {float(total.sum()):.2f} s", xlabel="Epoch", ylabel="Seconden")
            if len(data) <= 10:
                axes[0].set_xticks(data["epoch"])
            axes[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.94, fontsize=8, ncol=2)
        timing = run.timing
        estimate = _number(timing.get("estimated_100_epoch_seconds"))
        if run.reference_complete:
            total = measured_completed_seconds(run)
            if total is None:
                _empty(axes[1], "100 epochs afgerond\nGemeten totale duur ontbreekt")
            else:
                _empty(axes[1], f"100 afgeronde epochs\n\n{total / 60:.1f} minuten\n({total / 3600:.2f} uur)\n\nGemeten som van epochtijden\nTrain + test + controles + checkpoints")
            axes[1].set_title("Gemeten totaal; geen projectie")
        elif not timing.get("available") or estimate is None:
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


def measured_completed_seconds(run):
    """Use recorded epoch durations; never substitute a projected duration."""
    data = run.times
    if not data.empty and {"epoch", "total_epoch_seconds"} <= set(data):
        epochs = pd.to_numeric(data["epoch"], errors="coerce")
        values = pd.to_numeric(data["total_epoch_seconds"], errors="coerce")
        if (run.last_epoch is not None and len(data) == run.last_epoch
                and epochs.sort_values().tolist() == list(range(1, run.last_epoch + 1))
                and np.isfinite(values).all() and (values >= 0).all()):
            return float(values.sum())
    keys = ((run.timing, "measured_100_epoch_seconds"),) if run.reference_complete else ()
    keys += ((run.summary, "measured_completed_epoch_seconds"),
             (run.timing, "measured_completed_epoch_seconds"))
    for source, key in keys:
        value = _number(source.get(key))
        if value is not None and value >= 0:
            return value
    return None


def timing_markdown(run):
    total = measured_completed_seconds(run)
    prefix = "**Synthetische tijdgegevens voor softwarecontrole.** " if run.synthetic else ""
    measured = (f"**Gemeten som over {run.last_epoch} afgeronde epochs: {total:.2f} s ({total / 60:.2f} minuten).** "
                if total is not None else "Een bruikbare gemeten totale epochduur ontbreekt. ")
    if run.reference_complete:
        return (prefix + measured + "Alle honderd epochs zijn aanwezig. De gemeten duur omvat train, test, epochcontroles en checkpointschrijven; een eerdere projectie wordt hiervoor niet gebruikt. "
                "Download, installatie, data-inladen, eindrapportage en sessiepauzes zijn niet inbegrepen.")
    t = run.timing
    if not t.get("available"):
        return prefix + measured + "Er is nog geen opgeslagen looptijdschatting beschikbaar; epoch100 is niet gemeten."
    median = _number(t.get("median_epoch_seconds"))
    warm = f"{median:.1f} s" if median is not None else "niet gerapporteerd"
    return (prefix + measured + f"Warme epochs volgens het tijdsrapport: **{t.get('steady_epochs', [])}**; mediaan: **{warm}**. "
            "De afzonderlijke projectie geldt voor een **nieuwe volledige run van honderd epochs vanaf initialisatie**. Zij is geen gemeten eindtijd en geen resterende hervattijd. "
            "Het waargenomen minimum/maximum van de warme epochs levert een scenario-bereik, geen betrouwbaarheidsinterval. "
            "Bij een gedeeltelijke run zijn latere LR-fasen en langdurige thermische omstandigheden nog niet volledig gemeten. "
            "Download, installatie, data-inladen en eindrapportage zijn niet inbegrepen. **Geen accuracy voor epoch100 wordt voorspeld.**")



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
                "input_sha256": {str(p.relative_to(run.path)) if p.is_relative_to(run.path) else str(p):
                                 hashlib.sha256(p.read_bytes()).hexdigest() for p in run.input_files},
                "figures": [name + ".png" for name in figures], "issues": run.issues}
    (destination / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination



LAYER_BLOCKS = {'layers.3': 64, 'layers.6': 80, 'layers.10': 160,
                'layers.13': 288, 'layers.16': 288, 'layers.20': 3200}

TOTAL_BLOCKS = 4080

LAYER_COLORS = ['#245A81', '#33846D', '#C98220', '#8657A6', '#B44040', '#5D8790']

REFERENCE_INITIAL_SHA256 = 'a8fcd20c40a204ba24cabfa36258f9501d55b9d4d4bdfb8ba742b8c827af127f'

REFERENCE_REPORT_SHA256 = {32: 'be771afb34ad1043a07a77e0ceee50f591e3155578fc8a7a04960c60f3dda6a8', 64: 'f53511694e0762f55cf5b3fb8b85e9e0a2738b91a415b825418b10cb5398db0c'}  # Filled from the delivered pinned reference files.

COMPARABLE_CONFIG_KEYS = (
    'mode','weight_bits','weight_formats','codec_revision','weight_preprocess',
    'activation_bits','activation_policy','master_dtype','gradient_dtype','adam_moments_dtype',
    'mx_block_size','mx_scale_format','mx_group_axis','mx_rounding','numeric_kernel',
    'truncation','mantissa_truncation','exponent_truncation','shared_scale_compensation',
    'source_weight_bits','exponent_mapping','direct_lower_quantization','exponent_bias_compensation',
    'distillation','loss','passes_per_training_batch','optimizer_steps_per_training_batch',
    'training_mode_order','test_mode_order','legacy_abits_and_wbits','model_bank_structure',
    'bn_bank_mapping','batchnorm_policy','extra_FP32_teacher_forward','dataset','train_split',
    'test_split','extra_used','train_examples','test_examples','validation_examples','model','expand',
    'planned_epochs','batch_size','optimizer','lr','weight_decay','scheduler_milestones',
    'scheduler_gamma','workers','pin_memory','drop_last','train_shuffle','test_shuffle','seed',
    'evaluation_policy','selection_policy','backend_policy','rng_policy','checkpoint_policy',
)


def _validate_scales(run, records):
    """Independently recompute hard choices and epoch summaries from saved θ."""
    if run.test.empty:
        return
    def fail(message):
        run.scales_safe = False
        run.issue('FOUT', 'Schaaldiagnostiek: ' + message)
    blocks, epochs = run.scale_blocks, run.scale_epochs
    block_required = {'epoch', 'layer', 'output_channel', 'block_index', 'theta', 'offset', 'factor'}
    epoch_required = {'epoch', 'layer', 'blocks', 'theta_min', 'theta_max', 'theta_mean', 'theta_std',
        'theta_changed_blocks', 'theta_update_l2', 'count_x32', 'count_x64', 'fraction_x32',
        'net_choice_changes', 'choice_flips_during_epoch', 'unique_blocks_switched_during_epoch',
        'projected_values_during_epoch', 'scale_grad_l2_min', 'scale_grad_l2_max',
        'scale_grad_nonzero_blocks_max', 'scale_lr'}
    if blocks.empty or not block_required <= set(blocks):
        fail('scale_block_state.csv ontbreekt of mist verplichte kolommen.')
        return
    if epochs.empty or not epoch_required <= set(epochs):
        fail('scale_epoch_metrics.csv ontbreekt of mist verplichte kolommen.')
        return
    for frame, columns in [(blocks, block_required - {'layer'}), (epochs, epoch_required - {'layer'})]:
        for key in columns:
            frame[key] = pd.to_numeric(frame[key], errors='coerce')
        if not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
            fail('niet-eindige of ontbrekende numerieke θ-, keuze- of gradientwaarden.')
            return
    for frame in [blocks, epochs]:
        if set(frame.layer) != set(LAYER_BLOCKS):
            fail('de zes verwachte MX-lagen ontbreken of verschillen.')
            return
    bkeys = ['epoch', 'layer', 'output_channel', 'block_index']
    if blocks.duplicated(bkeys).any() or epochs.duplicated(['epoch', 'layer']).any():
        fail('dubbele blokstaten of laag/epochstatistieken.')
        return
    for field in ['epoch', 'output_channel', 'block_index', 'offset', 'factor']:
        if not ((blocks[field] >= 0) & (blocks[field] % 1 == 0)).all():
            fail('ongeldige integerkolom ' + field)
    if sorted(blocks.epoch.unique()) != list(range(run.last_epoch + 1)):
        fail('blokstaat moet epoch0 en iedere volledig afgeronde epoch bevatten.')
    if sorted(epochs.epoch.unique()) != list(range(1, run.last_epoch + 1)):
        fail('laagstatistieken bevatten niet iedere afgeronde epoch.')
    if not blocks.theta.between(5, 6).all():
        fail('FP32-proxy buiten [5,6].')
    desired = np.where(blocks.theta < 5.5, 5, 6)
    if not np.array_equal(blocks.offset.to_numpy(), desired):
        fail('harde offset volgt niet θ<5,5→5, anders6 (tie naar6).')
    if not np.array_equal(blocks.factor.to_numpy(), np.where(desired == 5, 32, 64)):
        fail('factor komt niet overeen met de harde offset.')
    if not blocks.loc[blocks.epoch == 0, 'theta'].eq(6).all():
        if (run.synthetic and run.report.get('synthetic_fixture_only') is True
                and run.config.get('synthetic_initial_theta_mixed') is True):
            run.issue('SYNTHETISCH', 'Deze expliciete softwarefixture start met gemengde θ-waarden. '
                      'Dit controleert wissels en exports; het is niet de onderzoeksinitialisatie θ=6.')
        else:
            fail('beginstaat van de schaalproxies is niet exact6.')
    if not run.scales_safe:
        return
    block_index = {}
    grouped_blocks = {(int(e), layer): group.sort_values(['output_channel', 'block_index'])
                      for (e, layer), group in blocks.groupby(['epoch', 'layer'], sort=False)}
    for epoch in range(run.last_epoch + 1):
        for layer, count in LAYER_BLOCKS.items():
            state = grouped_blocks.get((epoch, layer), blocks.iloc[:0])
            if len(state) != count:
                fail(f'{layer}/epoch{epoch}: verwacht {count} blokken, gevonden {len(state)}.')
                continue
            channels = sorted(state.output_channel.astype(int).unique())
            per_channel = state.groupby('output_channel').block_index.apply(lambda s: sorted(s.astype(int)))
            if (channels != list(range(len(channels))) or per_channel.apply(len).nunique() != 1
                    or any(v != list(range(len(v))) for v in per_channel)):
                fail(f'{layer}/epoch{epoch}: bloklayout is niet aaneengesloten per uitvoerchannel.')
                continue
            block_index[(epoch, layer)] = state
    if not run.scales_safe:
        return
    steps = math.ceil(run.config['train_examples'] / run.config['batch_size'])
    history_states = {0: run.history.get('initial_scale_state')}
    history_states.update({int(r['epoch']): r.get('scale_state') for r in records})
    for epoch in range(run.last_epoch + 1):
        saved = history_states[epoch]
        if not isinstance(saved, dict) or set(saved) != set(LAYER_BLOCKS):
            fail(f'epoch{epoch}: θ-blokstaat ontbreekt in history.json.')
            continue
        for layer in LAYER_BLOCKS:
            state = block_index[(epoch, layer)]
            array = np.asarray(saved[layer], dtype=float)
            channels = state.output_channel.nunique()
            if array.shape != (channels, len(state) // channels) or not np.array_equal(array.reshape(-1), state.theta.to_numpy()):
                fail(f'{layer}/epoch{epoch}: CSV-proxies verschillen van history.json.')
    for epoch in range(1, run.last_epoch + 1):
        record = next((r for r in records if r.get('epoch') == epoch), {})
        logged = record.get('scale_epoch_metrics')
        if not isinstance(logged, list) or len(logged) != 6:
            fail(f'epoch{epoch}: zes schaalstatistieken ontbreken in history.json.')
            logged = []
        for layer, count in LAYER_BLOCKS.items():
            matches = epochs.loc[(epochs.epoch == epoch) & (epochs.layer == layer)]
            if len(matches) != 1:
                fail(f'{layer}/epoch{epoch}: geen unieke statistiekrij.')
                continue
            row = matches.iloc[0]
            before, after = block_index[(epoch - 1, layer)], block_index[(epoch, layer)]
            t0, t1 = before.theta.to_numpy(), after.theta.to_numpy()
            n32 = int((after.offset == 5).sum())
            expected = {'blocks': count, 'theta_min': float(t1.min()), 'theta_max': float(t1.max()),
                        'theta_mean': float(t1.mean()), 'theta_std': float(t1.std(ddof=0)),
                        'theta_changed_blocks': int((t1 != t0).sum()),
                        'theta_update_l2': float(np.linalg.norm(t1 - t0)),
                        'count_x32': n32, 'count_x64': count - n32, 'fraction_x32': n32 / count,
                        'net_choice_changes': int((after.offset.to_numpy() != before.offset.to_numpy()).sum())}
            for key, value in expected.items():
                tol = 2e-6 if key in ('theta_mean', 'theta_std', 'theta_update_l2') else 1e-8
                if not _same_number(row[key], value, tolerance=tol):
                    fail(f'{layer}/epoch{epoch}: {key} strookt niet met de opgeslagen blokstaten.')
            counters = ['theta_changed_blocks', 'count_x32', 'count_x64', 'net_choice_changes',
                        'choice_flips_during_epoch', 'unique_blocks_switched_during_epoch',
                        'projected_values_during_epoch', 'scale_grad_nonzero_blocks_max']
            if any(row[k] < 0 or row[k] % 1 != 0 for k in counters):
                fail(f'{layer}/epoch{epoch}: ongeldige schaaltellingen.')
            flips, unique = row.choice_flips_during_epoch, row.unique_blocks_switched_during_epoch
            if not (row.net_choice_changes <= unique <= count and unique <= flips <= count * steps):
                fail(f'{layer}/epoch{epoch}: netto wijzigingen, unieke wissels en totaal flips zijn inconsistent.')
            if row.projected_values_during_epoch > count * steps or row.scale_grad_nonzero_blocks_max > count:
                fail(f'{layer}/epoch{epoch}: grensprojecties of gradientblokken overschrijden de mogelijke telling.')
            if not (0 <= row.scale_grad_l2_min <= row.scale_grad_l2_max):
                fail(f'{layer}/epoch{epoch}: ongeldige gradientnormgrenzen.')
            scale_lr = expected_lr(run.config, epoch, scale=True)
            if not math.isclose(float(row.scale_lr), scale_lr, rel_tol=1e-9, abs_tol=1e-14):
                fail(f'{layer}/epoch{epoch}: schaal-leersnelheid verschilt van de LR-fase.')
            old = next((x for x in logged if x.get('layer') == layer), None)
            if old is None or any(not _same_number(row[k], old.get(k), tolerance=1e-8) for k in epoch_required - {'layer'}):
                fail(f'{layer}/epoch{epoch}: schaalstatistiek-CSV verschilt van history.json.')
    if run.scales_safe:
        final = epochs.loc[epochs.epoch == run.last_epoch]
        if final.count_x32.sum() == 0:
            run.issue('UITKOMST', 'Alle blokken gebruiken op de laatste epoch nog ×64; dit is een meetuitkomst, geen technische FAIL. Bekijk ook θ-verplaatsingen en gradients.')
        if epochs.scale_grad_l2_max.eq(0).all():
            run.issue('UITKOMST', 'Alle gerapporteerde schaalgradients zijn nul; eindig en technisch geldig, maar er is geen schaalleersignaal waargenomen.')


def scale_learning_markdown(run):
    if not run.scales_safe or run.scale_epochs.empty:
        return '**Geen bevestigde schaaldiagnostiek beschikbaar.** Bekijk de meldingen.'
    last = run.scale_epochs.loc[run.scale_epochs.epoch == run.last_epoch]
    n32, n64 = int(last.count_x32.sum()), int(last.count_x64.sum())
    flips = int(run.scale_epochs.choice_flips_during_epoch.sum())
    changed = int(last.theta_changed_blocks.sum())
    return (f'Op de laatste afgeronde epoch gebruiken **{n32:,} blokken ×32** en **{n64:,} blokken ×64** '
            f'({100*n32/TOTAL_BLOCKS:.2f}% ×32; {TOTAL_BLOCKS:,} blokken totaal). '
            f'In de laatste epoch veranderden {changed:,} FP32-hulpwaarden. '
            f'Over de opgeslagen training werden {flips:,} harde omschakelingen geteld.\n\n'
            'Een blok kan meerdere keren wisselen: totaal flips, unieke wisselende blokken binnen een epoch '
            'en netto verschil tussen twee epochgrenzen zijn verschillende metingen. '
            'Geen omschakeling betekent niet automatisch dat de FP32-hulpwaarden of gradients nul zijn. '
            'Dit zijn trainingsdiagnostieken; zij bewijzen geen accuracyverbetering of optimale keuze.')


def scale_table(run):
    if not run.scales_safe or run.scale_epochs.empty:
        return pd.DataFrame({'Melding': ['Schaalstatistieken ontbreken of zijn inconsistent.']})
    return run.scale_epochs.copy()


def plot_scale_choices(run):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        ax.set_title(_title(run, 'Geleerde harde schaalkeuzes per MX-laag'), fontsize=14)
        if not run.scales_safe or run.scale_epochs.empty:
            _empty(ax, 'Geen bevestigde schaalkeuzes beschikbaar')
            return fig
        for (layer, _), color in zip(LAYER_BLOCKS.items(), LAYER_COLORS):
            rows = run.scale_epochs.loc[run.scale_epochs.layer == layer].sort_values('epoch')
            initial = run.scale_blocks.loc[(run.scale_blocks.epoch == 0) & (run.scale_blocks.layer == layer)]
            initial_fraction = 100 * float(initial.factor.eq(32).mean())
            ax.plot([0] + rows.epoch.tolist(), [initial_fraction] + (100*rows.fraction_x32).tolist(), marker='o', color=color, label=layer)
        ax.set(xlabel='Afgeronde epoch (0 = initialisatie)', ylabel='Blokken met ×32 [%]', ylim=(-2, 102),
               xticks=epoch_ticks(run.last_epoch, zero=True))
        ax.legend(frameon=False, ncol=3, loc='upper center')
        return fig


def plot_theta_distributions(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        fig.suptitle(_title(run, 'FP32-hulpwaarden θ per MX-laag'), fontsize=14, fontweight='bold')
        if not run.scales_safe or run.scale_blocks.empty:
            for ax in axes.flat: _empty(ax, 'Geen bevestigde θ-staat')
            return fig
        chosen = [(0, 'Initialisatie', '#99A3AD'), (run.observed_best_epoch, 'E4-best', BLUE)]
        if run.last_epoch != run.observed_best_epoch:
            chosen.append((run.last_epoch, 'Laatste', AMBER))
        for ax, (layer, count) in zip(axes.flat, LAYER_BLOCKS.items()):
            for epoch, label, color in chosen:
                values = run.scale_blocks.loc[(run.scale_blocks.epoch == epoch) & (run.scale_blocks.layer == layer), 'theta'].to_numpy()
                ax.hist(values, bins=np.linspace(5, 6, 26), weights=np.full(len(values), 100 / count),
                        histtype='step', lw=1.8, color=color, label=f'{label}: epoch {epoch}')
            ax.axvline(5.5, color=RED, ls=':', lw=1.2)
            ax.set(title=f'{layer} · {count} blokken', xlabel='θ (forward: <5,5 → ×32; anders ×64)',
                   ylabel='Blokken per bin [%]', xlim=(4.98, 6.02), ylim=(0, 105))
            ax.legend(frameon=False, fontsize=8)
        return fig


def plot_scale_switches(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Harde omschakelingen en FP32-verplaatsing'), fontsize=14, fontweight='bold')
        if not run.scales_safe or run.scale_epochs.empty:
            for ax in axes: _empty(ax, 'Geen bevestigde schaaldiagnostiek')
            return fig
        totals = run.scale_epochs.groupby('epoch', sort=True)[['net_choice_changes', 'choice_flips_during_epoch', 'unique_blocks_switched_during_epoch']].sum()
        for key, label, color in [('choice_flips_during_epoch','Alle omschakelingen',BLUE),
                                  ('unique_blocks_switched_during_epoch','Unieke blokken met een wissel',GREEN),
                                  ('net_choice_changes','Netto verschil tussen epochgrenzen',AMBER)]:
            axes[0].plot(totals.index, totals[key], marker='o', label=label, color=color)
        axes[0].set(xlabel='Epoch', ylabel='Aantal', xticks=epoch_ticks(run.last_epoch), ylim=(0, None))
        axes[0].legend(frameon=False, fontsize=8)
        for layer, color in zip(LAYER_BLOCKS, LAYER_COLORS):
            r = run.scale_epochs.loc[run.scale_epochs.layer == layer].sort_values('epoch')
            axes[1].plot(r.epoch, r.theta_update_l2, marker='o', label=layer, color=color)
        axes[1].set(xlabel='Epoch', ylabel='L2-norm netto θ-update per laag', xticks=epoch_ticks(run.last_epoch), ylim=(0,None))
        axes[1].legend(frameon=False, ncol=2, fontsize=8)
        for ax in axes: mark_lr_phases(ax, run.last_epoch)
        return fig


def plot_scale_gradients(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Schaalgradients en grensprojecties'), fontsize=14, fontweight='bold')
        if not run.scales_safe or run.scale_epochs.empty:
            for ax in axes: _empty(ax, 'Geen bevestigde gradientdiagnostiek')
            return fig
        for layer, color in zip(LAYER_BLOCKS, LAYER_COLORS):
            r = run.scale_epochs.loc[run.scale_epochs.layer == layer].sort_values('epoch')
            axes[0].plot(r.epoch, r.scale_grad_l2_max, marker='o', label=layer, color=color)
            axes[1].plot(r.epoch, r.projected_values_during_epoch, marker='o', label=layer, color=color)
        axes[0].set(xlabel='Epoch', ylabel='Max. L2-norm schaalgradient per update', yscale='symlog', xticks=epoch_ticks(run.last_epoch))
        axes[0].set_yscale('symlog', linthresh=1e-8)
        axes[1].set(xlabel='Epoch', ylabel='Aantal θ-waarden geprojecteerd na Adam', xticks=epoch_ticks(run.last_epoch), ylim=(0,None))
        for ax in axes:
            ax.legend(frameon=False, ncol=2, fontsize=8)
            mark_lr_phases(ax, run.last_epoch)
        return fig


@dataclass
class FixedReference:
    path: Path
    factor: int
    report: dict = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list[str] = field(default_factory=list)
    safe: bool = False
    compatible: bool = False

    @property
    def test(self):
        if not self.safe or self.metrics.empty: return pd.DataFrame()
        return self.metrics.loc[self.metrics.phase == 'test'].sort_values(['epoch','weight_bits'])

    def best_epoch(self, through_epoch=100):
        if self.test.empty: return None
        high = self.test.loc[(self.test.weight_bits == 8) & (self.test.epoch <= through_epoch)].sort_values('epoch')
        return int(high.loc[high.top1_percent.idxmax(),'epoch']) if not high.empty else None

    def metric(self, epoch, bit):
        if self.test.empty or epoch is None: return None
        r = self.test.loc[(self.test.epoch == epoch) & (self.test.weight_bits == bit)]
        return r.iloc[0] if len(r) == 1 else None


def load_fixed_references(run, reference_root=None):
    root = Path(reference_root).resolve() if reference_root is not None else run.path / 'fixed_references'
    references = {}
    expected_files = {'run_report.json','epoch_metrics.csv','epoch_times.csv','checkpoint_selection.json',
                      'selected_metrics.csv','training_summary.json'}
    for factor in (32,64):
        path = root / f'scale{factor}'
        ref = FixedReference(path,factor)
        references[factor] = ref
        try:
            mp = path/'REFERENCE.json'
            manifest = json.loads(mp.read_text(encoding='utf-8'))
            if (manifest.get('schema') != 'mx-learned-scale32-64-fixed-reference-v1'
                    or manifest.get('factor') != factor
                    or manifest.get('completed_epochs') != 100 or manifest.get('synthetic_fixture_only') is not False
                    or manifest.get('initial_model_sha256') != REFERENCE_INITIAL_SHA256
                    or set(manifest.get('files_sha256',{})) != expected_files
                    or manifest['files_sha256']['run_report.json'] != REFERENCE_REPORT_SHA256[factor]):
                raise ValueError(f'Geen geldige gepinde ×{factor}-referentie.')
            inputs = [mp]
            for name,digest in manifest['files_sha256'].items():
                p = path/name
                if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                    raise ValueError('Referentiebestand wijkt af van gepinde hash: '+name)
                inputs.append(p)
            report = json.loads((path/'run_report.json').read_text())
            c = report.get('config',{})
            if (report.get('schema') != f'mx-triple-scale{factor}-reference100-report-v1'
                    or report.get('status') != 'PASS' or report.get('synthetic_fixture_only') is not False
                    or report.get('full_training_completed') is not True or report.get('completed_epoch') != 100
                    or report.get('initial_model_sha256') != REFERENCE_INITIAL_SHA256
                    or c.get('scale_factor') != factor or c.get('scale_exponent_offset') != {32:5,64:6}[factor]
                    or c.get('learned_shared_scale') is not False):
                raise ValueError('Referentie bevestigt geen echte volledige vaste-schaaltraining.')
            full = pd.read_csv(path/'epoch_metrics.csv',float_precision='round_trip')
            # Validate all measured reference epochs. Every partial-run plot and
            # best selection is then filtered to the common observed endpoint.
            selected = full.loc[full.epoch.between(1,100)].copy()
            probe = RunAnalysis(path,report=report,metrics=selected)
            _validate_metrics(probe,[])
            if not probe.metrics_safe or len(probe.metrics)!=600 or len(probe.test)!=300:
                raise ValueError('Referentie mist geldige volledige honderd epochs.')
            ref.report,ref.metrics,ref.safe = report,probe.metrics,True
            run.input_files.extend(p for p in inputs if p not in run.input_files)
            if run.synthetic:
                ref.issues.append('Huidige run is synthetisch; geen onderzoeksvergelijking met echte referenties.')
            elif not run.provenance_confirmed or run.test.empty:
                ref.issues.append('Nog geen bevestigde trainingsmetrics om te vergelijken.')
            else:
                different = [k for k in COMPARABLE_CONFIG_KEYS if k not in run.config or k not in c or run.config[k]!=c[k]]
                if different: ref.issues.append('Overige trainingsconfiguratie verschilt: '+', '.join(different))
                if run.config.get('base_parameters') != c.get('parameters'):
                    ref.issues.append('Aantal oorspronkelijke modelparameters verschilt.')
                if run.report.get('initial_base_model_sha256') != REFERENCE_INITIAL_SHA256:
                    ref.issues.append('Dezelfde oorspronkelijke modelinitialisatie zonder θ niet bevestigd.')
                for field in ('environment','execution_settings','dataset_selection','transforms'):
                    if run.report.get(field) != report.get(field):
                        ref.issues.append('Verschillende '+field+'; vergelijking niet automatisch bevestigd.')
                if run.report.get('source',{}).get('workcopy_files_sha256') != report.get('source',{}).get('workcopy_files_sha256'):
                    ref.issues.append('Upstream-bronhashes verschillen.')
                if run.report.get('dataset_files',{}).get('files') != report.get('dataset_files',{}).get('files'):
                    ref.issues.append('Ruwe datasetbestanden verschillen.')
                if any(x['Niveau'] in ('FOUT','RUNFOUT') for x in run.issues):
                    ref.issues.append('Huidige run bevat analysefouten; vergelijking geblokkeerd.')
                ref.compatible = not ref.issues
        except (OSError,ValueError,KeyError,TypeError,AttributeError,pd.errors.ParserError) as exc:
            ref.issues.append(str(exc))
    return references


def comparison_markdown(run, references):
    text = ('### Geleerde blokkeuze tegenover vaste ×32 en vaste ×64\n\n'
            'De vaste referenties zijn de bestaande volledige honderd-epochtrainingen. '
            'Bij een onderbroken geleerde run beperken we iedere referentiecurve en iedere E4M3-selectie '
            'tot dezelfde reeds gemeten epochrange. E4M3 kiest per training één gezamenlijk checkpoint; '
            'E2M3 en E2M1 gebruiken de scores uit datzelfde checkpoint. ')
    through = min(100, run.last_epoch) if run.last_epoch else 0
    for factor, ref in references.items():
        if ref.safe and ref.compatible and through:
            text += f'\n\n**×{factor}: vergelijking bevestigd voor epochs 1–{through}.**'
        else:
            text += f'\n\n**×{factor}: vergelijking nog niet beschikbaar of niet bevestigd.**'
        if ref.issues:
            text += '\n\n' + '\n'.join('- ' + v for v in ref.issues)
    text += ('\n\nGelijke oorspronkelijke modelstaat en instellingen geven geen bitexacte CUDA-garantie. '
             'Eén run per methode geeft geen significantieclaim. De geleerde variant heeft extra schaalparameters.')
    return text


def comparison_table(run, references):
    rows = []
    if run.test.empty:
        return pd.DataFrame()
    through = min(100, run.last_epoch)
    endpoint_label = 'Epoch 100' if run.reference_complete else f'Laatste gezamenlijke epoch ({through}; onvoltooid)'
    for factor, ref in references.items():
        if not ref.safe or not ref.compatible:
            continue
        for label, e, er in [(endpoint_label, through, through),
                            (f'Beste E4M3 binnen epochs 1–{through}', run.observed_best_epoch, ref.best_epoch(through))]:
            for bit in DISPLAY_BITS:
                cur, base = run.metric(e, FORMATS[bit]), ref.metric(er, bit)
                current = float(cur.top1_percent) if cur is not None else np.nan
                reference = float(base.top1_percent) if base is not None else np.nan
                rows.append({'Controlepunt': label, 'Referentie': f'Vast ×{factor}', 'Gewichtsformaat': FORMATS[bit],
                             'Epoch geleerd': e, 'Epoch referentie': er, 'Geleerd accuracy [%]': current,
                             'Referentie accuracy [%]': reference, 'Verschil geleerd − vast [pp]': current-reference,
                             'Geleerd harde CE': float(cur.objective_loss) if cur is not None else np.nan,
                             'Referentie harde CE': float(base.objective_loss) if base is not None else np.nan})
    return pd.DataFrame(rows)


def plot_scale_comparison_curves(run, references):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Geleerde schaalkeuze en vaste referenties — testaccuracy'), fontsize=14, fontweight='bold')
        usable = [r for r in references.values() if r.safe and r.compatible]
        if run.test.empty or not usable:
            for ax in axes:
                _empty(ax, 'Geen bevestigde vergelijking beschikbaar')
            return fig
        through = min(100, run.last_epoch)
        for ax, bit in zip(axes, DISPLAY_BITS):
            cur = run.test.loc[(run.test.weight_bits == bit) & (run.test.epoch <= through)]
            ax.plot(cur.epoch, cur.top1_percent, color=BLUE, lw=1.8, marker='o', markersize=2, label='Geleerd per blok')
            for ref in usable:
                data = ref.test.loc[(ref.test.weight_bits == bit) & (ref.test.epoch <= through)]
                ax.plot(data.epoch, data.top1_percent, color=GREEN if ref.factor == 32 else AMBER,
                        lw=1.5, marker='s', markersize=2, ls='--', label=f'Vast ×{ref.factor}')
            ax.set(title=LABELS[bit] + '/A32', xlabel='Afgeronde epoch', ylabel='Testaccuracy [%]',
                   xticks=epoch_ticks(through), ylim=(0, 100))
            mark_lr_phases(ax, through)
            ax.legend(frameon=False, fontsize=8)
        fig.supxlabel('Dezelfde gemeten epochrange; afzonderlijke runs. Stippellijnen: LR-dalingen na epochs 50/75/90.', fontsize=9, color='#525D65')
        return fig


def plot_scale_comparison_checkpoints(run, references):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
        fig.suptitle(_title(run, 'Geleerd en vaste referenties — twee gezamenlijke checkpointkeuzes'), fontsize=14, fontweight='bold')
        usable = [r for r in references.values() if r.safe and r.compatible]
        if run.test.empty or not usable:
            for ax in axes:
                _empty(ax, 'Geen bevestigde vergelijking beschikbaar')
            return fig
        through = min(100, run.last_epoch)
        for ax, is_best in zip(axes, [False, True]):
            label = (f'Beste E4M3 binnen epochs 1–{through}' if is_best else
                     'Epoch 100' if run.reference_complete else f'Epoch {through} · training onvoltooid')
            current_epoch = run.observed_best_epoch if is_best else through
            series = [('Geleerd', current_epoch, [run.metric(current_epoch, FORMATS[b]) for b in DISPLAY_BITS], BLUE)]
            for ref in usable:
                ep = ref.best_epoch(through) if is_best else through
                series.append((f'Vast ×{ref.factor}', ep, [ref.metric(ep, b) for b in DISPLAY_BITS], GREEN if ref.factor == 32 else AMBER))
            width = .75 / len(series)
            for k, (name, ep, metrics, color) in enumerate(series):
                x = np.arange(3) + (k - (len(series)-1)/2) * width
                vals = [float(v.top1_percent) for v in metrics]
                ax.bar(x, vals, width=.9*width, color=color, label=f'{name}: ep{ep}')
                for pos, val in zip(x, vals):
                    ax.text(pos, val+1, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            ax.set(title=label, xticks=range(3), xticklabels=[LABELS[b] for b in DISPLAY_BITS], ylabel='Testaccuracy [%]', ylim=(0,116))
            ax.set_yticks(range(0,101,20))
            ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=3)
        fig.supxlabel('Per training komen alle drie modi uit hetzelfde door E4M3 geselecteerde checkpoint.', fontsize=10, color='#525D65')
        return fig
