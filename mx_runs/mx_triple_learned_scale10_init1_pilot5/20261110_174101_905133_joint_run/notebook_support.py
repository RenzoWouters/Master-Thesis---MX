"""Read-only analysis of saved ten-choice per-block scale MX/A32 pilot results.

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
OFFSET_CHOICES = tuple(range(10))
SCALE_FACTORS = tuple(2 ** x for x in OFFSET_CHOICES)
FACTOR_COLORS = ['#173F5F', '#20639B', '#258EA6', '#38A18C', '#75AF62',
                 '#B5B948', '#DDA43A', '#DC7939', '#C75452', '#8A3D6F']
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
        # This notebook belongs to a runner bounded at epoch5. Never turn an
        # incorrectly edited summary into a claimed epoch100 measurement.
        return False

    @property
    def pilot_complete(self):
        return (self.report.get("status") == "PASS" and self.summary.get("pilot_completed") is True
                and self.last_epoch == 5 and self.metrics_safe and self.selection_safe and self.scales_safe
                and not any(x['Niveau'] in ('FOUT', 'RUNFOUT') for x in self.issues)
                and all(self.test.loc[self.test["weight_format"] == fmt, "epoch"].astype(int).tolist()
                        == list(range(1, 6)) for fmt in FORMATS.values()))

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


def _validate_contract(run):
    if not run.report:
        return
    failures = []
    if run.report.get('schema') not in ('mx-triple-learned-scale10-pilot5-report-v1', 'mx-triple-learned-scale10-init1-pilot5-report-v1'):
        failures.append('Dit is niet het rapport van de geleerde tienkeuze-schaalpilot.')
    c = run.config
    expected = {'schema': ('mx-triple-learned-scale10-pilot5-protocol-v1' if run.report.get('schema') == 'mx-triple-learned-scale10-pilot5-report-v1' else 'mx-triple-learned-scale10-init1-pilot5-protocol-v1'),
                'activation_bits': 32, 'master_dtype': 'float32', 'mode': 'joint',
                'truncation': True, 'exponent_truncation': True, 'mantissa_truncation': True,
                'source_weight_bits': 8, 'shared_scale_compensation': True,
                'scale_exponent_offset_choices': list(OFFSET_CHOICES), 'scale_factor_choices': list(SCALE_FACTORS),
                'direct_lower_quantization': False, 'exponent_bias_compensation': False,
                'learned_shared_scale': True, 'distillation': True,
                'extra_FP32_teacher_forward': False,
                'planned_epochs': (100 if run.report.get('schema') == 'mx-triple-learned-scale10-pilot5-report-v1' else 5), 'base_parameters': 132954,
                'learned_scale_parameters': 4080, 'parameters': 137034,
                'scale_proxy_dtype': 'float32', 'scale_proxy_init': (6.0 if run.report.get('schema') == 'mx-triple-learned-scale10-pilot5-report-v1' else 0.0),
                'scale_proxy_bounds': [0.0, 9.0], 'scale_thresholds': [v + .5 for v in range(9)],
                'scale_tie_break': 'higher_offset', 'scale_granularity': 'per_MX_block',
                'source_scale_learned': False, 'scale_gradient_normalization': 'none',
                'scale_projection': 'after_each_Adam_step', 'scale_lr': .001}
    if run.report.get('schema') == 'mx-triple-learned-scale10-pilot5-report-v1':
        expected['pilot_stop_epoch'] = 5
    else:
        expected.update(pilot_epochs=5, reference_end_epoch=100, scale_learning_end_epoch=5, scale_freeze_epoch=None)
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
                        'source_scale_bytes_copied': (bit == 8 or _all_zero_offsets(run, expected_epoch)),
                        'scale_exponent_offset': 0 if bit == 8 else None,
                        'scale_factor': 1 if bit == 8 else None,
                        'scale_exponent_offset_choices': [0] if bit == 8 else list(OFFSET_CHOICES),
                        'scale_factor_choices': [1] if bit == 8 else list(SCALE_FACTORS),
                        'shared_scale_compensation': bit != 8 and not _all_zero_offsets(run, expected_epoch),
                        'learned_shared_scale': bit != 8,
                        'fp32_scale_proxies_absent': True,
                        'packed_scale_choices_in_source': True,
                        'packed_scale_choices_bytes': 2040,
                        'truncation': bit != 8, 'is_source_export': bit == 8,
                        'derived_from_saved_source': bit != 8,
                        'master_weights_read_to_derive': False, 'saved_and_reloaded': True}
            if any(k not in detail or detail[k] != v for k, v in expected.items()):
                different = [k for k, v in expected.items() if k not in detail or detail[k] != v]
                run.issue('FOUT', f'{role}/{bit}: exportcontract wijkt af: ' + ', '.join(different))
            if detail.get('source_file_sha256') != modes['8'].get('sha256'):
                run.issue('FOUT', f'{role}/{bit}: afleiding verwijst niet naar de opgeslagen E4M3-bronhash.')



def _all_zero_offsets(run, epoch):
    if run.scale_blocks.empty or epoch is None or not {'epoch', 'offset'} <= set(run.scale_blocks):
        return False
    rows = run.scale_blocks.loc[run.scale_blocks.epoch == epoch]
    return len(rows) == 4080 and bool(rows.offset.eq(0).all())


def resolve_run_dir(run_dir=None, start=None):
    """Locate one run without silently preferring an older PASS over a new FAIL.

    An explicit existing run wins. A copied feedback folder is relocatable even
    when its notebook still contains the original Windows output path. For
    automatic project discovery, check-only commands do not hide training runs.
    """
    here = Path(start or Path.cwd()).expanduser().resolve()
    helper = Path(__file__).resolve().parent
    if run_dir is not None:
        explicit = Path(run_dir).expanduser()
        if explicit.is_dir() and (explicit / 'run_report.json').is_file():
            return explicit.resolve()
        if explicit.is_dir() and (explicit.parent / 'run_report.json').is_file():
            return explicit.parent.resolve()
        for local in (here, helper):
            if (local / 'run_report.json').is_file():
                return local
        if explicit.is_dir():
            return explicit.resolve()
        raise FileNotFoundError('RUN_DIR bevat geen run_report.json: ' + str(explicit))
    for local in (here, helper):
        if (local / 'run_report.json').is_file():
            return local
        if local.name == 'runner_source' and (local.parent / 'run_report.json').is_file():
            return local.parent
    roots = dict.fromkeys([here, *here.parents, helper, *helper.parents])
    candidates = {}
    for root in roots:
        parent = root / 'mx_runs' / 'mx_triple_learned_scale10_pilot5'
        if not parent.is_dir():
            continue
        for report_path in parent.glob('*/run_report.json'):
            try:
                report = json.loads(report_path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                continue
            if report.get('schema') == 'mx-triple-learned-scale10-pilot5-report-v1':
                candidates[report_path.parent] = report
    training = {p: r for p, r in candidates.items()
                if r.get('command') in ('run', 'resume') or p.name.endswith(('_run', '_resume'))}
    usable = training or candidates
    if usable:
        return max(usable, key=lambda p: (p.name, p.stat().st_mtime_ns))
    return helper  # Empty package template: load_run reports missing data clearly.


def load_run(run_dir=None) -> RunAnalysis:
    """Read saved JSON/CSV only; do not infer missing mode scores or load checkpoints."""
    run = RunAnalysis(resolve_run_dir(run_dir))
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
        if run.summary.get("pilot_completed") is True and run.last_epoch < 5:
            run.issue("FOUT", "De samenvatting claimt een voltooide pilot vóór epoch5.")
        if run.summary.get("full_100_epoch_training_completed") is True and not run.reference_complete:
            run.issue("FOUT", "Deze vijf-epochpilot kan geen voltooide honderd-epochtraining bevestigen.")
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
    if not ((data["epoch"] >= 1) & (data["epoch"] <= 5) & (data["epoch"] % 1 == 0)).all():
        fail("Ongeldige epochnummers: deze pilot bevat uitsluitend epochs 1 t/m 5.")
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
                             'offset_choices': list(OFFSET_CHOICES), 'scale_factors': list(SCALE_FACTORS), 'learned_shared_scale': True,
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
        expected_summary = {'schema': ('mx-triple-learned-scale10-pilot5-completion-v1' if run.report.get('schema') == 'mx-triple-learned-scale10-pilot5-report-v1' else 'mx-triple-learned-scale10-init1-pilot5-completion-v1'),
                            'shared_scale_compensation': True, 'learned_shared_scale': True,
                            'scale_exponent_offset_choices': list(OFFSET_CHOICES), 'scale_factor_choices': list(SCALE_FACTORS),
                                                        'all_mantissa_scale_bytes_equal_E2M3': True}
        if any(run.summary.get(k) != v for k, v in expected_summary.items()):
            run.issue('FOUT', 'Samenvatting bevestigt het geleerde tienkeuzebeleid niet correct.')



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
    if run.pilot_complete:
        endpoint += " De vijf-epochpilot is volgens het runrapport voltooid."
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
             ('Geleerde correctie', 'Eén FP32 θ per B32-groep; θ∈[0,9], begin6; afronden naar 0..9, halve waarden naar boven; factor2^offset.'),
             ('Gedeelde lagere schaal', 'E2M3 en E2M1 delen dezelfde blokkeuze. De E4M3-bronschaal zelf wordt niet geleerd.'),
             ('Mastergradient', 'Identity-STE naar G(W), zonder extra schaalfactor; bestaande G(W)-afgeleide behouden.'),
             ('Schaalgradient', 'ln(2) × som(gradient_q × werkelijk gebruikt q) per blok, opgeteld over E2M3 en E2M1.'),
             ('Gradientnormalisatie', 'Geen extra LSQ-normalisatie; geen gemiddelde in plaats van de blocksom.'),
             ('Schaalgrenzen', 'θ na Adam projecteren op [0,9]; E8M0-code in breder integer-type; geen codeclipping of wrap.'),
             ('Trainbare parameters', f"{c.get('base_parameters', '?')} basis + {c.get('learned_scale_parameters', '?')} schaalproxies = {c.get('parameters', '?')}"),
             ('Activaties / masters / proxies', 'A32 / FP32 / FP32'),
             ('Optimizer', 'Eén Adam-update na drie passes; afzonderlijke parametergroep voor schaalproxies.'),
             ('Distillatie', c.get('loss', 'E4M3 hard CE; E2M3 detached E4M3; E2M1 detached E2M3.')),
             ('BN-banken', str(c.get('bn_bank_mapping', 'niet gerapporteerd'))),
             ('Backend', f"cuDNN benchmark={backend.get('cudnn_benchmark', '?')}; deterministisch={backend.get('deterministic_algorithms', '?')}"),
             ('Data', f"{c.get('dataset', '?')}; train={c.get('train_examples', '?')}; test={c.get('test_examples', '?')}; extra={c.get('extra_used', '?')}"),
             ('Groep / schaalformaat', f"{c.get('mx_block_size', '?')} / {c.get('mx_scale_format', '?')}"),
             ('Batch / seed', f"{c.get('batch_size', '?')} / {c.get('seed', '?')}"),
             ('Pilot / gepland LR-schema', f"{c.get('pilot_stop_epoch', '?')} / {c.get('planned_epochs', '?')} epochs"),
             ('Checkpointselectie', run.selection.get('policy', 'selectiebestand ontbreekt'))]
    return pd.DataFrame(items, columns=['Onderdeel', 'Opgeslagen configuratie'])




def checkpoint_table(run):
    rows = []
    roles = (("last_epoch", "Epoch 5" if run.pilot_complete else "Laatste afgeronde epoch (pilot onvoltooid)", "last", run.last_epoch),
             ("best_E4M3_test", "Beste checkpoint gekozen door E4M3 binnen de pilot", "best", run.observed_best_epoch))
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
            ax.set_xticks(range(1, run.last_epoch + 1))
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
            if run.last_epoch <= 10:
                ax.set_xticks(range(1, run.last_epoch + 1))
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
                ax.set_xticks(range(1, run.last_epoch + 1))
        return fig



def plot_checkpoint_accuracy(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), constrained_layout=True)
        fig.suptitle(_title(run, "Truncatie met geleerde schaal ×32/×64 — hetzelfde checkpoint per groep"), fontsize=14, fontweight="bold")
        table = checkpoint_table(run)
        for ax, role, title in zip(axes, ("last_epoch", "best_E4M3_test"),
                                   ("Epoch 5" if run.pilot_complete else "Laatste afgeronde epoch", "Beste E4M3-checkpoint binnen de pilot")):
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
        endpoint = "Vijf-epochpilot: alle drie modi gebruiken per groep hetzelfde checkpoint. Epoch 100 wordt hier niet gemeten."
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
            "Vijf epochs meten de LR-overgangen op 50/75/90 en langdurige thermische omstandigheden nog niet. "
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


def _hard_offsets_array(theta):
    values = np.asarray(theta, dtype=np.float64)
    lower = np.floor(values)
    return (lower + (values - lower >= .5)).astype(np.int64)


def _validate_scales(run, records):
    """Recompute every hard choice, histogram and endpoint from saved FP32 θ."""
    if run.test.empty:
        return
    def fail(message):
        run.scales_safe = False
        run.issue('FOUT', 'Schaaldiagnostiek: ' + message)
    blocks, epochs = run.scale_blocks, run.scale_epochs
    block_required = {'epoch', 'layer', 'output_channel', 'block_index', 'theta', 'offset', 'factor'}
    count_keys = {f'count_x{factor}' for factor in SCALE_FACTORS}
    fraction_keys = {f'fraction_x{factor}' for factor in SCALE_FACTORS}
    endpoint_keys = {'choice_min_blocks', 'choice_max_blocks', 'theta_at_min_blocks', 'theta_at_max_blocks'}
    epoch_required = {'epoch', 'layer', 'blocks', 'theta_min', 'theta_max', 'theta_mean', 'theta_std',
        'theta_changed_blocks', 'theta_update_l2', 'net_choice_changes', 'choice_flips_during_epoch',
        'unique_blocks_switched_during_epoch', 'projected_values_during_epoch',
        'scale_grad_l2_min', 'scale_grad_l2_max', 'scale_grad_nonzero_blocks_max', 'scale_lr',
        'choice_jump_distance_during_epoch', 'max_choice_jump'} | count_keys | fraction_keys | endpoint_keys
    if blocks.empty or not block_required <= set(blocks):
        fail('scale_block_state.csv ontbreekt of mist verplichte kolommen.')
        return
    if epochs.empty or not epoch_required <= set(epochs):
        fail('scale_epoch_metrics.csv ontbreekt of mist verplichte kolommen: ' + ', '.join(sorted(epoch_required - set(epochs))))
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
    if not blocks.theta.between(0, 9).all():
        fail('FP32-proxy buiten [0,9].')
    desired = _hard_offsets_array(blocks.theta)
    if not np.array_equal(blocks.offset.to_numpy(), desired):
        fail('harde offset volgt niet afronding naar het dichtstbijzijnde geheel met halve waarden naar boven.')
    if not np.array_equal(blocks.factor.to_numpy(), np.exp2(desired)):
        fail('factor komt niet overeen met 2 tot de harde offset.')
    if not blocks.loc[blocks.epoch == 0, 'theta'].eq(float(run.config.get('scale_proxy_init', 0))).all():
        if (run.synthetic and run.report.get('synthetic_fixture_only') is True
                and run.config.get('synthetic_initial_theta_mixed') is True):
            run.issue('SYNTHETISCH', 'Deze softwarefixture start met gemengde θ; deze gemengde initialisatie is geen onderzoeksresultaat.')
        else:
            fail('beginstaat van de schaalproxies wijkt af van scale_proxy_init.')
    if not run.scales_safe:
        return
    block_index = {}
    for epoch in range(run.last_epoch + 1):
        for layer, count in LAYER_BLOCKS.items():
            state = blocks.loc[(blocks.epoch == epoch) & (blocks.layer == layer)].sort_values(['output_channel', 'block_index'])
            if len(state) != count:
                fail(f'{layer}/epoch{epoch}: verwacht {count} blokken, gevonden {len(state)}.')
                continue
            channels = sorted(state.output_channel.astype(int).unique())
            per_channel = state.groupby('output_channel').block_index.apply(lambda v: sorted(v.astype(int)))
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
            offsets = after.offset.to_numpy()
            expected = {'blocks': count, 'theta_min': float(t1.min()), 'theta_max': float(t1.max()),
                        'theta_mean': float(t1.mean()), 'theta_std': float(t1.std(ddof=0)),
                        'theta_changed_blocks': int((t1 != t0).sum()),
                        'theta_update_l2': float(np.linalg.norm(t1 - t0)),
                        'net_choice_changes': int((offsets != before.offset.to_numpy()).sum()),
                        'choice_min_blocks': int((offsets == 0).sum()), 'choice_max_blocks': int((offsets == 9).sum()),
                        'theta_at_min_blocks': int((t1 == 0).sum()), 'theta_at_max_blocks': int((t1 == 9).sum())}
            for offset, factor in zip(OFFSET_CHOICES, SCALE_FACTORS):
                n = int((offsets == offset).sum())
                expected[f'count_x{factor}'] = n
                expected[f'fraction_x{factor}'] = n / count
            for key, value in expected.items():
                tol = 2e-6 if key in ('theta_mean', 'theta_std', 'theta_update_l2') else 1e-8
                if not _same_number(row[key], value, tolerance=tol):
                    fail(f'{layer}/epoch{epoch}: {key} strookt niet met de opgeslagen blokstaten.')
            counters = {'theta_changed_blocks', 'net_choice_changes', 'choice_flips_during_epoch',
                        'unique_blocks_switched_during_epoch', 'projected_values_during_epoch',
                        'scale_grad_nonzero_blocks_max', 'choice_jump_distance_during_epoch', 'max_choice_jump'} | count_keys | endpoint_keys
            if any(row[k] < 0 or row[k] % 1 != 0 for k in counters):
                fail(f'{layer}/epoch{epoch}: ongeldige schaaltellingen.')
            flips, unique = row.choice_flips_during_epoch, row.unique_blocks_switched_during_epoch
            if not (row.net_choice_changes <= unique <= count and unique <= flips <= count * steps):
                fail(f'{layer}/epoch{epoch}: netto wijzigingen, unieke wissels en totaal flips zijn inconsistent.')
            distance, maximum = row.choice_jump_distance_during_epoch, row.max_choice_jump
            if not (flips <= distance <= 9 * flips and 0 <= maximum <= 9
                    and ((flips == 0 and maximum == 0) or (flips > 0 and maximum >= 1))
                    and maximum <= distance <= maximum * flips
                    and np.abs(offsets - before.offset.to_numpy()).sum() <= distance):
                fail(f'{layer}/epoch{epoch}: sprongafstanden of grootste sprong zijn inconsistent.')
            if row.projected_values_during_epoch > count * steps or row.scale_grad_nonzero_blocks_max > count:
                fail(f'{layer}/epoch{epoch}: grensprojecties of gradientblokken overschrijden de mogelijke telling.')
            if not (0 <= row.scale_grad_l2_min <= row.scale_grad_l2_max):
                fail(f'{layer}/epoch{epoch}: ongeldige gradientnormgrenzen.')
            scale_lr = run.config.get('scale_lr', .001)
            if not _same_number(row.scale_lr, scale_lr):
                fail(f'{layer}/epoch{epoch}: schaal-leersnelheid verschilt van de configuratie.')
            old = next((x for x in logged if x.get('layer') == layer), None)
            if old is None or any(not _same_number(row[k], old.get(k), tolerance=1e-8) for k in epoch_required - {'layer'}):
                fail(f'{layer}/epoch{epoch}: schaalstatistiek-CSV verschilt van history.json.')
    if run.scales_safe:
        final = epochs.loc[epochs.epoch == run.last_epoch]
        if final[f'count_x{int(2**int(run.config.get("scale_proxy_init", 0)))}'].sum() == TOTAL_BLOCKS:
            run.issue('UITKOMST', 'Alle groepen kiezen op de laatste epoch nog de beginfactor. Bekijk ook de continue θ-verplaatsing; dit is geen technische FAIL.')
        if epochs.scale_grad_l2_max.eq(0).all():
            run.issue('UITKOMST', 'Alle schaalgradients zijn nul; er is geen schaalleersignaal waargenomen.')
        if run.summary.get('scale_equality_scope') == 'last_epoch':
            actual = bool(final.count_x1.sum() == TOTAL_BLOCKS)
            if run.summary.get('all_lower_scale_bytes_equal_current_E4_source') is not actual:
                fail('claim over laatste lagere schalen verschilt van de opgeslagen offsets.')


def scale_learning_markdown(run):
    if not run.scales_safe or run.scale_epochs.empty:
        return '**Geen bevestigde schaaldiagnostiek beschikbaar.** Bekijk de meldingen.'
    last = run.scale_epochs.loc[run.scale_epochs.epoch == run.last_epoch]
    counts = {factor: int(last[f'count_x{factor}'].sum()) for factor in SCALE_FACTORS}
    chosen = ', '.join(f'×{factor}: **{count:,}**' for factor, count in counts.items())
    flips = int(run.scale_epochs.choice_flips_during_epoch.sum())
    changed = int(last.theta_changed_blocks.sum())
    low, high = int(last.choice_min_blocks.sum()), int(last.choice_max_blocks.sum())
    clamp_low, clamp_high = int(last.theta_at_min_blocks.sum()), int(last.theta_at_max_blocks.sum())
    return (f'**Laatste afgeronde epoch {run.last_epoch} — aantallen per factor:** {chosen}. '
            f'Totaal: {TOTAL_BLOCKS:,} groepen. In de laatste epoch veranderden {changed:,} FP32-hulpwaarden. '
            f'Over de opgeslagen training werden {flips:,} harde keuzewisselingen geteld.\n\n'
            f'**Uiterste keuzes:** {low:,} groepen kiezen ×1 en {high:,} kiezen ×512. '
            f'**Exact op de projectiegrens:** {clamp_low:,} θ-waarden zijn 0 en {clamp_high:,} zijn 9. '
            'Een uiterste keuze betekent dus niet automatisch dat de continue parameter op de grens vastzit.\n\n'
            'Een groep kan binnen een epoch meerdere keren wisselen. Het aantal wissels, het aantal unieke '
            'wisselende groepen en het verschil tussen de twee epochgrenzen zijn afzonderlijke metingen. '
            'Een sprong van offset5 naar7 telt als één wissel met sprongafstand2. '
            'Geen harde wissel sluit een continue θ-verplaatsing niet uit; geen van deze diagnostieken bewijst accuracywinst.')


def scale_table(run):
    if not run.scales_safe or run.scale_epochs.empty:
        return pd.DataFrame({'Melding': ['Schaalstatistieken ontbreken of zijn inconsistent.']})
    return run.scale_epochs.copy()


def plot_scale_choices(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.1))
        fig.subplots_adjust(top=.87, bottom=.19, hspace=.42, wspace=.25)
        fig.suptitle(_title(run, 'Tien mogelijke schaalfactoren · verdeling per laag en epoch'),
                     fontsize=14, fontweight='bold', y=.97)
        if not run.scales_safe or run.scale_epochs.empty:
            for ax in axes.flat: _empty(ax, 'Geen bevestigde schaalkeuzes beschikbaar')
            return fig
        for ax, (layer, count) in zip(axes.flat, LAYER_BLOCKS.items()):
            states = run.scale_blocks.loc[run.scale_blocks.layer == layer]
            x = np.arange(run.last_epoch + 1)
            bottom = np.zeros(len(x))
            for factor, color in zip(SCALE_FACTORS, FACTOR_COLORS):
                values = np.array([100 * states.loc[states.epoch == epoch, 'factor'].eq(factor).sum() / count for epoch in x])
                ax.bar(x, values, bottom=bottom, color=color, width=.72, label=f'×{factor}', linewidth=0)
                bottom += values
            ax.set(title=f'{layer} · {count:,} groepen', xlabel='Afgeronde epoch (0 = initialisatie)',
                   ylabel='Groepen [%]', xticks=x, ylim=(0, 100))
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, .035), title='Gekozen factor')
        return fig


def plot_theta_distributions(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.5), constrained_layout=True)
        fig.suptitle(_title(run, 'Continue FP32-hulpwaarden θ · volledige toegestane ruimte'), fontsize=14, fontweight='bold')
        if not run.scales_safe or run.scale_blocks.empty:
            for ax in axes.flat: _empty(ax, 'Geen bevestigde θ-staat')
            return fig
        chosen = [(0, 'Initialisatie', '#99A3AD'), (run.observed_best_epoch, 'E4-best', BLUE)]
        if run.last_epoch != run.observed_best_epoch:
            chosen.append((run.last_epoch, 'Laatste', AMBER))
        for ax, (layer, count) in zip(axes.flat, LAYER_BLOCKS.items()):
            for epoch, label, color in chosen:
                values = run.scale_blocks.loc[(run.scale_blocks.epoch == epoch) & (run.scale_blocks.layer == layer), 'theta'].to_numpy()
                ax.hist(values, bins=np.linspace(0, 9, 91), weights=np.full(len(values), 100 / count),
                        histtype='step', lw=1.6, color=color, label=f'{label}: epoch {epoch}')
            for threshold in np.arange(.5, 9, 1):
                ax.axvline(threshold, color='#B7C0C8', ls=':', lw=.7, zorder=0)
            ax.set(title=f'{layer} · {count:,} groepen', xlabel='θ; stippellijnen = keuzedrempels',
                   ylabel='Groepen per bin [%]', xlim=(-.08, 9.08), xticks=range(10), ylim=(0, 105))
            ax.legend(frameon=False, fontsize=8)
        return fig


def plot_scale_boundaries(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Bereik van de keuze tegenover de grens van de leerparameter'), fontsize=14, fontweight='bold')
        if not run.scales_safe or run.scale_epochs.empty:
            for ax in axes: _empty(ax, 'Geen bevestigde randdiagnostiek')
            return fig
        keys = ['choice_min_blocks', 'choice_max_blocks', 'theta_at_min_blocks', 'theta_at_max_blocks']
        totals = run.scale_epochs.groupby('epoch', sort=True)[keys].sum()
        for ax, side, factor, color in [(axes[0], 'min', 1, BLUE), (axes[1], 'max', 512, RED)]:
            ax.plot(totals.index, 100 * totals[f'choice_{side}_blocks'] / TOTAL_BLOCKS, 'o-', color=color, label=f'Keuze ×{factor}')
            ax.plot(totals.index, 100 * totals[f'theta_at_{side}_blocks'] / TOTAL_BLOCKS, 's--', color=GREEN,
                    label='θ exact ' + ('0' if side == 'min' else '9'))
            ax.set(title='Onderkant' if side == 'min' else 'Bovenkant', xlabel='Epoch', ylabel='Groepen [%]',
                   xticks=range(1, run.last_epoch + 1), ylim=(-1, 101))
            ax.legend(frameon=False)
        return fig


def storage_cost_table(run=None):
    """Analytical choice payload and theta state; not checkpoint size or VRAM."""
    return pd.DataFrame([
        {'Variant': 'Vaste factor', 'Keuzebits per groep': 0, 'Groepen': TOTAL_BLOCKS,
         'Compacte keuzebytes': 0, 'Extra FP32 theta+gradient+Adam [bytes]': 0},
        {'Variant': 'Twee leerbare keuzes', 'Keuzebits per groep': 1, 'Groepen': TOTAL_BLOCKS,
         'Compacte keuzebytes': 510, 'Extra FP32 theta+gradient+Adam [bytes]': 65280},
        {'Variant': 'Tien leerbare keuzes', 'Keuzebits per groep': 4, 'Groepen': TOTAL_BLOCKS,
         'Compacte keuzebytes': 2040, 'Extra FP32 theta+gradient+Adam [bytes]': 65280},
    ])


def plot_storage_cost(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Kosten van de keuze · analytische opslag, geen VRAM-meting'), fontsize=14, fontweight='bold')
        table = storage_cost_table(run)
        labels = ['Vast', '2 keuzes', '10 keuzes']
        for ax, column, divisor, ylabel in [
                (axes[0], 'Compacte keuzebytes', 1, 'Bytes keuze-informatie in flexibele bron'),
                (axes[1], 'Extra FP32 theta+gradient+Adam [bytes]', 1024, 'Extra trainingstensors [KiB]')]:
            values = table[column].to_numpy() / divisor
            ax.bar(labels, values, color=['#A4ADB4', GREEN, BLUE], width=.6)
            for x, value in enumerate(values):
                ax.text(x, value + max(values) * .025, f'{value:g}', ha='center', va='bottom')
            ax.set(ylabel=ylabel, ylim=(0, max(values) * 1.18))
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
        axes[0].set(xlabel='Epoch', ylabel='Aantal', xticks=range(1, run.last_epoch+1), ylim=(0, None))
        axes[0].legend(frameon=False, fontsize=8)
        for layer, color in zip(LAYER_BLOCKS, LAYER_COLORS):
            r = run.scale_epochs.loc[run.scale_epochs.layer == layer].sort_values('epoch')
            axes[1].plot(r.epoch, r.theta_update_l2, marker='o', label=layer, color=color)
        axes[1].set(xlabel='Epoch', ylabel='L2-norm netto θ-update per laag', xticks=range(1, run.last_epoch+1), ylim=(0,None))
        axes[1].legend(frameon=False, ncol=2, fontsize=8)
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
        axes[0].set(xlabel='Epoch', ylabel='Max. L2-norm schaalgradient per update', yscale='symlog', xticks=range(1,run.last_epoch+1))
        axes[0].set_yscale('symlog', linthresh=1e-8)
        axes[1].set(xlabel='Epoch', ylabel='Aantal θ-waarden geprojecteerd na Adam', xticks=range(1,run.last_epoch+1), ylim=(0,None))
        for ax in axes: ax.legend(frameon=False, ncol=2, fontsize=8)
        return fig


# Initialization ablation: the inherited strict pilot reader above remains read-only.
INIT64_REFERENCE_SHA256 = 'e2df45acecccb8e56115b4d01f90b11a5f43bb4c9bfa0291a4000401046f866e'
NEW_SCHEMA = 'mx-triple-learned-scale10-init1-pilot5-report-v1'
NEW_LABEL, OLD_LABEL = 'Start ×1 — nieuwe pilot', 'Start ×64 — bestaande pilot'
NEW_COLOR, OLD_COLOR = '#245A81', '#87939C'
_load_pilot_core = load_run


def load_run(run_dir=None):
    run = _load_pilot_core(run_dir)
    run.init64_reference = None
    run.reference_safe = False
    if run.report.get('schema') == NEW_SCHEMA and run.history.get('epochs'):
        guard = run.report.get('history_export_validation', {})
        path = run.path / 'history.json'
        expected = dict(file='history.json', bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        epochs=len(run.history['epochs']), exact_trace_match=True)
        if any(guard.get(key) != value for key, value in expected.items()):
            run.metrics_safe = False
            run.scales_safe = False
            run.issue('FOUT', 'De history.json-readbackcontrole ontbreekt of past niet bij het werkelijk gelezen bestand.')
    if run.test.empty:
        return run
    if run.synthetic:
        run.issue('SYNTHETISCH', 'De opgeslagen CPU-fixture is geen SVHN-run; de echte historische accuracies worden hiermee niet vergeleken.')
        return run
    directory = run.path / 'init64_reference'
    marker = directory / 'REFERENCE.json'
    try:
        if hashlib.sha256(marker.read_bytes()).hexdigest() != INIT64_REFERENCE_SHA256:
            raise ValueError('De vastgelegde init×64-referentie-identiteit wijkt af.')
        metadata = json.loads(marker.read_text(encoding='utf-8-sig'))
        for name, digest in metadata['files_sha256'].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Historisch bestand ontbreekt of is gewijzigd: ' + name)
        ref = _load_pilot_core(directory)
        if (ref.report.get('schema') != 'mx-triple-learned-scale10-pilot5-report-v1'
                or ref.report.get('status') != 'PASS' or ref.last_epoch != 5
                or ref.synthetic or not ref.provenance_confirmed or not ref.pilot_complete):
            raise ValueError('Verwacht de complete, echte init×64-pilot met vijf epochs.')
        for key in ('model','expand','dataset','seed','batch_size','train_examples','test_examples',
                    'activation_bits','master_dtype','optimizer','lr','scale_lr','weight_decay',
                    'scheduler_milestones','scheduler_gamma','mx_block_size','loss',
                    'scale_factor_choices','scale_proxy_bounds'):
            if run.config.get(key) != ref.config.get(key):
                raise ValueError('De initialisatievergelijking verschilt ook in ' + key + '.')
        start = run.history.get('initial_base_model_sha256')
        if not start or start != ref.history.get('initial_base_model_sha256'):
            raise ValueError('De oorspronkelijke FP32-gewichten/BN-toestand zijn niet dezelfde.')
        for new, old in zip(run.history['epochs'], ref.history['epochs']):
            for phase in ('train','test'):
                if new['coverage'][phase]['order_sha256'] != old['coverage'][phase]['order_sha256']:
                    raise ValueError(f'De {phase}-volgorde verschilt op epoch {new["epoch"]}.')
        run.init64_reference = ref
        run.reference_safe = True
        run.input_files.extend(ref.input_files + [marker])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        run.issue('FOUT', 'Historische vergelijking niet bevestigd: ' + str(exc))
    return run


def status_markdown(run):
    errors = sum(x['Niveau'] in ('FOUT','RUNFOUT') for x in run.issues)
    heading = ('SYNTHETISCHE SOFTWARECONTROLE — geen onderzoeksresultaat' if run.synthetic else
               'Vijf-epochpilot voltooid' if run.pilot_complete else 'Voorlopige analyse van de initialisatieproef')
    last = run.last_epoch if run.last_epoch is not None else 'geen'
    return (f'### {heading}\n\nRunstatus: **{run.report.get("status", "niet beschikbaar")}** · laatste afgeronde epoch: **{last}** · analysefouten: **{errors}**. '
            f'Historische init×64-vergelijking: **{"bevestigd" if getattr(run,"reference_safe",False) else "niet beschikbaar of niet bevestigd"}**.\n\n'
            'Deze proef loopt maximaal vijf epochs. Er is geen epoch-100-resultaat. Lage accuracy is een meetuitkomst; '
            'vijf epochs tonen de vroege beweging, niet of een initialisatie uiteindelijk convergeert. '
            'De notebook leest alleen JSON/CSV en start geen training.')


def protocol_table(run):
    c = run.config
    return pd.DataFrame([
        ('Onderzoeksvraag','Beïnvloedt starten bij ×1 in plaats van ×64 het verloop van gewichten en schaalkeuzes?'),
        ('Enige bedoelde trainingswijziging','Alle 4.080 continue schaalparameters beginnen op θ=0 in plaats van θ=6.'),
        ('Betekenis ×1','S_E2 = S_E4 × 1 per groep; dit is geen absolute schaal 1. De E4-bronschaal blijft dynamisch.'),
        ('Tien vrije keuzes','×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256, ×512'),
        ('Gewichtsroute','E4M3-bron → exponent-MSB-truncatie → E2M3 → mantissatruncatie → E2M1; lagere modi delen hun schaal.'),
        ('Leren en grenzen','θ wordt geleerd met de bestaande log-exponent-STE en na iedere Adamupdate op [0,9] geprojecteerd; geen freeze.'),
        ('Activaties / masters / groepen','A32 / FP32 / B32'),
        ('Leerinstellingen',f'Adam; gewichts-LR {c.get("lr", "?")}; schaal-LR {c.get("scale_lr", "?")}; oorspronkelijke mijlpalen 50/75/90.'),
        ('Vergelijking','Dezelfde oorspronkelijke FP32-/BN-initialisatie en seed; samplevolgordes gecontroleerd. CUDA is niet bitexact gegarandeerd.'),
        ('Checkpointselectie','Laatste endpoint epoch 5 en eerste hoogste E4M3-testaccuracy binnen hetzelfde beschikbare 1–5-venster; alle drie modi samen.'),
        ('Interpretatie','Dit is een verkennende initialisatiecontrole, geen bewijs voor een optimale factor of een optimaal aantal epochs.')
    ],columns=['Onderdeel','Afspraak'])


def _variants(run, require_comparison=False):
    variants = []
    if getattr(run,'reference_safe',False) and getattr(run,'init64_reference',None) is not None:
        variants.append((OLD_LABEL,run.init64_reference,OLD_COLOR))
    if run.metrics_safe and not run.test.empty:
        variants.append((NEW_LABEL,run,NEW_COLOR))
    return variants if not require_comparison or len(variants)==2 else []


def _limits(values, minimum_span=1.0):
    values = np.asarray(list(values),dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):return (0.,100.)
    lo,hi = float(values.min()),float(values.max())
    span = max(hi-lo,minimum_span)
    margin = max(.15,.08*span)
    mid = (lo+hi)/2
    return max(0.,min(lo-margin,mid-span/2-margin)),min(100.,max(hi+margin,mid+span/2+margin))


def initialization_checkpoint_table(run,role='endpoint'):
    columns=['Variant','Epoch','Precisie','weight_bits','Testaccuracy [%]','Correct','Testbeelden']
    if role not in ('endpoint','best'):raise ValueError('Onbekende checkpointselectie.')
    variants=_variants(run,require_comparison=True)
    if not variants or (role=='endpoint' and run.last_epoch!=5):return pd.DataFrame(columns=columns)
    rows=[]
    for label,source,color in variants:
        metrics=source.test.loc[source.test.epoch<=run.last_epoch]
        high=metrics.loc[metrics.weight_bits.eq(8)].sort_values('epoch')
        epoch=5 if role=='endpoint' else int(high.loc[high.top1_percent.idxmax(),'epoch'])
        for bit in DISPLAY_BITS:
            row=metrics.loc[metrics.epoch.eq(epoch)&metrics.weight_bits.eq(bit)].iloc[0]
            rows.append({'Variant':label,'Epoch':epoch,'Precisie':LABELS[bit], 'weight_bits':bit,
                         'Testaccuracy [%]':float(row.top1_percent),'Correct':int(row.correct1),'Testbeelden':int(row.n)})
    return pd.DataFrame(rows,columns=columns)


def _checkpoint_limits(run):
    values=[]
    for role in ('endpoint','best'):
        table=initialization_checkpoint_table(run,role)
        if not table.empty:values.extend(table['Testaccuracy [%]'])
    return _limits(values)


def plot_initialization_accuracy(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(14,4.9),sharey=True)
        fig.subplots_adjust(top=.78,bottom=.21,wspace=.12)
        fig.suptitle(_title(run,'Accuracyverloop · start ×1 tegenover start ×64'),fontsize=15,fontweight='bold',y=.98)
        variants=_variants(run)
        values=[float(v) for _,r,_ in variants for v in r.test.loc[r.test.epoch<=run.last_epoch,'top1_percent']]
        limits=_limits(values)
        for ax,bit in zip(axes,DISPLAY_BITS):
            if not variants:_empty(ax,'Nog geen bevestigde testepochs');continue
            for label,r,color in variants:
                rows=r.test.loc[r.test.weight_bits.eq(bit)&r.test.epoch.le(run.last_epoch)].sort_values('epoch')
                ax.plot(rows.epoch,rows.top1_percent,marker='o',color=color,label=label,lw=2)
            ax.set(title=LABELS[bit]+'/A32',xlabel='Afgeronde epoch',xticks=range(1,6),xlim=(.8,5.2),ylim=limits)
        axes[0].set_ylabel('Testaccuracy [%]')
        if variants:
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.02))
        fig.text(.5,.855,'Gelijke y-as voor de drie precisies; alle getoonde meetwaarden blijven zichtbaar.',ha='center',color='#56626A')
        return fig


def plot_initialization_checkpoints(run,role='endpoint'):
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(11.8,5.4));fig.subplots_adjust(left=.12,right=.96,top=.80,bottom=.24)
        title='Accuracy op epoch 5' if role=='endpoint' else 'E4M3-beste checkpoint binnen het beschikbare pilotvenster'
        fig.suptitle(_title(run,title+' · ingezoomde as'),fontsize=14,fontweight='bold',y=.98)
        table=initialization_checkpoint_table(run,role)
        if table.empty:
            _empty(ax,'Epoch 5 nog niet voltooid of historische vergelijking niet bevestigd' if role=='endpoint' else 'Nog geen bevestigde checkpointvergelijking');return fig
        for i,bit in enumerate(DISPLAY_BITS):
            rows=table.loc[table.weight_bits.eq(bit)]
            old=rows.loc[rows.Variant.eq(OLD_LABEL)].iloc[0];new=rows.loc[rows.Variant.eq(NEW_LABEL)].iloc[0]
            a,b=float(old['Testaccuracy [%]']),float(new['Testaccuracy [%]'])
            ax.plot([a,b],[i,i],color='#C5CDD2',lw=2.5)
            for value,row,color,marker,delta in [(a,old,OLD_COLOR,'o',.17),(b,new,NEW_COLOR,'D',-.17)]:
                ax.scatter([value],[i],color=color,marker=marker,s=65,zorder=3)
                ax.annotate(f'{value:.3f}%'.replace('.',','),(value,i+delta),ha='center',va='center',fontsize=10,color=color)
        ax.set(yticks=range(3),yticklabels=[LABELS[b]+'/A32' for b in DISPLAY_BITS],ylim=(2.55,-.55),xlim=_checkpoint_limits(run),xlabel='Testaccuracy [%] · ingezoomde as')
        from matplotlib.lines import Line2D
        handles=[]
        for label,color,marker in [(OLD_LABEL,OLD_COLOR,'o'),(NEW_LABEL,NEW_COLOR,'D')]:
            epoch=int(table.loc[table.Variant.eq(label),'Epoch'].iloc[0]);handles.append(Line2D([],[],ls='',marker=marker,color=color,label=f'{label} · epoch {epoch}',markersize=7))
        fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.085))
        fig.text(.5,.02,'Iedere variant gebruikt één geselecteerde epoch voor alle drie precisies. Beide checkpointfiguren delen hun asgrenzen.',ha='center',fontsize=9,color='#56626A')
        return fig


def theta_summary_table(run):
    columns=['Epoch','Laag','Groepen','Min θ','P05 θ','Mediaan θ','Gemiddelde θ','P95 θ','Max θ','Exact θ=0 [%]','Factor ×1 [%]']
    if not run.metrics_safe or not run.scales_safe or run.scale_blocks.empty:return pd.DataFrame(columns=columns)
    rows=[]
    for epoch,frame in run.scale_blocks.groupby('epoch'):
        for layer,subset in [('Alle lagen',frame)]+list(frame.groupby('layer')):
            x=subset.theta.to_numpy(dtype=float)
            rows.append([int(epoch),layer,len(x),float(x.min()),float(np.quantile(x,.05)),float(np.median(x)),float(x.mean()),float(np.quantile(x,.95)),float(x.max()),100*float(np.mean(x==0)),100*float(np.mean(subset.factor.to_numpy()==1))])
    return pd.DataFrame(rows,columns=columns)


def factor_distribution_table(run,epoch=None):
    columns=['Epoch','Factor','Groepen','Aandeel [%]']
    if not run.metrics_safe or not run.scales_safe or run.scale_blocks.empty:return pd.DataFrame(columns=columns)
    epoch=run.last_epoch if epoch is None else int(epoch)
    frame=run.scale_blocks.loc[run.scale_blocks.epoch.eq(epoch)]
    if frame.empty:return pd.DataFrame(columns=columns)
    return pd.DataFrame([{'Epoch':epoch,'Factor':f'×{f}','Groepen':int(frame.factor.eq(f).sum()),'Aandeel [%]':100*float(frame.factor.eq(f).mean())} for f in SCALE_FACTORS],columns=columns)


def plot_factor_evolution(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(13.2,6.4),sharey=True);fig.subplots_adjust(left=.07,right=.90,top=.82,bottom=.16,wspace=.16)
        fig.suptitle(_title(run,'Schaalkeuzes vanaf initialisatie · alle tien factoren'),fontsize=14,fontweight='bold',y=.97)
        variants=_variants(run,require_comparison=True)
        if not variants:
            for ax in axes:_empty(ax,'Historische vergelijking nog niet bevestigd')
            return fig
        for ax,(label,r,color) in zip(axes,variants):
            if not r.scales_safe or r.scale_blocks.empty:_empty(ax,'Geen bevestigde schaalstaten');continue
            epochs=list(range(run.last_epoch+1))
            matrix=np.array([[float(factor_distribution_table(r,e).loc[factor_distribution_table(r,e).Factor.eq(f'×{f}'),'Aandeel [%]'].iloc[0]) for e in epochs] for f in SCALE_FACTORS])
            image=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto')
            for i in range(10):
                for j in range(len(epochs)):
                    ax.text(j,i,f'{matrix[i,j]:.1f}'.replace('.',','),ha='center',va='center',color='white' if matrix[i,j]>55 else '#233746',fontsize=8)
            ax.set(title=label,xticks=range(len(epochs)),xticklabels=epochs,yticks=range(10),yticklabels=[f'×{f}' for f in SCALE_FACTORS],xlabel='Epoch · 0 = initialisatie')
            ax.grid(False)
        cbax=fig.add_axes([.92,.19,.015,.59]);fig.colorbar(image,cax=cbax,label='Groepen [%]')
        fig.text(.5,.035,'Cellen tonen percentages; ook ongebruikte factoren blijven zichtbaar. Er worden geen keuzes tussen epochs geïnterpoleerd.',ha='center',fontsize=9,color='#56626A')
        return fig


def plot_factor_counts(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(15.6,6.8),sharex=True,sharey=True);fig.subplots_adjust(left=.055,right=.99,top=.78,bottom=.14,wspace=.18)
        fig.suptitle(_title(run,'Gekozen factoren · aantal groepen én aandeel'),fontsize=15,fontweight='bold',y=.97)
        items=[('Nieuwe start · epoch 0',run,0),('Nieuwe pilot · laatste epoch',run,run.last_epoch),('Bestaande ×64-pilot · dezelfde epoch',getattr(run,'init64_reference',None),run.last_epoch)]
        for ax,(title,r,epoch) in zip(axes,items):
            ax.set_title(title+(f'\nEpoch {epoch}' if epoch else ''),fontsize=11)
            table=factor_distribution_table(r,epoch) if r is not None and epoch is not None else pd.DataFrame()
            if table.empty:_empty(ax,'Nog geen bevestigde schaalverdeling');continue
            ax.barh(range(10),table.Groepen,color=FACTOR_COLORS,height=.64)
            ax.scatter(np.zeros(10),range(10),s=7,color='#6C7880',zorder=3)
            for i,row in table.iterrows():
                count=int(row.Groepen);label=f'{count:,} | {row["Aandeel [%]"]:.2f}%'.replace(',','_').replace('.',',').replace('_','.')
                ax.text(count+50,i,label,va='center',fontsize=8.5)
            ax.set(yticks=range(10),yticklabels=[f'×{f}' for f in SCALE_FACTORS],xlim=(0,TOTAL_BLOCKS*1.32),ylim=(9.6,-.6),xlabel='Aantal groepen (totaal 4.080)')
        fig.text(.5,.035,'Nulbaseline voor aantallen. De verhouding tussen E2- en E4-schaal wordt gekozen; de E4-bronschaal blijft dynamisch.',ha='center',fontsize=9,color='#56626A')
        return fig


def plot_theta_evolution(run):
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(12,5.4));fig.subplots_adjust(top=.80,bottom=.22)
        fig.suptitle(_title(run,'Continue schaalparameter θ · beweging vóór een harde omschakeling'),fontsize=14,fontweight='bold',y=.98)
        variants=_variants(run)
        if not variants:_empty(ax,'Nog geen bevestigde schaalstaten');return fig
        for label,r,color in variants:
            table=theta_summary_table(r);table=table.loc[table.Laag.eq('Alle lagen')&table.Epoch.le(run.last_epoch)]
            if table.empty:continue
            x=table.Epoch.to_numpy();ax.fill_between(x,table['P05 θ'].to_numpy(),table['P95 θ'].to_numpy(),color=color,alpha=.17)
            ax.plot(x,table['Gemiddelde θ'],color=color,marker='o',lw=2,label=label+' · gemiddelde')
            ax.plot(x,table['Min θ'],color=color,ls=':',alpha=.7,lw=1);ax.plot(x,table['Max θ'],color=color,ls=':',alpha=.7,lw=1)
        for threshold in np.arange(.5,9,.5):
            if threshold%1==.5:ax.axhline(threshold,color='#D0D4D8',lw=.5,zorder=0)
        ax.set(xlabel='Afgeronde epoch · 0 = initialisatie',ylabel='FP32 θ · offset = afronden(θ)',xticks=range(6),xlim=(-.15,5.15),ylim=(-.15,9.15))
        ax.legend(loc='upper left',frameon=False)
        fig.text(.5,.045,'Band: P05–P95 over groepen; stippellijnen: minimum/maximum. Horizontale lijnen zijn de omschakeldrempels.',ha='center',fontsize=9,color='#56626A')
        return fig


def plot_theta_by_layer(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,3,figsize=(14,7.6),sharex=True,sharey=True);fig.subplots_adjust(top=.85,bottom=.13,hspace=.32,wspace=.14)
        fig.suptitle(_title(run,'Nieuwe ×1-pilot · continue θ per laag'),fontsize=15,fontweight='bold',y=.98)
        table=theta_summary_table(run)
        if table.empty:
            for ax in axes.flat:_empty(ax,'Nog geen bevestigde schaalstaten')
            return fig
        for ax,layer in zip(axes.flat,LAYER_BLOCKS):
            sub=table.loc[table.Laag.eq(layer)].sort_values('Epoch');x=sub.Epoch.to_numpy()
            ax.fill_between(x,sub['P05 θ'].to_numpy(),sub['P95 θ'].to_numpy(),color=NEW_COLOR,alpha=.18)
            ax.plot(x,sub['Gemiddelde θ'],color=NEW_COLOR,marker='o',lw=2)
            ax.plot(x,sub['Min θ'],color=NEW_COLOR,ls=':',lw=1);ax.plot(x,sub['Max θ'],color=NEW_COLOR,ls=':',lw=1)
            for threshold in np.arange(.5,9,1):ax.axhline(threshold,color='#D4D8DA',lw=.7)
            ax.set(title=layer+f' · {LAYER_BLOCKS[layer]} groepen',xlabel='Epoch',ylabel='θ',xticks=range(6),xlim=(-.15,5.15))
        low=float(table['Min θ'].min());high=float(table['Max θ'].max())
        span=max(.01,high-low);margin=.12*span
        for ax in axes.flat:ax.set_ylim(max(-.02,low-margin),min(9.1,high+margin))
        fig.text(.5,.025,'Gemiddelde met P05–P95-band en min/max. Gelijke θ-as voor alle lagen; geen harde keuze is geen bewijs van stilstaande θ.',ha='center',fontsize=9,color='#56626A')
        return fig


def plot_initialization_boundaries(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(13,5.2));fig.subplots_adjust(top=.79,bottom=.23,wspace=.26)
        fig.suptitle(_title(run,'Grenskeuze ×1 en echte begrenzing van θ zijn verschillende grootheden'),fontsize=14,fontweight='bold',y=.98)
        variants=_variants(run)
        if not variants:
            for ax in axes:_empty(ax,'Nog geen bevestigde schaaldiagnostiek')
            return fig
        for label,r,color in variants:
            table=theta_summary_table(r);table=table.loc[table.Laag.eq('Alle lagen')&table.Epoch.le(run.last_epoch)]
            if table.empty:continue
            axes[0].plot(table.Epoch,table['Factor ×1 [%]'],color=color,marker='o',label=label+' · factor ×1')
            axes[0].plot(table.Epoch,table['Exact θ=0 [%]'],color=color,ls='--',marker='x',label=label+' · exact θ=0')
            audit=r.scale_epochs.loc[r.scale_epochs.epoch.le(run.last_epoch)]
            if not audit.empty:
                grouped=audit.groupby('epoch').projected_values_during_epoch.sum()
                axes[1].plot(grouped.index,grouped.values,color=color,marker='o',label=label)
        axes[0].set(xlabel='Epoch',ylabel='Groepen [%]',xticks=range(6),ylim=(-2,102));axes[1].set(xlabel='Epoch',ylabel='Projecties binnen de epoch',xticks=range(1,6),ylim=(0,None))
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01),fontsize=8.5)
        fig.text(.5,.86,'Een groep kan meermaals geprojecteerd worden. Een projectie is geen telling van unieke vastgelopen groepen.',ha='center',fontsize=9,color='#56626A')
        return fig


def plot_initialization_updates(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,2,figsize=(13,8));fig.subplots_adjust(top=.84,bottom=.13,hspace=.40,wspace=.24)
        fig.suptitle(_title(run,'Schaalupdates en harde omschakelingen · diagnostiek per epoch'),fontsize=15,fontweight='bold',y=.98)
        variants=_variants(run)
        if not variants:
            for ax in axes.flat:_empty(ax,'Nog geen bevestigde schaaldiagnostiek')
            return fig
        for label,r,color in variants:
            if not r.scales_safe or r.scale_epochs.empty:continue
            table=r.scale_epochs.loc[r.scale_epochs.epoch.le(run.last_epoch)]
            grouped=table.groupby('epoch')
            axes[0,0].plot(grouped.theta_changed_blocks.sum().index,grouped.theta_changed_blocks.sum().values,color=color,marker='o',label=label)
            norm=grouped.theta_update_l2.apply(lambda x:float(np.sqrt(np.square(x).sum())))
            axes[0,1].plot(norm.index,norm.values,color=color,marker='o',label=label)
            axes[1,0].plot(grouped.choice_flips_during_epoch.sum().index,grouped.choice_flips_during_epoch.sum().values,color=color,marker='o',label=label+' · flips')
            axes[1,0].plot(grouped.unique_blocks_switched_during_epoch.sum().index,grouped.unique_blocks_switched_during_epoch.sum().values,color=color,marker='x',ls='--',label=label+' · unieke groepen')
            maxnorm=grouped.scale_grad_l2_max.max()
            axes[1,1].plot(maxnorm.index,maxnorm.values,color=color,marker='o',label=label)
        titles=['Gewijzigde continue parameters','Netto θ-verplaatsing over alle groepen','Keuzewisselingen binnen de epoch','Grootste geregistreerde laag-gradientnorm']
        ylabs=['Groepen','L2-norm van θ_eind − θ_begin','Aantal','Max. laag-L2-norm, over alle batches']
        for ax,title,ylabel in zip(axes.flat,titles,ylabs):ax.set(title=title,xlabel='Epoch',ylabel=ylabel,xticks=range(1,6),ylim=(0,None))
        axes[1,0].legend(frameon=False,fontsize=8)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.01))
        return fig


def initialization_cost_table(run=None):
    return pd.DataFrame([
        ('FP32 schaalparameters',4080,'Beide initialisaties hebben hetzelfde aantal trainbare parameters.'),
        ('Gepakte keuze-informatie [bytes]',2040,'Vier bits per B32-groep, tien mogelijke offsets.'),
        ('θ + gradient + twee Adam-momenten [bytes]',65280,'Analytisch 4 × 4.080 × 4; exclusief framework en tijdelijke buffers.'),
        ('Nieuwe trainingspasses per minibatch',3,'E4M3 → E2M3 → E2M1; dezelfde route als de init×64-pilot.')
    ],columns=['Onderdeel','Waarde','Betekenis'])
