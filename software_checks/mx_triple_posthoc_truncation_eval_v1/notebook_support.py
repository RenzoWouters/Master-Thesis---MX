"""Read-only analysis of saved triple-MX post-hoc truncation evaluations.

Only JSON/CSV/NPZ are read. No torch, model, checkpoint or dataset is imported.
An explicit save_figures call creates a fresh directory, never edits raw input.
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

PATHS = ("direct8", "direct6", "direct4", "round64", "mantissa64", "round86", "exponent86", "combined864")
ROLES = ("last_epoch", "best_E4M3_test")
BITS = {"direct8": 8, "direct6": 6, "direct4": 4, "round64": 4, "mantissa64": 4,
        "round86": 6, "exponent86": 6, "combined864": 4}
BANKS = {8: 8, 6: 32, 4: 4}
BASELINES = {p: "direct" + str(BITS[p]) for p in PATHS}
LABELS = {"direct8": "Direct E4M3", "direct6": "Direct E2M3", "direct4": "Direct E2M1",
          "round64": "Afronden E2M3 → E2M1", "mantissa64": "Mantissatruncatie E2M3 → E2M1",
          "round86": "Afronden E4M3 → E2M3; E4-schaal vast",
          "exponent86": "Exponenttruncatie E4M3 → E2M3",
          "combined864": "Exponent + mantissa E4M3 → E2M1"}
SHORT = {"direct8": "Direct E4M3", "direct6": "Direct E2M3", "direct4": "Direct E2M1",
         "round64": "Afronden 6→4", "mantissa64": "Mantissa 6→4",
         "round86": "Afronden 8→6", "exponent86": "Exponent 8→6", "combined864": "Combinatie 8→6→4"}
COLORS = {"direct8": "#245A81", "direct6": "#33846D", "direct4": "#C98220",
          "round64": "#D8B47B", "mantissa64": "#B5791B", "round86": "#8CBCAE",
          "exponent86": "#25755B", "combined864": "#805B9A"}
PRIMARY = ("mantissa64", "exponent86", "combined864")
DERIVED = PATHS[3:]
STYLE = {"axes.spines.top": False, "axes.spines.right": False,
         "axes.titleweight": "bold", "axes.labelsize": 10, "font.size": 10,
         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
         "axes.grid": True, "grid.alpha": .15}


@dataclass
class RunAnalysis:
    path: Path
    report: dict = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    codes: pd.DataFrame = field(default_factory=pd.DataFrame)
    reproduction: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list[dict] = field(default_factory=list)
    output_checks: list[dict] = field(default_factory=list)
    input_files: list[Path] = field(default_factory=list)
    metrics_safe: bool = True

    @property
    def synthetic(self):
        return self.report.get("synthetic_fixture_only") is True or self.report.get("synthetic_checks_only") is True

    @property
    def confirmed(self):
        return (self.report.get("status") == "PASS" and self.report.get("command") == "evaluate"
                and self.metrics_safe and len(self.metrics) == len(PATHS) * len(ROLES)
                and len(self.output_checks) == len(ROLES)
                and not any(item["Niveau"] in ("FOUT", "RUNFOUT", "ONTBREEKT") for item in self.issues))

    def issue(self, level, message):
        self.issues.append({"Niveau": level, "Melding": message})

    def epoch(self, role):
        if self.metrics.empty or not self.metrics_safe:
            return None
        rows = self.metrics.loc[self.metrics["role"] == role, "epoch"].drop_duplicates()
        return int(rows.iloc[0]) if len(rows) == 1 else None


def _finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def _safe_child(root, name):
    path = (root / str(name)).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Bestandspad moet binnen de resultaatmap blijven.")
    return path


def load_run(run_dir, verify_per_image=True):
    run = RunAnalysis(Path(run_dir).expanduser().resolve())
    files = {"run_report.json": "report", "metrics.csv": "metrics", "paired_metrics.csv": "pairs",
             "weight_pair_metrics.csv": "weights", "layer_code_metrics.csv": "codes",
             "historical_reproduction.csv": "reproduction"}
    for name, attr in files.items():
        path = run.path / name
        if not path.is_file():
            if name != "historical_reproduction.csv":
                run.issue("ONTBREEKT", name)
            continue
        run.input_files.append(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig")) if name.endswith(".json") else pd.read_csv(path)
            if attr == "report" and not isinstance(value, dict):
                raise ValueError("run_report.json moet een object zijn.")
            setattr(run, attr, value)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            run.issue("FOUT", f"{name}: {exc}")
    if run.report.get("command") == "check":
        run.issues = [x for x in run.issues if x["Niveau"] != "ONTBREEKT"]
        run.issue("INFO", "Dit is een softwarecheck; er zijn geen post-hoctestaccuracies.")
        return run
    if run.report.get("optimizer_steps") != 0:
        run.issue("FOUT", "Het rapport bevestigt geen nul optimizerupdates.")
    if run.report.get("batchnorm_recalibration") is not False:
        run.issue("FOUT", "Het rapport bevestigt niet dat BN-herkalibratie uitstaat.")
    if run.report.get("synthetic_fixture_only") is not False and not run.synthetic:
        run.issue("FOUT", "Herkomst ontbreekt: niet als echte SVHN-evaluatie bevestigen.")
    _validate_metrics(run)
    _validate_pairs(run)
    _validate_roles(run)
    _validate_weight_rows(run)
    if verify_per_image and run.metrics_safe and not run.metrics.empty:
        _verify_outputs(run)
    elif not verify_per_image:
        run.issue("LET OP", "Per-beeldbestanden zijn niet gecontroleerd; deze weergave is alleen oriënterend.")
    for warning in run.report.get("warnings", []):
        run.issue("RUNWAARSCHUWING", warning.get("message", str(warning)) if isinstance(warning, dict) else str(warning))
    if run.report.get("error"):
        run.issue("RUNFOUT", str(run.report["error"]))
    return run


def _validate_metrics(run):
    if run.metrics.empty:
        run.metrics_safe = False
        return
    required = {"role", "epoch", "path", "weight_bits", "activation_bits", "bn_bank_key", "n",
                "correct1", "correct5", "top1_percent", "top5_percent", "hard_cross_entropy",
                "optimizer_steps", "batchnorm_recalibrated"}
    if not required <= set(run.metrics):
        run.issue("FOUT", "Metricvelden ontbreken: " + ", ".join(sorted(required - set(run.metrics))))
        run.metrics_safe = False
        return
    data = run.metrics
    numbers = ("epoch", "weight_bits", "activation_bits", "bn_bank_key", "n", "correct1", "correct5",
               "top1_percent", "top5_percent", "hard_cross_entropy", "optimizer_steps")
    for key in numbers:
        data[key] = pd.to_numeric(data[key], errors="coerce")
    if not np.isfinite(data[list(numbers)].to_numpy(dtype=float)).all():
        run.issue("FOUT", "Niet-eindige of ontbrekende essentiële metrics.")
        run.metrics_safe = False
        return
    if set(data["role"]) != set(ROLES) or data.duplicated(["role", "path"]).any():
        run.issue("FOUT", "Verwacht exact twee checkpointrollen zonder dubbele routes.")
        run.metrics_safe = False
    for role in ROLES:
        frame = data.loc[data["role"] == role]
        if set(frame["path"]) != set(PATHS) or len(frame) != len(PATHS):
            run.issue("FOUT", f"{role}: verwacht precies de acht vastgelegde evaluatieroutes.")
            run.metrics_safe = False
        if len(frame["epoch"].unique()) != 1 or (frame["epoch"] % 1 != 0).any():
            run.issue("FOUT", f"{role}: routes komen niet uit één volledige checkpointepoch.")
            run.metrics_safe = False
        if frame["n"].nunique() != 1:
            run.issue("FOUT", f"{role}: routes hebben een verschillend aantal testbeelden.")
            run.metrics_safe = False
    if not (data["activation_bits"] == 32).all() or not (data["optimizer_steps"] == 0).all():
        run.issue("FOUT", "De evaluatie moet A32 en nul optimizerupdates gebruiken.")
    if not data["batchnorm_recalibrated"].map(_bool).eq(False).all():
        run.issue("FOUT", "De metricrijen bevestigen geen bevroren BN-statistieken.")
    for path in PATHS:
        frame = data.loc[data["path"] == path]
        if not (frame["weight_bits"] == BITS[path]).all() or not (frame["bn_bank_key"] == BANKS[BITS[path]]).all():
            run.issue("FOUT", path + ": verkeerde doelprecisie of BN-bank.")
    if ((data["n"] <= 0) | (data["n"] % 1 != 0) | (data["correct1"] < 0)
            | (data["correct5"] < data["correct1"]) | (data["correct5"] > data["n"])
            | (data["hard_cross_entropy"] < 0)).any():
        run.issue("FOUT", "Ongeldige aantallen of cross-entropy.")
    for name, count in (("top1_percent", "correct1"), ("top5_percent", "correct5")):
        if not np.allclose(data[name], 100 * data[count] / data["n"], rtol=0, atol=1e-8):
            run.issue("FOUT", name + " strookt niet met correcte aantallen.")
    if run.metrics_safe and run.epoch("last_epoch") != 100:
        run.issue("FOUT", "De vaste laatste referentierol moet epoch 100 zijn.")
    if not run.synthetic and not data["n"].eq(26032).all():
        run.issue("FOUT", "Een echte proef moet alle 26.032 SVHN-testbeelden bevatten.")


def _validate_pairs(run):
    if run.pairs.empty:
        return
    required = {"role", "epoch", "reference_path", "other_path", "n", "accuracy_difference_pp",
                "prediction_changes", "wrong_to_correct", "correct_to_wrong", "wrong_to_other_wrong", "net_correct_change"}
    if not required <= set(run.pairs):
        run.issue("FOUT", "Onvolledige gepaarde uitvoer: " + ", ".join(sorted(required - set(run.pairs))))
        return
    for _, row in run.pairs.iterrows():
        if row["reference_path"] not in PATHS or row["other_path"] not in PATHS or row["role"] not in ROLES:
            run.issue("FOUT", "Onbekende route of rol in de gepaarde vergelijking.")
            continue
        values = [row[k] for k in ("n", "accuracy_difference_pp", "prediction_changes", "wrong_to_correct",
                                    "correct_to_wrong", "wrong_to_other_wrong", "net_correct_change")]
        if not all(_finite(x) is not None for x in values) or row["n"] <= 0:
            run.issue("FOUT", "Niet-eindige of ongeldige gepaarde tellingen.")
            continue
        if row["net_correct_change"] != row["wrong_to_correct"] - row["correct_to_wrong"]:
            run.issue("FOUT", "Netto correcte verandering strookt niet met de gepaarde overgangen.")
        if row["prediction_changes"] != row["wrong_to_correct"] + row["correct_to_wrong"] + row["wrong_to_other_wrong"]:
            run.issue("FOUT", "Aantal gewijzigde voorspellingen strookt niet met de overgangen.")
        if not math.isclose(row["accuracy_difference_pp"], 100 * row["net_correct_change"] / row["n"], abs_tol=1e-8):
            run.issue("FOUT", "Gepaarde accuracydelta strookt niet met de correcte aantallen.")
        if run.metrics_safe and not run.metrics.empty:
            frame = run.metrics.loc[run.metrics["role"] == row["role"]].set_index("path")
            if row["reference_path"] in frame.index and row["other_path"] in frame.index:
                expected = frame.loc[row["other_path"], "top1_percent"] - frame.loc[row["reference_path"], "top1_percent"]
                if row["epoch"] != run.epoch(row["role"]) or not math.isclose(expected, row["accuracy_difference_pp"], abs_tol=1e-8):
                    run.issue("FOUT", "Gepaarde rij gebruikt een andere epoch of inconsistente accuracydelta.")


def _validate_roles(run):
    roles = run.report.get("roles", {})
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        run.issue("FOUT", "Rapport bevat niet precies de twee vaste checkpointrollen.")
        return
    selection = run.report.get("reference", {}).get("checkpoint_selection", {})
    if not isinstance(selection, dict) or not {"best", "last"} <= set(selection):
        run.issue("FOUT", "De oorspronkelijke best/last-checkpointselectie ontbreekt.")
        selection = {}
    for role in ROLES:
        details = roles[role]
        if run.metrics_safe and not run.metrics.empty and details.get("epoch") != run.epoch(role):
            run.issue("FOUT", role + ": checkpointepoch verschilt tussen rapport en metrics.")
        original = selection.get("last" if role == "last_epoch" else "best", {})
        if original and (details.get("epoch") != original.get("epoch")
                         or details.get("checkpoint_sha256") != original.get("sha256")):
            run.issue("FOUT", role + ": oorspronkelijke checkpointselectie is niet behouden.")
        for key in ("all_model_and_bn_states_unchanged", "all_models_no_grad",
                    "six_FP32_master_parameters_absent_all_paths"):
            if details.get(key) is not True:
                run.issue("FOUT", role + ": audit niet bevestigd: " + key)
        if details.get("optimizer_steps") != 0:
            run.issue("FOUT", role + ": geen nul optimizerupdates bevestigd.")


def _validate_weight_rows(run):
    layers = {"layers.3", "layers.6", "layers.10", "layers.13", "layers.16", "layers.20"}
    required = {"role", "epoch", "layer", "path", "target_bits", "bn_bank_key", "n_weights",
                "mean_abs", "rms", "zero_count", "scales_equal_source"}
    if run.codes.empty or not required <= set(run.codes):
        run.issue("FOUT", "Laagdiagnostiek ontbreekt of mist code-/schaalvelden.")
    else:
        data = run.codes
        if data.duplicated(["role", "layer", "path"]).any():
            run.issue("FOUT", "Dubbele laagdiagnostiek voor hetzelfde checkpoint en route.")
        for role in ROLES:
            for route in PATHS:
                rows = data.loc[(data["role"] == role) & (data["path"] == route)]
                if set(rows["layer"]) != layers or len(rows) != len(layers):
                    run.issue("FOUT", f"{role}/{route}: niet precies de zes MX-lagen.")
                if not rows["target_bits"].eq(BITS[route]).all() or not rows["bn_bank_key"].eq(BANKS[BITS[route]]).all():
                    run.issue("FOUT", f"{role}/{route}: laagdiagnostiek gebruikt een verkeerde doelmodus.")
                if run.metrics_safe and not run.metrics.empty and not rows["epoch"].eq(run.epoch(role)).all():
                    run.issue("FOUT", f"{role}/{route}: laagdiagnostiek gebruikt een andere epoch.")
        if not data["scales_equal_source"].map(_bool).eq(True).all():
            run.issue("FOUT", "Niet alle routes behouden hun vastgelegde bronschaal.")
        numeric = data[["n_weights", "mean_abs", "rms", "zero_count"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric < 0).any().any() or (numeric["zero_count"] > numeric["n_weights"]).any():
            run.issue("FOUT", "Ongeldige gewichtsamplitudes of nultellingen.")
    pair_required = {"role", "epoch", "layer", "reference_path", "other_path", "mae", "rmse", "max_abs_difference"}
    if run.weights.empty or not pair_required <= set(run.weights):
        run.issue("FOUT", "Gewichtsvergelijkingen ontbreken of zijn onvolledig.")
    else:
        values = run.weights[["mae", "rmse", "max_abs_difference"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any().any():
            run.issue("FOUT", "Ongeldige numerieke gewichtsafwijkingen.")


def _verify_outputs(run):
    for role in ROLES:
        details = run.report.get("roles", {}).get(role, {})
        if not details.get("per_image_file"):
            run.issue("FOUT", role + ": per-beeldbestand ontbreekt in het rapport.")
            continue
        try:
            path = _safe_child(run.path, details["per_image_file"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != details.get("per_image_sha256"):
                raise ValueError("Checksum per-beeldbestand wijkt af.")
            run.input_files.append(path)
            with np.load(path, allow_pickle=False) as archive:
                labels, indices = archive["labels"], archive["indices"]
                n = len(labels)
                if labels.shape != (n,) or indices.shape != (n,) or not np.array_equal(indices, np.arange(n)):
                    raise ValueError("Indices dekken de testset niet exact eenmaal in volgorde.")
                if not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels > 9)).any():
                    raise ValueError("Ongeldige labels.")
                if n != details.get("n_test"):
                    raise ValueError("Aantal per-beeldrijen wijkt af van het rolrapport.")
                if details.get("label_sha256") and hashlib.sha256(labels.tobytes()).hexdigest() != details["label_sha256"]:
                    raise ValueError("Labelhash wijkt af.")
                predictions = {}
                for route in PATHS:
                    pred = archive[route + "__predictions"]
                    logits = archive[route + "__logits"]
                    ce = archive[route + "__per_image_ce"]
                    if pred.shape != (n,) or logits.shape != (n, 10) or ce.shape != (n,):
                        raise ValueError(route + ": onjuiste vorm van per-beeldarrays.")
                    if not np.issubdtype(pred.dtype, np.integer) or ((pred < 0) | (pred > 9)).any():
                        raise ValueError(route + ": ongeldige voorspellingen.")
                    if not np.isfinite(logits).all() or not np.isfinite(ce).all() or (ce < 0).any():
                        raise ValueError(route + ": ongeldige logits of losses.")
                    # torch.topk tie ordering can differ from NumPy argmax.
                    if not np.array_equal(logits[np.arange(n), pred], logits.max(axis=1)):
                        raise ValueError(route + ": voorspelling is geen logitmaximum.")
                    rows = run.metrics.loc[(run.metrics["role"] == role) & (run.metrics["path"] == route)]
                    if len(rows) != 1:
                        raise ValueError(route + ": geen unieke metricrij.")
                    row = rows.iloc[0]
                    if n != row["n"] or int((pred == labels).sum()) != row["correct1"]:
                        raise ValueError(route + ": correcte aantallen wijken af.")
                    if not math.isclose(float(ce.astype(np.float64).mean()), row["hard_cross_entropy"], rel_tol=0, abs_tol=1e-10):
                        raise ValueError(route + ": per-beeld-CE wijkt af van de metric.")
                    predictions[route] = pred
                if not run.pairs.empty and {"role", "reference_path", "other_path"} <= set(run.pairs):
                    for _, row in run.pairs.loc[run.pairs["role"] == role].iterrows():
                        if row["reference_path"] not in predictions or row["other_path"] not in predictions:
                            continue
                        ref, other = predictions[row["reference_path"]], predictions[row["other_path"]]
                        rc, oc = ref == labels, other == labels
                        actual = {"wrong_to_correct": int((~rc & oc).sum()), "correct_to_wrong": int((rc & ~oc).sum()),
                                  "wrong_to_other_wrong": int((~rc & ~oc & (ref != other)).sum()),
                                  "prediction_changes": int((ref != other).sum())}
                        if any(row.get(k) != value for k, value in actual.items()):
                            raise ValueError("Gepaarde tellingen wijken af van de per-beeldvoorspellingen.")
            run.output_checks.append({"Rol": role, "Epoch": run.epoch(role), "Testbeelden": n,
                                      "NPZ-checksum": "PASS", "Acht routes": "PASS", "Gepaarde tellingen": "PASS"})
        except (OSError, KeyError, ValueError, TypeError, IndexError) as exc:
            run.issue("FOUT", role + ": " + str(exc))


def diagnostics_table(run):
    return pd.DataFrame(run.issues or [{"Niveau": "OK", "Melding": "Alle ingelezen consistentiecontroles geslaagd."}])


def status_markdown(run):
    origin = "**SYNTHETISCHE SOFTWARECONTROLE — GEEN SVHN-RESULTATEN.**\n\n" if run.synthetic else ""
    if run.confirmed:
        state = "**Post-hocevaluatie bevestigd.** Beide vaste checkpointrollen en alle acht routes zijn gecontroleerd."
    else:
        state = "**Geen volledig bevestigde post-hocevaluatie.** Bekijk de meldingen voordat je scores interpreteert."
    return (origin + state + "\n\nDeze notebook leest opgeslagen rapporten en per-beeldarrays. "
            "Zij voert geen training, modelevaluatie of BN-herkalibratie uit. De selectie blijft epoch 100 "
            "en het eerder gekozen E4M3-bestcheckpoint; er wordt geen nieuw bestcheckpoint gezocht.")


def route_table():
    method = {"direct8": "Direct uit FP32 G(W)", "direct6": "Direct uit FP32 G(W)", "direct4": "Direct uit FP32 G(W)",
              "round64": "RNE op E2M3-elementen", "mantissa64": "Twee mantissa-LSB's verwijderen",
              "round86": "RNE op E4M3-elementen; saturatiecontrole", "exponent86": "Twee hoogste exponentbits behouden",
              "combined864": "Eerst exponent-MSB-truncatie; daarna mantissa-LSB-truncatie"}
    scale = {"direct8": "Eigen directe E4M3-schaal", "direct6": "Eigen directe E2M3-schaal", "direct4": "Eigen directe E2M1-schaal",
             "round64": "E2M3-bronschaal behouden", "mantissa64": "E2M3-bronschaal behouden",
             "round86": "E4M3-bronschaal behouden", "exponent86": "E4M3-bronschaal behouden", "combined864": "E4M3-bronschaal behouden"}
    return pd.DataFrame([{"Route": p, "Beschrijving": LABELS[p], "Doelbits": BITS[p], "A": 32,
                          "BN-bank": str(BANKS[BITS[p]]), "Bewerking": method[p], "Schaal": scale[p],
                          "Directe vergelijking": BASELINES[p]} for p in PATHS])


def accuracy_table(run):
    if run.metrics.empty or not run.metrics_safe:
        return pd.DataFrame()
    data = run.metrics.copy()
    data["Beschrijving"] = data["path"].map(LABELS)
    data["_role"] = data["role"].map({role: i for i, role in enumerate(ROLES)})
    data["_path"] = data["path"].map({path: i for i, path in enumerate(PATHS)})
    return data.sort_values(["_role", "_path"])[["role", "epoch", "path", "Beschrijving", "weight_bits", "bn_bank_key",
                                                "correct1", "n", "top1_percent", "hard_cross_entropy"]]


def delta_table(run):
    rows = []
    if not run.metrics_safe or run.metrics.empty:
        return pd.DataFrame()
    for role in ROLES:
        frame = run.metrics.loc[run.metrics["role"] == role].set_index("path")
        for path in DERIVED:
            if path not in frame.index or BASELINES[path] not in frame.index:
                continue
            row, baseline = frame.loc[path], frame.loc[BASELINES[path]]
            rows.append({"role": role, "epoch": int(row["epoch"]), "path": path, "Beschrijving": LABELS[path],
                         "Directe referentie": LABELS[BASELINES[path]],
                         "Delta accuracy [pp]": row["top1_percent"] - baseline["top1_percent"],
                         "Netto correcte beelden": int(row["correct1"] - baseline["correct1"])})
    return pd.DataFrame(rows)


def _title(run, text):
    return ("SYNTHETISCHE CONTROLE · " if run.synthetic else "") + text


def _role_title(run, role):
    epoch = run.epoch(role)
    return ("Vast eindpunt" if role == "last_epoch" else "Eerder gekozen E4M3-best") + (f" — epoch {epoch}" if epoch else "")


def _empty(ax, message):
    ax.set_axis_off()
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", wrap=True,
            fontsize=11, color="#525D65", bbox=dict(boxstyle="round,pad=.9", facecolor="#F1F4F6", edgecolor="none"))


def plot_checkpoint_accuracy(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(15.3, 6.2), constrained_layout=True)
        fig.suptitle(_title(run, "Exponent- en mantissatruncatie op dezelfde checkpoints"), fontsize=14, fontweight="bold")
        for ax, role in zip(axes, ROLES):
            ax.set_title(_role_title(run, role))
            if not run.confirmed:
                _empty(ax, "Geen volledig bevestigde resultaten\nZie status en consistentiecontroles")
                continue
            data = run.metrics.loc[run.metrics["role"] == role].set_index("path").loc[list(PATHS)]
            ys = np.arange(len(PATHS))
            ax.barh(ys, data["top1_percent"], color=[COLORS[p] for p in PATHS], height=.66)
            for y, score in zip(ys, data["top1_percent"]):
                ax.text(min(score + 1, 100.2), y, f"{score:.4f}%", va="center", fontsize=9)
            ax.set(yticks=ys, yticklabels=[SHORT[p] for p in PATHS], xlabel="Testaccuracy [%]", xlim=(0, 110))
            ax.set_xticks(range(0, 101, 20))
            ax.invert_yaxis()
            for y in (2.5, 4.5, 6.5):
                ax.axhline(y, color="#D5DCE0", linewidth=.8)
        fig.supxlabel("Directe controles · mantissa · exponent met vaste E4-schaal · combinatie | A32, bevroren doel-BN", fontsize=10, color="#525D65")
        return fig


def plot_accuracy_deltas(run):
    data = delta_table(run)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        fig.suptitle(_title(run, "Verschil tegenover directe kwantisatie naar dezelfde doelprecisie"), fontsize=14, fontweight="bold")
        for ax, role in zip(axes, ROLES):
            ax.set_title(_role_title(run, role))
            if not run.confirmed or data.empty:
                _empty(ax, "Geen bevestigde accuracyverschillen")
                continue
            frame = data.loc[data["role"] == role].set_index("path").loc[list(DERIVED)]
            values = frame["Delta accuracy [pp]"].to_numpy()
            ys = np.arange(len(DERIVED))
            ax.barh(ys, values, color=[COLORS[p] for p in DERIVED], height=.63)
            ax.axvline(0, color="#525D65", linewidth=.8)
            lo, hi = min(float(values.min()), 0.), max(float(values.max()), 0.)
            margin = max((hi - lo) * .18, .12)
            ax.set(xlim=(lo - margin, hi + margin), yticks=ys, yticklabels=[SHORT[p] for p in DERIVED], xlabel="Route − directe doelmodus [procentpunt]")
            ax.invert_yaxis()
            for y, value in zip(ys, values):
                ax.annotate(f"{value:+.4f}", (value, y), xytext=((5 if value >= 0 else -5), 0),
                            textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=9)
        return fig


def plot_paired_changes(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
        fig.suptitle(_title(run, "Welke voorspellingen veranderen door truncatie?"), fontsize=14, fontweight="bold")
        for ax, role in zip(axes, ROLES):
            ax.set_title(_role_title(run, role))
            if not run.confirmed or run.pairs.empty:
                _empty(ax, "Geen bevestigde gepaarde tellingen")
                continue
            rows = []
            for p in PRIMARY:
                data = run.pairs.loc[(run.pairs["role"] == role) & (run.pairs["other_path"] == p)
                                     & (run.pairs["reference_path"] == BASELINES[p])]
                if len(data) == 1:
                    rows.append(data.iloc[0])
            if len(rows) != len(PRIMARY):
                _empty(ax, "Primaire gepaarde vergelijkingen ontbreken")
                continue
            ys = np.arange(len(PRIMARY))
            ax.barh(ys - .17, [r["wrong_to_correct"] for r in rows], height=.3, color="#33846D", label="Fout → correct")
            ax.barh(ys + .17, [-r["correct_to_wrong"] for r in rows], height=.3, color="#B44040", label="Correct → fout")
            ax.axvline(0, color="#525D65", linewidth=.8)
            limit = max([1] + [r[k] for r in rows for k in ("wrong_to_correct", "correct_to_wrong")]) * 1.15
            ax.set(yticks=ys, yticklabels=[SHORT[p] for p in PRIMARY], xlabel="Aantal beelden; verlies links, winst rechts",
                   xlim=(-limit, limit))
            ax.invert_yaxis()
            ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, -.32), ncol=2)
        return fig


def plot_weight_differences(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, "Gedecodeerde gewichtsafwijking tegenover de directe doelmodus"), fontsize=14, fontweight="bold")
        required = {"role", "layer", "reference_path", "other_path", "mae"}
        for ax, role in zip(axes, ROLES):
            ax.set_title(_role_title(run, role))
            if not run.confirmed or run.weights.empty or not required <= set(run.weights):
                _empty(ax, "Geen bruikbare gewichtsvergelijkingen")
                continue
            plotted = False
            for route in PRIMARY:
                frame = run.weights.loc[(run.weights["role"] == role) & (run.weights["other_path"] == route)
                                         & (run.weights["reference_path"] == BASELINES[route])]
                if frame.empty:
                    continue
                ax.plot(frame["layer"], frame["mae"], marker="o", color=COLORS[route], label=SHORT[route])
                plotted = True
            if plotted:
                ax.set(xlabel="MX-convolutielaag", ylabel="MAE van gedecodeerde gewichten", ylim=(0, None))
                ax.tick_params(axis="x", rotation=30)
                ax.legend(frameon=False)
            else:
                _empty(ax, "Primaire gewichtsvergelijkingen ontbreken")
        return fig


def plot_weight_magnitudes(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, "Gewichtsgrootte na de codebewerking, met de bronschaal behouden"), fontsize=14, fontweight="bold")
        data = run.codes
        magnitude_column = next((x for x in ("decoded_rms", "rms", "decoded_mean_abs", "mean_abs") if x in data), None)
        for ax, role in zip(axes, ROLES):
            ax.set_title(_role_title(run, role))
            if not run.confirmed or data.empty or not {"role", "layer", "path"} <= set(data) or magnitude_column is None:
                _empty(ax, "Geen gewichtsamplitudes in deze rapportage")
                continue
            plotted = False
            for route in PRIMARY:
                frame = data.loc[(data["role"] == role) & (data["path"] == route), ["layer", magnitude_column]]
                base = data.loc[(data["role"] == role) & (data["path"] == BASELINES[route]), ["layer", magnitude_column]]
                merged = frame.merge(base, on="layer", suffixes=("_route", "_direct"), validate="one_to_one")
                if merged.empty:
                    continue
                numerator = merged[magnitude_column + "_route"].to_numpy(dtype=float)
                denominator = merged[magnitude_column + "_direct"].to_numpy(dtype=float)
                ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
                ax.plot(merged["layer"], ratio, marker="o", color=COLORS[route], label=SHORT[route])
                plotted = True
            if plotted:
                ax.axhline(1, color="#525D65", linestyle="--", linewidth=.8)
                ax.set(xlabel="MX-convolutielaag", ylabel="RMS-ratio route / directe doelmodus" if "rms" in magnitude_column else "Gemiddelde |w|-ratio route / directe doelmodus", ylim=(0, None))
                ax.tick_params(axis="x", rotation=30)
                ax.legend(frameon=False)
            else:
                _empty(ax, "Geen vergelijkbare amplituderijen")
        fig.supxlabel("Dit is een diagnose van de opgeslagen gewichten; er wordt geen schaalcorrectie toegepast.", fontsize=10, color="#525D65")
        return fig


def audit_table(run):
    rows = []
    for role, details in run.report.get("roles", {}).items():
        rows.append({"Rol": role, "Epoch": details.get("epoch"), "Model/BN ongewijzigd": details.get("all_model_and_bn_states_unchanged"),
                     "Geen gradients": details.get("all_models_no_grad"), "Optimizerupdates": details.get("optimizer_steps"),
                     "Zes FP32-masters afwezig": details.get("six_FP32_master_parameters_absent_all_paths"),
                     "Checkpoint-SHA256": details.get("checkpoint_sha256")})
    return pd.DataFrame(rows)


def timing_markdown(run):
    seconds = _finite(run.report.get("session_wall_seconds"))
    measured = f"**Gemeten evaluatiesessie: {seconds / 60:.2f} minuten.** " if seconds is not None else "Geen gemeten sessieduur beschikbaar. "
    return measured + "Deze tijd omvat de evaluatieroutes en administratie van deze proef. Zij is geen native-MX-snelheids- of energiebenchmark."


def save_figures(run, figures):
    """Save only on explicit request, in a new directory; preserve all inputs."""
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.input_files}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    target = run.path / ("analysis_" + stamp)
    target.mkdir(exist_ok=False)
    for name, figure in figures.items():
        if not name or not all(c.isalnum() or c in "_-" for c in name):
            raise ValueError("Ongeldige figuurnaam.")
        figure.savefig(target / (name + ".png"), dpi=180, bbox_inches="tight")
        figure.savefig(target / (name + ".pdf"), bbox_inches="tight")
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.input_files}
    if before != after:
        raise RuntimeError("Een invoerbestand is tijdens de analyse gewijzigd.")
    provenance = {"schema": "mx-triple-posthoc-analysis-v1", "synthetic_fixture_only": run.synthetic,
                  "source_directory": str(run.path), "confirmed_evaluation": run.confirmed,
                  "input_sha256": before, "figures": list(figures), "raw_inputs_unchanged": True,
                  "training_or_model_evaluation_performed": False}
    (target / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return target
