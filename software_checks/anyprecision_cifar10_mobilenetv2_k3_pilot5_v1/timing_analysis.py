"""Transparent extrapolation from complete five-precision update timings.

Percentiles describe observed minibatch times. Their extrapolations are scenario
estimates, not confidence intervals or guarantees of sustained laptop speed.
"""
from __future__ import annotations

import math
import statistics


BATCH_SIZE = 128
TRAIN_EXAMPLES = 50_000
VALIDATION_EXAMPLES = 10_000
TRAIN_BATCHES = math.ceil(TRAIN_EXAMPLES / BATCH_SIZE)
VALIDATION_BATCHES = math.ceil(VALIDATION_EXAMPLES / BATCH_SIZE)
TIMING_FIELDS = ("wall_s", "data_wait_s", "h2d_s", "compute_s")


def _quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _stats(values):
    if not values:
        return None
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p10": _quantile(values, .10), "p90": _quantile(values, .90),
            "min": min(values), "max": max(values)}


def _validate_rows(rows):
    cleaned = []
    for index, row in enumerate(rows):
        batch = row.get("batch_size")
        if isinstance(batch, bool) or not isinstance(batch, int) or not 1 <= batch <= BATCH_SIZE:
            raise ValueError(f"Row {index}: invalid batch_size; expected 1..128.")
        result = {"step": row.get("step", index + 1), "batch_size": batch}
        for name in TIMING_FIELDS:
            value = row.get(name, 0.0 if name in ("data_wait_s", "h2d_s") else None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Row {index}: {name} must be a finite nonnegative duration.")
            result[name] = float(value)
        if result["wall_s"] <= 0 or result["compute_s"] <= 0:
            raise ValueError(f"Row {index}: wall and compute timings must be positive.")
        if "precision_passes" in row and row["precision_passes"] != 5:
            raise ValueError("Each timing row must cover all five precision passes.")
        cleaned.append(result)
    return cleaned


def _describe(rows, warmup):
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("Warmup row counts must be nonnegative integers.")
    rows = _validate_rows(rows)
    after_warmup = rows[warmup:]
    eligible = [row for row in after_warmup if row["batch_size"] == BATCH_SIZE]
    partial = [row for row in after_warmup if row["batch_size"] != BATCH_SIZE]
    return {"observed_rows": len(rows), "observed_samples": sum(row["batch_size"] for row in rows),
            "warmup_requested_rows": warmup, "warmup_discarded_rows": min(warmup, len(rows)),
            "partial_rows_not_used_for_forecast": len(partial),
            "eligible_rows": len(eligible), "batch_size": BATCH_SIZE,
            "eligible_steps": [row["step"] for row in eligible],
            "available": bool(eligible),
            "stats": {name: _stats([row[name] for row in eligible]) for name in TIMING_FIELDS}}


def summarize_timing(train_rows, validation_rows=(), *, warmup_train_steps=10,
                     warmup_validation_steps=1, checkpoint_seconds=0.0, synthetic=False):
    """Summarize timings and estimate 5/100/400 complete epochs.

    Real-data ``wall_s`` must encompass data wait, host-to-device transfer,
    all five forward/backward passes, one Adam update and measured host overhead.
    Validation rows likewise cover all five precision forwards. In a synthetic
    check only ``compute_s`` is extrapolated and total epoch time stays unknown.

    A full epoch is conservatively counted as 391 train + 79 validation batches
    of 128. Thus the final train batch of 80 and validation batch of 16 are each
    counted as full batches; no precision multiplier is applied a second time.
    """
    if (isinstance(checkpoint_seconds, bool) or not isinstance(checkpoint_seconds, (int, float))
            or not math.isfinite(checkpoint_seconds) or checkpoint_seconds < 0):
        raise ValueError("checkpoint_seconds must be a finite nonnegative duration.")
    training = _describe(list(train_rows), warmup_train_steps)
    validation = _describe(list(validation_rows), warmup_validation_steps)
    basis = "compute_s" if synthetic else "wall_s"

    def scaled(description, count, field, statistic):
        stats = description["stats"][field]
        return None if stats is None else count * stats[statistic]

    train_seconds = scaled(training, TRAIN_BATCHES, basis, "mean")
    train_low = scaled(training, TRAIN_BATCHES, basis, "p10")
    train_high = scaled(training, TRAIN_BATCHES, basis, "p90")
    val_seconds = None if synthetic else scaled(validation, VALIDATION_BATCHES, "wall_s", "mean")
    val_low = None if synthetic else scaled(validation, VALIDATION_BATCHES, "wall_s", "p10")
    val_high = None if synthetic else scaled(validation, VALIDATION_BATCHES, "wall_s", "p90")
    complete = not synthetic and train_seconds is not None and val_seconds is not None
    checkpoint = float(checkpoint_seconds)
    total = train_seconds + val_seconds + checkpoint if complete else None
    low = train_low + val_low + checkpoint if complete else None
    high = train_high + val_high + checkpoint if complete else None

    def hours(value, epochs):
        return None if value is None else value * epochs / 3600.0

    projections = {}
    for epochs in (5, 100, 400):
        projections[str(epochs)] = {"epochs": epochs,
            "training_hours": hours(train_seconds, epochs),
            "validation_hours": hours(val_seconds, epochs),
            "checkpoint_hours": None if synthetic else hours(checkpoint, epochs),
            "total_hours": hours(total, epochs),
            "scenario_low_hours": hours(low, epochs),
            "scenario_high_hours": hours(high, epochs),
            "training_scenario_low_hours": hours(train_low, epochs),
            "training_scenario_high_hours": hours(train_high, epochs)}
    return {"schema_version": 1,
            "kind": "synthetic_compute_only" if synthetic else "real_data",
            "precision_passes_per_update": 5, "precision_multiplier_applied": 1,
            "training": training, "validation": validation,
            "full_epoch": {"train_examples": TRAIN_EXAMPLES, "validation_examples": VALIDATION_EXAMPLES,
                           "train_batches": TRAIN_BATCHES, "validation_batches": VALIDATION_BATCHES,
                           "batch_size": BATCH_SIZE, "train_last_batch": 80,
                           "validation_last_batch": 16, "checkpoints_per_epoch": 1},
            "per_epoch": {"training_seconds": train_seconds, "training_only_seconds": train_seconds,
                          "validation_seconds": val_seconds,
                          "checkpoint_seconds": None if synthetic else checkpoint,
                          "total_seconds": total, "scenario_low_seconds": low,
                          "scenario_high_seconds": high},
            "projections": projections,
            "estimate_complete": complete,
            "timing_basis": basis,
            "tail_batch_policy": "conservative: last partial batch counted as a full 128-image batch",
            "scenario_definition": "p10/p90 of observed steady full-batch times, extrapolated; not a confidence interval",
            "notes": [
                "One row already covers all five precisions. No extra factor of five is applied.",
                "One complete training epoch means all 50,000 train images; validation means all 10,000 official test images.",
                "Initial imports, dataset checksum/loading, worker startup, first CUDA warmup and notebook time are excluded from the extrapolation.",
                "Report/JSON output outside the supplied wall_s intervals is not included; the caller reports total loop wall time separately.",
                "Checkpoint estimate assumes one measured write per epoch; changed checkpoint policy changes the total.",
                "A short run cannot establish sustained thermal performance or final accuracy.",
                "Synthetic results exclude data loading, validation and checkpoint cost; use real-data results for a complete estimate."
                if synthetic else "Real-data estimate uses measured end-to-end minibatch wall time, including data wait and transfers.",
            ]}
