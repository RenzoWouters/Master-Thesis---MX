"""Read-only source identity checks for the six agreed epoch-100 models.

The experiment is a data-independent address/byte simulation, not a model
forward. We therefore verify original checkpoint/export bytes by SHA256 and
never deserialize .pt/pickle files. A PASS does not claim a new numerical
revalidation of their weights or an execution on physical MX hardware.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
CATALOG_FILE = PACKAGE_ROOT / "data" / "source_catalog.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(path=None):
    catalog = json.loads(Path(path or CATALOG_FILE).read_text(encoding="utf-8"))
    if catalog.get("schema") != "mx-precision-switch-source-catalog-v1":
        raise ValueError("Onbekende broncatalogus.")
    ids = [item["model_id"] for item in catalog["models"]]
    expected = ["github5", "dual_direct", "dual_mantissa", "triple_direct",
                "triple_scale64_b32", "triple_scale64_b64"]
    if ids != expected or catalog.get("epoch") != 100:
        raise ValueError("Broncatalogus bevat niet de zes afgesproken epoch-100-modellen.")
    return catalog


def _read_report(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _report_errors(report, spec):
    errors = []
    checks = [
        (report.get("schema") == spec["report_schema"], "Verkeerd reportschema/model."),
        (report.get("status") == "PASS", "Brontraining heeft geen PASS-status."),
        (report.get("command") in ("run", "resume"), "Bron is geen training/hervatting."),
        (report.get("synthetic_fixture_only") is False, "Synthetische of onbekende bron verboden."),
        (report.get("full_training_completed") is True, "Brontraining is niet volledig voltooid."),
        (report.get("completed_epoch") == 100, "Brontraining is niet op epoch 100 voltooid."),
        (report.get("config") == spec["config"], "Trainingconfiguratie wijkt af van de vastgelegde bron."),
        (report.get("checkpoint_selection", {}).get("last") == spec["checkpoint_selection"]["last"],
         "Epoch-100-checkpointidentiteit wijkt af van de vastgelegde bron."),
        (report.get("mx_exports", {}).get("last_epoch", {}) == spec["last_mode_exports"],
         "MX-exportidentiteiten wijken af van de vastgelegde bron."),
    ]
    errors.extend(message for ok, message in checks if not ok)
    return errors


def _candidate_report(project_root, spec, override=None):
    """Prefer the pinned run; bounded discovery also supports moved run folders."""
    root = Path(project_root)
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            return None, ["Bronoverride moet een niet-leeg pad naar runmap of run_report.json zijn."]
        target = Path(override)
        if not target.is_absolute():
            target = root / target
        report = target if target.name == "run_report.json" else target / "run_report.json"
        if not report.is_file():
            return None, [f"run_report.json ontbreekt bij expliciete bron: {report}"]
        return report, []
    exact = root / spec["run_relative_path"] / "run_report.json"
    if exact.is_file():
        return exact, []
    candidates = []
    for family in ("mx_runs", "reference_runs"):
        base = root / family
        if not base.is_dir():
            continue
        # Only ordinary family/run/report locations: embedded reference reports
        # and copied feedback packages are not independent training candidates.
        for report_path in sorted(base.glob("*/*/run_report.json")):
            try:
                report = _read_report(report_path)
                last = report.get("checkpoint_selection", {}).get("last", {})
                if (report.get("schema") == spec["report_schema"]
                        and last.get("sha256") == spec["checkpoint_selection"]["last"]["sha256"]):
                    candidates.append(report_path)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) > 1:
        return None, ["Meerdere kopieen van deze bron gevonden; wijs de runmap expliciet aan in sources.json: "
                      + ", ".join(str(path.parent) for path in candidates)]
    return None, [f"Verwachte bron ontbreekt: {exact}. Gebruik sources.json als de runmap verplaatst is."]


def _artifact_path(run_dir, artifact):
    target = run_dir / artifact["file"]
    if target.is_file():
        return target
    fallback = artifact.get("fallback_relative")
    if fallback and (run_dir / fallback).is_file():
        return run_dir / fallback
    return target


def verify_sources(project_root, overrides=None, hash_files=True):
    """Return PASS/FAIL plus six per-model checks; never modify original sources.

    ``overrides`` is an optional mapping model_id -> run directory/report path,
    or the path to a JSON file containing that mapping. It only relocates the
    pinned models: it cannot replace their recorded configuration or hashes.
    ``hash_files=False`` validates report metadata only and labels that scope.
    """
    catalog = load_catalog()
    if isinstance(overrides, (str, Path)):
        overrides = json.loads(Path(overrides).read_text(encoding="utf-8-sig"))
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError("sources.json moet een object met modelnamen en runpaden bevatten.")
    known_ids = {item["model_id"] for item in catalog["models"]}
    unknown = sorted(set(overrides) - known_ids)
    if unknown:
        raise ValueError("Onbekende modelnamen in sources.json: " + ", ".join(unknown))
    result = {
        "schema": "mx-precision-switch-source-verification-v1",
        "status": "PASS", "project_root": str(Path(project_root).resolve()),
        "epoch": 100, "hash_files": bool(hash_files),
        "verification_scope": "report_and_artifact_sha256" if hash_files else "report_metadata_only",
        "pickle_deserialized": False, "numerical_weight_revalidation": False,
        "models": {}, "errors": [],
    }
    for spec in catalog["models"]:
        mid = spec["model_id"]
        entry = {"model_id": mid, "label": spec["label"], "status": "PASS", "epoch": 100,
                 "packed_export_available": spec["packed_export_available"],
                 "layout_scope": spec["layout_scope"], "artifacts": [], "errors": []}
        result["models"][mid] = entry
        path, errors = _candidate_report(project_root, spec, overrides.get(mid))
        entry["errors"].extend(errors)
        if path is not None:
            entry["report_path"] = str(path.resolve())
            entry["run_directory"] = str(path.parent.resolve())
            entry["reference_report_sha256"] = spec["reference_report_sha256"]
            try:
                report = _read_report(path)
                entry["report_sha256"] = sha256_file(path)
                entry["report_bytes_match_reference"] = entry["report_sha256"] == spec["reference_report_sha256"]
                entry["errors"].extend(_report_errors(report, spec))
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                entry["errors"].append("Rapport niet leesbaar: " + str(exc))
            if not entry["errors"]:
                for artifact in spec["artifacts"]:
                    target = _artifact_path(path.parent, artifact)
                    check = {**artifact, "path": str(target.resolve()), "status": "NOT_HASHED"}
                    entry["artifacts"].append(check)
                    if not hash_files:
                        check["exists"] = target.is_file()
                        continue
                    if not target.is_file():
                        check["status"] = "MISSING"
                        entry["errors"].append("Bronbestand ontbreekt: " + str(target))
                        continue
                    try:
                        check["actual_bytes"] = target.stat().st_size
                        check["actual_sha256"] = sha256_file(target)
                        if check["actual_sha256"] != artifact["sha256"]:
                            check["status"] = "HASH_MISMATCH"
                            entry["errors"].append("SHA256 wijkt af: " + str(target))
                        elif "bytes" in artifact and check["actual_bytes"] != artifact["bytes"]:
                            check["status"] = "SIZE_MISMATCH"
                            entry["errors"].append("Bestandsgrootte wijkt af: " + str(target))
                        else:
                            check["status"] = "PASS"
                    except OSError as exc:
                        check["status"] = "READ_ERROR"
                        entry["errors"].append("Bronbestand niet leesbaar: " + str(exc))
        if entry["errors"]:
            entry["status"] = "FAIL"
            result["errors"].extend(mid + ": " + error for error in entry["errors"])
    if result["errors"]:
        result["status"] = "FAIL"
    result["verified_file_count"] = sum(
        artifact["status"] == "PASS" for entry in result["models"].values() for artifact in entry["artifacts"])
    return result


def reference_source_summary():
    """Metadata for an explicitly labelled reference-only geometry execution."""
    catalog = load_catalog()
    return {
        "status": "REFERENCE_ONLY", "epoch": 100, "hash_files": False,
        "verification_scope": "frozen_reference_metadata_only",
        "pickle_deserialized": False, "numerical_weight_revalidation": False,
        "verified_file_count": 0, "errors": [],
        "models": {item["model_id"]: {
            "model_id": item["model_id"], "label": item["label"], "status": "REFERENCE_ONLY",
            "epoch": 100, "run_directory": item["original_output_directory"],
            "report_sha256": item["reference_report_sha256"],
            "packed_export_available": item["packed_export_available"],
            "layout_scope": item["layout_scope"],
            "artifacts": [{**artifact, "status": "NOT_HASHED"} for artifact in item["artifacts"]],
        } for item in catalog["models"]},
    }
