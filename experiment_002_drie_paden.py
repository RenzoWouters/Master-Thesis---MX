"""Experiment 002: drie omzettingspaden, met de opgeslagen bron van experiment 001.

A / direct:              oorspronkelijke invoer -> E2M1 met RNE.
B / rounded_from_source: opgeslagen E2M3 -> decoderen -> E2M1 met RNE.
C / truncated:           dezelfde opgeslagen E2M3 -> twee mantissabits verwijderen.

RNE = round-to-nearest, ties-to-even. Shared scale blijft X = 1, code 127.
De E2M3-bron wordt NIET opnieuw uit de oorspronkelijke invoer berekend.
Alleen pad A, de analyse en de controles mogen de oorspronkelijke invoer lezen.
De zelftests gebruiken eigen testcodes, los van het ingelezen onderzoeksblok.

Onderzoeksbasis: experiment_001_mantissa.py versie 1.0.0 en de twee JSON-logs
uit een ongewijzigde run daarvan. Formaatdefinities worden uit die module
hergebruikt; dit is geen nieuwe of algemene MX-implementatie.
Alle berekeningen zijn exacte Fraction-berekeningen; floats alleen voor tekst.
Geen externe pakketten, GPU, dataset, training of hardwaremeting nodig.

Uitvoeren vanuit de projectmap, met het juiste pad naar de eerdere run:
    python experiment_002_drie_paden.py --bronmap "results/experiment_001_mantissa/<run>"

De twee .py-bestanden moeten naast elkaar staan. Oude bestanden worden alleen
gelezen. Uitvoer: een nieuwe map onder results/experiment_002_drie_paden met
resultaten.json en exacte kopieen van beide ingelezen logs als controlemateriaal.
JSON is een onderzoekslog, geen bitgepakt MX-opslagformaat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

try:
    import experiment_001_mantissa as experiment_001
    from experiment_001_mantissa import (
        MXBlock, decode_element, quantize_rne, require, save_json, truncate_block,
    )
except ModuleNotFoundError as error:
    print(
        "FOUT: plaats het ongewijzigde experiment_001_mantissa.py naast dit bestand. "
        f"Importmelding: {error}", file=sys.stderr,
    )
    raise SystemExit(1) from error

VERSION = "1.0.0"
EXPERIMENT = "002_drie_paden"
EXPECTED_PREVIOUS_ROW_KEYS = {
    "index", "input_decimal", "input_exact", "source_bits", "truncated_bits", "direct_bits",
    "source_value_exact", "truncated_value_exact", "direct_value_exact", "gap_exact",
    "codes_equal", "input_exceeds_source_range", "input_exceeds_target_range",
}

# Validatie van de EERDERE proef. Deze beschrijvingen sturen de rekenfuncties niet.
EXPECTED_PREVIOUS_CONFIG = {
    "source": "MXFP6_E2M3", "target": "MXFP4_E2M1", "block_size": 32,
    "shared_scale": "1", "shared_scale_E8M0_code": 127,
    "scale_selection": "fixed_by_experiment_not_recomputed",
    "source_rounding": "nearest_ties_even", "reference_rounding": "nearest_ties_even",
    "truncation": "discard_two_lowest_mantissa_bits", "overflow": "saturate",
    "input_scope": "finite_exact_rationals", "missing_values": 0, "excluded_values": 0,
    "subnormals": "supported", "random_seed": None,
    "storage": "JSON_debug_representation_not_bitpacked_MX",
}


def round_block_from_source(source: MXBlock) -> MXBlock:
    """Nieuw pad B: rond alleen de opgeslagen bronwaarden naar E2M1 af.

    Het oorspronkelijke tekenbit blijft behouden, ook voor negatieve nul.
    Fraction bewaart zelf geen negatieve nul; daarom lezen we het teken uit
    de bronCODE en ronden we de magnitude af. Afronding mag het exponentveld
    veranderen, anders dan de mantissatruncatie. De shared scale blijft vast.
    """
    if source.mantissa_bits != 3 or source.scale_code != 127:
        raise ValueError("Pad B vereist E2M3 met vaste schaalcode 127.")
    target_codes = []
    for source_code in source.element_codes:
        magnitude = abs(decode_element(source_code, 3))
        magnitude_code = quantize_rne(magnitude, 1)
        sign_bit = (source_code >> 5) << 3
        target_codes.append(sign_bit | magnitude_code)
    return MXBlock(1, source.scale_code, tuple(target_codes))


def self_test_rounding() -> None:
    """Test pad B voor alle 64 broncodes met een andere referentieberekening.

    Alleen voor E2M3 -> E2M1: de weggevallen twee bits geven de positie binnen
    een doelinterval. Rest 2 is een tie; verhoog dan alleen een oneven doelcode.
    De referentie gebruikt geen decode_element of quantize_rne. Boven de grootste
    eindige doelwaarde satureren we. Dit is een test, niet het experimentele pad.
    """
    for start in (0, 32):
        source = MXBlock(3, 127, tuple(range(start, start + 32)))
        before = source.element_codes
        target = round_block_from_source(source)
        require(source.element_codes == before, "Pad B wijzigt de broncodes")
        require(target.scale_code == source.scale_code, "Pad B wijzigt de schaal")
        for src, dst in zip(source.element_codes, target.element_codes, strict=True):
            magnitude_code = src & 0b011111
            lower_code, remainder = divmod(magnitude_code, 4)
            increment = remainder > 2 or (remainder == 2 and lower_code % 2 == 1)
            expected = min(lower_code + int(increment), 7) | ((src >> 5) << 3)
            require(dst == expected, f"Pad B: verkeerde afronding voor {src:06b}")
            require(dst >> 3 == src >> 5, "Pad B: tekenbit is gewijzigd")
    negative_zero = MXBlock(3, 127, (0b100000,) * 32)
    require(round_block_from_source(negative_zero).element_codes == (0b1000,) * 32,
            "Pad B verliest het teken van negatieve nul")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def check_equal(actual: object, expected: object, label: str) -> None:
    """Strikte vergelijking: bijvoorbeeld False is niet hetzelfde type als 0."""
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Controle mislukt ({label}): {actual!r}; verwacht {expected!r}.")


def parse_json_object(content: bytes, label: str) -> dict:
    """Accepteer een JSON-object, geen NaN/Infinity of dubbele veldnamen."""
    def reject_constant(text: str) -> None:
        raise ValueError(f"{label}: niet-eindige JSON-waarde {text}.")

    def unique_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: dubbel JSON-veld {key!r}.")
            result[key] = value
        return result

    result = json.loads(content.decode("utf-8-sig"), parse_constant=reject_constant,
                        object_pairs_hook=unique_keys)
    if not isinstance(result, dict):
        raise ValueError(f"{label}: verwacht een JSON-object.")
    return result


def load_previous_run(run_dir: Path, dependency_bytes: bytes) -> tuple:
    """Lees broncodes en originele invoer; regenereer de onderzoeksbron niet."""
    source_path = run_dir / "bronblok.json"
    results_path = run_dir / "resultaten.json"
    source_bytes = source_path.read_bytes()
    results_bytes = results_path.read_bytes()
    stored = parse_json_object(source_bytes, "bronblok.json")
    previous = parse_json_object(results_bytes, "resultaten.json")

    check_equal(previous["experiment"], "001_mantissa", "eerder experiment")
    check_equal(previous["script_version"], "1.0.0", "eerdere scriptversie")
    check_equal(experiment_001.VERSION, "1.0.0", "geimporteerde moduleversie")
    check_equal(previous["self_tests"], "passed", "eerdere zelftests")
    check_equal(previous["source_block_sha256"], sha256_bytes(source_bytes), "bronhash")
    check_equal(previous["script_sha256"], sha256_bytes(dependency_bytes),
                "hash experiment_001_mantissa.py; gebruik het ongewijzigde script")
    for key, expected in EXPECTED_PREVIOUS_CONFIG.items():
        check_equal(previous["config"][key], expected, f"config.{key}")

    check_equal(stored["format"], "MXFP6_E2M3", "bronformaat")
    check_equal(stored["mantissa_bits"], 3, "bronmantissabits")
    check_equal(stored["scale_code"], 127, "bronschaal")
    if not isinstance(stored["element_codes"], list):
        raise ValueError("element_codes moet een JSON-lijst zijn.")
    source = MXBlock(3, 127, tuple(stored["element_codes"]))
    previous_rows = previous["rows"]
    if not isinstance(previous_rows, list) or len(previous_rows) != 32:
        raise ValueError("Verwacht precies 32 rijen uit experiment 001.")

    values = []
    for i, row in enumerate(previous_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Rij {i}: verwacht een JSON-object.")
        check_equal(set(row), EXPECTED_PREVIOUS_ROW_KEYS, f"velden eerdere rij {i}")
        check_equal(row["index"], i, f"rijvolgorde {i}")
        check_equal(row["input_decimal"], experiment_001.INPUT_TEXT[i], f"invoer rij {i}")
        if not isinstance(row["input_exact"], str):
            raise ValueError(f"Rij {i}: input_exact moet breuktekst zijn.")
        value = Fraction(row["input_exact"])
        check_equal(value, Fraction(row["input_decimal"]), f"invoerbreuk rij {i}")
        values.append(value)
    return source, tuple(values), previous, source_bytes, results_bytes


def compare_blocks(left: MXBlock, right: MXBlock) -> dict:
    """Afstanden tussen twee doelblokken; gemiddelden over ALLE elementen."""
    require(left.mantissa_bits == right.mantissa_bits == 1, "Vergelijk E2M1-blokken")
    require(left.scale_code == right.scale_code == 127, "Vergelijk bij schaal X=1")
    gaps = []
    code_indices, numeric_indices = [], []
    for i, (lc, rc) in enumerate(zip(left.element_codes, right.element_codes, strict=True)):
        gap = abs(decode_element(lc, 1) - decode_element(rc, 1))
        gaps.append(gap)
        if lc != rc:
            code_indices.append(i)
        if gap != 0:
            numeric_indices.append(i)
    return {
        "n_code_mismatches": len(code_indices),
        "n_numeric_mismatches": len(numeric_indices),
        "mean_absolute_gap_exact": str(sum(gaps) / len(gaps)),
        "max_absolute_gap_exact": str(max(gaps)),
        "code_mismatch_indices": code_indices,
        "numeric_mismatch_indices": numeric_indices,
    }


def analyse(values: tuple[Fraction, ...], previous: dict, source: MXBlock,
            direct: MXBlock, rounded_from_source: MXBlock, truncated: MXBlock) -> tuple:
    """Bereken rijen en meetwaarden; controleer alle oude rijvelden en meetwaarden."""
    rows = []
    errors = {name: [] for name in ("source", "direct", "rounded_from_source", "truncated")}
    for i, value in enumerate(values):
        codes = {"source": source.element_codes[i], "direct": direct.element_codes[i],
                 "rounded_from_source": rounded_from_source.element_codes[i],
                 "truncated": truncated.element_codes[i]}
        decoded = {name: decode_element(code, 3 if name == "source" else 1)
                   for name, code in codes.items()}
        row = {"index": i, "input_decimal": previous["rows"][i]["input_decimal"],
               "input_exact": str(value)}
        for name, code in codes.items():
            row[f"{name}_bits"] = format(code, "06b" if name == "source" else "04b")
            row[f"{name}_value_exact"] = str(decoded[name])
            error = abs(decoded[name] - value)
            errors[name].append(error)
            row[f"error_{name}_vs_input_exact"] = str(error)
        row.update({
            # Deze twee veldnamen blijven gelijk aan experiment 001 (C tegenover A).
            "gap_exact": str(abs(decoded["truncated"] - decoded["direct"])),
            "codes_equal": codes["truncated"] == codes["direct"],
            "gap_direct_rounded_from_source_exact":
                str(abs(decoded["direct"] - decoded["rounded_from_source"])),
            "gap_rounded_from_source_truncated_exact":
                str(abs(decoded["rounded_from_source"] - decoded["truncated"])),
            "direct_rounded_from_source_codes_equal":
                codes["direct"] == codes["rounded_from_source"],
            "rounded_from_source_truncated_codes_equal":
                codes["rounded_from_source"] == codes["truncated"],
            "input_exceeds_source_range": abs(value) > Fraction(15, 2),
            "input_exceeds_target_range": abs(value) > 6,
        })
        # Geen oude uitkomst overnemen: vergelijk haar met onze nieuwe berekening.
        for key, old_value in previous["rows"][i].items():
            check_equal(row[key], old_value, f"regressie rij {i}, {key}")
        rows.append(row)

    comparisons = {
        "direct_vs_rounded_from_source": compare_blocks(direct, rounded_from_source),
        "rounded_from_source_vs_truncated": compare_blocks(rounded_from_source, truncated),
        "direct_vs_truncated": compare_blocks(direct, truncated),
    }
    metrics = {"n_values": len(values)}
    for name, path_errors in errors.items():
        metrics[f"mae_{name}_vs_input_exact"] = str(sum(path_errors) / len(values))
    metrics["comparisons"] = comparisons
    metrics["n_input_exceeds_source_range"] = sum(r["input_exceeds_source_range"] for r in rows)
    metrics["n_input_exceeds_target_range"] = sum(r["input_exceeds_target_range"] for r in rows)

    old_gap = comparisons["direct_vs_truncated"]
    baseline = {
        "n_values": len(values),
        "n_code_mismatches": old_gap["n_code_mismatches"],
        "n_numeric_mismatches": old_gap["n_numeric_mismatches"],
        "mae_source_vs_input_exact": metrics["mae_source_vs_input_exact"],
        "mae_truncated_vs_input_exact": metrics["mae_truncated_vs_input_exact"],
        "mae_direct_vs_input_exact": metrics["mae_direct_vs_input_exact"],
        "mean_absolute_qt_gap_exact": old_gap["mean_absolute_gap_exact"],
        "max_absolute_qt_gap_exact": old_gap["max_absolute_gap_exact"],
    }
    check_equal(set(previous["metrics"]), set(baseline), "velden eerdere meetwaarden")
    for key, computed in baseline.items():
        check_equal(computed, previous["metrics"][key], f"regressie meetwaarde {key}")
    return rows, metrics, baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronmap", required=True, type=Path,
                        help="Map van experiment 001 met bronblok.json en resultaten.json.")
    args = parser.parse_args()
    previous_dir = args.bronmap.expanduser().resolve(strict=True)
    if not previous_dir.is_dir():
        raise ValueError("--bronmap moet een map zijn, geen bestand.")
    script = Path(__file__).resolve()
    dependency_path = Path(experiment_001.__file__).resolve()
    check_equal(dependency_path, script.with_name("experiment_001_mantissa.py"),
                "de twee Pythonbestanden moeten naast elkaar staan")
    script_bytes = script.read_bytes()
    dependency_bytes = dependency_path.read_bytes()

    experiment_001.self_test()
    self_test_rounding()
    source, values, previous, source_bytes, previous_bytes = load_previous_run(
        previous_dir, dependency_bytes,
    )

    # Bronafhankelijke paden ontvangen GEEN values of previous.
    source_before = (source.mantissa_bits, source.scale_code, source.element_codes)
    rounded_from_source = round_block_from_source(source)   # B, nieuw
    truncated = truncate_block(source)                     # C, ongewijzigd
    direct = MXBlock(1, source.scale_code, tuple(quantize_rne(v, 1) for v in values))  # A
    require((source.mantissa_bits, source.scale_code, source.element_codes) == source_before,
            "Het ingelezen bronblok is gewijzigd")
    rows, metrics, baseline = analyse(values, previous, source, direct,
                                      rounded_from_source, truncated)

    # Ook bestanden en beide scripts moeten tijdens de berekening ongewijzigd blijven.
    for path, original in ((previous_dir / "bronblok.json", source_bytes),
                           (previous_dir / "resultaten.json", previous_bytes),
                           (dependency_path, dependency_bytes), (script, script_bytes)):
        require(path.read_bytes() == original, f"Bestand tijdens uitvoering gewijzigd: {path}")

    warnings = []
    if metrics["n_input_exceeds_target_range"]:
        warnings.append(f"{metrics['n_input_exceeds_target_range']}/{len(values)} invoerwaarden "
                        "liggen buiten het doelbereik; behouden in alle gemiddelden.")
    output_root = script.parent / "results" / "experiment_002_drie_paden"
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")
    run_dir = Path(tempfile.mkdtemp(prefix=stamp, dir=output_root))
    # Exacte kopieen in de NIEUWE map, zonder oude bestanden aan te passen.
    for name, content in (("bronblok.json", source_bytes),
                          ("resultaten_experiment_001.json", previous_bytes)):
        with (run_dir / name).open("xb") as stream:
            stream.write(content)

    save_json(run_dir / "resultaten.json", {
        "experiment": EXPERIMENT, "script_version": VERSION,
        "status": "exploratory_synthetic_test_not_model_benchmark",
        "run_id": run_dir.name,
        "script_sha256": sha256_bytes(script_bytes),
        "dependency_script_sha256": sha256_bytes(dependency_bytes),
        "source_block_sha256": sha256_bytes(source_bytes),
        "previous_results_sha256": sha256_bytes(previous_bytes),
        "provenance": {"previous_run_directory": str(previous_dir),
                       "dependency_script": dependency_path.name,
                       "dependency_version": experiment_001.VERSION,
                       "input_values_from": "previous results rows[*].input_exact",
                       "source_regenerated_from_inputs": False},
        "environment": {"python": sys.version, "system": platform.system(),
                        "machine": platform.machine(), "executable": sys.executable},
        "config": {
            **EXPECTED_PREVIOUS_CONFIG,
            "rounded_from_source_rounding": "nearest_ties_even",
            "rounded_from_source_sign": "preserve_source_sign_including_negative_zero",
            "rounded_from_source_exponent": "may_change_due_to_rounding",
            "path_A": "direct: original input -> RNE E2M1",
            "path_B": "rounded_from_source: stored E2M3 -> decode -> RNE E2M1",
            "path_C": "truncated: stored E2M3 -> discard two mantissa bits",
        },
        "self_tests": {"experiment_001": "passed", "new_rounding_all_64_codes": "passed",
                       "negative_zero_sign": "passed", "source_unchanged": "passed"},
        "baseline_check": {"status": "passed", "rows_compared": len(rows),
                           "metrics_compared": len(baseline), "recomputed_metrics": baseline},
        "warnings": warnings, "metrics": metrics, "rows": rows,
    })

    print("Experiment 002: drie paden naar MXFP4 E2M1 | shared scale: 1 | CPU")
    print("Zelftests: OK (experiment 001 + alle 64 broncodes voor pad B)")
    print(f"Bestaande bron ingelezen uit: {previous_dir}")
    print(f"Controle experiment 001: OK ({len(rows)} rijen, {len(baseline)} meetwaarden)")
    print("A = direct | B = rounded_from_source | C = truncated")
    for label, key in (("A versus B", "direct_vs_rounded_from_source"),
                       ("B versus C", "rounded_from_source_vs_truncated"),
                       ("A versus C", "direct_vs_truncated")):
        comparison = metrics["comparisons"][key]
        gap = Fraction(comparison["mean_absolute_gap_exact"])
        print(f"{label}: {comparison['n_code_mismatches']}/{len(values)} codeverschillen; "
              f"gemiddeld absoluut verschil: {gap} = {float(gap):.6f}")
    for label, key in (("bron", "source"), ("A", "direct"),
                       ("B", "rounded_from_source"), ("C", "truncated")):
        error = Fraction(metrics[f"mae_{key}_vs_input_exact"])
        print(f"MAE {label} tegenover invoer: {error} = {float(error):.10f}")
    for warning in warnings:
        print(f"AANDACHTSPUNT: {warning}")
    print(f"Resultaten opgeslagen in: {run_dir}")
    print("Geen netwerk, training, snelheidsmeting of energiemeting uitgevoerd.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, AssertionError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"FOUT: {error}", file=sys.stderr)
        print("Controleer --bronmap en gebruik de ongewijzigde bestanden van experiment 001.",
              file=sys.stderr)
        sys.exit(1)
