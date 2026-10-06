"""Experiment 003: drie paden van MXFP8 E4M3 naar MXFP6 E2M3, schaal X=1.

A / direct: oorspronkelijke exacte invoer -> E2M3, nearest/ties-to-even (RNE).
B / rounded_from_source: opgeslagen E4M3 -> decoderen -> E2M3, RNE.
C / naive_exponent_truncation: S E3 E2 E1 E0 M2 M1 M0 -> S E3 E2 M2 M1 M0.
C verwijdert ALLEEN de twee laagste exponentbits. Geen biascorrectie, geen
mantissatruncatie, geen afronding en geen saturatiebewerking in pad C.
De uitkomst van C wordt volgens de gewone E2M3-bias 1 geinterpreteerd.

Eigen verkennende proefopstelling (geen resultaat of recept uit een paper):
- twee APART gerapporteerde blokken van 32 vaste synthetische waarden;
- basisblok: dezelfde oorspronkelijke invoer als in experiment 001/002;
- grensblok: extra gekozen waarden voor kleine magnitudes, RNE-grenzen,
  bron/doel-bereikgrenzen en het besproken voorbeeld 5.5, met beide tekens;
- nieuwe E4M3-bron eenmalig per run coderen, opslaan, herlezen en niet wijzigen;
- B en C ontvangen uitsluitend die bron, nooit de oorspronkelijke invoer;
- shared scale E8M0-code 127 (X=1), in ieder blok en ieder pad onveranderd;
- eindige Fraction-invoer, RNE met saturatie voor bron, A en B;
- subnormalen en negatieve nul in de codes blijven ondersteund;
- E4M3-NaN-codes 0x7f/0xff expliciet afwijzen, niet als getallen interpreteren.

Bronnen voor de formaten (NIET voor de naieve bitbewerking):
1. OCP Microscaling Formats (MX) Specification, v1.0, september 2023,
   sec. 5.2-5.4, p. 10-14, met name tabellen 2, 4 en 7.
   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
2. Rouhani et al., Microscaling Data Formats for Deep Learning, 2023,
   projectpaper p. 2-3, sec. 2 en tabel 1. Geen toepassing van automatische
   schaalkeuze uit Algorithm 1: de schaal wordt hier bewust vastgehouden.

Zelfstandig bestand: geen imports uit experiment 001/002 of externe pakketten.
Exacte rekenkern; float alleen voor leesbare terminaltekst. Geen netwerk,
training, tijd-/energiebenchmark of claim van volledige MX-conformiteit:
alleen bovenstaande eindige conversies en experimenteel pad C vallen in scope.
JSON-logs zijn geen bitgepakte MX-opslag. Alle uitvoer gaat naar een nieuwe map.

Gebruik: python experiment_003_exponent.py
Alleen controles: python experiment_003_exponent.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

VERSION = "1.0.0"
EXPERIMENT = "003_exponent"
BLOCK_SIZE = 32
SCALE_CODE = 127
SCALE = Fraction(1)
SPEC_URL = "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf"

# Ongewijzigde oorspronkelijke invoer uit experiment 001, NIET de oude broncodes.
BASE_INPUT = (
    "-7.5", "-6.5", "-5.75", "-3.875", "-2.875", "-1.875", "-1.625", "-1.375",
    "-0.875", "-0.6875", "-0.375", "-0.3125", "-0.1875", "-0.125", "-0.0625", "0",
    "0.03125", "0.0625", "0.125", "0.1875", "0.3125", "0.375", "0.6875", "0.875",
    "1.375", "1.625", "1.875", "2.875", "3.875", "5.75", "6.5", "7.5",
)
# Voor uitvoering gekozen, geen steekproef van netwerkgewichten.
# 0.061/0.064 en 0.183/0.189: aan weerszijden van E2M3-afrondingsgrenzen.
EDGE_POSITIVE = (
    "0.0005", "0.001953125", "0.015625", "0.061", "0.064", "0.183", "0.189",
    "0.9375", "1", "5.5", "7.5", "7.75", "8", "240", "448", "500",
)
INPUTS = {
    "basis_32": BASE_INPUT,
    "grenzen_32": tuple("-" + text for text in reversed(EDGE_POSITIVE)) + EDGE_POSITIVE,
}
PATHS = ("direct", "rounded_from_source", "naive_exponent_truncation")
PAIRS = (("A versus B", PATHS[0], PATHS[1]),
         ("B versus C", PATHS[1], PATHS[2]),
         ("A versus C", PATHS[0], PATHS[2]))


def require(condition: bool, message: str) -> None:
    """Blijft ook actief bij python -O."""
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class ElementFormat:
    name: str
    exponent_bits: int
    bias: int
    max_magnitude_code: int
    mantissa_bits: int = 3

    @property
    def sign_shift(self) -> int:
        return self.exponent_bits + self.mantissa_bits

    @property
    def total_bits(self) -> int:
        return self.sign_shift + 1


E4M3 = ElementFormat("E4M3", 4, 7, 126)
E2M3 = ElementFormat("E2M3", 2, 1, 31)


class NonFiniteCodeError(ValueError):
    """NaN is buiten de eindige proefscope; afwijzen in plaats van stil vervangen."""


def split_fields(code: int, fmt: ElementFormat) -> tuple[int, int, int]:
    if fmt not in (E4M3, E2M3):
        raise ValueError("Alleen de vastgelegde E4M3- en E2M3-formaten zijn toegestaan.")
    if type(code) is not int or not 0 <= code < (1 << fmt.total_bits):
        raise ValueError(f"Ongeldige {fmt.name}-code: {code!r}")
    sign = code >> fmt.sign_shift
    exponent = (code >> 3) & ((1 << fmt.exponent_bits) - 1)
    mantissa = code & 7
    if fmt == E4M3 and exponent == 15 and mantissa == 7:
        raise NonFiniteCodeError(f"E4M3-code {code:08b} is NaN, geen eindige waarde.")
    return sign, exponent, mantissa


def power_of_two(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def decode_element(code: int, fmt: ElementFormat) -> Fraction:
    """Decodeer een eindige elementcode. Fraction zelf bewaart geen negatieve nul."""
    sign, exponent, mantissa = split_fields(code, fmt)
    fraction = Fraction(mantissa, 8)
    if exponent == 0:
        magnitude = fraction * power_of_two(1 - fmt.bias)
    else:
        magnitude = (1 + fraction) * power_of_two(exponent - fmt.bias)
    return -magnitude if sign else magnitude


@lru_cache(maxsize=2)
def positive_values(fmt: ElementFormat) -> tuple[Fraction, ...]:
    """Exacte kandidatentabel, opgebouwd uit alle eindige niet-negatieve codes."""
    return tuple(decode_element(code, fmt) for code in range(fmt.max_magnitude_code + 1))


def quantize_rne(value: Fraction, fmt: ElementFormat, *, negative_zero: bool = False) -> int:
    """Dichtste eindige waarde, ties-to-even, saturatie en behoud van nul-teken."""
    if not isinstance(value, Fraction) or type(negative_zero) is not bool:
        raise ValueError("Verwacht exacte Fraction-invoer en een booleaanse nul-tekenvlag.")
    if fmt not in (E4M3, E2M3):
        raise ValueError("Onbekend elementformaat.")
    table = positive_values(fmt)
    magnitude = abs(value)
    code = min(range(len(table)), key=lambda c: (abs(magnitude - table[c]), c & 1))
    sign = int(value < 0 or (value == 0 and negative_zero))
    return (sign << fmt.sign_shift) | code


@dataclass(frozen=True)
class MXBlock:
    fmt: ElementFormat
    scale_code: int
    element_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.scale_code) is not int or self.scale_code != SCALE_CODE:
            raise ValueError("Experiment 003 ondersteunt uitsluitend schaalcode 127 (X=1).")
        if not isinstance(self.element_codes, tuple) or len(self.element_codes) != BLOCK_SIZE:
            raise ValueError("Een blok moet precies 32 codes in een tuple bevatten.")
        for code in self.element_codes:
            split_fields(code, self.fmt)


def round_source_code(source_code: int) -> int:
    """Pad B, een element: numerieke hercodering uit de opgeslagen bron, niet uit x."""
    sign, _, _ = split_fields(source_code, E4M3)
    magnitude = abs(decode_element(source_code, E4M3))
    return (sign << E2M3.sign_shift) | quantize_rne(magnitude, E2M3)


def truncate_exponent_code(source_code: int) -> int:
    """Pad C: behoud S, de HOOGSTE twee exponentbits en alle drie mantissabits.

    Dit is NIET source_code >> 2: dat zou ook mantissabits verwijderen.
    Dit is een naieve bitbewerking, geen numerieke RNE-conversie naar E2M3.
    """
    sign, exponent, mantissa = split_fields(source_code, E4M3)
    return (sign << E2M3.sign_shift) | ((exponent >> 2) << 3) | mantissa


def round_block_from_source(source: MXBlock) -> MXBlock:
    if source.fmt != E4M3:
        raise ValueError("Pad B vereist een E4M3-bron.")
    return MXBlock(E2M3, source.scale_code, tuple(round_source_code(c) for c in source.element_codes))


def truncate_exponent_block(source: MXBlock) -> MXBlock:
    if source.fmt != E4M3:
        raise ValueError("Pad C vereist een E4M3-bron.")
    return MXBlock(E2M3, source.scale_code, tuple(truncate_exponent_code(c) for c in source.element_codes))


def self_test() -> dict:
    """Gerichte afrondingstests plus uitputtende controles van de kleine coderuimtes."""
    # Onafhankelijke integerwaardetabellen: E4M3 in eenheden 1/512, E2M3 in 1/8.
    units6 = tuple(range(8)) + tuple(range(8, 16)) + tuple(range(16, 32, 2)) + tuple(range(32, 64, 4))
    units8 = tuple(range(8)) + tuple((8 + m) * (2 ** (e - 1))
                                   for e in range(1, 16) for m in range(8)
                                   if (e, m) != (15, 7))
    boundaries_tested = 0
    for fmt, units, denominator in ((E2M3, units6, 8), (E4M3, units8, 512)):
        for code in range(1 << fmt.total_bits):
            magnitude_code = code & ((1 << fmt.sign_shift) - 1)
            if fmt == E4M3 and magnitude_code == 127:
                try:
                    decode_element(code, fmt)
                except NonFiniteCodeError:
                    pass
                else:
                    raise AssertionError("E4M3 NaN werd als eindig getal toegelaten.")
                continue
            expected = Fraction(units[magnitude_code], denominator)
            if code >> fmt.sign_shift:
                expected = -expected
            require(decode_element(code, fmt) == expected, f"Decoder {fmt.name}: {code}")
            actual = quantize_rne(expected, fmt, negative_zero=bool(code >> fmt.sign_shift))
            require(actual == code, f"Rondgang, inclusief nul-teken, {fmt.name}: {code}")
        table = tuple(Fraction(u, denominator) for u in units)
        for c, (low, high) in enumerate(zip(table, table[1:])):
            # Zowel positieve als negatieve kwartpunten en middelpunten.
            midpoint = (low + high) / 2
            tie = c if c % 2 == 0 else c + 1
            for value, expected_code in (((3 * low + high) / 4, c),
                                         (midpoint, tie), ((low + 3 * high) / 4, c + 1)):
                require(quantize_rne(value, fmt) == expected_code, "Positieve afrondingsgrens")
                require(quantize_rne(-value, fmt) == ((1 << fmt.sign_shift) | expected_code),
                        "Negatieve afrondingsgrens")
            boundaries_tested += 1
        require(quantize_rne(Fraction(10**6), fmt) == fmt.max_magnitude_code, "Saturatie +")
        require(quantize_rne(Fraction(-10**6), fmt) == ((1 << fmt.sign_shift) | fmt.max_magnitude_code),
                "Saturatie -")
    finite_codes = [c for c in range(256) if (c & 127) != 127]
    for c in finite_codes:
        sign, e, m = split_fields(c, E4M3)
        # B-referentie via biascorrectie of integerafronding voor subnormalen.
        # Gebruikt GEEN decode_element of quantize_rne.
        if e >= 10:
            expected_magnitude = 31
        elif e >= 7:
            expected_magnitude = ((e - 6) << 3) | m
        else:
            u512 = m if e == 0 else (8 + m) * (2 ** (e - 1))
            low, remainder = divmod(u512, 64)
            expected_magnitude = low + int(remainder > 32 or (remainder == 32 and low % 2 == 1))
        require(round_source_code(c) == ((sign << 5) | expected_magnitude), f"Pad B: {c}")
        # C-referentie via selectie uit een bitTEKST, niet via shifts.
        text = f"{c:08b}"
        expected_c = int(text[0] + text[1:3] + text[5:8], 2)
        actual_c = truncate_exponent_code(c)
        require(actual_c == expected_c, f"Pad C bitselectie: {c}")
        require((actual_c & 7) == m and actual_c >> 5 == sign, "C wijzigt mantissa/teken")
    for c in (127, 255):
        for function in (round_source_code, truncate_exponent_code):
            try:
                function(c)
            except NonFiniteCodeError:
                pass
            else:
                raise AssertionError("Een conversie accepteerde een NaN-broncode.")
    # Blokinterfaces: laatste blok alleen voor deze test met nulcodes aanvullen.
    for start in range(0, len(finite_codes), BLOCK_SIZE):
        codes = tuple(finite_codes[start:start + BLOCK_SIZE])
        codes += (0,) * (BLOCK_SIZE - len(codes))
        source = MXBlock(E4M3, SCALE_CODE, codes)
        for convert in (round_block_from_source, truncate_exponent_block):
            target = convert(source)
            require(source.element_codes == codes and target.scale_code == source.scale_code,
                    "Bron of shared scale gewijzigd")
    # Het eerder besproken voorbeeld en signed zero, zonder vaste print-uitkomsten.
    c55 = quantize_rne(Fraction("5.5"), E4M3)
    require(c55 == 0b01001011, "E4M3-codering 5.5")
    require(round_source_code(c55) == 0b011011, "Pad B 5.5")
    require(truncate_exponent_code(c55) == 0b010011, "Pad C 5.5")
    require(round_source_code(128) == 32 and truncate_exponent_code(128) == 32, "Negatieve nul")
    return {"status": "passed", "source_bit_patterns_examined": 256,
            "finite_source_codes_checked_for_B_and_C": len(finite_codes),
            "nan_codes_rejected": [127, 255], "target_codes_checked": 64,
            "positive_adjacent_intervals_tested_with_both_signs": boundaries_tested,
            "source_and_shared_scale_unchanged": True,
            "scope": "finite encodings, selected rounding probes; not all rational inputs"}


def save_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def mean_text(values: list[Fraction]) -> str | None:
    return str(sum(values, Fraction(0)) / len(values)) if values else None


def compare_paths(left_codes: tuple[int, ...], right_codes: tuple[int, ...]) -> dict:
    left = tuple(decode_element(c, E2M3) for c in left_codes)
    right = tuple(decode_element(c, E2M3) for c in right_codes)
    gaps = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return {"n_values": len(gaps),
            "n_code_mismatches": sum(a != b for a, b in zip(left_codes, right_codes, strict=True)),
            "n_numeric_mismatches": sum(g != 0 for g in gaps),
            "mean_absolute_gap_exact": mean_text(gaps),
            "max_absolute_gap_exact": str(max(gaps)),
            "code_mismatch_indices": [i for i, (a, b) in enumerate(zip(left_codes, right_codes, strict=True)) if a != b],
            "numeric_mismatch_indices": [i for i, g in enumerate(gaps) if g != 0]}


def path_metrics(inputs: tuple[Fraction, ...], decoded: tuple[Fraction, ...], indices: list[int]) -> dict:
    errors = [abs(decoded[i] - inputs[i]) for i in indices]
    relative = [abs(decoded[i] - inputs[i]) / abs(inputs[i]) for i in indices if inputs[i] != 0]
    # Alle ongeordende paren van verschillende invoerwaarden, niet alleen buren.
    reversals = [(i, j) for p, i in enumerate(indices) for j in indices[p + 1:]
                 if (inputs[i] - inputs[j]) * (decoded[i] - decoded[j]) < 0]
    return {"n_values": len(indices), "mae_vs_input_exact": mean_text(errors),
            "max_absolute_error_exact": str(max(errors)) if errors else None,
            "mean_absolute_relative_error_nonzero_input_exact": mean_text(relative),
            "max_absolute_relative_error_nonzero_input_exact": str(max(relative)) if relative else None,
            "relative_error_denominator": len(relative),
            "zero_inputs_not_used_for_relative_error": sum(inputs[i] == 0 for i in indices),
            "n_nonzero_inputs_mapped_to_zero": sum(inputs[i] != 0 and decoded[i] == 0 for i in indices),
            "n_strict_order_reversal_pairs_vs_input": len(reversals),
            "strict_order_reversal_pairs_vs_input": reversals}


def analyse_block(texts: tuple[str, ...], source: MXBlock, targets: dict[str, MXBlock]) -> dict:
    inputs = tuple(Fraction(t) for t in texts)
    blocks = {"source": source, **targets}
    decoded = {name: tuple(SCALE * decode_element(c, block.fmt) for c in block.element_codes)
               for name, block in blocks.items()}
    source_max = positive_values(E4M3)[-1]
    target_max = positive_values(E2M3)[-1]
    metrics = {name: path_metrics(inputs, values, list(range(BLOCK_SIZE)))
               for name, values in decoded.items()}
    comparisons = {left + "_vs_" + right: compare_paths(targets[left].element_codes, targets[right].element_codes)
                   for _, left, right in PAIRS}
    rows = []
    for i, value in enumerate(inputs):
        row = {"index": i, "input_decimal": texts[i], "input_exact": str(value),
               "input_exceeds_source_range": abs(value) > source_max,
               "input_exceeds_target_range": abs(value) > target_max,
               "nonzero_input_below_target_min_positive": 0 < abs(value) < Fraction(1, 8)}
        for name, block in blocks.items():
            c = block.element_codes[i]
            s, e, m = split_fields(c, block.fmt)
            y = decoded[name][i]
            row[name] = {"code": c, "bits": f"{c:0{block.fmt.total_bits}b}",
                         "sign": s, "exponent_field": e, "mantissa_field": m,
                         "value_exact": str(y), "absolute_error_vs_input_exact": str(abs(y - value)),
                         "absolute_relative_error_vs_input_exact": str(abs(y - value) / abs(value)) if value != 0 else None,
                         "negative_zero": s == 1 and y == 0,
                         "subnormal": e == 0 and m != 0}
        row["gaps_exact"] = {left + "_vs_" + right: str(abs(decoded[left][i] - decoded[right][i]))
                              for _, left, right in PAIRS}
        row["C_magnitude_greater_than_source"] = abs(decoded[PATHS[2]][i]) > abs(decoded["source"][i])
        rows.append(row)
    partitions = {"within_target_range": [i for i, v in enumerate(inputs) if abs(v) <= target_max],
                  "outside_target_range": [i for i, v in enumerate(inputs) if abs(v) > target_max]}
    by_range = {group: {"indices": indices,
                       "paths": {name: path_metrics(inputs, vals, indices) for name, vals in decoded.items()}}
                for group, indices in partitions.items()}
    warnings = []
    for name, maximum in (("bron", source_max), ("doel", target_max)):
        n = sum(abs(v) > maximum for v in inputs)
        if n:
            warnings.append(f"{n}/32 invoerwaarden boven het {name}bereik; niet uitgesloten.")
    expanded = sum(row["C_magnitude_greater_than_source"] for row in rows)
    if expanded:
        warnings.append(f"Pad C vergroot bij {expanded}/32 waarden de magnitude t.o.v. de opgeslagen bron.")
    n_reversals = metrics[PATHS[2]]["n_strict_order_reversal_pairs_vs_input"]
    if n_reversals:
        warnings.append(f"Pad C: {n_reversals} strikt omgekeerde waardeparen t.o.v. de invoer (geen zelftestfout).")
    return {"n_values": len(inputs), "metrics": metrics, "comparisons": comparisons,
            "range_counts": {"input_exceeds_source_range": sum(abs(v) > source_max for v in inputs),
                             "input_exceeds_target_range": sum(abs(v) > target_max for v in inputs)},
            "metrics_by_input_target_range": by_range,
            "n_C_magnitude_greater_than_source": expanded,
            "warnings": warnings, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--self-test", action="store_true", help="Alleen zelftests, geen uitvoerbestanden.")
    parser.add_argument("--output-dir", type=Path, help="Optionele bovenmap voor nieuwe runmappen.")
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print("Zelftests: OK | 254 eindige E4M3-codes | 2 NaN-codes afgewezen | 64 E2M3-codes")
        return
    for name, texts in INPUTS.items():
        require(len(texts) == BLOCK_SIZE, f"Blok {name} heeft geen 32 waarden.")
        tuple(Fraction(text) for text in texts)  # Verkeerde invoer stopt de run, niet uitsluiten.
    script = Path(__file__).resolve()
    script_bytes = script.read_bytes()
    output_root = args.output_dir or (script.parent / "results" / "experiment_003_exponent")
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")
    run_dir = Path(tempfile.mkdtemp(prefix=stamp, dir=output_root)).resolve()
    input_path = run_dir / "invoer.json"
    save_json(input_path, {"design": "two fixed synthetic blocks, reported separately",
                          "blocks": INPUTS, "missing_values": 0, "excluded_values": 0})
    source_path = run_dir / "bronblokken.json"
    # NIEUWE E4M3-bron eenmaal coderen. Oude E2M3-bestanden worden niet gebruikt.
    save_json(source_path, {
        "element_format": asdict(E4M3), "scale_code": SCALE_CODE, "block_size": BLOCK_SIZE,
        "blocks": {name: {"element_codes": [quantize_rne(Fraction(text), E4M3) for text in texts]}
                   for name, texts in INPUTS.items()},
    })
    source_digest = sha256(source_path.read_bytes())
    with source_path.open(encoding="utf-8") as stream:
        stored = json.load(stream)
    require(stored["element_format"] == asdict(E4M3) and stored["scale_code"] == SCALE_CODE,
            "Opgeslagen bronmetadata wijkt af.")
    results = {}
    for name, texts in INPUTS.items():
        source = MXBlock(E4M3, stored["scale_code"], tuple(stored["blocks"][name]["element_codes"]))
        before = source.element_codes
        # B en C krijgen UITSLUITEND de opgeslagen bron; A mag texts gebruiken.
        rounded = round_block_from_source(source)
        truncated = truncate_exponent_block(source)
        direct = MXBlock(E2M3, SCALE_CODE, tuple(quantize_rne(Fraction(text), E2M3) for text in texts))
        require(source.element_codes == before, "Broncodes gewijzigd tijdens omzetting.")
        results[name] = analyse_block(texts, source, dict(zip(PATHS, (direct, rounded, truncated), strict=True)))
    require(sha256(source_path.read_bytes()) == source_digest, "Bronbestand gewijzigd.")
    require(script.read_bytes() == script_bytes, "Script gewijzigd tijdens de uitvoering.")
    save_json(run_dir / "resultaten.json", {
        "experiment": EXPERIMENT, "script_version": VERSION, "run_id": run_dir.name,
        "status": "completed_exploratory_synthetic_test_not_model_benchmark",
        "script_sha256": sha256(script_bytes), "source_blocks_sha256": source_digest,
        "input_file_sha256": sha256(input_path.read_bytes()),
        "environment": {"python": sys.version, "system": platform.system(), "machine": platform.machine()},
        "provenance": {"standalone_script": True, "new_E4M3_source_created_once_per_run": True,
                       "B_and_C_inputs": "only reloaded stored source codes and format/scale metadata",
                       "basis_32_inputs": "literal copy of experiment 001 INPUT_TEXT; not its E2M3 source",
                       "grenzen_32_inputs": "new explicitly chosen diagnostic values, not measured data"},
        "format_reference": {"title": "OCP Microscaling Formats (MX) Specification",
                             "version": "1.0 (September 2023)", "pages": "10-14", "url": SPEC_URL},
        "config": {"source": asdict(E4M3), "target": asdict(E2M3), "block_size": BLOCK_SIZE,
                   "shared_scale": str(SCALE), "shared_scale_E8M0_code": SCALE_CODE,
                   "scale_selection": "fixed, not optimized or recomputed",
                   "source_A_B_rounding": "nearest_ties_even_with_finite_saturation",
                   "path_A": "direct original input -> RNE E2M3",
                   "path_B": "stored E4M3 -> decode -> RNE E2M3; preserve source sign",
                   "path_C": "S E3 E2 E1 E0 M2 M1 M0 -> S E3 E2 M2 M1 M0; NO bias compensation",
                   "C_scope": "naive field-bit deletion, not a standard numerical conversion",
                   "input_scope": "finite exact rationals", "NaN_policy": "reject, not numeric",
                   "missing_values": 0, "excluded_input_values": 0, "random_seed": None,
                   "mean_denominator": "all 32 inputs per block unless explicit subset",
                   "relative_error_zero_policy": "null; excluded only from relative-error means",
                   "range_flags": "compare original magnitude to max finite; not a saturation event counter",
                   "storage": "JSON debug logs, not packed MX"},
        "self_tests": tests, "blocks": results,
    })
    print("Experiment 003: E4M3 -> E2M3 | shared scale: 1 | alleen CPU")
    print("Zelftests: OK (254 eindige broncodes; 2 NaN-codes expliciet afgewezen)")
    print("A = direct | B = rounded_from_source | C = naive_exponent_truncation")
    for name, result in results.items():
        print(f"\nBlok {name}: {result['n_values']} waarden (apart gerapporteerd)")
        for label, left, right in PAIRS:
            comparison = result["comparisons"][left + "_vs_" + right]
            gap = Fraction(comparison["mean_absolute_gap_exact"])
            print(f"{label}: {comparison['n_code_mismatches']}/{result['n_values']} codeverschillen; "
                  f"gemiddeld absoluut verschil: {gap} = {float(gap):.6f}")
        for label, key in (("bron", "source"), ("A", PATHS[0]), ("B", PATHS[1]), ("C", PATHS[2])):
            error = Fraction(result["metrics"][key]["mae_vs_input_exact"])
            print(f"MAE {label} tegenover invoer: {error} = {float(error):.8f}")
        for warning in result["warnings"]:
            print(f"AANDACHTSPUNT: {warning}")
    example = next(row for row in results["grenzen_32"]["rows"] if row["input_decimal"] == "5.5")
    print("\nBerekend voorbeeld 5.5: " + ", ".join(
        f"{label}={float(Fraction(example[key]['value_exact'])):g}"
        for label, key in (("bron", "source"), ("A", PATHS[0]), ("B", PATHS[1]), ("C", PATHS[2]))))
    print(f"Resultaten opgeslagen in: {run_dir}")
    print("Geen netwerk, training, accuracy-, snelheids- of energiemeting uitgevoerd.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, AssertionError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"FOUT: {error}", file=sys.stderr)
        sys.exit(1)
