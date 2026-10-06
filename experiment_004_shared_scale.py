"""Experiment 004: een schaalbit weglaten, met E4M3-elementen in alle paden.

Eigen verkennende proef, geen netwerk of modelaccuracy-/hardwaremeting.
Schaalvariant: c7 = c8 >> 1; interpreteer c7 als E8M0-code (c7 << 1).
Dit is een eigen 7-bit opslagcodering, NIET een standaard E7M0-formaat.
De impliciete nul is geen opgeslagen achtste bit. JSON is NIET bitgepakt.

Bron: S_i = X * Q_E4M3(x_i / X), eenmaal per conditie opslaan en herlezen.
A / direct:              X' * Q_E4M3(x_i / X').
B / rounded_from_source: X' * Q_E4M3(S_i / X'), alleen uit opgeslagen bron.
C / scale_only:          X' * P_i, elementcodes EXACT overnemen uit de bron.
Alle drie de doelpaden gebruiken hetzelfde X'. Alleen C wijzigt uitsluitend
het schaalveld; A/B mogen de elementcodes hercoderen, niet hun bitbreedte.
Q = round-to-nearest, ties-to-even, met eindige saturatie.

Ontwerp vastgelegd voor uitvoering:
- schalen X = 1/4, 1/2, 1, 2, 4 (E8M0-codes 125 t/m 129);
- twee blokken van 32 vaste waarden, overgenomen uit experiment 003;
- dezelfde OORSPRONKELIJKE invoer per blok bij alle vijf schalen: geen x * X;
- dus 10 aparte condities; geen gepoold gemiddelde of onafhankelijke steekproef;
- X wordt niet uit de data gekozen of geoptimaliseerd;
- naast absolute fouten ook fouten gedeeld door de oorspronkelijke schaal X;
- relatieve fouten met aparte, expliciete uitsluiting van nulle noemers;
- bereikoverschrijdingen niet verwijderen; wel apart rapporteren;
- elk B/C-pad ontvangt alleen een herlezen bronblok, nooit de invoerwaarden.

Numerieke basis:
- E4M3 zoals in experiment_003_exponent.py v1.0.0: bias 7, max 448,
  kleinste positieve waarde 1/512, NaN-codes 127 en 255 afgewezen.
- E8M0: X = 2**(c-127), eindige codes 0..254; code 255 is NaN en wordt
  VOOR de bitbewerking afgewezen (anders zou NaN een eindige schaal worden).
- exacte Fraction-rekenkunde, ook voor X * P; geen FP32-overloopemulatie.
- de 7-bit codering heeft geen NaN-code; scope is uitsluitend eindige schalen.

Formaatbronnen (niet de bron van onze experimentele 7-bit bewerking):
OCP Microscaling Formats (MX) Specification v1.0, Sep 2023, sec. 5.2/5.4.
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
NVIDIA CUDA Math API 12.8.1, __nv_fp8_e8m0: schaalcodes, bias en NaN.
https://docs.nvidia.com/cuda/archive/12.8.1/cuda-math-api/cuda_math_api/struct____nv__fp8__e8m0.html
Projectpaper: Microscaling Data Formats for Deep Learning, p. 2-3.

Zelfstandig; alleen Pythonstandaardbibliotheek, geen imports uit oude scripts.
Gebruik: python experiment_004_shared_scale.py
Alleen controles: python experiment_004_shared_scale.py --self-test
Uitvoer gaat naar een NIEUWE map. Eerdere scripts en resultaten blijven staan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

VERSION = "1.0.0"
EXPERIMENT = "004_shared_scale"
BLOCK_SIZE = 32
SCALE_CODES = (125, 126, 127, 128, 129)
ELEMENT_MAX = Fraction(448)
ELEMENT_MIN_POSITIVE = Fraction(1, 512)
PATHS = ("direct", "rounded_from_source", "scale_only")
PAIRS = (("A-B", PATHS[0], PATHS[1]), ("B-C", PATHS[1], PATHS[2]),
         ("A-C", PATHS[0], PATHS[2]))
BASE_INPUT = (
    "-7.5", "-6.5", "-5.75", "-3.875", "-2.875", "-1.875", "-1.625", "-1.375",
    "-0.875", "-0.6875", "-0.375", "-0.3125", "-0.1875", "-0.125", "-0.0625", "0",
    "0.03125", "0.0625", "0.125", "0.1875", "0.3125", "0.375", "0.6875", "0.875",
    "1.375", "1.625", "1.875", "2.875", "3.875", "5.75", "6.5", "7.5",
)
EDGE_POSITIVE = (
    "0.0005", "0.001953125", "0.015625", "0.061", "0.064", "0.183", "0.189",
    "0.9375", "1", "5.5", "7.5", "7.75", "8", "240", "448", "500",
)
INPUTS = {"basis_32": BASE_INPUT,
          "grenzen_32": tuple("-" + t for t in reversed(EDGE_POSITIVE)) + EDGE_POSITIVE}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def power_of_two(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def decode_scale(code: int) -> Fraction:
    """E8M0 heeft geen nul-schaal: ook code 0 is strikt positief."""
    if type(code) is not int or not 0 <= code <= 254:
        raise ValueError("Verwacht eindige E8M0-code 0..254; 255 (NaN) is niet toegestaan.")
    return power_of_two(code - 127)


def shorten_scale(code: int) -> int:
    """Bewaar zeven hoge bits. Controleer NaN VOOR we een bit verwijderen."""
    decode_scale(code)
    return code >> 1


def reconstruct_scale_code(short_code: int) -> int:
    if type(short_code) is not int or not 0 <= short_code <= 127:
        raise ValueError("De verkorte schaalcode moet een 7-bit integer (0..127) zijn.")
    return short_code << 1


def split_element(code: int) -> tuple[int, int, int]:
    if type(code) is not int or not 0 <= code < 256:
        raise ValueError(f"Ongeldige E4M3-code: {code!r}")
    if (code & 127) == 127:
        raise ValueError(f"E4M3-code {code:08b} is NaN, buiten de eindige proefscope.")
    return code >> 7, (code >> 3) & 15, code & 7


def decode_element(code: int) -> Fraction:
    """E4M3 zoals in experiment 003. Fraction bewaart geen negatieve nul."""
    sign, exponent, mantissa = split_element(code)
    f = Fraction(mantissa, 8)
    magnitude = f * power_of_two(-6) if exponent == 0 else (1 + f) * power_of_two(exponent - 7)
    return -magnitude if sign else magnitude


@lru_cache(maxsize=1)
def positive_values() -> tuple[Fraction, ...]:
    return tuple(decode_element(c) for c in range(127))


def quantize_rne(value: Fraction, *, negative_zero: bool = False) -> int:
    """Kies de dichtste eindige E4M3-code; even laatste bit bij een tie."""
    if not isinstance(value, Fraction) or type(negative_zero) is not bool:
        raise ValueError("Verwacht Fraction-invoer en een booleaanse nul-tekenvlag.")
    table = positive_values()
    magnitude = abs(value)
    code = min(range(127), key=lambda c: (abs(magnitude - table[c]), c & 1))
    sign = int(value < 0 or (value == 0 and negative_zero))
    return (sign << 7) | code


def validate_elements(codes: tuple[int, ...]) -> None:
    if not isinstance(codes, tuple) or len(codes) != BLOCK_SIZE:
        raise ValueError("Een blok vereist een tuple met precies 32 elementcodes.")
    for code in codes:
        split_element(code)


@dataclass(frozen=True)
class SourceBlock:
    """E4M3-elementen met de oorspronkelijke 8-bit schaalcode."""
    scale_code: int
    element_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        decode_scale(self.scale_code)
        validate_elements(self.element_codes)

    @property
    def scale(self) -> Fraction:
        return decode_scale(self.scale_code)


@dataclass(frozen=True)
class TargetBlock:
    """E4M3-elementen met ONZE 7-bit schaalcodering, nul impliciet."""
    scale_code_7bit: int
    element_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        reconstruct_scale_code(self.scale_code_7bit)
        validate_elements(self.element_codes)

    @property
    def scale_code(self) -> int:
        return reconstruct_scale_code(self.scale_code_7bit)

    @property
    def scale(self) -> Fraction:
        return decode_scale(self.scale_code)


def direct_block(values: tuple[Fraction, ...], short_code: int) -> TargetBlock:
    """Pad A gebruikt de originele invoer en dezelfde opgelegde X' als B/C."""
    target_scale = decode_scale(reconstruct_scale_code(short_code))
    return TargetBlock(short_code, tuple(quantize_rne(x / target_scale) for x in values))


def round_block_from_source(source: SourceBlock) -> TargetBlock:
    """Pad B ontvangt alleen de bron; hercodeer S_i / X' naar E4M3."""
    short_code = shorten_scale(source.scale_code)
    target_scale = decode_scale(reconstruct_scale_code(short_code))
    ratio = source.scale / target_scale
    codes = tuple(quantize_rne(ratio * decode_element(c), negative_zero=bool(c >> 7))
                  for c in source.element_codes)
    return TargetBlock(short_code, codes)


def change_scale_only(source: SourceBlock) -> TargetBlock:
    """Pad C: alle elementcodes bit voor bit gelijk; alleen schaal verkorten."""
    return TargetBlock(shorten_scale(source.scale_code), source.element_codes)


def decoded_block(block: SourceBlock | TargetBlock) -> tuple[Fraction, ...]:
    return tuple(block.scale * decode_element(c) for c in block.element_codes)


def self_test() -> dict:
    """Alle schaalcodes; E4M3-codes; twee schaalverhoudingen en blokinterfaces."""
    # Onafhankelijke integerwaardetabel in eenheden 1/512.
    units = tuple(range(8)) + tuple((8 + m) * 2 ** (e - 1)
                                   for e in range(1, 16) for m in range(8)
                                   if (e, m) != (15, 7))
    finite = tuple(c for c in range(256) if (c & 127) != 127)
    for c in finite:
        sign, u = c >> 7, units[c & 127]
        expected = Fraction(-u if sign else u, 512)
        require(decode_element(c) == expected, f"Decoder: {c}")
        require(quantize_rne(expected, negative_zero=bool(sign)) == c, f"Rondgang: {c}")
    for i in range(126):
        low, high = Fraction(units[i], 512), Fraction(units[i + 1], 512)
        tie = i if i % 2 == 0 else i + 1
        for value, expected in (((3 * low + high) / 4, i),
                                ((low + high) / 2, tie), ((low + 3 * high) / 4, i + 1)):
            require(quantize_rne(value) == expected, "Afronding positieve grens")
            require(quantize_rne(-value) == (128 | expected), "Afronding negatieve grens")
    require(quantize_rne(Fraction(1000)) == 126, "Saturatie +")
    require(quantize_rne(Fraction(-1000)) == 254, "Saturatie -")
    for c in range(255):
        expected_scale = Fraction(2 ** c, 2 ** 127)
        require(decode_scale(c) == expected_scale, f"Schaaldecoder: {c}")
        short = shorten_scale(c)
        full = reconstruct_scale_code(short)
        require(short == int(f"{c:08b}"[:7], 2), "Schaalbitselectie")
        require(full == c - c % 2, "Schaalreconstructie")
        ratio = decode_scale(full) / expected_scale
        require(ratio == (Fraction(1, 2) if c % 2 else Fraction(1)), "Schaalverhouding")
    for short in range(128):
        require(reconstruct_scale_code(short) == 2 * short, "Alle 7-bit codes")
    invalid_calls = (
        lambda: decode_element(127), lambda: decode_element(255),
        lambda: quantize_rne(1.0), lambda: split_element(True),
        lambda: decode_scale(255), lambda: shorten_scale(255),
        lambda: shorten_scale(-1), lambda: decode_scale(True),
        lambda: reconstruct_scale_code(128), lambda: reconstruct_scale_code(-1),
        lambda: SourceBlock(127, (0,) * 31), lambda: TargetBlock(128, (0,) * 32),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("Ongeldige invoer niet afgewezen")
    # Blokcontrole met alle 254 elementcodes bij de vijf proefschalen plus
    # uiterste schaalcodes. Alleen testblokken aanvullen, nooit onderzoeksdata.
    tested_scales = (0, 1, 125, 126, 127, 128, 129, 254)
    for scale_code in tested_scales:
        for start in range(0, len(finite), BLOCK_SIZE):
            part = finite[start:start + BLOCK_SIZE]
            codes = part + (0,) * (BLOCK_SIZE - len(part))
            source = SourceBlock(scale_code, codes)
            b, c = round_block_from_source(source), change_scale_only(source)
            require(source.element_codes == codes and source.scale_code == scale_code, "Bron gewijzigd")
            require(b.scale_code_7bit == c.scale_code_7bit, "Verschillende doelschalen")
            require(c.element_codes == codes, "C wijzigt elementcodes")
            for src, bc, cc in zip(codes, b.element_codes, c.element_codes, strict=True):
                mag = src & 127
                # Bij X' = X/2 moet B een E4M3-magnitude verdubbelen.
                # Dit kan exact t/m 224. Grotere magnitudes satureren naar 448.
                expected_mag = mag if scale_code % 2 == 0 else (
                    mag * 2 if mag < 8 else min(mag + 8, 126))
                require(bc == ((src & 128) | expected_mag), "B onafhankelijke bitreferentie")
                expected_ratio = Fraction(1, 2) if scale_code % 2 else Fraction(1)
                require(c.scale * decode_element(cc) == expected_ratio * source.scale * decode_element(src),
                        "C voldoet niet aan de theoretische schaalverhouding")
                require(bc >> 7 == src >> 7 and cc >> 7 == src >> 7, "Teken, inclusief nul")
    example = SourceBlock(127, (quantize_rne(Fraction("1.375")),) * 32)
    require(decoded_block(round_block_from_source(example))[0] == Fraction("1.375"), "Voorbeeld B")
    require(decoded_block(change_scale_only(example))[0] == Fraction("0.6875"), "Voorbeeld C")
    require(quantize_rne(Fraction("0.0005")) == 0, "Onder nulgrens bron")
    tiny = (Fraction("0.0005"),) * 32
    a = direct_block(tiny, shorten_scale(127))
    require(a.element_codes[0] == 1, "A met fijnere absolute subnormaalstap")
    require(round_block_from_source(SourceBlock(127, (0,) * 32)).element_codes[0] == 0,
            "B kan bronverlies naar nul niet herstellen")
    return {"status": "passed", "finite_E4M3_codes_checked": len(finite),
            "E4M3_nan_codes_rejected": [127, 255], "finite_E8M0_codes_checked": 255,
            "E8M0_nan_code_rejected_before_truncation": 255,
            "short_scale_codes_checked": 128, "rounding_intervals_both_signs": 126,
            "all_element_codes_tested_at_scale_codes": list(tested_scales),
            "C_element_codes_unchanged": True, "source_unchanged": True,
            "scope": "finite codes and selected rounding inputs; not all rational inputs"}


def mean_text(values: list[Fraction]) -> str | None:
    return str(sum(values, Fraction(0)) / len(values)) if values else None


def metrics(reference: tuple[Fraction, ...], actual: tuple[Fraction, ...],
            scale: Fraction, indices: list[int]) -> dict:
    errors = [abs(actual[i] - reference[i]) for i in indices]
    relative = [abs(actual[i] - reference[i]) / abs(reference[i])
                for i in indices if reference[i] != 0]
    reversals = [(i, j) for p, i in enumerate(indices) for j in indices[p + 1:]
                 if (reference[i] - reference[j]) * (actual[i] - actual[j]) < 0]
    return {"n_values": len(indices), "mae_exact": mean_text(errors),
            "mae_divided_by_source_scale_exact": mean_text([e / scale for e in errors]),
            "max_absolute_error_exact": str(max(errors)) if errors else None,
            "mean_absolute_relative_error_nonzero_reference_exact": mean_text(relative),
            "relative_error_denominator": len(relative),
            "zero_references_excluded_only_from_relative_error": len(indices) - len(relative),
            "n_numeric_mismatches": sum(actual[i] != reference[i] for i in indices),
            "n_nonzero_references_mapped_to_zero": sum(reference[i] != 0 and actual[i] == 0 for i in indices),
            "n_strict_order_reversal_pairs": len(reversals), "strict_order_reversal_pairs": reversals}


def compare_paths(a: TargetBlock, b: TargetBlock, source_scale: Fraction) -> dict:
    require(a.scale_code_7bit == b.scale_code_7bit, "Vergelijkingspaden moeten dezelfde X' gebruiken")
    aa, bb = decoded_block(a), decoded_block(b)
    gaps = [abs(x - y) for x, y in zip(aa, bb, strict=True)]
    return {"n_values": len(gaps),
            "n_element_code_mismatches": sum(x != y for x, y in zip(a.element_codes, b.element_codes, strict=True)),
            "n_numeric_mismatches": sum(g != 0 for g in gaps),
            "mean_absolute_gap_exact": mean_text(gaps),
            "mean_absolute_gap_divided_by_source_scale_exact": mean_text([g / source_scale for g in gaps]),
            "max_absolute_gap_exact": str(max(gaps)),
            "numeric_mismatch_indices": [i for i, g in enumerate(gaps) if g != 0]}


def analyse_case(values: tuple[Fraction, ...], source: SourceBlock,
                 targets: dict[str, TargetBlock]) -> dict:
    indices = list(range(BLOCK_SIZE))
    x_scale, target_scale = source.scale, targets[PATHS[0]].scale
    ratio = target_scale / x_scale
    blocks = {"source": source, **targets}
    decoded = {name: decoded_block(block) for name, block in blocks.items()}
    s = decoded["source"]
    input_metrics = {name: metrics(values, y, x_scale, indices) for name, y in decoded.items()}
    source_metrics = {name: metrics(s, decoded[name], x_scale, indices) for name in PATHS}
    groups = {"within_target_range": [i for i, v in enumerate(values) if abs(v) <= ELEMENT_MAX * target_scale],
              "outside_target_range": [i for i, v in enumerate(values) if abs(v) > ELEMENT_MAX * target_scale]}
    by_range = {group: {"indices": ii, "metrics_vs_input": {
        name: metrics(values, y, x_scale, ii) for name, y in decoded.items()}} for group, ii in groups.items()}
    source_exceeds = sum(abs(v) > ELEMENT_MAX * x_scale for v in values)
    target_exceeds = len(groups["outside_target_range"])
    b_exceeds = sum(abs(v) > ELEMENT_MAX * target_scale for v in s)
    require(targets[PATHS[2]].element_codes == source.element_codes, "C verandert elementcodes")
    require(all(decoded[PATHS[2]][i] == ratio * s[i] for i in indices), "C verhouding onjuist")
    for block in targets.values():
        require(block.scale == target_scale, "Doelschaal verschilt tussen paden")
    if ratio == 1:
        require(all(block.element_codes == source.element_codes for block in targets.values()),
                "Ongewijzigde schaal moet alle paden gelijk maken")
    warnings = []
    if source_exceeds:
        warnings.append(f"{source_exceeds}/32 oorspronkelijke waarden buiten het bronbereik; behouden.")
    if target_exceeds:
        warnings.append(f"{target_exceeds}/32 oorspronkelijke waarden buiten het doelbereik; behouden.")
    if b_exceeds:
        warnings.append(f"Pad B: {b_exceeds}/32 opgeslagen bronwaarden buiten het doelbereik; saturatie nodig.")
    rows = []
    for i, value in enumerate(values):
        row = {"index": i, "input_exact": str(value),
               "input_exceeds_source_range": abs(value) > ELEMENT_MAX * x_scale,
               "input_exceeds_target_range": abs(value) > ELEMENT_MAX * target_scale,
               "B_source_value_exceeds_target_range": abs(s[i]) > ELEMENT_MAX * target_scale,
               "C_expected_scale_ratio_exact": str(ratio),
               "C_measured_ratio_to_source_exact": str(decoded[PATHS[2]][i] / s[i]) if s[i] else None}
        for name, block in blocks.items():
            code, y = block.element_codes[i], decoded[name][i]
            sign, exponent, mantissa = split_element(code)
            row[name] = {"element_code": code, "element_bits": f"{code:08b}",
                         "element_value_exact": str(decode_element(code)), "value_exact": str(y),
                         "sign": sign, "exponent_field": exponent, "mantissa_field": mantissa,
                         "negative_zero": bool(sign and y == 0),
                         "element_is_subnormal": exponent == 0 and mantissa != 0,
                         "absolute_error_vs_input_exact": str(abs(y - value)),
                         "absolute_error_vs_input_divided_by_source_scale_exact": str(abs(y - value) / x_scale),
                         "absolute_error_vs_source_exact": str(abs(y - s[i]))}
        rows.append(row)
    short_code = targets[PATHS[0]].scale_code_7bit
    return {"source_scale_code_8bit": source.scale_code, "source_scale_bits": f"{source.scale_code:08b}",
            "target_scale_code_7bit": short_code, "target_scale_bits_7bit": f"{short_code:07b}",
            "reconstructed_E8M0_code": short_code << 1, "reconstructed_E8M0_bits": f"{short_code << 1:08b}",
            "source_scale_exact": str(x_scale), "target_scale_exact": str(target_scale),
            "scale_ratio_exact": str(ratio), "n_values": len(values),
            "metrics_vs_input": input_metrics, "metrics_vs_stored_source": source_metrics,
            "comparisons": {left + "_vs_" + right: compare_paths(targets[left], targets[right], x_scale)
                            for _, left, right in PAIRS},
            "metrics_by_input_target_range": by_range,
            "range_counts": {"original_input_exceeds_source_range": source_exceeds,
                             "original_input_exceeds_target_range": target_exceeds,
                             "B_source_value_exceeds_target_range": b_exceeds},
            "n_element_codes_changed_from_source": {name: sum(a != b for a, b in zip(
                source.element_codes, targets[name].element_codes, strict=True)) for name in PATHS},
            "invariants": {"C_element_codes_unchanged": True, "C_value_ratio_verified": True,
                           "all_target_scales_equal": True},
            "warnings": warnings, "rows": rows}


def save_json(path: Path, data: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true", help="Alleen controles, geen uitvoerbestanden.")
    parser.add_argument("--output-dir", type=Path, help="Optionele bovenmap voor nieuwe uitvoermappen.")
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print("Zelftests: OK | 255 eindige schaalcodes | 254 eindige E4M3-codes | NaN afgewezen")
        return
    input_values = {name: tuple(Fraction(t) for t in texts) for name, texts in INPUTS.items()}
    require(all(len(v) == BLOCK_SIZE for v in input_values.values()), "Elk invoerblok vereist 32 waarden")
    script = Path(__file__).resolve()
    script_bytes = script.read_bytes()
    root = args.output_dir or script.parent / "results" / "experiment_004_shared_scale"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")
    run_dir = Path(tempfile.mkdtemp(prefix=stamp, dir=root))
    input_path, source_path = run_dir / "invoer.json", run_dir / "bronblokken.json"
    save_json(input_path, {"design": "same fixed original inputs at each scale, no rescaling of inputs",
                          "blocks": INPUTS, "source_scale_codes": SCALE_CODES,
                          "source_scale_values_exact": [str(decode_scale(c)) for c in SCALE_CODES],
                          "missing_values": 0, "excluded_values": 0})
    input_digest = sha256(input_path.read_bytes())
    # Een bron per conditie. Alleen dit gedeelte mag broncodes uit x berekenen.
    stored_sources = {}
    for c in SCALE_CODES:
        for name, values in input_values.items():
            case_id = f"scale_{c}_{name}"
            codes = tuple(quantize_rne(v / decode_scale(c)) for v in values)
            source = SourceBlock(c, codes)
            stored_sources[case_id] = {"input_block": name, "scale_code": c,
                                       "element_codes": source.element_codes}
    save_json(source_path, {"element_format": "E4M3", "block_size": BLOCK_SIZE,
                            "scale_format": "E8M0", "cases": stored_sources})
    source_digest = sha256(source_path.read_bytes())
    with source_path.open(encoding="utf-8") as stream:
        reloaded = json.load(stream)
    results = {}
    for case_id, stored in reloaded["cases"].items():
        source = SourceBlock(stored["scale_code"], tuple(stored["element_codes"]))
        before = (source.scale_code, source.element_codes)
        short_code = shorten_scale(source.scale_code)
        values = input_values[stored["input_block"]]
        # B en C krijgen alleen de herlezen bron. A en analyse mogen x gebruiken.
        targets = {PATHS[0]: direct_block(values, short_code),
                   PATHS[1]: round_block_from_source(source),
                   PATHS[2]: change_scale_only(source)}
        require((source.scale_code, source.element_codes) == before, "Bron in geheugen gewijzigd")
        results[case_id] = {"input_block": stored["input_block"], **analyse_case(values, source, targets)}
    require(sha256(source_path.read_bytes()) == source_digest, "Bronbestand gewijzigd")
    require(sha256(input_path.read_bytes()) == input_digest, "Invoerbestand gewijzigd")
    require(script.read_bytes() == script_bytes, "Script veranderde tijdens uitvoering")
    # Bewaar exacte scriptbytes: toekomstige herberekening hoeft geen oud script te raden.
    with (run_dir / script.name).open("xb") as stream:
        stream.write(script_bytes)
    payload = {
        "experiment": EXPERIMENT, "script_version": VERSION, "run_id": run_dir.name,
        "status": "completed_exploratory_synthetic_test_not_model_benchmark",
        "script_sha256": sha256(script_bytes), "source_blocks_sha256": source_digest,
        "input_file_sha256": input_digest,
        "environment": {"python": sys.version, "system": platform.system(), "machine": platform.machine()},
        "provenance": {"standalone": True, "input_templates": "literal values from experiment 003",
                       "same_original_inputs_across_scales": True, "source_created_once_per_condition": True,
                       "source_reloaded_before_B_C": True, "B_C_inputs": "stored block only",
                       "old_experiments_read_or_modified_at_runtime": False},
        "config": {"element_format_source_and_target": "E4M3", "exponent_bits": 4, "mantissa_bits": 3,
                   "element_bias": 7, "block_size": BLOCK_SIZE, "source_scale_format": "E8M0",
                   "source_scale_codes": list(SCALE_CODES), "scale_selection": "fixed, not optimized",
                   "target_scale_encoding": "custom 7-bit prefix, reconstruct E8M0 via appended zero",
                   "scale_truncation": "c7 = c8 >> 1; c8_reconstructed = c7 << 1",
                   "target_scale_nan_support": False,
                   "source_and_A_B_rounding": "nearest_ties_even_finite_saturation",
                   "path_A": "X' * Q_E4M3(x / X')",
                   "path_B": "X' * Q_E4M3((X * decoded_stored_element) / X')",
                   "path_C": "X' * decoded_stored_element; all element codes unchanged",
                   "numeric_scope": "finite exact rationals; no FP32 arithmetic-limit simulation",
                   "normalization": "divide absolute error/gap by ORIGINAL scale X, not X'",
                   "relative_error": "exclude zero reference only, report denominator",
                   "range_flags": "magnitude comparisons, not a count of all saturation/rounding events",
                   "ordering_scope": "within each block; no cross-block ordering test",
                   "missing_values": 0, "excluded_values": 0, "random_seed": None,
                   "storage": "JSON research log; not bitpacked; no measured hardware savings"},
        "format_reference": {"OCP_MX": "v1.0 Sep 2023, sec. 5.2/5.4",
                             "E8M0_reference": "NVIDIA CUDA Math API 12.8.1 __nv_fp8_e8m0",
                             "custom_scale_truncation": "own experimental design, not prescribed by sources"},
        "self_tests": tests, "n_conditions": len(results),
        "mean_scope": "32 elements per condition; named range subsets additionally; no pooled mean",
        "cases": results,
    }
    save_json(run_dir / "resultaten.json", payload)
    print("Experiment 004: shared scale 8 -> 7 bits | E4M3 blijft E4M3 | alleen CPU")
    print("Zelftests: OK (255 eindige schaalcodes; 254 eindige E4M3-codes; NaN afgewezen)")
    print("A = direct | B = rounded_from_source | C = scale_only")
    print("Twee vaste invoerblokken bij vijf schalen: 10 afzonderlijke condities.")
    for case_id, result in results.items():
        print(f"\n{case_id}: X={result['source_scale_exact']} -> X'={result['target_scale_exact']}"
              f" | verhouding={result['scale_ratio_exact']}")
        print(f"Schaalbits: {result['source_scale_bits']} -> {result['target_scale_bits_7bit']}"
              f" -> {result['reconstructed_E8M0_bits']} (impliciete nul)")
        for label, left, right in PAIRS:
            comp = result["comparisons"][left + "_vs_" + right]
            gap = Fraction(comp["mean_absolute_gap_exact"])
            print(f"{label}: {comp['n_element_code_mismatches']}/32 codeverschillen; "
                  f"gemiddeld absoluut verschil={float(gap):.8f}")
        for label, name in (("bron", "source"), ("A", PATHS[0]), ("B", PATHS[1]), ("C", PATHS[2])):
            measure = result["metrics_vs_input"][name]
            print(f"MAE {label} t.o.v. invoer: {float(Fraction(measure['mae_exact'])):.8f}"
                  f" | MAE/X: {float(Fraction(measure['mae_divided_by_source_scale_exact'])):.8f}")
        rel = result["metrics_vs_stored_source"][PATHS[2]]
        ratio_error = rel["mean_absolute_relative_error_nonzero_reference_exact"]
        print(f"C: 0/32 elementcodes gewijzigd; relatieve fout t.o.v. niet-nulle bronwaarden: {ratio_error}")
        for warning in result["warnings"]:
            print(f"AANDACHTSPUNT: {warning}")
    example = next(row for row in results["scale_127_basis_32"]["rows"] if row["input_exact"] == "11/8")
    print("\nBerekend voorbeeld bij X=1, X'=1/2, invoer 1.375: " + ", ".join(
        f"{label}={float(Fraction(example[name]['value_exact'])):g}"
        for label, name in (("bron", "source"), ("A", PATHS[0]), ("B", PATHS[1]), ("C", PATHS[2]))))
    print(f"Resultaten opgeslagen in: {run_dir}")
    print("Geen netwerk, training, accuracy-, snelheids-, opslag- of energiemeting uitgevoerd.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, AssertionError, KeyError, TypeError, ZeroDivisionError) as error:
        print(f"FOUT: {error}", file=sys.stderr)
        sys.exit(1)
