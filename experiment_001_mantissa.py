"""Experiment 001: mantissatruncatie van MXFP6 E2M3 naar MXFP4 E2M1.

Verkennende numerieke proef, geen training en geen hardwarebenchmark.
Eigen, expliciet gekozen proefopstelling:
- een vast blok van 32 synthetische, eindige invoerwaarden;
- shared scale X = 1 (E8M0-code 127), onveranderd in alle paden;
- bron: round-to-nearest, ties-to-even naar E2M3;
- truncatie: twee laagste mantissabits weg uit de opgeslagen broncodes;
- referentie: rechtstreeks round-to-nearest, ties-to-even naar E2M1;
- saturatie naar de grootste eindige waarde bij overschrijding;
- subnormalen blijven ondersteund, inclusief afronding naar nul.

Formaatbron: OCP Microscaling Formats (MX) Specification, v1.0, Sep 2023,
sec. 5.2-5.4, p. 10-14 (met name tabellen 4, 5 en 7).
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
Projectbron: Rouhani et al., Microscaling Data Formats for Deep Learning,
2023, p. 2-3, sec. 2 en tabel 1. De proefopstelling zelf is geen paperresultaat.

Alle berekeningen gebruiken exacte breuken; floats dienen alleen voor weergave.
Er zijn geen externe pakketten nodig. Er is geen toeval in de numerieke proef;
een random seed is daarom niet van toepassing. Elke run krijgt een nieuwe map.
JSON is hier een leesbaar onderzoekslog, geen compact MX-geheugenformaat.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

VERSION = "1.0.0"
INPUT_TEXT = (
    "-7.5", "-6.5", "-5.75", "-3.875", "-2.875", "-1.875", "-1.625", "-1.375",
    "-0.875", "-0.6875", "-0.375", "-0.3125", "-0.1875", "-0.125", "-0.0625", "0",
    "0.03125", "0.0625", "0.125", "0.1875", "0.3125", "0.375", "0.6875", "0.875",
    "1.375", "1.625", "1.875", "2.875", "3.875", "5.75", "6.5", "7.5",
)


def require(condition: bool, message: str) -> None:
    """Een test die ook actief blijft als Python met -O wordt gestart."""
    if not condition:
        raise AssertionError(message)


def decode_element(code: int, mantissa_bits: int) -> Fraction:
    """Lees een unsigned bitpatroon S EE M... als een exacte getalwaarde."""
    if mantissa_bits not in (1, 3):
        raise ValueError("Deze proef ondersteunt alleen E2M1 en E2M3.")
    if type(code) is not int or not 0 <= code < (1 << (3 + mantissa_bits)):
        raise ValueError("Bitpatroon buiten het gekozen elementformaat.")
    sign = code >> (2 + mantissa_bits)
    exponent = (code >> mantissa_bits) & 0b11
    fraction = Fraction(code & ((1 << mantissa_bits) - 1), 1 << mantissa_bits)
    # Beide formaten hebben bias 1. Exponentveld 0 betekent subnormaal/nul.
    magnitude = fraction if exponent == 0 else (1 + fraction) * (1 << (exponent - 1))
    return -magnitude if sign else magnitude


def quantize_rne(value: Fraction, mantissa_bits: int) -> int:
    """Dichtste code; bij gelijke afstand wint de code met even laagste bit.

    Alleen eindige Fraction-invoer. Buiten het eindige bereik wordt verzadigd.
    Het enumereren van alle positieve codes is bewust eenvoudig, niet snel.
    Negatieve waarden die naar nul afronden, behouden hun tekenbit.
    """
    if not isinstance(value, Fraction) or mantissa_bits not in (1, 3):
        raise ValueError("Verwacht een Fraction en mantissa_bits 1 of 3.")
    magnitude = abs(value)
    magnitude_code = min(
        range(1 << (2 + mantissa_bits)),
        key=lambda code: (abs(magnitude - decode_element(code, mantissa_bits)), code & 1),
    )
    return (int(value < 0) << (2 + mantissa_bits)) | magnitude_code


@dataclass(frozen=True)
class MXBlock:
    """Onveranderlijke bronrepresentatie: alleen codes en formaatmetadata."""
    mantissa_bits: int
    scale_code: int
    element_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.mantissa_bits not in (1, 3) or self.scale_code != 127:
            raise ValueError("Deze proef vereist E2M1/E2M3 met vaste schaalcode 127.")
        if not isinstance(self.element_codes, tuple) or len(self.element_codes) != 32:
            raise ValueError("Verwacht een tuple met precies 32 elementcodes.")
        for code in self.element_codes:
            decode_element(code, self.mantissa_bits)


def truncate_block(source: MXBlock) -> MXBlock:
    """Leid E2M1 uitsluitend uit opgeslagen E2M3-codes af, zonder herkwantisatie."""
    if source.mantissa_bits != 3:
        raise ValueError("De bron moet E2M3 zijn.")
    # Codes zijn unsigned containers, GEEN signed integergewichten.
    # S EE MMM -> S EE M. Teken- en exponentveld behouden hun waarde.
    return MXBlock(1, source.scale_code, tuple(code >> 2 for code in source.element_codes))


def self_test() -> None:
    """Controleer codering, alle afrondingsgrenzen en alle 64 truncatiecodes."""
    expected_fp4 = tuple(Fraction(s) for s in ("0", "0.5", "1", "1.5", "2", "3", "4", "6"))
    require(tuple(decode_element(c, 1) for c in range(8)) == expected_fp4, "FP4-waardetabel")
    require(decode_element(1, 3) == Fraction(1, 8), "FP6 kleinste subnormaal")
    require(decode_element(8, 3) == 1, "FP6 kleinste normaal")
    require(decode_element(31, 3) == Fraction(15, 2), "FP6 grootste normaal")
    for m in (1, 3):
        count = 1 << (2 + m)
        for code in range(count):
            value = decode_element(code, m)
            require(quantize_rne(value, m) == code, f"Rondgang m={m}, code={code}")
            if code:
                require(quantize_rne(-value, m) == (count | code), "Negatieve rondgang")
        for code in range(count - 1):
            low = decode_element(code, m)
            high = decode_element(code + 1, m)
            midpoint = (low + high) / 2
            expected = code if code % 2 == 0 else code + 1
            require(quantize_rne(midpoint, m) == expected, "Positieve tie-to-even")
            require(quantize_rne(-midpoint, m) == (count | expected), "Negatieve tie-to-even")
            require(quantize_rne((3 * low + high) / 4, m) == code, "Onder afrondingsgrens")
            require(quantize_rne((low + 3 * high) / 4, m) == code + 1, "Boven afrondingsgrens")
        require(quantize_rne(Fraction(100), m) == count - 1, "Positieve saturatie")
        require(quantize_rne(Fraction(-100), m) == 2 * count - 1, "Negatieve saturatie")
    for start in (0, 32):
        source = MXBlock(3, 127, tuple(range(start, start + 32)))
        before = source.element_codes
        target = truncate_block(source)
        require(source.element_codes == before, "Bron is gewijzigd")
        require(target.scale_code == source.scale_code, "Schaal is gewijzigd")
        for source_code, target_code in zip(source.element_codes, target.element_codes, strict=True):
            require(source_code >> 5 == target_code >> 3, "Tekenbit is gewijzigd")
            require((source_code >> 3) & 3 == (target_code >> 1) & 3, "Exponent is gewijzigd")
            magnitude = abs(decode_element(source_code, 3))
            # Onafhankelijke definitie: grootste doelwaarde <= bronmagnitude.
            reference = max(c for c in range(8) if decode_element(c, 1) <= magnitude)
            reference |= (source_code >> 5) << 3
            require(target_code == reference, "Truncatie wijkt af van referentie")


def save_json(path: Path, data: object) -> None:
    # Exclusief aanmaken: een bestaand bestand nooit overschrijven.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    self_test()
    values = tuple(Fraction(text) for text in INPUT_TEXT)
    require(len(values) == 32, "Het synthetische blok moet 32 waarden bevatten.")
    script = Path(__file__).resolve()
    output_root = script.parent / "results" / "experiment_001_mantissa"
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_")
    run_dir = Path(tempfile.mkdtemp(prefix=stamp, dir=output_root))

    # Stap A: codeer de hoge precisie eenmalig en bewaar alleen die bron.
    source_path = run_dir / "bronblok.json"
    save_json(source_path, {
        "format": "MXFP6_E2M3", "mantissa_bits": 3, "scale_code": 127,
        "element_codes": [quantize_rne(value, 3) for value in values],
    })
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    with source_path.open(encoding="utf-8") as stream:
        stored = json.load(stream)
    source = MXBlock(stored["mantissa_bits"], stored["scale_code"], tuple(stored["element_codes"]))

    # Stap B: dit pad krijgt uitsluitend de opgeslagen bron, NIET values.
    truncated = truncate_block(source)
    # Stap C: alleen de vergelijkingsreferentie mag de originele invoer gebruiken.
    direct = MXBlock(1, 127, tuple(quantize_rne(value, 1) for value in values))
    require(hashlib.sha256(source_path.read_bytes()).hexdigest() == source_digest, "Bronbestand gewijzigd")

    rows = []
    errors_source, errors_truncated, errors_direct, errors_gap = [], [], [], []
    for i, value in enumerate(values):
        source_value = decode_element(source.element_codes[i], 3)
        truncated_value = decode_element(truncated.element_codes[i], 1)
        direct_value = decode_element(direct.element_codes[i], 1)
        errors_source.append(abs(source_value - value))
        errors_truncated.append(abs(truncated_value - value))
        errors_direct.append(abs(direct_value - value))
        errors_gap.append(abs(truncated_value - direct_value))
        rows.append({
            "index": i, "input_decimal": INPUT_TEXT[i], "input_exact": str(value),
            "source_bits": f"{source.element_codes[i]:06b}",
            "truncated_bits": f"{truncated.element_codes[i]:04b}",
            "direct_bits": f"{direct.element_codes[i]:04b}",
            "source_value_exact": str(source_value),
            "truncated_value_exact": str(truncated_value),
            "direct_value_exact": str(direct_value),
            "gap_exact": str(abs(truncated_value - direct_value)),
            "codes_equal": truncated.element_codes[i] == direct.element_codes[i],
            "input_exceeds_source_range": abs(value) > Fraction(15, 2),
            "input_exceeds_target_range": abs(value) > 6,
        })
    metrics = {
        "n_values": len(values),
        "n_code_mismatches": sum(not row["codes_equal"] for row in rows),
        "n_numeric_mismatches": sum(error != 0 for error in errors_gap),
        "mae_source_vs_input_exact": str(sum(errors_source) / len(values)),
        "mae_truncated_vs_input_exact": str(sum(errors_truncated) / len(values)),
        "mae_direct_vs_input_exact": str(sum(errors_direct) / len(values)),
        "mean_absolute_qt_gap_exact": str(sum(errors_gap) / len(values)),
        "max_absolute_qt_gap_exact": str(max(errors_gap)),
    }
    save_json(run_dir / "resultaten.json", {
        "experiment": "001_mantissa", "script_version": VERSION,
        "status": "exploratory_synthetic_test_not_model_benchmark",
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "source_block_sha256": source_digest,
        "environment": {"python": sys.version, "system": platform.system(), "machine": platform.machine()},
        "config": {
            "source": "MXFP6_E2M3", "target": "MXFP4_E2M1", "block_size": 32,
            "shared_scale": "1", "shared_scale_E8M0_code": 127,
            "scale_selection": "fixed_by_experiment_not_recomputed",
            "source_rounding": "nearest_ties_even", "reference_rounding": "nearest_ties_even",
            "truncation": "discard_two_lowest_mantissa_bits", "overflow": "saturate",
            "input_scope": "finite_exact_rationals", "missing_values": 0, "excluded_values": 0,
            "subnormals": "supported", "random_seed": None,
            "storage": "JSON_debug_representation_not_bitpacked_MX",
        },
        "self_tests": "passed", "metrics": metrics, "rows": rows,
    })
    print("Experiment 001: MXFP6 E2M3 -> MXFP4 E2M1")
    print("Zelftests: OK (inclusief alle 64 bronbitpatronen)")
    print("Blok: 32 vaste waarden | shared scale: 1 | alleen CPU")
    print(f"Verschillen truncatie vs directe afronding: {metrics['n_code_mismatches']}/32")
    print(f"Gemiddeld absoluut verschil tussen beide paden: {float(sum(errors_gap) / 32):.6f}")
    print("Voorbeeld input 1.375: bron 1.375, truncatie 1.0, directe afronding 1.5")
    print(f"Resultaten opgeslagen in: {run_dir}")
    print("Geen netwerk, training, snelheidsmeting of energiemeting uitgevoerd.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, AssertionError) as error:
        print(f"FOUT: {error}", file=sys.stderr)
        sys.exit(1)
