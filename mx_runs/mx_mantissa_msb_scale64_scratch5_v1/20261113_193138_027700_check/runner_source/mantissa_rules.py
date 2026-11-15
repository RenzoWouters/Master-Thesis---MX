"""Exact, finite E2M3 -> E2M1 code mappings at an unchanged shared scale.

The formats use one sign bit, two exponent bits (bias 1), and respectively
three or one mantissa bits. Every encoding is finite; exponent zero contains
zero/subnormal values. The sign of a zero encoding is retained by the mappings,
although its decoded Fraction value is zero.

``msb_bestexp`` retains the ORIGINAL mantissa LSB, fixes sign and shared scale,
and searches all four target exponents for minimum absolute numerical error.
It is a constrained numerical optimum, not an accuracy optimum for a network.
No tensor libraries or external dependencies are required.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache


METHODS = ('lsb_trunc', 'rne', 'msb_bestexp')
LABELS = {
    'lsb_trunc': 'Laagste twee mantissabits verwijderen',
    'rne': 'Afronden naar dichtstbijzijnde (RNE)',
    'msb_bestexp': 'Mantissa-LSB behouden + beste exponent',
}


def _check_code(code: int, bits: int) -> None:
    if type(bits) is not int or bits not in (4, 6):
        raise ValueError('bits must be 6 (E2M3) or 4 (E2M1).')
    if type(code) is not int or not 0 <= code < (1 << bits):
        raise ValueError(f'code must be an integer in 0..{(1 << bits) - 1}.')


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f'Unknown method {method!r}; choose one of {METHODS!r}.')


def decode_code(code: int, bits: int) -> Fraction:
    """Return the exact numerical value, with a shared scale of one.

    Signed zero decodes to Fraction(0); its sign remains available in the code.
    Both formats have exponent bias 1 and no infinity or NaN encodings.
    """
    _check_code(code, bits)
    mantissa_bits = bits - 3
    sign = (code >> (bits - 1)) & 1
    exponent = (code >> mantissa_bits) & 3
    mantissa = code & ((1 << mantissa_bits) - 1)
    if exponent == 0:
        magnitude = Fraction(mantissa, 1 << mantissa_bits)
    else:
        magnitude = Fraction((1 << mantissa_bits) + mantissa,
                             1 << mantissa_bits) * (1 << (exponent - 1))
    return -magnitude if sign else magnitude


def map_code(code: int, method: str) -> int:
    """Map one six-bit E2M3 encoding to a four-bit E2M1 encoding.

    RNE ties select the target magnitude code with an even least significant
    bit. MSB/exponent-search ties first minimize |E' - E| and then E'.
    """
    _check_code(code, 6)
    _check_method(method)
    if method == 'lsb_trunc':
        return code >> 2

    sign = (code >> 5) & 1
    magnitude = abs(decode_code(code, 6))
    if method == 'rne':
        # Positive magnitude codes are ordered by value. Adjacent ties have
        # opposite parity; the last key is only a deterministic fallback.
        target_magnitude = min(range(8), key=lambda candidate: (
            abs(decode_code(candidate, 4) - magnitude), candidate & 1,
            candidate))
    else:
        retained_lsb = code & 1
        original_exponent = (code >> 3) & 3
        target_exponent = min(range(4), key=lambda exponent: (
            abs(decode_code((exponent << 1) | retained_lsb, 4) - magnitude),
            abs(exponent - original_exponent), exponent))
        target_magnitude = (target_exponent << 1) | retained_lsb
    return (sign << 3) | target_magnitude


@lru_cache(maxsize=len(METHODS))
def _lookup(method: str) -> tuple[int, ...]:
    return tuple(map_code(code, method) for code in range(64))


def lookup(method: str) -> tuple[int, ...]:
    """Return an immutable, 64-entry source-code-indexed lookup table."""
    _check_method(method)
    return _lookup(method)


def _error_comparison(error: Fraction, baseline: Fraction) -> str:
    return 'better' if error < baseline else 'worse' if error > baseline else 'equal'


def numeric_rows() -> list[dict]:
    """Return 192 JSON-serializable diagnostic rows, ordered by code/method.

    Error is target minus source. Magnitude error is |target| minus |source|.
    Relative absolute error is undefined for zero sources and is then None.
    All decisions and intermediate arithmetic use exact Fractions; these
    dyadic values and absolute/squared errors have exact float representations.
    """
    rows = []
    for source_code in range(64):
        source = decode_code(source_code, 6)
        baseline_error = abs(decode_code(map_code(source_code, 'lsb_trunc'), 4)
                             - source)
        for method in METHODS:
            target_code = map_code(source_code, method)
            target = decode_code(target_code, 4)
            error = target - source
            absolute_error = abs(error)
            rows.append({
                'source_code': source_code,
                'source_bits': format(source_code, '06b'),
                'source_sign': source_code >> 5,
                'source_exponent': (source_code >> 3) & 3,
                'source_mantissa': source_code & 7,
                'source_value': float(source),
                'method': method,
                'target_code': target_code,
                'target_bits': format(target_code, '04b'),
                'target_sign': target_code >> 3,
                'target_exponent': (target_code >> 1) & 3,
                'target_mantissa': target_code & 1,
                'target_value': float(target),
                'error': float(error),
                'abs_error': float(absolute_error),
                'squared_error': float(error * error),
                'magnitude_error': float(abs(target) - abs(source)),
                'relative_abs_error': (float(absolute_error / abs(source))
                                       if source else None),
                'exact': error == 0,
                'comparison_to_lsb': _error_comparison(absolute_error, baseline_error),
            })
    return rows


def _population_metrics(codes: tuple[int, ...], method: str) -> dict:
    sources = [decode_code(code, 6) for code in codes]
    targets = [decode_code(map_code(code, method), 4) for code in codes]
    errors = [abs(target - source) for source, target in zip(sources, targets)]
    baseline = [abs(decode_code(map_code(code, 'lsb_trunc'), 4) - source)
                for code, source in zip(codes, sources)]
    relative = [error / abs(source) for source, error in zip(sources, errors)
                if source != 0]
    count = len(codes)
    exact_metrics = {
        'mae': sum(errors, Fraction(0)) / count,
        'mse': sum((error * error for error in errors), Fraction(0)) / count,
        'max_abs_error': max(errors),
        'magnitude_bias': sum((abs(target) - abs(source)
                               for source, target in zip(sources, targets)),
                              Fraction(0)) / count,
        'mean_relative_abs_error': (sum(relative, Fraction(0)) / len(relative)
                                    if relative else None),
    }
    result = {key: float(value) if value is not None else None
              for key, value in exact_metrics.items()}
    result.update({
        'relative_error_count': len(relative),
        'exact_count': sum(error == 0 for error in errors),
        'better_than_lsb': sum(error < base for error, base in zip(errors, baseline)),
        'worse_than_lsb': sum(error > base for error, base in zip(errors, baseline)),
        'equal_to_lsb': sum(error == base for error, base in zip(errors, baseline)),
        'exact_fractions': {key: str(value) if value is not None else None
                            for key, value in exact_metrics.items()},
    })
    return result


def _monotonicity(method: str) -> dict:
    values = [decode_code(map_code(code, method), 4) for code in range(32)]
    violations = [
        {'source_code_before': code, 'source_code_after': code + 1,
         'target_value_before': float(values[code]),
         'target_value_after': float(values[code + 1])}
        for code in range(31) if values[code + 1] < values[code]
    ]
    return {
        'ordering': 'source magnitude codes 0..31, increasing decoded value',
        'nondecreasing': not violations,
        'decreases': len(violations),
        'violations': violations,
    }


def _exhaustive_checks() -> dict:
    rne_nearest = True
    rne_ties_even = True
    msb_optimal = True
    msb_tiebreak = True
    msb_retains_lsb = True
    for code in range(64):
        source = decode_code(code, 6)
        sign = code >> 5
        rne_target = map_code(code, 'rne')
        all_target_errors = [abs(decode_code(target, 4) - source)
                             for target in range(16)]
        rne_nearest &= abs(decode_code(rne_target, 4) - source) == min(all_target_errors)
        same_sign_targets = [(sign << 3) | magnitude for magnitude in range(8)]
        closest = [target for target in same_sign_targets
                   if abs(decode_code(target, 4) - source) == min(all_target_errors)]
        if len(closest) > 1:
            rne_ties_even &= (rne_target & 1) == 0

        msb_target = map_code(code, 'msb_bestexp')
        original_exponent = (code >> 3) & 3
        candidates = [(sign << 3) | (exponent << 1) | (code & 1)
                      for exponent in range(4)]
        candidate_errors = {target: abs(decode_code(target, 4) - source)
                            for target in candidates}
        best_error = min(candidate_errors.values())
        best_candidates = [target for target in candidates
                           if candidate_errors[target] == best_error]
        tie_winner = min(best_candidates, key=lambda target: (
            abs(((target >> 1) & 3) - original_exponent), (target >> 1) & 3))
        msb_optimal &= candidate_errors[msb_target] == best_error
        msb_tiebreak &= msb_target == tie_winner
        msb_retains_lsb &= (msb_target & 1) == (code & 1)
    return {
        'codes_checked': 64,
        'lsb_mapping_equals_code_shift': all(map_code(code, 'lsb_trunc') == code >> 2
                                           for code in range(64)),
        'all_target_codes_in_range': all(0 <= map_code(code, method) < 16
                                        for code in range(64) for method in METHODS),
        'all_signs_preserved': all((map_code(code, method) >> 3) == (code >> 5)
                                  for code in range(64) for method in METHODS),
        'signed_zero_codes_preserved': all(map_code(0, method) == 0
                                          and map_code(32, method) == 8
                                          for method in METHODS),
        'rne_is_globally_nearest': rne_nearest,
        'rne_ties_choose_even_magnitude_code': rne_ties_even,
        'msb_preserves_original_mantissa_lsb': msb_retains_lsb,
        'msb_minimizes_error_over_all_four_exponents': msb_optimal,
        'msb_exponent_tiebreak_respected': msb_tiebreak,
    }


def numeric_summary() -> dict:
    """Return exact-count and numerical-error diagnostics for four populations.

    The populations are uniformly weighted CODE populations, not an observed
    model-weight distribution. The 31- and 24-code subsets use positive codes.
    Relative means always exclude zero sources and state their denominator.
    Exact Fraction strings accompany aggregate floats for reproducibility.
    """
    populations = {
        'all64': (tuple(range(64)), 'All 64 encodings, including both signed zeros'),
        'positive32': (tuple(range(32)), 'Positive-sign encodings, including +0'),
        'positive_nonzero31': (tuple(range(1, 32)), 'Positive nonzero encodings'),
        'positive_normal24': (tuple(range(8, 32)), 'Positive normal encodings; exponent 1..3'),
    }
    return {
        'schema': 'mx-e2m3-e2m1-mantissa-rules-numeric-summary-v1',
        'methods': list(METHODS),
        'labels': dict(LABELS),
        'assumptions': {
            'source_format': 'E2M3',
            'target_format': 'E2M1',
            'exponent_bias': 1,
            'shared_scale': 1,
            'shared_scale_unchanged': True,
            'population_weighting': 'uniform within each reported code population',
            'population_is_model_weight_distribution': False,
            'magnitude_bias_definition': 'mean(abs(target) - abs(source))',
            'relative_error_definition': 'abs(target - source) / abs(source); nonzero source only',
            'msb_constraint': 'original mantissa LSB, sign and shared scale fixed',
            'msb_objective': 'minimum absolute numerical error over all four target exponents',
            'msb_tiebreak': 'minimum abs(target exponent - source exponent), then lowest target exponent',
            'network_accuracy_optimality_claimed': False,
        },
        'populations': {
            name: {'count': len(codes), 'description': description,
                   'methods': {method: _population_metrics(codes, method)
                               for method in METHODS}}
            for name, (codes, description) in populations.items()
        },
        'monotonicity': {method: _monotonicity(method) for method in METHODS},
        'exhaustive_checks': _exhaustive_checks(),
    }


if __name__ == '__main__':
    import json
    print(json.dumps(numeric_summary(), indent=2, ensure_ascii=False, allow_nan=False))
