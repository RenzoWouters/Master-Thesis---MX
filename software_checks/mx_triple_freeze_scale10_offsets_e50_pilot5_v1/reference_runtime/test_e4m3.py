"""Small, deterministic codec tests; no dataset, model training or file writes.

The scalar oracle below does not call codec functions and uses exact rational
element values. Tests cover every finite E4M3 bit pattern and both NaN codes.
The STE test checks the declared surrogate derivative, not the true derivative
of the discontinuous quantizer.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import io
import json
import math

import torch

import mx_e4m3 as mx


def _magnitude(code: int) -> Fraction:
    """Independent exact scalar E4M3 decoding of a nonnegative finite code."""
    if not 0 <= code < 127:
        raise ValueError('Nonnegative finite E4M3 code required.')
    exponent, mantissa = divmod(code, 8)
    if exponent == 0:
        return Fraction(mantissa, 512)
    return Fraction(8 + mantissa, 8) * Fraction(2) ** (exponent - 7)


_LEVELS = tuple(_magnitude(code) for code in range(127))


def _decode(code: int) -> float:
    if not 0 <= code <= 255 or code % 128 == 127:
        raise ValueError('Finite E4M3 code required.')
    magnitude = float(_LEVELS[code % 128])
    return math.copysign(magnitude, -1.0 if code >= 128 else 1.0)


def _encode(value: float) -> int:
    """Exhaustive nearest-level oracle, with even-code tie breaking."""
    if not math.isfinite(value):
        raise ValueError('Finite input required.')
    target = Fraction.from_float(abs(value))
    code = min(range(127), key=lambda c: (abs(target - _LEVELS[c]), c % 2))
    return code + (128 if math.copysign(1.0, value) < 0 else 0)


def _oracle_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar grouping, source-subnormal handling and scale-selection oracle."""
    rows = weight.detach().cpu().reshape(weight.shape[0], -1).tolist()
    codes, scales, decoded = [], [], []
    tiny = float(torch.finfo(torch.float32).tiny)
    for row in rows:
        source = [math.copysign(0.0, x) if 0 < abs(x) < tiny else x for x in row]
        padded = source + [0.0] * ((-len(source)) % 32)
        code_row, scale_row, decoded_row = [], [], []
        for offset in range(0, len(padded), 32):
            block = padded[offset:offset + 32]
            maximum = max(abs(x) for x in block)
            k = 0 if maximum == 0 else math.frexp(maximum)[1] - 9
            k = min(127, max(-127, k))
            block_codes = [_encode(math.ldexp(x, -k)) for x in block]
            code_row.append(block_codes)
            scale_row.append(k + 127)
            decoded_row.extend(math.ldexp(_decode(c), k) for c in block_codes)
        codes.append(code_row)
        scales.append(scale_row)
        decoded.append(decoded_row[:len(row)])
    return (torch.tensor(codes, dtype=torch.uint8), torch.tensor(scales, dtype=torch.uint8),
            torch.tensor(decoded, dtype=torch.float32).reshape(weight.shape))


def _exact(a: torch.Tensor, b: torch.Tensor) -> bool:
    a, b = a.detach().cpu(), b.detach().cpu()
    return (a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
            and (not a.is_floating_point() or torch.equal(torch.signbit(a), torch.signbit(b))))


def _comparison_details(actual: torch.Tensor, expected: torch.Tensor,
                        labels: list | None = None) -> dict:
    """JSON-safe exact diagnostic, including signed zero and FP64 bit patterns."""
    a, b = actual.detach().cpu(), expected.detach().cpu()
    result = {'actual_dtype': str(a.dtype), 'expected_dtype': str(b.dtype),
              'actual_shape': list(a.shape), 'expected_shape': list(b.shape),
              'exact': _exact(a, b)}
    if a.dtype != b.dtype or a.shape != b.shape:
        return result
    a, b = a.contiguous().flatten(), b.contiguous().flatten()
    numeric = a != b
    signs = torch.signbit(a) != torch.signbit(b)
    mismatches = torch.nonzero(numeric | signs).flatten().tolist()
    error = float((a - b).abs().max()) if a.numel() else 0.0
    result.update(numeric_mismatch_count=int(numeric.sum()),
                  signbit_mismatch_count=int(signs.sum()),
                  max_abs_error=error if math.isfinite(error) else repr(error),
                  first_mismatches=[])
    for index in mismatches[:16]:
        av, bv = float(a[index]), float(b[index])
        abits = int(a[index].view(torch.int64)) & ((1 << 64) - 1)
        bbits = int(b[index].view(torch.int64)) & ((1 << 64) - 1)
        entry = {'index': index, 'label': labels[index] if labels is not None else index,
                 'actual_hex': av.hex(), 'expected_hex': bv.hex(),
                 'actual_fp64_bits': f'0x{abits:016x}',
                 'expected_fp64_bits': f'0x{bbits:016x}',
                 'actual_signbit': bool(torch.signbit(a[index])),
                 'expected_signbit': bool(torch.signbit(b[index])),
                 'same_sign_ulp_distance': abs(abits - bbits)
                    if math.isfinite(av) and math.isfinite(bv)
                    and bool(torch.signbit(a[index])) == bool(torch.signbit(b[index])) else None}
        result['first_mismatches'].append(entry)
    return result


def legacy_decode_diagnostics(device: str = 'cpu') -> dict:
    """Measure the v1 arithmetic; diagnostic only, never used for v2 weights.

    PyTorch 2.10 implements ldexp as multiplication by pow(2, exponent).
    Retaining this tiny probe lets the laptop identify the actual v1 mismatch.
    Exact v2 tests determine PASS; a legacy mismatch is explicitly reported.
    """
    target = torch.device(device)
    finite = [c for c in range(256) if c not in (127, 255)]
    c = torch.tensor(finite, dtype=torch.int64, device=target)
    exponent, mantissa = (c >> 3) & 15, c & 7
    fraction = mantissa.to(torch.float64) / 8.0
    magnitude = torch.where(exponent == 0, fraction / 64.0,
                            torch.ldexp(1.0 + fraction, (exponent - 7).int()))
    values = torch.where((c & 128) != 0, -magnitude, magnitude)
    expected = torch.tensor([_decode(code) for code in finite], dtype=torch.float64)
    k = torch.arange(-127, 128, dtype=torch.int32, device=target)
    old_powers = torch.ldexp(torch.ones(255, dtype=torch.float64, device=target), k)
    expected_powers = torch.tensor([math.ldexp(1.0, i) for i in range(-127, 128)],
                                   dtype=torch.float64)
    return {'purpose': 'non-gating v1 arithmetic diagnostic; no model or dataset',
            'device': str(target), 'torch_version': torch.__version__,
            'finite_code_decode': _comparison_details(values, expected, finite),
            'e8m0_powers': _comparison_details(old_powers, expected_powers, list(range(-127, 128)))}


def run_tests(device: str = 'cpu', report: dict | None = None) -> dict:
    """Run codec tests on the requested device and return a JSON-safe report."""
    target = torch.device(device)
    checks: dict[str, bool] = {}
    rejected: list[str] = []
    if report is None:
        report = {}
    report.update(status='RUNNING', device=str(target), torch_version=torch.__version__,
                  format=mx.FORMAT, codec_revision=mx.CODEC_REVISION,
                  checks=checks, expected_rejection_cases=rejected,
                  training_performed=False, dataset_used=False)

    def check(condition: bool, label: str) -> None:
        checks[label] = bool(condition)
        if not condition:
            report.update(status='FAIL', failed_check=label)
            details = report.get('exact_comparisons', {}).get(label)
            suffix = '' if details is None else '; details=' + json.dumps(details, ensure_ascii=False)
            raise AssertionError('E4M3 codec check failed: ' + label + suffix)

    def reject(action, label: str) -> None:
        try:
            action()
        except ValueError:
            rejected.append(label)
        else:
            report.update(status='FAIL', failed_check='unexpectedly_accepted_' + label)
            raise AssertionError('E4M3 codec unexpectedly accepted: ' + label)

    # Exhaustive decoding and exact re-encoding, including both signed zeros.
    finite = [c for c in range(256) if c not in (127, 255)]
    code_tensor = torch.tensor(finite, dtype=torch.uint8, device=target)
    expected = torch.tensor([_decode(c) for c in finite], dtype=torch.float64)
    values = mx.decode_elements(code_tensor)
    report['exact_comparisons'] = {
        'all_254_finite_codes_match_independent_decode':
            _comparison_details(values, expected, finite)}
    check(_exact(values, expected), 'all_254_finite_codes_match_independent_decode')
    check(_exact(mx.encode_elements(values), code_tensor), 'all_254_finite_codes_roundtrip')

    # Exhaustive exact scale construction, including both ends of E8M0.
    scale_exponents = torch.arange(-127, 128, dtype=torch.int32, device=target)
    powers = mx._scale_power_of_two(scale_exponents)
    expected_powers = torch.tensor([math.ldexp(1.0, k) for k in range(-127, 128)],
                                   dtype=torch.float64)
    report['exact_comparisons']['all_255_scale_powers_are_exact'] = _comparison_details(
        powers, expected_powers, list(range(-127, 128)))
    check(_exact(powers, expected_powers), 'all_255_scale_powers_are_exact')
    products = values[:, None] * powers[None, :]
    expected_products = torch.tensor([[math.ldexp(_decode(c), k) for k in range(-127, 128)]
                                     for c in finite], dtype=torch.float64)
    report['exact_comparisons']['all_64770_code_scale_products_are_exact_fp64'] = (
        _comparison_details(products, expected_products))
    check(_exact(products, expected_products), 'all_64770_code_scale_products_are_exact_fp64')
    check(_decode(1) == 2 ** -9 and _decode(8) == 2 ** -6 and _decode(126) == 448.0,
          'element_boundaries_are_2pow_minus9_2pow_minus6_and_448')
    for code in (127, 255):
        reject(lambda c=code: mx.decode_elements(torch.tensor([c], dtype=torch.uint8, device=target)),
               f'nan_element_code_{code}')
    reject(lambda: mx.decode_elements(torch.tensor([0], dtype=torch.int64, device=target)),
           'non_uint8_element_codes')

    # Every midpoint plus the adjacent FP64 numbers on both sides and signs.
    probes, oracle_codes, midpoint_codes = [], [], []
    for lower in range(126):
        midpoint = float((_LEVELS[lower] + _LEVELS[lower + 1]) / 2)
        for sign in (1.0, -1.0):
            tie = math.copysign(midpoint, sign)
            even = lower if lower % 2 == 0 else lower + 1
            midpoint_codes.append((tie, even + (128 if sign < 0 else 0)))
            for value in (math.nextafter(tie, -math.inf), tie, math.nextafter(tie, math.inf)):
                probes.append(value)
                oracle_codes.append(_encode(value))
    check(_exact(mx.encode_elements(torch.tensor(probes, dtype=torch.float64, device=target)),
                 torch.tensor(oracle_codes, dtype=torch.uint8)),
          'all_midpoints_and_adjacent_fp64_values_match_scalar_oracle')
    check(_exact(mx.encode_elements(torch.tensor([x for x, _ in midpoint_codes],
                                                dtype=torch.float64, device=target)),
                 torch.tensor([c for _, c in midpoint_codes], dtype=torch.uint8)),
          'all_252_signed_midpoints_round_ties_to_even')

    boundary_inputs = [0.0, -0.0, 2 ** -10, -(2 ** -10), 2 ** -9, -(2 ** -9),
                       2 ** -6, -(2 ** -6), 448.0, -448.0, 449.0, -449.0,
                       480.0, -480.0, 1e300, -1e300]
    check(_exact(mx.encode_elements(torch.tensor(boundary_inputs, dtype=torch.float64, device=target)),
                 torch.tensor([_encode(x) for x in boundary_inputs], dtype=torch.uint8)),
          'signed_zero_subnormal_normal_and_saturation_boundaries')
    for name, value in [('nan', float('nan')), ('positive_inf', float('inf')),
                        ('negative_inf', -float('inf'))]:
        reject(lambda x=value: mx.encode_elements(torch.tensor([x], dtype=torch.float64, device=target)),
               'element_input_' + name)

    # Each channel and partial block must get its own independent scale.
    row0 = [i / 16 for i in range(-16, 16)] + [4.0, -7.0, -0.0]
    row1 = [128.0 * i for i in range(-16, 16)] + [0.0, 0.0, -0.0]
    row0[7] = 0.1
    row1[17] = 130.0
    grouped = torch.tensor([row0, row1], dtype=torch.float32, device=target).reshape(2, 5, 7)
    before = grouped.clone()
    encoded = mx.encode_tensor(grouped)
    oracle_c, oracle_s, oracle_q = _oracle_tensor(grouped)
    check(_exact(encoded.codes, oracle_c) and _exact(encoded.scale_codes, oracle_s),
          'channel_and_partial_block_layout_match_scalar_oracle')
    check(_exact(encoded.decode(), oracle_q), 'grouped_decode_matches_scalar_oracle')
    check(_exact(encoded.scale_codes, torch.tensor([[119, 121], [130, 127]], dtype=torch.uint8)),
          'four_independent_block_scales')
    check(bool((encoded.codes[:, -1, 3:] == 0).all()), 'padding_is_positive_zero')
    check(_exact(grouped, before), 'encoding_does_not_mutate_source')
    noncontiguous = grouped.transpose(1, 2)
    other = mx.encode_tensor(noncontiguous)
    other_c, other_s, other_q = _oracle_tensor(noncontiguous)
    check(_exact(other.codes, other_c) and _exact(other.scale_codes, other_s)
          and _exact(other.decode(), other_q), 'noncontiguous_source_layout')

    # Scale clipping, powers of two and FP32 subnormal flushing.
    tiny = torch.finfo(torch.float32).tiny
    fp32_max = torch.finfo(torch.float32).max
    below_one = float(torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)))
    maxima = [0.0, tiny, below_one, 1.0, 256.0, 2 ** 127, fp32_max]
    scale_input = torch.zeros((len(maxima), 32), dtype=torch.float32, device=target)
    scale_input[:, 0] = torch.tensor(maxima, dtype=torch.float32, device=target)
    scale_input[:, 1] = -scale_input[:, 0]
    scaled = mx.encode_tensor(scale_input)
    scale_c, scale_s, scale_q = _oracle_tensor(scale_input)
    check(_exact(scaled.codes, scale_c) and _exact(scaled.scale_codes, scale_s)
          and _exact(scaled.decode(), scale_q), 'scale_extremes_match_scalar_oracle')
    check(_exact(scaled.scale_codes.flatten(),
                 torch.tensor([127, 0, 118, 119, 127, 246, 246], dtype=torch.uint8)),
          'zero_lower_clamp_power_boundaries_and_fp32_max_scales')
    smallest = 2 ** -149
    largest_subnormal = tiny - smallest
    subnormal_input = torch.tensor([[smallest, -smallest, largest_subnormal,
                                     -largest_subnormal, 0.0, -0.0]],
                                   dtype=torch.float32, device=target)
    subnormal = mx.encode_tensor(subnormal_input)
    expected_zero = torch.tensor([[0.0, -0.0, 0.0, -0.0, 0.0, -0.0]], dtype=torch.float32)
    check(_exact(subnormal.decode(), expected_zero), 'fp32_subnormals_flush_to_signed_zero')
    check(int(subnormal.scale_codes[0, 0]) == 127, 'flushed_zero_block_uses_scale_one')

    # A syntactically finite MX combination can exceed FP32 emulator range.
    upper_scale = torch.tensor([[254]], dtype=torch.uint8, device=target)
    upper_codes = torch.full((1, 1, 32), 1, dtype=torch.uint8, device=target)
    upper = mx.MXTensor(upper_codes, upper_scale, (1, 32))
    check(_exact(upper.decode(), torch.full((1, 32), float(2 ** 118), dtype=torch.float32)),
          'upper_finite_e8m0_scale_decodes_when_combination_fits_fp32')
    overflow = mx.MXTensor(torch.full((1, 1, 32), 126, dtype=torch.uint8, device=target),
                           upper_scale, (1, 32))
    reject(overflow.decode, 'finite_mx_combination_outside_fp32_range')

    # Payloads preserve exact codes and scales and have independent storage.
    saved_codes, saved_scales = encoded.codes.clone(), encoded.scale_codes.clone()
    payload = encoded.payload()
    check(set(payload) == {'format', 'block_size', 'shape', 'layout', 'packed_codes', 'scale_codes'}
          and payload['layout'] == 'row_Cin_Kh_Kw; one_code_per_byte'
          and payload['packed_codes'].dtype == torch.uint8,
          'payload_is_explicit_byte_codes_without_masters')
    rebuilt = mx.MXTensor.from_payload(payload, device=target)
    check(_exact(rebuilt.codes, saved_codes) and _exact(rebuilt.scale_codes, saved_scales)
          and _exact(rebuilt.decode(), encoded.decode()), 'payload_roundtrip_exact')
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    on_disk = torch.load(buffer, map_location='cpu', weights_only=True)
    reloaded = mx.MXTensor.from_payload(on_disk, device=target)
    check(_exact(reloaded.codes, saved_codes) and _exact(reloaded.scale_codes, saved_scales),
          'torch_serialization_roundtrip_exact')
    payload['packed_codes'][0, 0, 0] ^= 128
    payload['scale_codes'][0, 0] += 1
    check(_exact(encoded.codes, saved_codes) and _exact(encoded.scale_codes, saved_scales)
          and _exact(rebuilt.codes, saved_codes) and _exact(rebuilt.scale_codes, saved_scales),
          'payload_mutation_does_not_modify_source_or_rebuilt_tensor')
    payload2 = encoded.payload()
    rebuilt2 = mx.MXTensor.from_payload(payload2, device=target)
    rebuilt2.codes[0, 0, 0] ^= 128
    rebuilt2.scale_codes[0, 0] += 1
    check(_exact(payload2['packed_codes'], saved_codes) and _exact(payload2['scale_codes'], saved_scales),
          'rebuilt_mutation_does_not_modify_payload')

    for field, bad in [('format', 'MXFP6_E2M3'), ('layout', 'unspecified'), ('block_size', 16)]:
        invalid = encoded.payload()
        invalid[field] = bad
        reject(lambda p=invalid: mx.MXTensor.from_payload(p, device=target), 'payload_' + field)
    hidden = encoded.payload()
    hidden['master_weight'] = grouped.detach().cpu().clone()
    reject(lambda: mx.MXTensor.from_payload(hidden), 'payload_hidden_master')
    for code in (127, 255):
        invalid = encoded.payload()
        invalid['packed_codes'][0, 0, 0] = code
        reject(lambda p=invalid: mx.MXTensor.from_payload(p), f'payload_nan_code_{code}')
    invalid = encoded.payload()
    invalid['scale_codes'][0, 0] = 255
    reject(lambda: mx.MXTensor.from_payload(invalid), 'payload_nan_scale')
    for code in (1, 128):
        invalid_padding = encoded.payload()
        invalid_padding['packed_codes'][0, -1, -1] = code
        reject(lambda p=invalid_padding: mx.MXTensor.from_payload(p), f'padding_code_{code}')
    wrong_shape = encoded.payload()
    wrong_shape['shape'] = [3, 35]
    reject(lambda: mx.MXTensor.from_payload(wrong_shape), 'payload_shape_mismatch')

    reject(lambda: mx.encode_tensor(torch.zeros((1, 32), dtype=torch.float64, device=target)),
           'non_fp32_master_input')
    reject(lambda: mx.encode_tensor(torch.empty((1, 0), dtype=torch.float32, device=target)),
           'empty_master_input')
    for name, value in [('nan', float('nan')), ('inf', float('inf'))]:
        bad_weight = torch.tensor([[value]], dtype=torch.float32, device=target)
        reject(lambda w=bad_weight: mx.encode_tensor(w), 'nonfinite_master_' + name)

    # Identity STE remains exact also when a forward value saturates.
    master = torch.linspace(-1.9, 2.1, 32, dtype=torch.float32, device=target).reshape(1, 32)
    master[0, -1] = 511.0
    master.requires_grad_(True)
    master_before = master.detach().clone()
    quantized = mx.fake_quantize_mx(master)
    check(_exact(quantized, mx.encode_tensor(master).decode()), 'ste_forward_is_exact_codec_output')
    check(float(quantized[0, -1].detach()) == 448.0, 'ste_test_includes_saturated_forward')
    upstream = torch.arange(-16, 16, dtype=torch.float32, device=target).reshape(1, 32) / 3
    quantized.backward(upstream)
    check(_exact(master.grad, upstream), 'ste_backward_is_identity_including_saturation')
    check(_exact(master.detach(), master_before), 'ste_does_not_mutate_fp32_master')

    diagnostics = mx.distortion(grouped, encoded)
    expected_error = oracle_q.to(torch.float64) - grouped.detach().cpu().to(torch.float64)
    check(diagnostics['n_weights'] == 70 and diagnostics['blocks'] == 4
          and diagnostics['padded_slots'] == 58 and diagnostics['packed_weight_and_scale_bytes'] == 132
          and diagnostics['fp32_weight_bytes'] == 280,
          'diagnostics_count_actual_weights_padding_and_byte_payload')
    check(math.isclose(diagnostics['mae'], float(expected_error.abs().mean()), rel_tol=1e-12, abs_tol=1e-15)
          and diagnostics['max_abs_error'] == float(expected_error.abs().max()),
          'distortion_matches_independent_decoded_values')
    check(mx.distortion(scale_input, scaled)['scale_range_clamped_blocks'] == 1,
          'diagnostics_report_lower_scale_clipping')
    check(mx.distortion(subnormal_input, subnormal)['fp32_subnormal_input_count'] == 4,
          'diagnostics_report_fp32_subnormal_inputs')
    check(_exact(encoded.codes, saved_codes) and _exact(encoded.scale_codes, saved_scales),
          'all_readonly_operations_preserve_encoded_source')

    report.update(status='PASS', finite_bit_patterns_checked=254,
                  finite_scale_powers_checked=255, exact_fp64_code_scale_products_checked=64770,
                  nan_bit_patterns_rejected=[127, 255], signed_midpoint_ties_checked=252,
                  adjacent_midpoint_probes_checked=504)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', help='cpu or cuda; no training is performed')
    args = parser.parse_args()
    print(json.dumps(run_tests(args.device), indent=2, ensure_ascii=False))
