"""Independent numeric verification of the new direct E5M2 and E3M2 codecs.

The scalar oracle uses the OCP value equation, scalar Python math, explicit
float32 bit extraction for scale selection, and a full nearest-level search.
It does not call a codec to calculate expected values. Tests cover every finite
element code, every adjacent midpoint and both neighbours, all 255 finite
E8M0 codes, float32 power boundaries, row padding, packing and the identity STE.
No dataset is opened, no accuracy is claimed and no model is trained here.

Source: OCP Microscaling Formats (MX) Specification v1.0, pp. 10--14,
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
The scale-selection/finite-input/FP32-subnormal rules are project policies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import random
import struct
from typing import Callable

import torch
import mx_e5m2
import mx_e3m2

SPEC = {
    'E5M2': {'codec': mx_e5m2, 'ebits': 5, 'mbits': 2, 'bias': 15,
             'emax': 15, 'positive_codes': 124, 'block_bytes': 33},
    'E3M2': {'codec': mx_e3m2, 'ebits': 3, 'mbits': 2, 'bias': 3,
             'emax': 4, 'positive_codes': 32, 'block_bytes': 25},
}
SOURCE = {'document': 'OCP Microscaling Formats (MX) Specification v1.0',
          'url': 'https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf',
          'element_pages': {'E5M2': 11, 'E3M2': 12},
          'scope_note': 'Scale selection and FP32 input-subnormal flush follow the project reference.'}
F32_TINY = math.ldexp(1.0, -126)
F32_MAX = struct.unpack('>f', bytes.fromhex('7f7fffff'))[0]


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _float32(value: float) -> float:
    return struct.unpack('>f', struct.pack('>f', value))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack('>I', struct.pack('>f', value))[0]


def _f32_from_bits(bits: int) -> float:
    return struct.unpack('>f', struct.pack('>I', bits))[0]


def _decode_scalar(code: int, s: dict) -> float:
    """Literal OCP normal/subnormal equation, independent of codec internals."""
    mbits, ebits, bias = s['mbits'], s['ebits'], s['bias']
    sign = -1.0 if code & (1 << (mbits + ebits)) else 1.0
    exponent = (code >> mbits) & ((1 << ebits) - 1)
    mantissa = code & ((1 << mbits) - 1)
    if exponent == 0:
        magnitude = math.ldexp(mantissa / (1 << mbits), 1 - bias)
    else:
        magnitude = math.ldexp(1.0 + mantissa / (1 << mbits), exponent - bias)
    return math.copysign(magnitude, sign)


def _encode_scalar(value: float, s: dict) -> int:
    _require(math.isfinite(value), 'Scalar oracle expects finite input.')
    sign_code = (1 << (s['ebits'] + s['mbits'])) if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    last = s['positive_codes'] - 1
    # Saturate first: extremely large binary64 values would lose level
    # differences when subtracting them from every finite codebook value.
    if magnitude >= _decode_scalar(last, s):
        return last | sign_code
    code = min(range(s['positive_codes']),
               key=lambda candidate: (abs(magnitude - _decode_scalar(candidate, s)), candidate & 1))
    return code | sign_code


def _same_bits(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    _require(actual.dtype == expected.dtype and actual.shape == expected.shape,
             label + ': dtype or shape mismatch.')
    integer = torch.int64 if actual.dtype == torch.float64 else torch.int32
    if actual.dtype in (torch.float32, torch.float64):
        same = torch.equal(actual.detach().cpu().contiguous().view(integer),
                           expected.detach().cpu().contiguous().view(integer))
    else:
        same = torch.equal(actual.detach().cpu(), expected.detach().cpu())
    _require(same, label + ': numerical bits differ, including possible signed zero.')


def _reject(call: Callable, label: str) -> None:
    try:
        call()
    except (ValueError, TypeError, RuntimeError, AssertionError):
        return
    raise AssertionError(label + ': invalid input was accepted.')


def _oracle_tensor(weight: torch.Tensor, s: dict):
    shape = tuple(weight.shape)
    rows = weight.detach().cpu().reshape(shape[0], -1).tolist()
    expected_codes, expected_scales, decoded = [], [], []
    for row in rows:
        row = [math.copysign(0.0, x) if 0 < abs(x) < F32_TINY else x for x in row]
        cols = len(row)
        padded = row + [0.0] * ((-cols) % 32)
        row_codes, row_scales, row_decoded = [], [], []
        for pos in range(0, len(padded), 32):
            block = padded[pos:pos + 32]
            maximum = max(abs(x) for x in block)
            # All nonzero source values are normal binary32 after the flush.
            # Extract the binary32 exponent rather than calling torch.frexp.
            exponent = 0 if maximum == 0 else ((_f32_bits(maximum) >> 23) & 255) - 127 - s['emax']
            exponent = min(127, max(-127, exponent))
            codes = [_encode_scalar(math.ldexp(x, -exponent), s) for x in block]
            row_codes.append(codes)
            row_scales.append(exponent + 127)
            row_decoded.extend(_float32(math.ldexp(_decode_scalar(c, s), exponent)) for c in codes)
        expected_codes.append(row_codes)
        expected_scales.append(row_scales)
        decoded.append(row_decoded[:cols])
    return (torch.tensor(expected_codes, dtype=torch.uint8),
            torch.tensor(expected_scales, dtype=torch.uint8),
            torch.tensor(decoded, dtype=torch.float32).reshape(shape))


def _check_tensor(weight: torch.Tensor, s: dict, name: str) -> dict:
    codec = s['codec']
    original = weight.detach().clone()
    codes, scales, decoded = _oracle_tensor(weight, s)
    encoded = codec.encode_tensor(weight)
    _same_bits(encoded.codes, codes, name + ' element codes')
    _same_bits(encoded.scale_codes, scales, name + ' scale codes')
    _same_bits(encoded.decode(), decoded, name + ' decoded values')
    _same_bits(weight, original, name + ' source unmodified')
    payload = encoded.payload()
    _require(payload['packed_codes'].numel() + payload['scale_codes'].numel()
             == encoded.scale_codes.numel() * s['block_bytes'], name + ' payload byte count')
    restored = codec.MXTensor.from_payload(payload, weight.device)
    _same_bits(restored.codes, encoded.codes, name + ' packed code roundtrip')
    _same_bits(restored.scale_codes, encoded.scale_codes, name + ' packed scale roundtrip')
    _same_bits(restored.decode(), encoded.decode(), name + ' packed value roundtrip')
    stats = codec.distortion(weight, encoded)
    _require(all(math.isfinite(value) for value in stats.values() if isinstance(value, (float, int))),
             name + ' nonfinite diagnostic')
    _require(stats['packed_weight_and_scale_bytes'] == encoded.scale_codes.numel() * s['block_bytes'],
             name + ' diagnostic byte count')
    _require(stats['n_weights'] == weight.numel(), name + ' diagnostic weight count')
    return {'n_weights': weight.numel(), 'blocks': encoded.scale_codes.numel(),
            'padded_slots': encoded.codes.numel() - weight.numel()}


def _test_one(name: str, s: dict, device: torch.device) -> dict:
    codec = s['codec']
    report = {'status': 'RUNNING', 'format': codec.FORMAT}
    sign_mask = 1 << (s['ebits'] + s['mbits'])
    finite_codes = list(range(s['positive_codes'])) + [sign_mask | x for x in range(s['positive_codes'])]
    input_codes = torch.tensor(finite_codes, dtype=torch.uint8, device=device)
    expected = torch.tensor([_decode_scalar(x, s) for x in finite_codes], dtype=torch.float64)
    actual = codec.decode_elements(input_codes)
    _same_bits(actual, expected, name + ' exhaustive finite decode')
    _same_bits(codec.encode_elements(actual), input_codes, name + ' exhaustive finite roundtrip')
    report['all_finite_codes'] = len(finite_codes)

    values = [0.0, -0.0, math.nextafter(0.0, math.inf), -math.nextafter(0.0, math.inf),
              codec.MAX_FINITE, -codec.MAX_FINITE, codec.MAX_FINITE * 2, -codec.MAX_FINITE * 2,
              float.fromhex('0x1.fffffffffffffp+1023'), -float.fromhex('0x1.fffffffffffffp+1023')]
    for code in range(s['positive_codes'] - 1):
        midpoint = (_decode_scalar(code, s) + _decode_scalar(code + 1, s)) / 2
        for x in [math.nextafter(midpoint, -math.inf), midpoint, math.nextafter(midpoint, math.inf)]:
            values += [x, -x]
    rng = random.Random(20261115)
    values += [math.ldexp(rng.uniform(-2.0, 2.0), rng.randint(-35, 25)) for _ in range(512)]
    expected_codes = torch.tensor([_encode_scalar(x, s) for x in values], dtype=torch.uint8)
    observed = codec.encode_elements(torch.tensor(values, dtype=torch.float64, device=device))
    _same_bits(observed, expected_codes, name + ' midpoint/neighbour/saturation oracle')
    report['midpoints_both_signs'] = 2 * (s['positive_codes'] - 1)
    report['midpoint_neighbour_and_other_scalar_cases'] = len(values)

    for invalid in [float('nan'), float('inf'), -float('inf')]:
        _reject(lambda value=invalid: codec.encode_elements(torch.tensor([value], device=device)),
                name + ' nonfinite element input')
        _reject(lambda value=invalid: codec.encode_tensor(torch.tensor([[value]], device=device)),
                name + ' nonfinite tensor input')
    invalid_codes = sorted(set(range(256)) - set(finite_codes))
    _reject(lambda: codec.decode_elements(torch.tensor([0], dtype=torch.int64, device=device)),
            name + ' invalid code dtype')
    for code in invalid_codes:
        _reject(lambda value=code: codec.decode_elements(torch.tensor([value], dtype=torch.uint8, device=device)),
                name + ' invalid reserved/high-bit code')
    report['invalid_element_codes_rejected'] = len(invalid_codes)

    # Exercise every finite E8M0 scale with a nonzero local subnormal and
    # its negative counterpart; all products fit finite binary32 here.
    scales = torch.arange(255, dtype=torch.uint8, device=device).reshape(255, 1).expand(255, 2).clone()
    codes = torch.empty((255, 2, 32), dtype=torch.uint8, device=device)
    codes[:, 0, :] = 1
    codes[:, 1, :] = sign_mask | 1
    scaled = codec.MXTensor(codes, scales, (255, 64))
    expected_scaled = torch.tensor([
        [_float32(math.ldexp(_decode_scalar(1, s), exponent))] * 32
        + [_float32(math.ldexp(_decode_scalar(sign_mask | 1, s), exponent))] * 32
        for exponent in range(-127, 128)], dtype=torch.float32)
    _same_bits(scaled.decode(), expected_scaled, name + ' all E8M0 scaled decode')
    _same_bits(codec.MXTensor.from_payload(scaled.payload(), device).decode(), expected_scaled,
               name + ' all E8M0 packed decode')
    _same_bits(codec._scale_power_of_two(torch.arange(-127, 128, dtype=torch.int32, device=device)),
               torch.tensor([math.ldexp(1.0, x) for x in range(-127, 128)], dtype=torch.float64),
               name + ' all E8M0 exact powers')
    _reject(lambda: codec.MXTensor(codes[:1], torch.full((1, 2), 255, dtype=torch.uint8, device=device), (1, 64)).decode(),
            name + ' NaN scale code')
    overflow_codes = torch.full((1, 1, 32), s['positive_codes'] - 1, dtype=torch.uint8, device=device)
    _reject(lambda: codec.MXTensor(overflow_codes, torch.tensor([[254]], dtype=torch.uint8, device=device), (1, 32)).decode(),
            name + ' decoded value beyond finite FP32')
    report['finite_E8M0_codes_checked'] = 255

    edge_rows = []
    for exponent in range(-126, 128):
        power = math.ldexp(1.0, exponent)
        bits = _f32_bits(power)
        edge_rows.append([power, -power, _f32_from_bits(bits - 1), _f32_from_bits(bits + 1)]
                         + [power, -power, 0.0, -0.0] * 7)
    boundaries = torch.tensor(edge_rows, dtype=torch.float32, device=device)
    report['float32_power_boundaries'] = _check_tensor(boundaries, s, name + ' float32 powers')

    tiny_sub = _f32_from_bits(1)
    largest_sub = _f32_from_bits(0x007fffff)
    values = [0.0, -0.0, tiny_sub, -tiny_sub, largest_sub, -largest_sub, F32_TINY, -F32_TINY,
              F32_MAX, -F32_MAX, 1.0, -1.0, .25, -.25, 7.0, -7.0, 31.0]
    padded_weight = torch.tensor([values + [0.0] * 18,
                                 [-0.0] * 35,
                                 [_float32(rng.uniform(-1, 1)) for _ in range(35)]],
                                dtype=torch.float32, device=device)
    report['mixed_range_padding_signed_zero'] = _check_tensor(padded_weight, s, name + ' mixed range')
    # Four-dimensional convolution layout and a noncontiguous input view.
    conv = torch.tensor([_float32(rng.uniform(-.5, .5)) for _ in range(3 * 5 * 3 * 3)],
                        dtype=torch.float32, device=device).reshape(3, 5, 3, 3).transpose(2, 3)
    report['convolution_row_layout'] = _check_tensor(conv, s, name + ' convolution')

    zero = torch.tensor([[0.0, -0.0, tiny_sub, -tiny_sub]], dtype=torch.float32, device=device)
    encoded_zero = codec.encode_tensor(zero)
    _require(int(encoded_zero.scale_codes.item()) == 127, name + ' zero block must use scale 1')
    _same_bits(encoded_zero.codes[0, 0, :4], torch.tensor([0, sign_mask, 0, sign_mask], dtype=torch.uint8),
               name + ' scalar subnormal flush sign')
    _reject(lambda: codec.encode_tensor(torch.ones((1, 32), dtype=torch.float64, device=device)),
            name + ' FP64 input must not silently replace FP32 policy')
    _reject(lambda: codec.encode_tensor(torch.ones(32, dtype=torch.float32, device=device)),
            name + ' missing output-channel dimension')
    _reject(lambda: codec.encode_tensor(torch.empty((1, 0), dtype=torch.float32, device=device)),
            name + ' empty weight tensor')

    payload = codec.encode_tensor(conv).payload()
    corrupt = deepcopy(payload)
    corrupt['hidden_master'] = torch.zeros(1)
    _reject(lambda: codec.MXTensor.from_payload(corrupt, device), name + ' hidden payload fields')
    corrupt = deepcopy(payload)
    corrupt['format'] = 'WRONG_FORMAT'
    _reject(lambda: codec.MXTensor.from_payload(corrupt, device), name + ' wrong format')
    corrupt = deepcopy(payload)
    corrupt['layout'] = 'WRONG_LAYOUT'
    _reject(lambda: codec.MXTensor.from_payload(corrupt, device), name + ' wrong layout')
    padded = codec.encode_tensor(padded_weight)
    corrupt_codes = padded.codes.clone()
    corrupt_codes[0, -1, -1] = 1
    _reject(lambda: codec.MXTensor(corrupt_codes, padded.scale_codes, padded.shape).validate(),
            name + ' nonzero padding')

    leaf = torch.tensor([[_float32(rng.uniform(-.4, .4)) for _ in range(35)] for _ in range(2)],
                        dtype=torch.float32, device=device, requires_grad=True)
    before = leaf.detach().clone()
    upstream = torch.arange(1, leaf.numel() + 1, dtype=torch.float32, device=device).reshape_as(leaf)
    q = codec.fake_quantize_mx(leaf)
    _same_bits(q, codec.encode_tensor(leaf.detach()).decode(), name + ' STE exact forward')
    (q * upstream).sum().backward()
    _same_bits(leaf.grad, upstream, name + ' STE identity gradient')
    _same_bits(leaf.detach(), before, name + ' STE master unmodified')
    report['identity_STE_and_exact_forward'] = True

    if name == 'E3M2':
        packed_codes = torch.tensor(list(range(64)) * 4, dtype=torch.uint8, device=device).reshape(2, 4, 32)
        packed = codec.pack6(packed_codes)
        oracle_bytes = []
        for block in packed_codes.cpu().reshape(-1, 32).tolist():
            block_bytes = []
            for start in range(0, 32, 4):
                word = sum(block[start + j] << (6 * j) for j in range(4))
                block_bytes.extend(word.to_bytes(3, 'little'))
            oracle_bytes.append(block_bytes)
        expected_packed = torch.tensor(oracle_bytes, dtype=torch.uint8).reshape(2, 4, 24)
        _same_bits(packed, expected_packed, name + ' independent little-endian packed bytes')
        _same_bits(codec.unpack6(packed), packed_codes, name + ' exhaustive pack roundtrip')
        _reject(lambda: codec.pack6(torch.full((1, 32), 64, dtype=torch.uint8, device=device)),
                name + ' high bits in pack input')
        _reject(lambda: codec.unpack6(torch.zeros((1, 23), dtype=torch.uint8, device=device)),
                name + ' wrong packed width')
        report['packing_layout_and_all_64_codes'] = True
    else:
        _require(payload['packed_codes'].shape[-1] == 32, name + ' one byte per element')
        report['one_byte_per_element'] = True
    report['status'] = 'PASS'
    return report


def run_tests(device: str | torch.device = 'cpu', report: dict | None = None) -> dict:
    result = report if report is not None else {}
    result.update(status='RUNNING', device=str(device), source=SOURCE, synthetic_only=True, formats={})
    try:
        for name, specification in SPEC.items():
            result['formats'][name] = _test_one(name, specification, torch.device(device))
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    result['status'] = 'PASS'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(run_tests(args.device), indent=2, ensure_ascii=False, allow_nan=False))
