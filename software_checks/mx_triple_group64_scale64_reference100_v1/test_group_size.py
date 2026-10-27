"""Independent B64 geometry, packed-bit and old-payload rejection checks.

No dataset or model training. All three routes keep the same element codec
rules; this test exercises row-local grouping and the exact stored bytes.
"""
from __future__ import annotations

import copy
import math

import torch

from group_config import BLOCK_SIZE, packed_code_bytes, packed_group_bytes
import mx_e4m3 as high
import mx_format as middle
import mx_e2m1 as low
import trunc_adapter as adapter


SHAPES = ((16, 12, 3, 3), (16, 16, 3, 3), (32, 16, 3, 3),
          (32, 32, 3, 3), (32, 32, 3, 3), (128, 32, 5, 5))


def _reject(call):
    try:
        call()
    except (ValueError, RuntimeError, KeyError):
        return
    raise AssertionError('Incompatible group metadata or packing was accepted.')


def _scalar_pack(codes, bits):
    """Independent one-row little-endian bitstream, avoiding codec helpers."""
    accumulator = sum(int(value) << (bits * i) for i, value in enumerate(codes))
    return list(accumulator.to_bytes(len(codes) * bits // 8, byteorder='little'))


def run_tests(device='cpu'):
    d = torch.device(device)
    assert BLOCK_SIZE == 64
    assert (high.BLOCK_SIZE, middle.BLOCK_SIZE, low.BLOCK_SIZE) == (64, 64, 64)
    assert [packed_code_bytes(b) for b in (8, 6, 4)] == [64, 48, 32]
    assert [packed_group_bytes(b) for b in (8, 6, 4)] == [65, 49, 33]
    for b in (16, 32, 64):
        assert [packed_group_bytes(bits, b) for bits in (8, 6, 4)] == [
            1 + b, 1 + 3 * b // 4, 1 + b // 2]
    checks = {'group_size': 64, 'packed_bytes_per_group': {'8': 65, '6': 49, '4': 33}}

    # Every scalar code appears at least once; bytes are compared to a scalar
    # bitstream oracle, not merely tested with mutually inverse pack/unpack.
    for bit, module in ((6, middle), (4, low)):
        values = list(range(1 << bit)) * (64 // (1 << bit))
        blocks = torch.tensor(values, dtype=torch.uint8, device=d).reshape(-1, 64)
        pack = module.pack6 if bit == 6 else module.pack4
        unpack = module.unpack6 if bit == 6 else module.unpack4
        packed = pack(blocks)
        expected = [_scalar_pack(row, bit) for row in blocks.cpu().tolist()]
        assert packed.cpu().tolist() == expected
        assert torch.equal(unpack(packed), blocks)
        for old_size in (16, 32):
            _reject(lambda m=pack, n=old_size: m(torch.zeros((1, n), dtype=torch.uint8, device=d)))
            _reject(lambda m=unpack, b=bit, n=old_size: m(torch.zeros((1, n*b//8), dtype=torch.uint8, device=d)))
    checks['all_E2_codes_pack_match_independent_bitstream'] = True
    checks['B16_B32_packed_shapes_rejected'] = True

    # A maximum in the first group must not change the adjacent B64 group or
    # a different output row. Source scale exponents here follow by inspection:
    # max256 -> 0; max1 -> -8; max4 -> -6; max0.25 -> -10; max2 -> -7.
    w = torch.zeros((2, 131), dtype=torch.float32, device=d)
    w[0, :64] = 256.; w[0, 64:128] = 1.; w[0, 128:] = 4.
    w[1, :64] = .25; w[1, 64:128] = 2.; w[1, 128:] = -0.
    mx = high.encode_tensor(w)
    assert mx.codes.shape == (2, 3, 64)
    assert mx.scale_codes.cpu().tolist() == [[127, 119, 121], [117, 120, 127]]
    assert bool((mx.codes[:, -1, 3:] == 0).all())
    assert torch.equal(mx.decode(), w)
    assert torch.equal(torch.signbit(mx.decode()), torch.signbit(w))
    changed = w.clone(); changed[0, :64] *= 128.
    other = high.encode_tensor(changed)
    assert torch.equal(mx.codes[:, 1:], other.codes[:, 1:])
    assert torch.equal(mx.scale_codes[:, 1:], other.scale_codes[:, 1:])
    assert torch.equal(mx.codes[1], other.codes[1])
    assert torch.equal(mx.scale_codes[1], other.scale_codes[1])
    checks['independent_adjacent_groups_rows_partial_padding_signed_zero'] = True

    chain = adapter.code_chain(mx)
    for bit, encoded in chain.items():
        payload = encoded.payload()
        assert payload['block_size'] == 64
        restored = adapter.codec(bit).MXTensor.from_payload(payload, d)
        assert torch.equal(restored.codes, encoded.codes)
        assert torch.equal(restored.scale_codes, encoded.scale_codes)
        assert torch.equal(restored.decode(), encoded.decode())
        for bad_size in (16, 32, True, 64.0):
            bad = copy.deepcopy(payload); bad['block_size'] = bad_size
            _reject(lambda b=bad, c=adapter.codec(bit): c.MXTensor.from_payload(b, d))
        if bit != 8:
            assert torch.equal(encoded.scale_codes.int(), mx.scale_codes.int() + 6)
    assert torch.equal(chain[4].codes, chain[6].codes >> 2)
    assert torch.equal(chain[4].scale_codes, chain[6].scale_codes)
    checks['payload_roundtrips_and_incompatible_group_metadata_rejected'] = True
    checks['fixed64_scale_and_mantissa_scale_inheritance'] = True

    # Shape-only oracle for this architecture, independently checked against
    # the actual encoded payload sizes on a deterministic synthetic tensor.
    groups = padding = weights = 0
    observed_bytes = {8: 0, 6: 0, 4: 0}
    for shape in SHAPES:
        rows, cols = shape[0], math.prod(shape[1:])
        n = math.prod(shape)
        weight = torch.arange(n, dtype=torch.float32, device=d).remainder(97).sub(48).reshape(shape) / 32.
        encoded_chain = adapter.encode_chain(weight)
        expected_groups = rows * math.ceil(cols / 64)
        for bit, encoded in encoded_chain.items():
            assert encoded.scale_codes.numel() == expected_groups
            payload = encoded.payload()
            observed_bytes[bit] += payload['packed_codes'].numel() + payload['scale_codes'].numel()
            diag = adapter.codec(bit).distortion(weight, encoded)
            assert diag['block_size'] == 64
            assert diag['packed_weight_and_scale_bytes'] == expected_groups * (1 + bit * 8)
        weights += n
        groups += expected_groups
        padding += rows * ((-cols) % 64)
    assert (weights, groups, padding) == (129472, 2160, 8768)
    assert observed_bytes == {8: 140400, 6: 105840, 4: 71280}
    checks['actual_six_layer_payloads_match_independent_geometry'] = {
        'weights': weights, 'groups': groups, 'padding_slots': padding,
        'bytes_by_mode': {str(k): v for k, v in observed_bytes.items()}}
    checks['status'] = 'PASS'
    checks['no_dataset_or_training'] = True
    return checks


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(run_tests(args.device), indent=2))
