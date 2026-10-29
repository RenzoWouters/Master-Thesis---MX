"""Independent 8x8 geometry, transpose, packed-bit and old-payload rejection checks.

No dataset or model training. All three routes keep the same element codec
rules; this test exercises square tiling and the exact stored bytes.
"""
from __future__ import annotations

import copy
import math

import torch

from group_config import (BLOCK_SIZE, GROUP_SHAPE, packed_code_bytes, packed_group_bytes,
                          pack_groups, unpack_groups, padding_mask, code_shape, scale_shape)
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
    assert BLOCK_SIZE == 64 and GROUP_SHAPE == (8, 8)
    assert (high.BLOCK_SIZE, middle.BLOCK_SIZE, low.BLOCK_SIZE) == (64, 64, 64)
    assert [packed_code_bytes(b) for b in (8, 6, 4)] == [64, 48, 32]
    assert [packed_group_bytes(b) for b in (8, 6, 4)] == [65, 49, 33]
    for b in (16, 32, 64):
        assert [packed_group_bytes(bits, b) for bits in (8, 6, 4)] == [
            1 + b, 1 + 3 * b // 4, 1 + b // 2]
    checks = {'group_size': 64, 'group_shape': [8, 8], 'packed_bytes_per_group': {'8': 65, '6': 49, '4': 33}}

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

    # An asymmetric logical matrix crosses both tile boundaries and has
    # tails in both axes. Every expected tile scale follows by inspection.
    w = torch.zeros((17, 19), dtype=torch.float32, device=d)
    w[:8, :8] = 256.; w[:8, 8:16] = 1.; w[:8, 16:] = 4.
    w[8:16, :8] = .25; w[8:16, 8:16] = 2.; w[8:16, 16:] = -0.
    w[16:, :8] = -4.; w[16:, 8:16] = .25; w[16:, 16:] = 1.
    mx = high.encode_tensor(w)
    assert mx.codes.shape == (3, 3, 64)
    assert mx.scale_codes.cpu().tolist() == [[127, 119, 121], [117, 120, 127], [121, 117, 119]]
    assert bool((mx.codes[padding_mask(tuple(w.shape), d)] == 0).all())
    assert torch.equal(mx.decode(), w)
    assert torch.equal(torch.signbit(mx.decode()), torch.signbit(w))
    changed = w.clone(); changed[:8, :8] *= 128.
    other = high.encode_tensor(changed)
    outside = torch.ones((3, 3), dtype=torch.bool, device=d); outside[0, 0] = False
    assert torch.equal(mx.codes[outside], other.codes[outside])
    assert torch.equal(mx.scale_codes[outside], other.scale_codes[outside])
    checks['independent_adjacent_tiles_both_axis_padding_signed_zero'] = True

    # Verify element placement against explicit logical coordinates rather
    # than relying on mutually inverse pack/unpack implementations.
    indexed = torch.arange(17 * 19, dtype=torch.int64, device=d).reshape(17, 19)
    tiled = pack_groups(indexed)
    assert tuple(tiled.shape) == (3, 3, 64)
    for tr in range(3):
        for tc in range(3):
            expected = [int(indexed[8*tr+r, 8*tc+c]) if 8*tr+r < 17 and 8*tc+c < 19 else 0
                        for r in range(8) for c in range(8)]
            assert tiled[tr, tc].cpu().tolist() == expected
    assert torch.equal(unpack_groups(tiled, (17, 19)), indexed)
    noncontiguous = indexed.t()
    assert not noncontiguous.is_contiguous()
    assert torch.equal(unpack_groups(pack_groups(noncontiguous), (19, 17)), noncontiguous)
    checks['coordinate_oracle_and_noncontiguous_inverse_layout'] = True

    # A transposition swaps tile indices and transposes within each tile.
    # The tile's element set, maxabs and therefore scale must be unchanged.
    source = torch.arange(17 * 19, dtype=torch.float32, device=d).remainder(97).sub(48).reshape(17, 19) / 32.
    source[8, 8] = -0.
    for module in (high, middle, low):
        encoded = module.encode_tensor(source)
        transposed = module.encode_tensor(source.t())
        expected_codes = encoded.codes.reshape(3, 3, 8, 8).permute(1, 0, 3, 2).reshape(3, 3, 64)
        assert torch.equal(transposed.codes, expected_codes)
        assert torch.equal(transposed.scale_codes, encoded.scale_codes.t())
        assert torch.equal(transposed.decode(), encoded.decode().t())
        assert torch.equal(torch.signbit(transposed.decode()), torch.signbit(encoded.decode().t()))
    source_chain, transpose_chain = adapter.encode_chain(source), adapter.encode_chain(source.t())
    for bit in (8, 6, 4):
        assert torch.equal(transpose_chain[bit].decode(), source_chain[bit].decode().t())
        assert torch.equal(transpose_chain[bit].scale_codes, source_chain[bit].scale_codes.t())
    checks['all_three_codecs_and_fixed64_chain_transpose_equivalent'] = True

    chain = adapter.code_chain(mx)
    for bit, encoded in chain.items():
        payload = encoded.payload()
        assert payload['block_size'] == 64 and payload['layout'].startswith('tile8x8_Cout_CinKhKw;')
        restored = adapter.codec(bit).MXTensor.from_payload(payload, d)
        assert torch.equal(restored.codes, encoded.codes)
        assert torch.equal(restored.scale_codes, encoded.scale_codes)
        assert torch.equal(restored.decode(), encoded.decode())
        for bad_size in (16, 32, True, 64.0):
            bad = copy.deepcopy(payload); bad['block_size'] = bad_size
            _reject(lambda b=bad, c=adapter.codec(bit): c.MXTensor.from_payload(b, d))
        bad_layout = copy.deepcopy(payload)
        bad_layout['layout'] = {8: 'row_Cin_Kh_Kw; one_code_per_byte', 6: 'row_Cin_Kh_Kw; four_codes_lsb_first_in_three_bytes', 4: 'row_Cin_Kh_Kw; two_codes_low_nibble_first_in_one_byte'}[bit]
        _reject(lambda b=bad_layout, c=adapter.codec(bit): c.MXTensor.from_payload(b, d))
        for padding_code in (1, {8: 128, 6: 32, 4: 8}[bit]):
            bad_codes = encoded.codes.clone(); bad_codes[-1, -1, -1] = padding_code
            _reject(lambda b=bad_codes, c=adapter.codec(bit), e=encoded: c.MXTensor(b, e.scale_codes, e.shape).validate())
        if bit != 8:
            assert torch.equal(encoded.scale_codes.int(), mx.scale_codes.int() + 6)
    assert torch.equal(chain[4].codes, chain[6].codes >> 2)
    assert torch.equal(chain[4].scale_codes, chain[6].scale_codes)
    checks['payload_roundtrips_old1D64_layout_and_invalid_padding_rejected'] = True
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
        expected_groups = math.ceil(rows / 8) * math.ceil(cols / 8)
        for bit, encoded in encoded_chain.items():
            assert encoded.scale_codes.numel() == expected_groups
            payload = encoded.payload()
            observed_bytes[bit] += payload['packed_codes'].numel() + payload['scale_codes'].numel()
            diag = adapter.codec(bit).distortion(weight, encoded)
            assert diag['block_size'] == 64
            assert diag['packed_weight_and_scale_bytes'] == expected_groups * (1 + bit * 8)
        weights += n
        groups += expected_groups
        padding += expected_groups * 64 - n
    assert (weights, groups, padding) == (129472, 2024, 64)
    assert observed_bytes == {8: 131560, 6: 99176, 4: 66792}
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
