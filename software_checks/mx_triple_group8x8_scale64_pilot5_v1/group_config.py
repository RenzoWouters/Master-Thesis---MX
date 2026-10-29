"""Fixed 2D 8x8 geometry for this isolated MX experiment.

The logical matrix has rows=Cout and columns=prod(weight.shape[1:]).
Its two tails are padded independently to eight; a shared scale belongs to
one 8x8 tile. Code order is tile-row, tile-column, intra-tile-row, column.
This layout is distinct from the older 1x64 layout despite BLOCK_SIZE=64.
The scalar element formats and scale-selection rules remain unchanged.
"""
from __future__ import annotations

import math

BLOCK_SIZE = 64
GROUP_SHAPE = (8, 8)
GROUP_AXIS = '8 Cout rows x 8 flattened Cin*Kh*Kw columns; zero-pad both matrix tails'
LAYOUT_PREFIX = 'tile8x8_Cout_CinKhKw; tile_row_major; elements_row_major'


def matrix_shape(shape) -> tuple[int, int]:
    if len(shape) < 2 or not all(type(d) is int and d > 0 for d in shape):
        raise ValueError('Verwacht een niet-lege gewichtsvorm met minstens twee assen.')
    return shape[0], math.prod(shape[1:])


def scale_shape(shape) -> tuple[int, int]:
    rows, cols = matrix_shape(shape)
    return (rows + 7) // 8, (cols + 7) // 8


def code_shape(shape) -> tuple[int, int, int]:
    return (*scale_shape(shape), BLOCK_SIZE)


def pack_groups(tensor):
    """Reorder logical weights into [tile_rows, tile_cols, 64] groups."""
    import torch.nn.functional as F
    rows, cols = matrix_shape(tuple(tensor.shape))
    nr, nc = scale_shape(tuple(tensor.shape))
    padded = F.pad(tensor.reshape(rows, cols), (0, (-cols) % 8, 0, (-rows) % 8))
    return padded.reshape(nr, 8, nc, 8).permute(0, 2, 1, 3).reshape(nr, nc, BLOCK_SIZE).contiguous()


def unpack_groups(grouped, shape):
    """Inverse tile ordering, then remove both padding tails."""
    rows, cols = matrix_shape(shape)
    nr, nc = scale_shape(shape)
    if tuple(grouped.shape) != (nr, nc, BLOCK_SIZE):
        raise ValueError('Onjuiste 8x8-groepvorm bij decoderen.')
    matrix = grouped.reshape(nr, nc, 8, 8).permute(0, 2, 1, 3).reshape(nr * 8, nc * 8)
    return matrix[:rows, :cols].reshape(shape).contiguous()


def padding_mask(shape, device='cpu'):
    """True only for padded slots, used by diagnostics and independent tests."""
    import torch
    rows, cols = matrix_shape(shape)
    nr, nc = scale_shape(shape)
    row = torch.arange(nr * 8, device=device).reshape(nr, 1, 8, 1)
    col = torch.arange(nc * 8, device=device).reshape(1, nc, 1, 8)
    return ((row >= rows) | (col >= cols)).reshape(nr, nc, BLOCK_SIZE)


def validate_padding(codes, shape) -> None:
    """Padded integer codes must be +zero, never signed zero or nonzero."""
    rows, cols = matrix_shape(shape)
    nr, nc = scale_shape(shape)
    tile = codes.reshape(nr, nc, 8, 8)
    if rows % 8 and not bool((tile[-1, :, rows % 8:, :] == 0).all()):
        raise ValueError('Padding in Cout moet +nul zijn.')
    if cols % 8 and not bool((tile[:, -1, :, cols % 8:] == 0).all()):
        raise ValueError('Padding in Cin*Kh*Kw moet +nul zijn.')


def packed_code_bytes(bits: int, block_size: int = BLOCK_SIZE) -> int:
    """Whole-byte code payload per group, excluding its E8M0 scale byte."""
    if type(bits) is not int or bits not in (4, 6, 8):
        raise ValueError('Expected a 4-, 6-, or 8-bit element encoding.')
    if type(block_size) is not int or block_size <= 0 or block_size % 4:
        raise ValueError('Group size must be a positive multiple of four.')
    return block_size * bits // 8


def packed_group_bytes(bits: int, block_size: int = BLOCK_SIZE) -> int:
    """Element payload plus exactly one unsigned eight-bit scale."""
    return packed_code_bytes(bits, block_size) + 1
