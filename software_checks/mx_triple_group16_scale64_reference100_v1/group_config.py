"""Group geometry for this isolated B16 experiment.

Only grouping/padding and the resulting payload dimensions differ from B32.
Scalar element encodings and their rounding rules are unchanged. The value
is fixed in the package contract; old B32 payloads must never be reinterpreted.
"""

BLOCK_SIZE = 16
GROUP_AXIS = 'flatten Cin*Kh*Kw within each Cout, zero-pad tail'


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
