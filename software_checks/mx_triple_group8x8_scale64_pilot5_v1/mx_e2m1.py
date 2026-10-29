"""E2M1 extension forked from the validated E2M3 codec (see codec_e2m1.patch).

Readable finite-FP32 MXFP4 E2M1 codec. No activation quantization.

Format: 1 sign + 2 exponent + 1 fraction bit; bias=1; E8M0 scale;
64 elements per 8x8 tile across Cout and flattened Cin*Kh*Kw. RNE ties-to-even.
Paper algorithm 1 determines scales. Finite input only (NaN/Inf rejected).
Scalar FP32 subnormals are flushed to signed zero, following that paper's
example conversion. LOCAL E2M1 subnormals are supported. E8M0 exponent
range clipping is explicit. This is a numerical emulator, not a GPU kernel.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import Tensor

from group_config import (BLOCK_SIZE, GROUP_SHAPE, GROUP_AXIS, LAYOUT_PREFIX,
                          packed_code_bytes, packed_group_bytes, pack_groups, unpack_groups,
                          code_shape, scale_shape, validate_padding)
FORMAT = 'MXFP4_E2M1'
LAYOUT = LAYOUT_PREFIX + '; two_codes_low_nibble_first_in_one_byte'

def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)

def decode_elements(codes: Tensor) -> Tensor:
    """Decode [s ee m] codes to FP64, preserving signed zero."""
    require(codes.dtype == torch.uint8, 'Elementcodes moeten uint8 zijn.')
    require(bool((codes <= 15).all()), 'E2M1: alleen codes 0 t/m 15 toegestaan.')
    c = codes.to(torch.int64)
    exponent, mantissa = (c >> 1) & 3, c & 1
    fraction = mantissa.to(torch.float64) / 2.0
    magnitude = torch.where(exponent == 0, fraction,
                            torch.ldexp(1.0 + fraction, (exponent - 1).int()))
    return torch.where((c & 8) != 0, -magnitude, magnitude)

def encode_elements(values: Tensor) -> Tensor:
    """Nearest E2M1 value, even least-significant code bit at a tie.

    Lookup uses the 8 nonnegative values; not nearest integer rounding.
    A finite value outside the element range saturates to +/-6.0.
    """
    require(bool(torch.isfinite(values).all()), 'NaN/Inf in elementconversie.')
    values = values.to(torch.float64)
    levels = decode_elements(torch.arange(8, device=values.device, dtype=torch.uint8))
    magnitude = values.abs().clamp_max(6.0).contiguous()
    hi = torch.bucketize(magnitude, levels).clamp(0, 7)
    lo = (hi - 1).clamp_min(0)
    dl, dh = magnitude - levels[lo], levels[hi] - magnitude
    choose_hi = (dh < dl) | ((dh == dl) & ((hi & 1) == 0))
    unsigned = torch.where(choose_hi, hi, lo)
    return (unsigned | (torch.signbit(values).to(torch.int64) << 3)).to(torch.uint8)

@dataclass(frozen=True)
class MXTensor:
    """codes[tile_rows, tile_cols, 64], scales[tile_rows, tile_cols]."""
    codes: Tensor
    scale_codes: Tensor
    shape: tuple[int, ...]

    def validate(self) -> None:
        require(len(self.shape) >= 2 and all(type(d) is int and d > 0 for d in self.shape),
                'Ongeldige oorspronkelijke gewichtsvorm.')
        require(tuple(self.codes.shape) == code_shape(self.shape), 'Onjuiste 8x8-codevorm.')
        require(tuple(self.scale_codes.shape) == scale_shape(self.shape), 'Onjuiste 8x8-schaalvorm.')
        require(self.codes.dtype == self.scale_codes.dtype == torch.uint8, 'Codes moeten uint8 zijn.')
        require(self.codes.device == self.scale_codes.device, 'Codes en schalen op verschillend device.')
        require(bool((self.codes <= 15).all()), 'Ongeldige E2M1-code.')
        require(bool((self.scale_codes < 255).all()), 'E8M0 NaN-code 255 is niet toegestaan.')
        validate_padding(self.codes, self.shape)

    def decode(self) -> Tensor:
        self.validate()
        exponents = self.scale_codes.to(torch.int32) - 127
        values = torch.ldexp(decode_elements(self.codes), exponents[..., None])
        result = unpack_groups(values, self.shape).to(torch.float32).contiguous()
        require(bool(torch.isfinite(result).all()), 'Gedecodeerde MX-waarde buiten eindig FP32-bereik.')
        return result

    def payload(self) -> dict:
        """Packed 4-bit data: two element codes occupy exactly one byte.

        Ordering is explicitly our little-endian bit layout; MX does not
        prescribe the storage layout. One 64-element block = 32+1 bytes.
        There are no FP master weights in this object.
        """
        self.validate()
        return {'format': FORMAT, 'block_size': BLOCK_SIZE, 'shape': list(self.shape),
                'layout': LAYOUT,
                'packed_codes': pack4(self.codes).cpu(),
                'scale_codes': self.scale_codes.cpu().clone()}

    @staticmethod
    def from_payload(p: dict, device: str | torch.device = 'cpu') -> 'MXTensor':
        require(set(p) == {'format', 'block_size', 'shape', 'layout', 'packed_codes', 'scale_codes'},
                'Onverwachte velden in MX-bron; geen verborgen FP-gewichten toegestaan.')
        require(p.get('format') == FORMAT and type(p.get('block_size')) is int and p['block_size'] == BLOCK_SIZE, 'Onbekend MX-formaat.')
        require(p.get('layout') == LAYOUT, 'Onbekende layout.')
        result = MXTensor(unpack4(p['packed_codes'].to(device)),
                          p['scale_codes'].to(device), tuple(p['shape']))
        result.validate()
        return result

def encode_tensor(weight: Tensor) -> MXTensor:
    """Encode 8x8 tiles across Cout rows and flattened Cin*Kh*Kw columns.

    Scalar FP64 intermediates make the numerical rules explicit near powers
    of two and scale limits. Only finite FP32 input is accepted.
    """
    require(weight.dtype == torch.float32, 'Deze eerste codec verwacht FP32-gewichten.')
    require(weight.ndim >= 2 and weight.numel() > 0, 'Verwacht niet-lege gewichtstensor.')
    require(bool(torch.isfinite(weight).all()), 'NaN/Inf in brongewicht: stop, niet verbergen.')
    with torch.no_grad():
        w = weight.detach().to(torch.float64)
        scalar_subnormal = (w.abs() > 0) & (w.abs() < torch.finfo(torch.float32).tiny)
        w = torch.where(scalar_subnormal, torch.copysign(torch.zeros_like(w), w), w)
        blocks = pack_groups(w)
        maxima = blocks.abs().amax(dim=-1)
        # frexp(v) gives v = fraction * 2**power, fraction in [0.5, 1).
        # floor(log2(maxima)) - emax_elem = power - 1 - 2.
        _, power = torch.frexp(maxima)
        scale_exp = torch.where(maxima == 0, torch.zeros_like(power), power - 3)
        scale_exp = scale_exp.clamp(-127, 127).to(torch.int32)
        normalized = torch.ldexp(blocks, -scale_exp[..., None])
        codes = encode_elements(normalized)
        result = MXTensor(codes, (scale_exp + 127).to(torch.uint8), tuple(weight.shape))
        result.validate()
        return result

def pack4(codes: Tensor) -> Tensor:
    require(codes.dtype == torch.uint8 and codes.ndim >= 1 and codes.shape[-1] == BLOCK_SIZE, f'Verwacht {BLOCK_SIZE} uint8-codes per blok.')
    require(bool((codes <= 15).all()), 'Ongeldige E2M1-code.')
    q = codes.to(torch.int64).reshape(*codes.shape[:-1], BLOCK_SIZE // 2, 2)
    return (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)

def unpack4(data: Tensor) -> Tensor:
    require(data.dtype == torch.uint8 and data.ndim >= 1 and data.shape[-1] == packed_code_bytes(4), f'Verwacht {packed_code_bytes(4)} bytes per blok.')
    b = data.to(torch.int64)
    return torch.stack([b & 15, (b >> 4) & 15], -1).to(torch.uint8).reshape(*data.shape[:-1], BLOCK_SIZE)

def distortion(weight: Tensor, encoded: MXTensor) -> dict:
    """Diagnostics, not a speed/energy or whole-model storage claim."""
    q = encoded.decode()
    w = weight.detach().to(torch.float64)
    err = q.to(torch.float64) - w
    code = unpack_groups(encoded.codes, encoded.shape)
    scalar_sub = (w.abs() > 0) & (w.abs() < torch.finfo(torch.float32).tiny)
    src = torch.where(scalar_sub, torch.zeros_like(w), w)
    padded = pack_groups(src)
    normalized = torch.ldexp(padded, -(encoded.scale_codes.int() - 127)[..., None])
    maxima = padded.abs().amax(-1)
    _, power = torch.frexp(maxima)
    raw_exp = torch.where(maxima == 0, torch.zeros_like(power), power - 3)
    nblocks = encoded.scale_codes.numel()
    return {'n_weights': weight.numel(), 'blocks': nblocks, 'block_size': BLOCK_SIZE, 'group_shape': list(GROUP_SHAPE), 'group_axis': GROUP_AXIS,
            'padded_slots': encoded.codes.numel() - weight.numel(),
            'mae': float(err.abs().mean()), 'rmse': float(err.square().mean().sqrt()),
            'max_abs_error': float(err.abs().max()),
            'relative_l2': float(err.norm() / w.norm()) if float(w.norm()) else 0.0,
            'new_zero_count': int(((q == 0) & (weight != 0)).sum()),
            'local_subnormal_count': int((((code & 6) == 0) & ((code & 1) != 0)).sum()),
            'out_of_element_range_before_round': int((normalized.abs() > 6.0).sum()),
            'fp32_subnormal_input_count': int(scalar_sub.sum()),
            'scale_range_clamped_blocks': int(((raw_exp < -127) | (raw_exp > 127)).sum()),
            'scale_exponent_min': int(encoded.scale_codes.min()) - 127,
            'scale_exponent_max': int(encoded.scale_codes.max()) - 127,
            'packed_weight_and_scale_bytes': nblocks * packed_group_bytes(4),
            'fp32_weight_bytes': weight.numel() * 4}

class _MXIdentitySTE(torch.autograd.Function):
    """Exact quantized forward; intentionally approximate identity backward.

    Ordinary finite-difference gradcheck is NOT the test for an STE: the
    true derivative of quantization is not the proposed surrogate gradient.
    """
    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        return encode_tensor(value).decode()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        return (grad_output.clone(),)

def fake_quantize_mx(weight: Tensor) -> Tensor:
    return _MXIdentitySTE.apply(weight)
