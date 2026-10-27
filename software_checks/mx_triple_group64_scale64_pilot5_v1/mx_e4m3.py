"""Finite-FP32 MXFP8 E4M3 weight codec, adapted from the validated E2M3 codec.

Element layout is [s eeee mmm], exponent bias 7. Codes 0x7f and 0xff are
NaNs and are rejected; all other codes are finite, including signed zeros
and local subnormals. Maximum finite magnitude is 448. Finite overflow in
element encoding saturates to that maximum. Rounding is nearest, ties even.

Blocks contain 64 values within an output channel. Their E8M0 scale follows
the existing paper-algorithm policy, with emax_element=8 for E4M3. FP32 input
subnormals are flushed to signed zero as in the established E2M3 codec.
This module does not quantize activations, alter FP32 masters or implement
exponent truncation. Execution remains a FP32 numerical emulator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor


from group_config import BLOCK_SIZE, packed_code_bytes, packed_group_bytes
FORMAT = 'MXFP8_E4M3'
LAYOUT = 'row_Cin_Kh_Kw; one_code_per_byte'
MAX_FINITE = 448.0
CODEC_REVISION = 'e4m3-exact-binary-v2'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _scale_power_of_two(exponents: Tensor) -> Tensor:
    """Exact FP64 powers of two for internal integer exponents in [-127,127].

    Both E8M0 validation and scale selection enforce that range. Every power
    is a *normal* binary64 number: sign=0, biased exponent=k+1023, fraction=0.
    Reinterpreting those bits avoids a backend-dependent pow/ldexp evaluation.
    This is the existing scale, with no correction, offset or learned factor.
    """
    bits = (exponents.to(torch.int64) + 1023) << 52
    return bits.contiguous().view(torch.float64)


def decode_elements(codes: Tensor) -> Tensor:
    """Decode finite [s eeee mmm] codes to FP64, preserving signed zero."""
    require(codes.dtype == torch.uint8, 'Elementcodes moeten uint8 zijn.')
    require(bool(((codes & 127) != 127).all()),
            'E4M3 NaN-codes 127 en 255 zijn niet toegestaan.')
    c = codes.to(torch.int64)
    exponent, mantissa = (c >> 3) & 15, c & 7
    # Every finite E4M3 magnitude is an integer multiple of 2**-9.
    # Subnormal: M/512. Normal: (8+M)*2**(E-1)/512.
    # The largest numerator is 229376; all integer and FP64 operations are exact.
    units = torch.where(exponent == 0, mantissa,
                        (8 + mantissa) << (exponent - 1).clamp_min(0))
    magnitude = units.to(torch.float64) * (1.0 / 512.0)
    return torch.where((c & 128) != 0, -magnitude, magnitude)


def encode_elements(values: Tensor) -> Tensor:
    """Nearest finite E4M3 value, even least-significant code bit at a tie.

    The 127 nonnegative finite codes are monotone. A finite magnitude above
    448 saturates; input NaN/Inf is rejected instead of silently repaired.
    """
    require(bool(torch.isfinite(values).all()), 'NaN/Inf in elementconversie.')
    values = values.to(torch.float64)
    levels = decode_elements(torch.arange(127, device=values.device, dtype=torch.uint8))
    magnitude = values.abs().clamp_max(MAX_FINITE).contiguous()
    hi = torch.bucketize(magnitude, levels).clamp(0, 126)
    lo = (hi - 1).clamp_min(0)
    dl, dh = magnitude - levels[lo], levels[hi] - magnitude
    choose_hi = (dh < dl) | ((dh == dl) & ((hi & 1) == 0))
    unsigned = torch.where(choose_hi, hi, lo)
    return (unsigned | (torch.signbit(values).to(torch.int64) << 7)).to(torch.uint8)


@dataclass(frozen=True)
class MXTensor:
    """codes[C_out, blocks_per_row, BLOCK_SIZE], scales[C_out, blocks_per_row]."""

    codes: Tensor
    scale_codes: Tensor
    shape: tuple[int, ...]

    def validate(self) -> None:
        require(len(self.shape) >= 2 and all(type(d) is int and d > 0 for d in self.shape),
                'Ongeldige oorspronkelijke gewichtsvorm.')
        rows, cols = self.shape[0], math.prod(self.shape[1:])
        blocks = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        require(tuple(self.codes.shape) == (rows, blocks, BLOCK_SIZE), 'Onjuiste codevorm.')
        require(tuple(self.scale_codes.shape) == (rows, blocks), 'Onjuiste schaalvorm.')
        require(self.codes.dtype == self.scale_codes.dtype == torch.uint8, 'Codes moeten uint8 zijn.')
        require(self.codes.device == self.scale_codes.device, 'Codes en schalen op verschillend device.')
        require(bool(((self.codes & 127) != 127).all()),
                'E4M3 NaN-codes 127 en 255 zijn niet toegestaan.')
        require(bool((self.scale_codes < 255).all()), 'E8M0 NaN-code 255 is niet toegestaan.')
        padding = blocks * BLOCK_SIZE - cols
        if padding:
            require(bool((self.codes[:, -1, -padding:] == 0).all()), 'Padding moet +nul zijn.')

    def decode(self) -> Tensor:
        self.validate()
        exponents = self.scale_codes.to(torch.int32) - 127
        values = decode_elements(self.codes) * _scale_power_of_two(exponents[..., None])
        values = values.reshape(self.shape[0], -1)[:, :math.prod(self.shape[1:])]
        result = values.reshape(self.shape).to(torch.float32).contiguous()
        require(bool(torch.isfinite(result).all()), 'Gedecodeerde MX-waarde buiten eindig FP32-bereik.')
        return result

    def payload(self) -> dict:
        """One byte per element and one byte per scale: 64+1 bytes per block.

        The legacy key ``packed_codes`` is retained for schema consistency;
        E4M3 codes are already byte-sized and undergo no additional packing.
        Payload tensors own their storage and contain no FP32 master weights.
        """
        self.validate()
        return {'format': FORMAT, 'block_size': BLOCK_SIZE, 'shape': list(self.shape),
                'layout': LAYOUT, 'packed_codes': self.codes.detach().cpu().clone(),
                'scale_codes': self.scale_codes.detach().cpu().clone()}

    @staticmethod
    def from_payload(p: dict, device: str | torch.device = 'cpu') -> 'MXTensor':
        require(type(p) is dict and
                set(p) == {'format', 'block_size', 'shape', 'layout', 'packed_codes', 'scale_codes'},
                'Onverwachte velden in MX-bron; geen verborgen FP-gewichten toegestaan.')
        require(p.get('format') == FORMAT and type(p.get('block_size')) is int and p['block_size'] == BLOCK_SIZE, 'Onbekend MX-formaat.')
        require(p.get('layout') == LAYOUT, 'Onbekende layout.')
        result = MXTensor(p['packed_codes'].detach().to(device).clone(),
                          p['scale_codes'].detach().to(device).clone(), tuple(p['shape']))
        result.validate()
        return result


def encode_tensor(weight: Tensor) -> MXTensor:
    """Encode finite FP32 weights, grouping within each output channel.

    FP64 intermediates make behavior explicit near powers of two and scale
    limits. For nonzero blocks, scale exponent is floor(log2(maxabs))-8.
    E8M0 exponents are clipped to [-127,127]; an all-zero block gets scale 1.
    """
    require(weight.dtype == torch.float32, 'Deze codec verwacht FP32-gewichten.')
    require(weight.ndim >= 2 and weight.numel() > 0, 'Verwacht niet-lege gewichtstensor.')
    require(bool(torch.isfinite(weight).all()), 'NaN/Inf in brongewicht: stop, niet verbergen.')
    with torch.no_grad():
        w = weight.detach().to(torch.float64)
        scalar_subnormal = (w.abs() > 0) & (w.abs() < torch.finfo(torch.float32).tiny)
        w = torch.where(scalar_subnormal, torch.copysign(torch.zeros_like(w), w), w)
        rows, cols = w.shape[0], math.prod(w.shape[1:])
        padding = (-cols) % BLOCK_SIZE
        blocks = F.pad(w.reshape(rows, cols), (0, padding)).reshape(rows, -1, BLOCK_SIZE)
        maxima = blocks.abs().amax(dim=-1)
        # frexp gives maxabs = fraction * 2**power, fraction in [0.5,1).
        # floor(log2(maxabs)) - emax_element = power - 1 - 8.
        _, power = torch.frexp(maxima)
        scale_exp = torch.where(maxima == 0, torch.zeros_like(power), power - 9)
        scale_exp = scale_exp.clamp(-127, 127).to(torch.int32)
        normalized = blocks * _scale_power_of_two(-scale_exp[..., None])
        codes = encode_elements(normalized)
        result = MXTensor(codes, (scale_exp + 127).to(torch.uint8), tuple(weight.shape))
        result.validate()
        return result


def distortion(weight: Tensor, encoded: MXTensor) -> dict:
    """Numerical diagnostics; no speed, energy or whole-model storage claim."""
    require(tuple(weight.shape) == encoded.shape, 'Gewichts- en MX-vorm verschillen.')
    q = encoded.decode()
    w = weight.detach().to(torch.float64)
    err = q.to(torch.float64) - w
    cols = math.prod(encoded.shape[1:])
    code = encoded.codes.reshape(encoded.shape[0], -1)[:, :cols]
    scalar_sub = (w.abs() > 0) & (w.abs() < torch.finfo(torch.float32).tiny)
    src = torch.where(scalar_sub, torch.zeros_like(w), w)
    padded = F.pad(src.reshape(encoded.shape[0], -1), (0, (-cols) % BLOCK_SIZE)).reshape_as(encoded.codes)
    normalized = padded * _scale_power_of_two(-(encoded.scale_codes.int() - 127)[..., None])
    maxima = padded.abs().amax(-1)
    _, power = torch.frexp(maxima)
    raw_exp = torch.where(maxima == 0, torch.zeros_like(power), power - 9)
    nblocks = encoded.scale_codes.numel()
    return {'n_weights': weight.numel(), 'blocks': nblocks, 'block_size': BLOCK_SIZE,
            'padded_slots': encoded.codes.numel() - weight.numel(),
            'mae': float(err.abs().mean()), 'rmse': float(err.square().mean().sqrt()),
            'max_abs_error': float(err.abs().max()),
            'relative_l2': float(err.norm() / w.norm()) if float(w.norm()) else 0.0,
            'new_zero_count': int(((q == 0) & (weight != 0)).sum()),
            'local_subnormal_count': int((((code & 120) == 0) & ((code & 7) != 0)).sum()),
            'out_of_element_range_before_round': int((normalized.abs() > MAX_FINITE).sum()),
            'fp32_subnormal_input_count': int(scalar_sub.sum()),
            'scale_range_clamped_blocks': int(((raw_exp < -127) | (raw_exp > 127)).sum()),
            'scale_exponent_min': int(encoded.scale_codes.min()) - 127,
            'scale_exponent_max': int(encoded.scale_codes.max()) - 127,
            'packed_weight_and_scale_bytes': nblocks * packed_group_bytes(8),
            'fp32_weight_bytes': weight.numel() * 4}


class _MXIdentitySTE(torch.autograd.Function):
    """Exact quantized forward with an intentional identity surrogate backward."""

    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        return encode_tensor(value).decode()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        return (grad_output.clone(),)


def fake_quantize_mx(weight: Tensor) -> Tensor:
    return _MXIdentitySTE.apply(weight)
