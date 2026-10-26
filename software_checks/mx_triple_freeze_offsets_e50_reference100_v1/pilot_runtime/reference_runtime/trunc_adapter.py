"""E4M3 source with a learned binary exponent offset per MX block.

Forward: hard offset 5 (x32) below theta=5.5, otherwise 6 (x64).
Backward: identity to G(W), and ln(2)*sum(grad*q) to the FP32 theta
of each block, from the two lower modes only. Integer codes/scales are
cached without an autograd graph. The complete lower code route is unchanged.
"""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
from contextlib import contextmanager
import torch
from torch import Tensor
import torch.nn.functional as F
import joint_adapter as base
import mx_e4m3 as high
import posthoc_truncation as posthoc

TARGET_LAYERS, BN_LAYERS = base.TARGET_LAYERS, base.BN_LAYERS
MODES, BANK, FORMATS = base.MODES, base.BANK, base.FORMATS
require, codec = high.require, base.codec
inactive_bn = base.inactive_bn
effective_reference_weight = base.effective_reference_weight
WeightModeBatchNorm = base.WeightModeBatchNorm
exponent_truncate = posthoc.exponent_truncate
mantissa_truncate = posthoc.mantissa_truncate
SOURCE_SCHEMA = 'mx-triple-learned-scale32-64-training-source-v1'
DERIVED_SCHEMA = 'mx-triple-learned-scale32-64-training-derived-v1'
TRAINING_POLICY = 'joint_E4_RNE_E2_exponent_MSB_learned_binary_scale32_64_then_mantissa_identity_weight_STE_log_exponent_STE'
CHOICE_SCHEMA = 'mx-binary-scale-choice-v1'
OFFSET_CHOICES = (5, 6)
SCALE_FACTORS = (32, 64)
THETA_MIN, THETA_MAX, THETA_THRESHOLD, THETA_INITIAL = 5.0, 6.0, 5.5, 6.0
MX_WEIGHT_SHAPES = {'layers.3': (16, 12, 3, 3), 'layers.6': (16, 16, 3, 3),
                    'layers.10': (32, 16, 3, 3), 'layers.13': (32, 32, 3, 3),
                    'layers.16': (32, 32, 3, 3), 'layers.20': (128, 32, 5, 5)}
SOURCE_KEYS = {'schema', 'model', 'expand', 'activation_bits', 'weight_bits', 'weight_format',
               'bn_bank_key', 'supported_weight_modes', 'bank_mapping', 'weight_preprocess',
               'training_policy', 'mx_layers', 'scale_choices', 'auxiliary_fp32_state', 'provenance'}


def plain_metadata(value):
    if value is None or type(value) in (str, int, bool): return True
    if type(value) is float: return math.isfinite(value)
    if type(value) is list: return all(plain_metadata(v) for v in value)
    if type(value) is dict: return all(type(k) is str and plain_metadata(v) for k, v in value.items())
    return False


def exact_metadata(value, expected):
    if type(value) is not type(expected): return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(exact_metadata(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(exact_metadata(a, b) for a, b in zip(value, expected))
    return plain_metadata(value) and value == expected


def block_shape(weight_shape):
    require(len(weight_shape) >= 2 and all(type(v) is int and v > 0 for v in weight_shape),
            'Verwacht een positieve gewichtstensorvorm.')
    return (weight_shape[0], (math.prod(weight_shape[1:]) + high.BLOCK_SIZE - 1) // high.BLOCK_SIZE)


def theta_state_keys():
    return {name + '.quantize_fn.theta' for name in TARGET_LAYERS}


def base_state_dict(model):
    """Original model state, excluding only the six new scale parameters."""
    return {k: v for k, v in model.state_dict().items() if k not in theta_state_keys()}


def scale_parameters(model):
    return [model.get_submodule(name).quantize_fn.theta for name in TARGET_LAYERS]


@torch.no_grad()
def scale_snapshot(model):
    return {name: model.get_submodule(name).quantize_fn.theta.detach().cpu().clone()
            for name in TARGET_LAYERS}


def hard_offsets(theta: Tensor) -> Tensor:
    require(isinstance(theta, Tensor) and theta.dtype == torch.float32,
            'Schaalhulpparameters moeten FP32 zijn.')
    require(theta.ndim == 2 and theta.numel() > 0 and bool(torch.isfinite(theta).all()),
            'Onjuiste of niet-eindige schaalhulpparameters.')
    require(bool(((theta >= THETA_MIN) & (theta <= THETA_MAX)).all()),
            'Theta buiten [5,6]; projectie hoort direct na Adam.')
    return torch.where(theta.detach() < THETA_THRESHOLD, 5, 6).to(torch.int32)


def _validate_offsets(offsets, shape, device):
    require(isinstance(offsets, Tensor) and offsets.dtype in (torch.int32, torch.int64, torch.uint8),
            'Harde offsets moeten integerdata zijn.')
    require(tuple(offsets.shape) == tuple(shape) and offsets.device == device,
            'Harde schaaloffset heeft verkeerde vorm of device.')
    require(bool(((offsets == 5) | (offsets == 6)).all()), 'Alleen exponentoffset 5 of 6 toegestaan.')


def validate_scale_parameters(model):
    counts = {5: 0, 6: 0}; minima = []; maxima = []; total = 0
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name); q = layer.quantize_fn
        require(isinstance(q, JointMXQuantizer) and isinstance(q.theta, torch.nn.Parameter),
                'Ontbrekende geleerde schaalparameter: ' + name)
        require(tuple(q.theta.shape) == block_shape(tuple(layer.weight.shape))
                and q.theta.device == layer.weight.device, 'Onjuiste theta-layout: ' + name)
        offsets = hard_offsets(q.theta)
        total += offsets.numel()
        counts[5] += int((offsets == 5).sum()); counts[6] += int((offsets == 6).sum())
        minima.append(float(q.theta.detach().min())); maxima.append(float(q.theta.detach().max()))
    return {'parameter_tensors': len(TARGET_LAYERS), 'total_blocks': total,
            'theta_min': min(minima), 'theta_max': max(maxima),
            'choice32_blocks': counts[5], 'choice64_blocks': counts[6],
            'all_fp32_finite_in_range': True}


@torch.no_grad()
def project_scales_(model):
    """Project AFTER Adam; custom backward itself does not mask endpoint gradients."""
    require(cache_is_empty(model), 'Schaalprojectie binnen een gedeelde broncyclus.')
    counts = {}
    for name in TARGET_LAYERS:
        theta = model.get_submodule(name).quantize_fn.theta
        require(theta.dtype == torch.float32 and bool(torch.isfinite(theta).all()),
                'Niet-eindige theta na Adam: ' + name)
        counts[name] = int(((theta < THETA_MIN) | (theta > THETA_MAX)).sum())
        theta.clamp_(THETA_MIN, THETA_MAX)
    return {'projected_values': sum(counts.values()), 'projected_by_layer': counts}


def shifted_scale_codes(source: high.MXTensor, offsets: Tensor) -> Tensor:
    """Promoted exact source E8M0 code + hard offset, even for zero blocks."""
    require(isinstance(source, high.MXTensor), 'E4M3-bron vereist voor schaaloffset.')
    source.validate()
    _validate_offsets(offsets, source.scale_codes.shape, source.scale_codes.device)
    shifted = source.scale_codes.to(torch.int32) + offsets.to(torch.int32)
    require(bool((shifted <= 254).all()),
            'Geleerde lagere schaal buiten eindig E8M0-bereik; geen clipping of uint8-wrap.')
    return shifted.to(torch.uint8)


def code_chain(source: high.MXTensor, offsets: Tensor) -> dict:
    naive = exponent_truncate(source)
    shifted = shifted_scale_codes(source, offsets)
    middle = codec(6).MXTensor(naive.codes, shifted, source.shape)
    low = mantissa_truncate(middle)
    require(torch.equal(middle.scale_codes.to(torch.int32), source.scale_codes.to(torch.int32) + offsets.to(torch.int32))
            and torch.equal(middle.scale_codes, low.scale_codes),
            'Geleerde schaaloffset of mantissaschaal gewijzigd.')
    return {8: source, 6: middle, 4: low}


def encode_chain(effective: Tensor, offsets: Tensor) -> dict:
    return code_chain(high.encode_tensor(effective), offsets)


class _CachedCodeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, effective: Tensor, theta: Tensor, codes: Tensor, scales: Tensor, bit: int) -> Tensor:
        require(effective.dtype == theta.dtype == torch.float32, 'FP32 G(W) en theta vereist.')
        require(tuple(theta.shape) == block_shape(tuple(effective.shape)), 'Theta past niet bij gewichtsvorm.')
        decoded = codec(bit).MXTensor(codes, scales, tuple(effective.shape)).decode()
        ctx.bit = bit
        ctx.theta_shape = tuple(theta.shape)
        if bit in (6, 4):
            # Saved decoded REAL weights survive source-cache reset after pass 3.
            ctx.save_for_backward(decoded)
        return decoded

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient: Tensor):
        grad_theta = None
        if ctx.bit in (6, 4):
            (q,) = ctx.saved_tensors
            rows = q.shape[0]
            product = (gradient * q).reshape(rows, -1)
            product = F.pad(product, (0, (-product.shape[1]) % high.BLOCK_SIZE))
            grad_theta = product.reshape(*ctx.theta_shape, high.BLOCK_SIZE).sum(-1) * math.log(2.0)
        # Independent identity-STE toward G(W), without scale factor 32 or 64.
        return gradient.clone(), grad_theta, None, None, None


class JointMXQuantizer(base.JointMXQuantizer):
    def __init__(self, weight_shape, device=None):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.full(block_shape(tuple(weight_shape)), THETA_INITIAL,
                                                  dtype=torch.float32, device=device))
        self._reset()

    def _reset(self):
        self._triplet_active = False
        self._chain = self._source_key = self._versions = self._offsets = self._offset_version = None
        self._theta_version = None
        self._used_modes = []

    def _assert_cache_unchanged(self):
        require(self._versions == {b: (x.codes._version, x.scale_codes._version)
                                  for b, x in self._chain.items()}, 'Gecachte codes of schalen gewijzigd.')
        require(self.theta._version == self._theta_version and self._offsets._version == self._offset_version,
                'Theta of harde schaalkeuzes gewijzigd tussen precisiepasses.')

    def forward(self, weight: Tensor) -> Tensor:
        require(self.abit == self.wbit == 32, 'A32 en W32-voorbewerking moeten behouden blijven.')
        bit = self.weight_mode; codec(bit)
        effective = effective_reference_weight(weight)
        if self._triplet_active:
            require(bit not in self._used_modes, 'Dubbele modus in dezelfde broncyclus.')
            key = (id(weight), weight._version, weight.device, weight.dtype, tuple(weight.shape),
                   id(self.theta), self.theta._version, self.theta.device, tuple(self.theta.shape))
            if self._chain is None:
                self._offsets = hard_offsets(self.theta)
                self._chain = encode_chain(effective, self._offsets)
                self._source_key = key
                self._versions = {b: (x.codes._version, x.scale_codes._version) for b, x in self._chain.items()}
                self._theta_version = self.theta._version; self._offset_version = self._offsets._version
            else:
                require(key == self._source_key, 'FP32-master of theta gewijzigd tussen precisiepasses.')
                self._assert_cache_unchanged()
            encoded = self._chain[bit]
            self._used_modes.append(bit)
        else:
            encoded = encode_chain(effective, hard_offsets(self.theta))[bit]
        return _CachedCodeSTE.apply(effective, self.theta, encoded.codes, encoded.scale_codes, bit)


def select_mode(model, bit):
    codec(bit)
    require(not hasattr(model, '_frozen_weight_bit') or model._frozen_weight_bit == bit,
            'Bevroren export: reconstrueer de gewenste precisie uit de E4-bron.')
    base.select_mode(model, bit)


def adapt_copy(reference):
    model = base.adapt_copy(reference)
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        layer.quantize_fn = JointMXQuantizer(tuple(layer.weight.shape), layer.weight.device)
    select_mode(model, 8)
    return model


def cache_is_empty(model):
    return all(not q._triplet_active and q._chain is None and q._offsets is None and not q._used_modes
               for q in (model.get_submodule(n).quantize_fn for n in TARGET_LAYERS))


@contextmanager
def shared_source_triplets(model):
    quantizers = [model.get_submodule(n).quantize_fn for n in TARGET_LAYERS]
    require(all(isinstance(q, JointMXQuantizer) for q in quantizers) and cache_is_empty(model),
            'Onverwachte quantizer, oude of geneste broncache.')
    position = [0]; training = model.training
    order = (8, 6, 4) if training else (4, 6, 8)
    audit = {'source_triplets': 0, 'high_encodings': 0, 'exponent_derivations': 0,
             'mantissa_derivations': 0, 'scale_offsets_learned': 0,
             'mantissa_scale_copies_unchanged': 0,
             'offset_choices': list(OFFSET_CHOICES), 'scale_factors': list(SCALE_FACTORS),
             'learned_shared_scale': True, 'same_hard_offsets_across_lower_modes': True,
             'theta_unchanged_within_triplet': True,
             'source_cache_cleared': False, 'no_direct_lower_quantizer': True,
             'training_order': list(order), 'source_weight_bits': 8}

    def pre(module, inputs):
        require(module.training == training and module.weight_mode == order[position[0]],
                'Train/eval of passvolgorde gewijzigd binnen broncyclus.')
        if position[0] == 0:
            require(cache_is_empty(module), 'Onverwacht oude bron.')
            for q in quantizers: q._triplet_active = True

    def post(module, inputs, output):
        expected = list(order[:position[0] + 1])
        require(all(q._chain is not None and q._used_modes == expected for q in quantizers),
                'Niet alle modi gebruiken dezelfde laagbron.')
        for q in quantizers: q._assert_cache_unchanged()
        position[0] += 1
        if position[0] == len(order):
            for q in quantizers: q._reset()
            position[0] = 0
            audit['source_triplets'] += 1
            audit['high_encodings'] += len(TARGET_LAYERS)
            audit['exponent_derivations'] += len(TARGET_LAYERS)
            audit['mantissa_derivations'] += len(TARGET_LAYERS)
            audit['scale_offsets_learned'] += 2 * len(TARGET_LAYERS)
            audit['mantissa_scale_copies_unchanged'] += len(TARGET_LAYERS)
    handles = [model.register_forward_pre_hook(pre), model.register_forward_hook(post)]
    try:
        yield audit
        require(position[0] == 0 and cache_is_empty(model), 'Onvolledige broncyclus.')
        audit['source_cache_cleared'] = True
    finally:
        for handle in handles: handle.remove()
        for q in quantizers: q._reset()


def pack_scale_choices(offsets):
    """Flatten row-major; bit j%8 is offset[j]-5, with zero unused high bits."""
    require(isinstance(offsets, Tensor), 'Geen harde schaalkeuzes.')
    _validate_offsets(offsets, offsets.shape, offsets.device)
    require(offsets.ndim == 2 and offsets.numel() > 0, 'Schaalkeuzes moeten een niet-lege matrix zijn.')
    flags = (offsets.detach().cpu().reshape(-1).to(torch.int64) - 5)
    padding = (-flags.numel()) % 8
    flags = F.pad(flags, (0, padding))
    powers = 1 << torch.arange(8, dtype=torch.int64)
    packed = (flags.reshape(-1, 8) * powers).sum(-1).to(torch.uint8)
    return {'schema': CHOICE_SCHEMA, 'shape': list(offsets.shape), 'num_blocks': offsets.numel(),
            'bit_order': 'little', 'mapping': {'0': 5, '1': 6}, 'packed_choices': packed}


def unpack_scale_choices(payload, device='cpu'):
    require(type(payload) is dict and set(payload) ==
            {'schema', 'shape', 'num_blocks', 'bit_order', 'mapping', 'packed_choices'},
            'Onverwachte velden in gepakte schaalkeuzes.')
    require(payload['schema'] == CHOICE_SCHEMA and payload['bit_order'] == 'little'
            and exact_metadata(payload['mapping'], {'0': 5, '1': 6}), 'Afwijkende schaalkeuze-encoding.')
    shape = payload['shape']; n = payload['num_blocks']; packed = payload['packed_choices']
    require(type(shape) is list and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape)
            and type(n) is int and n == math.prod(shape), 'Onjuiste schaalkeuzevorm.')
    require(isinstance(packed, Tensor) and packed.dtype == torch.uint8
            and tuple(packed.shape) == ((n + 7) // 8,), 'Onjuiste gepakte schaalkeuzedata.')
    data = packed.detach().to(device=device, dtype=torch.int64)
    if n % 8:
        require(int(data[-1]) >> (n % 8) == 0, 'Niet-nul paddingbits in schaalkeuzes.')
    bits = ((data[:, None] >> torch.arange(8, device=device)) & 1).reshape(-1)[:n]
    return (bits.reshape(shape) + 5).to(torch.int32)


def source_payload(model, provenance):
    require(cache_is_empty(model), 'Export binnen onvolledige broncyclus.')
    validate_scale_parameters(model)
    encoded, choices, rows = {}, {}, []
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        effective = effective_reference_weight(layer.weight.detach())
        offsets = hard_offsets(layer.quantize_fn.theta)
        chain = encode_chain(effective, offsets)
        encoded[name] = chain[8].payload(); choices[name] = pack_scale_choices(offsets)
        for bit in (8, 6, 4):
            rows.append({'weight_bits': bit, 'weight_format': FORMATS[bit], 'layer': name,
                         'derivation': {8: 'E4_RNE_source', 6: 'E4_exponent_MSB_learned_scale32_64',
                                        4: 'E4_exponent_MSB_learned_scale32_64_then_mantissa_shift2'}[bit],
                         'scale_exponent_offset': 0 if bit == 8 else 'learned_per_block_5_or_6',
                         'scale_factor': 1 if bit == 8 else 'learned_per_block_32_or_64',
                         'choice32_blocks': int((offsets == 5).sum()),
                         'choice64_blocks': int((offsets == 6).sum()),
                         'packed_scale_choice_bytes': choices[name]['packed_choices'].numel(),
                         **codec(bit).distortion(effective, chain[bit])})
    remove = {n + '.weight' for n in TARGET_LAYERS} | theta_state_keys()
    auxiliary = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in remove}
    payload = {'schema': SOURCE_SCHEMA, 'model': 'svhnq', 'expand': 2, 'activation_bits': 32,
               'weight_bits': 8, 'weight_format': FORMATS[8], 'bn_bank_key': BANK[8],
               'supported_weight_modes': [8, 6, 4], 'bank_mapping': {'8': '8', '6': '32', '4': '4'},
               'weight_preprocess': 'already_G(W); do_not_repeat', 'training_policy': TRAINING_POLICY,
               'mx_layers': encoded, 'scale_choices': choices,
               'auxiliary_fp32_state': auxiliary, 'provenance': copy.deepcopy(provenance)}
    validate_source(payload)
    return payload, rows


def auxiliary_layout():
    """Exact state layout of the fixed svhnq expand=2 auxiliary tensors."""
    layout = {'layers.0.weight': ((12, 3, 5, 5), torch.float32),
              'layers.0.bias': ((12,), torch.float32),
              'fc.weight': ((10, 128), torch.float32), 'fc.bias': ((10,), torch.float32)}
    for name, channels in zip(BN_LAYERS, (16, 16, 32, 32, 32)):
        for bank in ('1', '2', '4', '8', '32'):
            for suffix in ('weight', 'bias', 'running_mean', 'running_var'):
                layout[name + '.bn_dict.' + bank + '.' + suffix] = ((channels,), torch.float32)
            layout[name + '.bn_dict.' + bank + '.num_batches_tracked'] = ((), torch.int64)
    return layout


def _validate_common(payload, bit):
    codec(bit)
    require(plain_metadata(payload['provenance']), 'Verborgen toestand in provenance.')
    require(payload['model'] == 'svhnq' and type(payload['expand']) is int and payload['expand'] == 2
            and type(payload['activation_bits']) is int and payload['activation_bits'] == 32,
            'Model/A32 gewijzigd.')
    require(type(payload['weight_bits']) is int and payload['weight_bits'] == bit and payload['weight_format'] == FORMATS[bit]
            and payload['bn_bank_key'] == BANK[bit], 'Afwijkend formaat of BN-bank.')
    require(exact_metadata(payload['supported_weight_modes'], [8, 6, 4])
            and exact_metadata(payload['bank_mapping'], {'8': '8', '6': '32', '4': '4'}), 'Modi/BN gewijzigd.')
    require(payload['weight_preprocess'] == 'already_G(W); do_not_repeat'
            and payload['training_policy'] == TRAINING_POLICY, 'Afleidingsbeleid gewijzigd.')
    require(type(payload['mx_layers']) is dict and tuple(payload['mx_layers']) == TARGET_LAYERS
            and type(payload['scale_choices']) is dict and tuple(payload['scale_choices']) == TARGET_LAYERS,
            'Verkeerde MX-laagselectie of schaalkeuzes.')
    for name, value in payload['mx_layers'].items():
        mx = codec(bit).MXTensor.from_payload(value)
        require(mx.shape == MX_WEIGHT_SHAPES[name], 'Onjuiste MX-laagvorm.')
        offsets = unpack_scale_choices(payload['scale_choices'][name])
        require(tuple(offsets.shape) == tuple(mx.scale_codes.shape), 'Schaalkeuzes passen niet bij MX-laag.')
        if bit == 8:
            shifted_scale_codes(mx, offsets)
        else:
            require(bool((mx.scale_codes.to(torch.int32) - offsets >= 0).all()),
                    'Afgeleide schaal kan niet uit een eindige E4-bronschaal ontstaan.')
    aux = payload['auxiliary_fp32_state']
    require(type(aux) is dict and all(type(k) is str and isinstance(v, Tensor) for k, v in aux.items()),
            'Onjuiste auxiliaire modelstaat.')
    require(not (({n + '.weight' for n in TARGET_LAYERS} | theta_state_keys()) & set(aux)),
            'Verborgen MX-master of FP32-theta in export.')
    require(not any('quantize_fn' in k or k.endswith('.theta') for k in aux),
            'Trainbare quantizerstaat mag niet in een compacte export staan.')
    expected = auxiliary_layout()
    require(set(aux) == set(expected), 'Onbekende of ontbrekende auxiliaire modelvelden.')
    require(all(tuple(aux[k].shape) == shape and aux[k].dtype == dtype
                and bool(torch.isfinite(aux[k]).all()) for k, (shape, dtype) in expected.items()),
            'Niet-eindige, verkeerd gevormde of onverwacht getypeerde auxiliaire staat.')


def validate_source(payload):
    require(type(payload) is dict and set(payload) == SOURCE_KEYS, 'Onverwachte bronvelden.')
    require(payload['schema'] == SOURCE_SCHEMA, 'Geen geleerde E4M3-bronexport.')
    _validate_common(payload, 8)


def derivation_metadata(bit, source_sha256):
    require(type(bit) is int and bit in (6, 4), 'Verwacht afgeleide modus6 of4.')
    require(type(source_sha256) is str and len(source_sha256) == 64
            and set(source_sha256) <= set('0123456789abcdef'), 'Ongeldige bron-SHA256.')
    return {'method': 'E4_exponent_MSB_learned_scale32_64' if bit == 6 else 'E4_exponent_MSB_learned_scale32_64_then_mantissa_shift2',
            'source_weight_bits': 8, 'target_weight_bits': bit, 'source_sha256': source_sha256,
            'dropped_exponent_bits': 2, 'dropped_mantissa_bits': 0 if bit == 6 else 2,
            'scales_unchanged': False, 'source_scale_exponent_offsets': list(OFFSET_CHOICES),
            'scale_factors': list(SCALE_FACTORS),
            'scale_policy': 'source_E8M0_code_plus_packed_binary_offset_5_or_6_no_clip_no_wrap',
            'scale_choices_policy': 'one_saved_bit_per_block_shared_by_E2M3_E2M1',
            'shared_scale_compensation': True, 'learned_shared_scale': True,
            'bias_compensation': False, 'masters_required': False, 'theta_required': False}


def derive_payload(source, bit, source_sha256):
    validate_source(source)
    metadata = derivation_metadata(bit, source_sha256)
    result = copy.deepcopy(source)
    result.update(schema=DERIVED_SCHEMA, weight_bits=bit, weight_format=FORMATS[bit], bn_bank_key=BANK[bit])
    result['mx_layers'] = {name: code_chain(high.MXTensor.from_payload(value),
        unpack_scale_choices(source['scale_choices'][name]))[bit].payload()
        for name, value in source['mx_layers'].items()}
    result['derivation'] = metadata
    validate_derived(result)
    return result


def validate_derived(payload):
    require(type(payload) is dict and set(payload) == SOURCE_KEYS | {'derivation'}, 'Onverwachte afgeleide velden.')
    require(payload['schema'] == DERIVED_SCHEMA, 'Geen geleerde afgeleide MX-export.')
    bit = payload['weight_bits']
    require(type(bit) is int and bit in (6, 4), 'Verkeerde afgeleide precisie.')
    _validate_common(payload, bit)
    d = payload['derivation']
    require(type(d) is dict and exact_metadata(d, derivation_metadata(bit, d.get('source_sha256'))),
            'Afwijkende bitbewerking of schaal-/biascompensatie.')


def export_payload(model, bit, provenance):
    codec(bit)
    source, rows = source_payload(model, provenance)
    return (source if bit == 8 else derive_payload(source, bit, '0' * 64)), [r for r in rows if r['weight_bits'] == bit]


def load_exported_model(models, payload, device, bit=None):
    source = payload.get('schema') == SOURCE_SCHEMA
    if source:
        validate_source(payload); bit = 8 if bit is None else bit; codec(bit)
    else:
        validate_derived(payload)
        bit = payload['weight_bits'] if bit is None else bit
        require(bit == payload['weight_bits'], 'Afgeleide export heeft vaste doelmodus.')
    # No learned quantizer is needed for inference reconstruction.
    model = base.adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2).to(device))
    expected = set(model.state_dict()) - {n + '.weight' for n in TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected, 'Verborgen master/theta of ontbrekende hulpstaat.')
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        mx = codec(8 if source else bit).MXTensor.from_payload(payload['mx_layers'][name], device)
        if source and bit != 8:
            mx = code_chain(mx, unpack_scale_choices(payload['scale_choices'][name], device))[bit]
        require(mx.shape == tuple(layer.weight.shape), 'Onjuiste gewichtsvorm.')
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, base.original_adapter.FrozenMXConv(layer, mx))
    missing, extra = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n + '.decoded_weight' for n in TARGET_LAYERS} and not extra, 'Onvolledige reconstructie.')
    select_mode(model, bit); model._frozen_weight_bit = bit
    return model.to(device).eval().requires_grad_(False)


def load_verified_payload(path):
    path = Path(path); marker = path.with_suffix(path.suffix + '.sha256')
    require(path.is_file() and marker.is_file(), 'MX-bestand of checksum ontbreekt.')
    with path.open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    require(digest == marker.read_text(encoding='ascii').strip(), 'MX-checksum wijkt af.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(type(payload) is dict, 'Geen MX-payload.')
    if payload.get('schema') == SOURCE_SCHEMA: validate_source(payload)
    else: validate_derived(payload)
    return payload
