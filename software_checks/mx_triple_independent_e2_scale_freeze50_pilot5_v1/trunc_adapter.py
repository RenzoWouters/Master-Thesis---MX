"""E4M3 code source with an independently learned absolute E2 E8M0 scale.

E4 keeps its existing dynamic amax scale. E2M3 and E2M1 share one absolute
scale per group, rounded from FP32 phi=log2(S_E2). Phi is initialized ONCE
to the fresh E4 exponent plus six, then has no ratio constraint to E4.
The lower element codes still come from exponent-MSB and mantissa truncation.
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
SOURCE_SCHEMA = 'mx-triple-independent-e2-scale-training-source-v1'
DERIVED_SCHEMA = 'mx-triple-independent-e2-scale-training-derived-v1'
TRAINING_POLICY = 'joint_E4_RNE_E2_exponent_MSB_independent_absolute_E8M0_then_mantissa_identity_weight_STE_log_scale_STE'
ROUNDING_POLICY = 'nearest_integer_half_up'
PHI_MIN, PHI_MAX = -127.0, 127.0
INITIAL_RELATIVE_OFFSET = 6
E2_EXPONENTS = tuple(range(-127, 128))
MX_WEIGHT_SHAPES = {'layers.3': (16, 12, 3, 3), 'layers.6': (16, 16, 3, 3),
                    'layers.10': (32, 16, 3, 3), 'layers.13': (32, 32, 3, 3),
                    'layers.16': (32, 32, 3, 3), 'layers.20': (128, 32, 5, 5)}
SOURCE_KEYS = {'schema', 'model', 'expand', 'activation_bits', 'weight_bits', 'weight_format',
               'bn_bank_key', 'supported_weight_modes', 'bank_mapping', 'weight_preprocess',
               'training_policy', 'mx_layers', 'lower_scale_codes', 'auxiliary_fp32_state', 'provenance'}


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


def phi_state_keys():
    return {name + '.quantize_fn.phi' for name in TARGET_LAYERS}


def base_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if k not in phi_state_keys()}


def scale_parameters(model):
    return [model.get_submodule(name).quantize_fn.phi for name in TARGET_LAYERS]


@torch.no_grad()
def scale_snapshot(model):
    return {name: model.get_submodule(name).quantize_fn.phi.detach().cpu().clone()
            for name in TARGET_LAYERS}


def hard_exponents(phi: Tensor) -> Tensor:
    require(isinstance(phi, Tensor) and phi.dtype == torch.float32,
            'Absolute schaalhulpparameters moeten FP32 zijn.')
    require(phi.ndim == 2 and phi.numel() > 0 and bool(torch.isfinite(phi).all()),
            'Onjuiste of niet-eindige absolute schaalhulpparameters.')
    require(bool(((phi >= PHI_MIN) & (phi <= PHI_MAX)).all()),
            'Phi buiten [-127,127]; projectie hoort direct na Adam.')
    # floor + fractional comparison avoids rounding a nextafter neighbor onto a tie.
    value = phi.detach()
    lower = torch.floor(value)
    return lower.to(torch.int32) + ((value - lower) >= 0.5).to(torch.int32)


def scale_codes(phi: Tensor) -> Tensor:
    return (hard_exponents(phi) + 127).to(torch.uint8)


def validate_lower_scale_codes(value, shape, device):
    require(isinstance(value, Tensor) and value.dtype == torch.uint8,
            'Onafhankelijke E2-schalen moeten uint8 E8M0-codes zijn.')
    require(tuple(value.shape) == tuple(shape) and value.device == device,
            'Onafhankelijke E2-schaal heeft verkeerde vorm of device.')
    require(bool((value < 255).all()), 'E8M0 NaN-code255 is niet toegestaan.')


def exponent_histogram(exponents):
    require(isinstance(exponents, Tensor) and exponents.dtype in (torch.int32, torch.int64),
            'Exponenthistogram verwacht integerexponenten.')
    require(bool(((exponents >= -127) & (exponents <= 127)).all()), 'Exponent buiten eindig E8M0-bereik.')
    counts = torch.bincount(exponents.detach().reshape(-1).to(torch.int64) + 127, minlength=255)
    return {str(e): int(n) for e, n in zip(E2_EXPONENTS, counts.cpu().tolist())}


def observed_offset_histogram(offsets):
    require(isinstance(offsets, Tensor) and offsets.dtype in (torch.int32, torch.int64),
            'Waargenomen exponentverschillen moeten integerdata zijn.')
    keys, counts = torch.unique(offsets.detach().cpu(), sorted=True, return_counts=True)
    return {str(int(k)): int(n) for k, n in zip(keys.tolist(), counts.tolist())}


def validate_scale_parameters(model):
    counts = {str(e): 0 for e in E2_EXPONENTS}
    minima, maxima = [], []
    total = at_min = at_max = 0
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name); q = layer.quantize_fn
        require(isinstance(q, JointMXQuantizer) and isinstance(q.phi, torch.nn.Parameter),
                'Ontbrekende onafhankelijke schaalparameter: ' + name)
        require(tuple(q.phi.shape) == block_shape(tuple(layer.weight.shape))
                and q.phi.device == layer.weight.device, 'Onjuiste phi-layout: ' + name)
        exponents = hard_exponents(q.phi)
        total += exponents.numel()
        for e, n in exponent_histogram(exponents).items(): counts[e] += n
        at_min += int((q.phi.detach() == PHI_MIN).sum())
        at_max += int((q.phi.detach() == PHI_MAX).sum())
        minima.append(float(q.phi.detach().min())); maxima.append(float(q.phi.detach().max()))
    require(sum(counts.values()) == total, 'Onvolledig histogram van E2-exponenten.')
    return {'parameter_tensors': len(TARGET_LAYERS), 'total_blocks': total,
            'phi_min': min(minima), 'phi_max': max(maxima), 'e2_exponent_counts': counts,
            'phi_at_min_blocks': at_min, 'phi_at_max_blocks': at_max,
            'all_fp32_finite_in_range': True}


@torch.no_grad()
def project_scales_(model):
    """Project only to finite E8M0 exponent range; no dynamic E4-based limit.

    E8M0 exponents126/127 are valid, but some E2 element products overflow
    FP32. The unchanged codec rejects those actual products, without clipping.
    """
    require(cache_is_empty(model), 'Schaalprojectie binnen een gedeelde broncyclus.')
    counts = {}
    for name in TARGET_LAYERS:
        phi = model.get_submodule(name).quantize_fn.phi
        require(phi.dtype == torch.float32 and bool(torch.isfinite(phi).all()), 'Niet-eindige phi na Adam: ' + name)
        counts[name] = int(((phi < PHI_MIN) | (phi > PHI_MAX)).sum())
        phi.clamp_(PHI_MIN, PHI_MAX)
    return {'projected_values': sum(counts.values()), 'projected_by_layer': counts}


def code_chain(source: high.MXTensor, lower_scale_codes: Tensor) -> dict:
    require(isinstance(source, high.MXTensor), 'E4M3-bron vereist.')
    source.validate()
    validate_lower_scale_codes(lower_scale_codes, source.scale_codes.shape, source.scale_codes.device)
    naive = exponent_truncate(source)
    middle = codec(6).MXTensor(naive.codes, lower_scale_codes.clone(), source.shape)
    low = mantissa_truncate(middle)
    require(torch.equal(middle.scale_codes, lower_scale_codes) and torch.equal(middle.scale_codes, low.scale_codes),
            'Onafhankelijke E2-schaal of mantissaschaal gewijzigd.')
    return {8: source, 6: middle, 4: low}


def encode_chain(effective: Tensor, lower_scale_codes: Tensor) -> dict:
    return code_chain(high.encode_tensor(effective), lower_scale_codes)


@torch.no_grad()
def scale_diagnostics(model):
    result = {}
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        phi = layer.quantize_fn.phi
        e4 = high.encode_tensor(effective_reference_weight(layer.weight.detach())).scale_codes.to(torch.int32) - 127
        e2 = hard_exponents(phi)
        result[name] = {'phi': phi.detach().cpu().tolist(), 'e4_exponents': e4.cpu().tolist(),
                        'e2_exponents': e2.cpu().tolist(), 'observed_relative_offsets': (e2-e4).cpu().tolist()}
    return result


class _CachedCodeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, effective: Tensor, phi: Tensor, codes: Tensor, scales: Tensor, bit: int) -> Tensor:
        require(effective.dtype == phi.dtype == torch.float32, 'FP32 G(W) en phi vereist.')
        require(tuple(phi.shape) == block_shape(tuple(effective.shape)), 'Phi past niet bij gewichtsvorm.')
        decoded = codec(bit).MXTensor(codes, scales, tuple(effective.shape)).decode()
        ctx.bit = bit; ctx.phi_shape = tuple(phi.shape)
        if bit in (6, 4): ctx.save_for_backward(decoded)
        return decoded

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient: Tensor):
        grad_phi = None
        if ctx.bit in (6, 4):
            (q,) = ctx.saved_tensors
            product = (gradient * q).reshape(q.shape[0], -1)
            product = F.pad(product, (0, (-product.shape[1]) % high.BLOCK_SIZE))
            grad_phi = product.reshape(*ctx.phi_shape, high.BLOCK_SIZE).sum(-1) * math.log(2.0)
        return gradient.clone(), grad_phi, None, None, None


class JointMXQuantizer(base.JointMXQuantizer):
    def __init__(self, weight: Tensor):
        super().__init__()
        # No forward, RNG draw or live autograd graph: initialization happens once.
        with torch.no_grad():
            source = high.encode_tensor(effective_reference_weight(weight.detach()))
            initial = source.scale_codes.to(torch.int32) - 127 + INITIAL_RELATIVE_OFFSET
            require(bool(((initial >= PHI_MIN) & (initial <= PHI_MAX)).all()),
                    'Eenmalige initialisatie E4+6 ligt buiten eindig E8M0; niet clippen.')
            self.phi = torch.nn.Parameter(initial.to(torch.float32).clone())
        self._reset()

    def _reset(self):
        self._triplet_active = False
        self._chain = self._source_key = self._versions = self._scales = self._scales_version = None
        self._phi_version = None
        self._used_modes = []

    def _assert_cache_unchanged(self):
        require(self._versions == {b: (x.codes._version, x.scale_codes._version) for b, x in self._chain.items()},
                'Gecachte codes of schalen gewijzigd.')
        require(self.phi._version == self._phi_version and self._scales._version == self._scales_version,
                'Phi of absolute E2-schalen gewijzigd tussen precisiepasses.')

    def forward(self, weight: Tensor) -> Tensor:
        require(self.abit == self.wbit == 32, 'A32 en W32-voorbewerking moeten behouden blijven.')
        bit = self.weight_mode; codec(bit)
        effective = effective_reference_weight(weight)
        if self._triplet_active:
            require(bit not in self._used_modes, 'Dubbele modus in dezelfde broncyclus.')
            key = (id(weight), weight._version, weight.device, weight.dtype, tuple(weight.shape),
                   id(self.phi), self.phi._version, self.phi.device, tuple(self.phi.shape))
            if self._chain is None:
                self._scales = scale_codes(self.phi)
                self._chain = encode_chain(effective, self._scales)
                self._source_key = key
                self._versions = {b: (x.codes._version, x.scale_codes._version) for b, x in self._chain.items()}
                self._phi_version = self.phi._version; self._scales_version = self._scales._version
            else:
                require(key == self._source_key, 'FP32-master of phi gewijzigd tussen precisiepasses.')
                self._assert_cache_unchanged()
            encoded = self._chain[bit]
            self._used_modes.append(bit)
        else:
            encoded = encode_chain(effective, scale_codes(self.phi))[bit]
        return _CachedCodeSTE.apply(effective, self.phi, encoded.codes, encoded.scale_codes, bit)


def select_mode(model, bit):
    codec(bit)
    require(not hasattr(model, '_frozen_weight_bit') or model._frozen_weight_bit == bit,
            'Bevroren export: reconstrueer de gewenste precisie uit de E4-bron.')
    base.select_mode(model, bit)


def adapt_copy(reference):
    model = base.adapt_copy(reference)
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        layer.quantize_fn = JointMXQuantizer(layer.weight)
    select_mode(model, 8)
    return model


def cache_is_empty(model):
    return all(not q._triplet_active and q._chain is None and q._scales is None and not q._used_modes
               for q in (model.get_submodule(n).quantize_fn for n in TARGET_LAYERS))


@contextmanager
def shared_source_triplets(model):
    quantizers = [model.get_submodule(n).quantize_fn for n in TARGET_LAYERS]
    require(all(isinstance(q, JointMXQuantizer) for q in quantizers) and cache_is_empty(model),
            'Onverwachte quantizer, oude of geneste broncache.')
    position = [0]; training = model.training
    order = (8, 6, 4) if training else (4, 6, 8)
    audit = {'source_triplets': 0, 'high_encodings': 0, 'exponent_derivations': 0,
             'mantissa_derivations': 0, 'learned_absolute_scales_applied': 0,
             'mantissa_scale_copies_unchanged': 0, 'e2_exponent_bounds': [-127, 127],
             'independent_absolute_lower_scale': True, 'learned_shared_scale': True,
             'same_absolute_scale_across_lower_modes': True, 'phi_unchanged_within_triplet': True,
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
            for key in ('high_encodings', 'exponent_derivations', 'mantissa_derivations', 'mantissa_scale_copies_unchanged'):
                audit[key] += len(TARGET_LAYERS)
            audit['learned_absolute_scales_applied'] += 2 * len(TARGET_LAYERS)
    handles = [model.register_forward_pre_hook(pre), model.register_forward_hook(post)]
    try:
        yield audit
        require(position[0] == 0 and cache_is_empty(model), 'Onvolledige broncyclus.')
        audit['source_cache_cleared'] = True
    finally:
        for handle in handles: handle.remove()
        for q in quantizers: q._reset()


def source_payload(model, provenance):
    require(cache_is_empty(model), 'Export binnen onvolledige broncyclus.')
    validate_scale_parameters(model)
    encoded, lower_scales, rows = {}, {}, []
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        effective = effective_reference_weight(layer.weight.detach())
        lower = scale_codes(layer.quantize_fn.phi)
        chain = encode_chain(effective, lower)
        encoded[name] = chain[8].payload(); lower_scales[name] = lower.detach().cpu().clone()
        relative = lower.to(torch.int32) - chain[8].scale_codes.to(torch.int32)
        for bit in (8, 6, 4):
            rows.append({'weight_bits': bit, 'weight_format': FORMATS[bit], 'layer': name,
                         'derivation': {8: 'E4_RNE_source', 6: 'E4_exponent_MSB_independent_E2_scale',
                                        4: 'E4_exponent_MSB_independent_E2_scale_then_mantissa_shift2'}[bit],
                         'scale_policy': 'dynamic_E4_amax' if bit == 8 else 'learned_absolute_E2_E8M0',
                         'lower_scale_bytes': lower.numel(),
                         'e2_exponent_counts': exponent_histogram(lower.to(torch.int32) - 127),
                         'observed_relative_offset_counts': observed_offset_histogram(relative),
                         **codec(bit).distortion(effective, chain[bit])})
    remove = {n + '.weight' for n in TARGET_LAYERS} | phi_state_keys()
    auxiliary = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in remove}
    payload = {'schema': SOURCE_SCHEMA, 'model': 'svhnq', 'expand': 2, 'activation_bits': 32,
               'weight_bits': 8, 'weight_format': FORMATS[8], 'bn_bank_key': BANK[8],
               'supported_weight_modes': [8, 6, 4], 'bank_mapping': {'8': '8', '6': '32', '4': '4'},
               'weight_preprocess': 'already_G(W); do_not_repeat', 'training_policy': TRAINING_POLICY,
               'mx_layers': encoded, 'lower_scale_codes': lower_scales,
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
            and type(payload['activation_bits']) is int and payload['activation_bits'] == 32, 'Model/A32 gewijzigd.')
    require(type(payload['weight_bits']) is int and payload['weight_bits'] == bit and payload['weight_format'] == FORMATS[bit]
            and payload['bn_bank_key'] == BANK[bit], 'Afwijkend formaat of BN-bank.')
    require(exact_metadata(payload['supported_weight_modes'], [8, 6, 4])
            and exact_metadata(payload['bank_mapping'], {'8': '8', '6': '32', '4': '4'}), 'Modi/BN gewijzigd.')
    require(payload['weight_preprocess'] == 'already_G(W); do_not_repeat'
            and payload['training_policy'] == TRAINING_POLICY, 'Afleidingsbeleid gewijzigd.')
    require(type(payload['mx_layers']) is dict and tuple(payload['mx_layers']) == TARGET_LAYERS
            and type(payload['lower_scale_codes']) is dict and tuple(payload['lower_scale_codes']) == TARGET_LAYERS,
            'Verkeerde MX-laagselectie of absolute E2-schalen.')
    for name, value in payload['mx_layers'].items():
        mx = codec(bit).MXTensor.from_payload(value)
        require(mx.shape == MX_WEIGHT_SHAPES[name], 'Onjuiste MX-laagvorm.')
        lower = payload['lower_scale_codes'][name]
        validate_lower_scale_codes(lower, mx.scale_codes.shape, mx.scale_codes.device)
        if bit in (6, 4):
            require(torch.equal(mx.scale_codes, lower), 'Afgeleide schaal verschilt van opgeslagen absolute E2-schaal.')
        # E8M0 code validity alone does not imply a finite decoded FP32 product.
        mx.decode()
        if bit == 8:
            chain = code_chain(mx, lower)
            chain[6].decode(); chain[4].decode()
    aux = payload['auxiliary_fp32_state']
    require(type(aux) is dict and all(type(k) is str and isinstance(v, Tensor) for k, v in aux.items()),
            'Onjuiste auxiliaire modelstaat.')
    require(not (({n + '.weight' for n in TARGET_LAYERS} | phi_state_keys()) & set(aux)),
            'Verborgen MX-master of FP32-phi in export.')
    require(not any('quantize_fn' in k or k.endswith(('.phi', '.theta')) for k in aux),
            'Trainbare quantizerstaat mag niet in een compacte export staan.')
    expected = auxiliary_layout()
    require(set(aux) == set(expected), 'Onbekende of ontbrekende auxiliaire modelvelden.')
    require(all(tuple(aux[k].shape) == shape and aux[k].dtype == dtype
                and bool(torch.isfinite(aux[k]).all()) for k, (shape, dtype) in expected.items()),
            'Niet-eindige, verkeerd gevormde of onverwacht getypeerde auxiliaire staat.')


def validate_source(payload):
    require(type(payload) is dict and set(payload) == SOURCE_KEYS, 'Onverwachte bronvelden.')
    require(payload['schema'] == SOURCE_SCHEMA, 'Geen onafhankelijke E2-schaalbronexport.')
    _validate_common(payload, 8)


def derivation_metadata(bit, source_sha256):
    require(type(bit) is int and bit in (6, 4), 'Verwacht afgeleide modus6 of4.')
    require(type(source_sha256) is str and len(source_sha256) == 64
            and set(source_sha256) <= set('0123456789abcdef'), 'Ongeldige bron-SHA256.')
    return {'method': 'E4_exponent_MSB_independent_absolute_E2_scale' if bit == 6 else 'E4_exponent_MSB_independent_absolute_E2_scale_then_mantissa_shift2',
            'source_weight_bits': 8, 'target_weight_bits': bit, 'source_sha256': source_sha256,
            'dropped_exponent_bits': 2, 'dropped_mantissa_bits': 0 if bit == 6 else 2,
            'scale_policy': 'saved_absolute_E2_E8M0_code_no_source_ratio_constraint_no_clip_no_wrap',
            'lower_scale_bits_per_block': 8, 'lower_scale_shared_by_E2M3_E2M1': True,
            'independent_absolute_lower_scale': True, 'learned_shared_scale': True,
            'bias_compensation': False, 'masters_required': False, 'phi_required': False}


def derive_payload(source, bit, source_sha256):
    validate_source(source)
    metadata = derivation_metadata(bit, source_sha256)
    result = copy.deepcopy(source)
    result.update(schema=DERIVED_SCHEMA, weight_bits=bit, weight_format=FORMATS[bit], bn_bank_key=BANK[bit])
    result['mx_layers'] = {name: code_chain(high.MXTensor.from_payload(value),
        source['lower_scale_codes'][name])[bit].payload() for name, value in source['mx_layers'].items()}
    result['derivation'] = metadata
    validate_derived(result)
    return result


def validate_derived(payload):
    require(type(payload) is dict and set(payload) == SOURCE_KEYS | {'derivation'}, 'Onverwachte afgeleide velden.')
    require(payload['schema'] == DERIVED_SCHEMA, 'Geen onafhankelijke E2-schaalafleiding.')
    bit = payload['weight_bits']
    require(type(bit) is int and bit in (6, 4), 'Verkeerde afgeleide precisie.')
    _validate_common(payload, bit)
    d = payload['derivation']
    require(type(d) is dict and exact_metadata(d, derivation_metadata(bit, d.get('source_sha256'))),
            'Afwijkende bitbewerking of schaalbeleid.')


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
    model = base.adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2).to(device))
    expected = set(model.state_dict()) - {n + '.weight' for n in TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected, 'Verborgen master/phi of ontbrekende hulpstaat.')
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        mx = codec(8 if source else bit).MXTensor.from_payload(payload['mx_layers'][name], device)
        if source and bit != 8: mx = code_chain(mx, payload['lower_scale_codes'][name].to(device))[bit]
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
