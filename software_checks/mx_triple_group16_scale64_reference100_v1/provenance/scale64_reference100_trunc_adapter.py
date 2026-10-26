"""One E4M3 source; exponent-MSB truncation with scale x64, then mantissa.

E4 scale bytes remain unchanged; each lower scale is source exponent plus6.
Each precision pass builds its own G(W) graph; the shared cache contains
detached integer codes. Backward is identity to G(W), with no factor64.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from contextlib import contextmanager
import torch
from torch import Tensor
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
SOURCE_SCHEMA = 'mx-triple-scale64-training-source-v1'
DERIVED_SCHEMA = 'mx-triple-scale64-training-derived-v1'
TRAINING_POLICY = 'joint_E4_RNE_E2_exponent_MSB_scale64_then_mantissa_truncation_identity_STE'
SCALE_EXPONENT_OFFSET = 6
SCALE_FACTOR = 64
SOURCE_KEYS = {'schema', 'model', 'expand', 'activation_bits', 'weight_bits', 'weight_format',
               'bn_bank_key', 'supported_weight_modes', 'bank_mapping', 'weight_preprocess',
               'training_policy', 'mx_layers', 'auxiliary_fp32_state', 'provenance'}


def plain_metadata(value):
    if value is None or type(value) in (str, int, float, bool): return True
    if type(value) is list: return all(plain_metadata(v) for v in value)
    if type(value) is dict: return all(type(k) is str and plain_metadata(v) for k, v in value.items())
    return False


def shifted_scale_codes(source: high.MXTensor) -> Tensor:
    """Exact source E8M0 exponent +6; includes zero blocks, never clip/wrap."""
    require(isinstance(source, high.MXTensor), 'E4M3-bron vereist voor schaaloffset.')
    source.validate()
    shifted = source.scale_codes.to(torch.int32) + SCALE_EXPONENT_OFFSET
    require(bool((shifted <= 254).all()),
            'Schaal x64 valt buiten eindig E8M0-bereik; geen clipping of uint8-wrap toegestaan.')
    return shifted.to(torch.uint8)


def code_chain(source: high.MXTensor) -> dict:
    """Exact post-hoc B6/B4: old bitmapping, lower scales source+6."""
    naive = exponent_truncate(source)
    shifted = shifted_scale_codes(source)
    middle = codec(6).MXTensor(naive.codes, shifted, source.shape)
    low = mantissa_truncate(middle)
    require(torch.equal(middle.scale_codes.to(torch.int32), source.scale_codes.to(torch.int32) + 6)
            and torch.equal(middle.scale_codes, low.scale_codes), 'Vaste schaaloffset +6 of mantissaschaal gewijzigd.')
    return {8: source, 6: middle, 4: low}


def encode_chain(effective: Tensor) -> dict:
    return code_chain(high.encode_tensor(effective))


class _CachedCodeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, effective: Tensor, codes: Tensor, scales: Tensor, bit: int) -> Tensor:
        require(effective.dtype == torch.float32, 'FP32 G(W) vereist.')
        # Exact decoded forward, without q + (w - w.detach()) cancellation.
        return codec(bit).MXTensor(codes, scales, tuple(effective.shape)).decode()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient: Tensor):
        return gradient.clone(), None, None, None


class JointMXQuantizer(base.JointMXQuantizer):
    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        self._triplet_active = False
        self._chain = self._source_key = self._versions = None
        self._used_modes = []

    def _assert_cache_unchanged(self):
        require(self._versions == {b: (x.codes._version, x.scale_codes._version)
                                  for b, x in self._chain.items()}, 'Gecachte codes of schalen gewijzigd.')

    def forward(self, weight: Tensor) -> Tensor:
        require(self.abit == self.wbit == 32, 'A32 en W32-voorbewerking moeten behouden blijven.')
        bit = self.weight_mode
        codec(bit)
        effective = effective_reference_weight(weight)
        if self._triplet_active:
            require(bit not in self._used_modes, 'Dubbele modus in dezelfde broncyclus.')
            key = (id(weight), weight._version, weight.device, weight.dtype, tuple(weight.shape))
            if self._chain is None:
                self._chain = encode_chain(effective)
                self._source_key = key
                self._versions = {b: (x.codes._version, x.scale_codes._version)
                                  for b, x in self._chain.items()}
            else:
                require(key == self._source_key, 'FP32-master gewijzigd tussen precisiepasses.')
                self._assert_cache_unchanged()
            encoded = self._chain[bit]
            self._used_modes.append(bit)
        else:
            encoded = encode_chain(effective)[bit]
        return _CachedCodeSTE.apply(effective, encoded.codes, encoded.scale_codes, bit)


def select_mode(model, bit):
    codec(bit)
    require(not hasattr(model, '_frozen_weight_bit') or model._frozen_weight_bit == bit,
            'Bevroren export: reconstrueer de gewenste precisie uit de E4-bron.')
    base.select_mode(model, bit)


def adapt_copy(reference):
    model = base.adapt_copy(reference)
    for name in TARGET_LAYERS:
        model.get_submodule(name).quantize_fn = JointMXQuantizer()
    select_mode(model, 8)
    return model


def cache_is_empty(model):
    return all(not q._triplet_active and q._chain is None and not q._used_modes
               for q in (model.get_submodule(n).quantize_fn for n in TARGET_LAYERS))


@contextmanager
def shared_source_triplets(model):
    """Wrap the unmodified author loss/update loop; one source for 8/6/4.

    Training order is 8,6,4 and test order 4,6,8. Exactly six E4 encodings
    occur per minibatch, and no E2 encoder. All caches clear after the third
    forward, before its backward and before the one Adam update.
    """
    quantizers = [model.get_submodule(n).quantize_fn for n in TARGET_LAYERS]
    require(all(isinstance(q, JointMXQuantizer) for q in quantizers) and cache_is_empty(model),
            'Onverwachte quantizer, oude of geneste broncache.')
    position = [0]
    training = model.training
    order = (8, 6, 4) if training else (4, 6, 8)
    audit = {'source_triplets': 0, 'high_encodings': 0, 'exponent_derivations': 0,
             'mantissa_derivations': 0, 'scale_offsets_plus6': 0,
             'mantissa_scale_copies_unchanged': 0,
             'scale_exponent_offset': SCALE_EXPONENT_OFFSET, 'scale_factor': SCALE_FACTOR,
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
            audit['scale_offsets_plus6'] += 2 * len(TARGET_LAYERS)
            audit['mantissa_scale_copies_unchanged'] += len(TARGET_LAYERS)
    handles = [model.register_forward_pre_hook(pre), model.register_forward_hook(post)]
    try:
        yield audit
        require(position[0] == 0 and cache_is_empty(model), 'Onvolledige broncyclus.')
        audit['source_cache_cleared'] = True
    finally:
        for handle in handles: handle.remove()
        for q in quantizers: q._reset()


def source_payload(model, provenance):
    """Archive only E4 source codes plus shared FP32 exception/BN state."""
    require(cache_is_empty(model), 'Export binnen onvolledige broncyclus.')
    encoded, rows = {}, []
    for name in TARGET_LAYERS:
        effective = effective_reference_weight(model.get_submodule(name).weight.detach())
        chain = encode_chain(effective)
        encoded[name] = chain[8].payload()
        for bit in (8, 6, 4):
            rows.append({'weight_bits': bit, 'weight_format': FORMATS[bit], 'layer': name,
                         'derivation': {8: 'E4_RNE_source', 6: 'E4_exponent_MSB_keep_scale64',
                                        4: 'E4_exponent_MSB_scale64_then_mantissa_shift2'}[bit],
                         'scale_exponent_offset': 0 if bit == 8 else SCALE_EXPONENT_OFFSET,
                         'scale_factor': 1 if bit == 8 else SCALE_FACTOR,
                         **codec(bit).distortion(effective, chain[bit])})
    remove = {n + '.weight' for n in TARGET_LAYERS}
    auxiliary = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in remove}
    payload = {'schema': SOURCE_SCHEMA, 'model': 'svhnq', 'expand': 2, 'activation_bits': 32,
               'weight_bits': 8, 'weight_format': FORMATS[8], 'bn_bank_key': BANK[8],
               'supported_weight_modes': [8, 6, 4], 'bank_mapping': {'8': '8', '6': '32', '4': '4'},
               'weight_preprocess': 'already_G(W); do_not_repeat', 'training_policy': TRAINING_POLICY,
               'mx_layers': encoded, 'auxiliary_fp32_state': auxiliary, 'provenance': copy.deepcopy(provenance)}
    validate_source(payload)
    return payload, rows


def _validate_common(payload, bit):
    codec(bit)
    require(plain_metadata(payload['provenance']), 'Verborgen toestand in provenance.')
    require(payload['model'] == 'svhnq' and payload['expand'] == 2 and payload['activation_bits'] == 32,
            'Model/A32 gewijzigd.')
    require(payload['weight_bits'] == bit and payload['weight_format'] == FORMATS[bit]
            and payload['bn_bank_key'] == BANK[bit], 'Afwijkend formaat of BN-bank.')
    require(payload['supported_weight_modes'] == [8, 6, 4]
            and payload['bank_mapping'] == {'8': '8', '6': '32', '4': '4'}, 'Modi/BN gewijzigd.')
    require(payload['weight_preprocess'] == 'already_G(W); do_not_repeat'
            and payload['training_policy'] == TRAINING_POLICY, 'Afleidingsbeleid gewijzigd.')
    require(type(payload['mx_layers']) is dict and tuple(payload['mx_layers']) == TARGET_LAYERS,
            'Verkeerde MX-laagselectie.')
    for mx in payload['mx_layers'].values(): codec(bit).MXTensor.from_payload(mx)
    aux = payload['auxiliary_fp32_state']
    require(type(aux) is dict and all(type(k) is str and isinstance(v, Tensor) for k, v in aux.items()),
            'Onjuiste auxiliaire modelstaat.')
    require(not ({n + '.weight' for n in TARGET_LAYERS} & set(aux)), 'Verborgen MX-master in export.')
    require(all((v.dtype in (torch.float32, torch.int64)) and bool(torch.isfinite(v).all())
                for v in aux.values()), 'Niet-eindige of onverwachte auxiliaire dtype.')


def validate_source(payload):
    require(type(payload) is dict and set(payload) == SOURCE_KEYS, 'Onverwachte bronvelden.')
    require(payload['schema'] == SOURCE_SCHEMA, 'Geen E4M3-bronexport.')
    _validate_common(payload, 8)


def derivation_metadata(bit, source_sha256):
    require(type(bit) is int and bit in (6, 4), 'Verwacht afgeleide modus6 of4.')
    require(type(source_sha256) is str and len(source_sha256) == 64
            and set(source_sha256) <= set('0123456789abcdef'), 'Ongeldige bron-SHA256.')
    return {'method': 'E4_exponent_MSB_keep_scale64' if bit == 6 else 'E4_exponent_MSB_scale64_then_mantissa_shift2',
            'source_weight_bits': 8, 'target_weight_bits': bit, 'source_sha256': source_sha256,
            'dropped_exponent_bits': 2, 'dropped_mantissa_bits': 0 if bit == 6 else 2,
            'scales_unchanged': False, 'source_scale_exponent_offset': SCALE_EXPONENT_OFFSET,
            'scale_factor': SCALE_FACTOR,
            'scale_policy': 'source_E8M0_exponent_plus6_no_clip_no_wrap',
            'shared_scale_compensation': True, 'learned_shared_scale': False,
            'bias_compensation': False, 'masters_required': False}


def derive_payload(source, bit, source_sha256):
    """Only stored E4 codes/scales/auxiliary state; no master or quantizer."""
    validate_source(source)
    metadata = derivation_metadata(bit, source_sha256)
    result = copy.deepcopy(source)
    result.update(schema=DERIVED_SCHEMA, weight_bits=bit, weight_format=FORMATS[bit], bn_bank_key=BANK[bit])
    result['mx_layers'] = {name: code_chain(high.MXTensor.from_payload(value))[bit].payload()
                           for name, value in source['mx_layers'].items()}
    result['derivation'] = metadata
    validate_derived(result)
    return result


def validate_derived(payload):
    require(type(payload) is dict and set(payload) == SOURCE_KEYS | {'derivation'}, 'Onverwachte afgeleide velden.')
    require(payload['schema'] == DERIVED_SCHEMA, 'Geen afgeleide MX-export.')
    bit = payload['weight_bits']
    require(type(bit) is int and bit in (6, 4), 'Verkeerde afgeleide precisie.')
    _validate_common(payload, bit)
    d = payload['derivation']
    require(type(d) is dict and d == derivation_metadata(bit, d.get('source_sha256')),
            'Afwijkende bitbewerking of schaal-/biascompensatie.')


def export_payload(model, bit, provenance):
    """Synthetic compatibility only. Production derives6/4 from reloaded source."""
    codec(bit)
    source, rows = source_payload(model, provenance)
    return (source if bit == 8 else derive_payload(source, bit, '0' * 64)), [r for r in rows if r['weight_bits'] == bit]


def load_exported_model(models, payload, device, bit=None):
    source = payload.get('schema') == SOURCE_SCHEMA
    if source:
        validate_source(payload)
        bit = 8 if bit is None else bit
        codec(bit)
    else:
        validate_derived(payload)
        bit = payload['weight_bits'] if bit is None else bit
        require(bit == payload['weight_bits'], 'Afgeleide export heeft vaste doelmodus.')
    model = adapt_copy(models.svhnq([1, 2, 4, 8, 32], expand=2).to(device))
    expected = set(model.state_dict()) - {n + '.weight' for n in TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected, 'Verborgen master of ontbrekende hulpstaat.')
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        mx = codec(8 if source else bit).MXTensor.from_payload(payload['mx_layers'][name], device)
        if source and bit != 8: mx = code_chain(mx)[bit]
        require(mx.shape == tuple(layer.weight.shape), 'Onjuiste gewichtsvorm.')
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, base.original_adapter.FrozenMXConv(layer, mx))
    missing, extra = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n + '.decoded_weight' for n in TARGET_LAYERS} and not extra, 'Onvolledige reconstructie.')
    select_mode(model, bit)
    model._frozen_weight_bit = bit
    return model.to(device).eval().requires_grad_(False)


def load_verified_payload(path):
    path = Path(path)
    marker = path.with_suffix(path.suffix + '.sha256')
    require(path.is_file() and marker.is_file(), 'MX-bestand of checksum ontbreekt.')
    with path.open('rb') as handle: digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    require(digest == marker.read_text(encoding='ascii').strip(), 'MX-checksum wijkt af.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(type(payload) is dict, 'Geen MX-payload.')
    if payload.get('schema') == SOURCE_SCHEMA: validate_source(payload)
    else: validate_derived(payload)
    return payload
