"""One E2M3 source per layer and paired minibatch; E2M1 = source codes >> 2.

No low-precision re-encoding or scale selection. The cache contains detached
integer codes, not an autograd graph. Each pass has its own G(W) graph so the
original sequential backwards remain valid. STE is identity w.r.t. G(W), not
1/4: the shift operates on a code, not on the numerical value.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from contextlib import contextmanager
import torch
from torch import nn, Tensor
import dual_adapter as base
import mx_format as high
import mx_e2m1 as low

TARGET_LAYERS, BN_LAYERS = base.TARGET_LAYERS, base.BN_LAYERS
MODES, BANK, FORMATS = base.MODES, base.BANK, base.FORMATS
require = high.require
codec = base.codec
inactive_bn = base.inactive_bn
effective_reference_weight = base.effective_reference_weight
WeightModeBatchNorm = base.WeightModeBatchNorm
SOURCE_SCHEMA = 'mx-truncation-aware-source-v1'
DERIVED_SCHEMA = 'mx-truncation-aware-low-derived-v1'
SOURCE_KEYS = {'schema','model','expand','activation_bits','weight_bits','weight_format',
               'bn_bank_key','supported_weight_modes','bank_mapping','weight_preprocess',
               'training_policy','mx_layers','auxiliary_fp32_state','provenance'}


def plain_metadata(value):
    if value is None or type(value) in (str,int,float,bool): return True
    if type(value) is list: return all(plain_metadata(v) for v in value)
    if type(value) is dict: return all(type(k) is str and plain_metadata(v) for k,v in value.items())
    return False


def truncate_codes(source: high.MXTensor) -> low.MXTensor:
    """Exactly the C path of the validated mantissa experiment."""
    source.validate()
    codes = (source.codes.to(torch.int64) >> 2).to(torch.uint8)
    result = low.MXTensor(codes, source.scale_codes.clone(), source.shape)
    result.validate()
    require(torch.equal(source.codes.long() >> 5, codes.long() >> 3), 'Teken gewijzigd.')
    require(torch.equal((source.codes.long() >> 3) & 3, (codes.long() >> 1) & 3), 'Exponent gewijzigd.')
    require(torch.equal(source.scale_codes, result.scale_codes), 'Schaal gewijzigd.')
    return result


def encode_pair(effective: Tensor) -> dict:
    h = high.encode_tensor(effective)
    return {6: h, 4: truncate_codes(h)}


class _CachedCodeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, effective: Tensor, codes: Tensor, scales: Tensor, bit: int) -> Tensor:
        require(effective.dtype == torch.float32, 'FP32 G(W) vereist.')
        return codec(bit).MXTensor(codes, scales, tuple(effective.shape)).decode()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad: Tensor):
        return grad.clone(), None, None, None


class DualMXQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_mode = 6
        self.abit = self.wbit = 32
        self._pair_active = False
        self._source = None
        self._source_key = None
        self._used_modes = []
        self._source_version = None

    def _reset(self):
        self._pair_active = False
        self._source = self._source_key = self._source_version = None
        self._used_modes = []

    def forward(self, weight: Tensor) -> Tensor:
        require(self.abit == self.wbit == 32, 'A32 en W32-voorbewerking behouden.')
        bit = self.weight_mode
        codec(bit)
        effective = effective_reference_weight(weight)
        if self._pair_active:
            require(bit not in self._used_modes, 'Dubbele modus binnen dezelfde broncyclus.')
            # Tensor._version is used only as an audited mutation guard, never as
            # numerical input. Adam is forbidden between the two forwards.
            key = (id(weight), weight._version, weight.device, weight.dtype, tuple(weight.shape))
            if self._source is None:
                self._source = high.encode_tensor(effective)
                self._source_key = key
                self._source_version = (self._source.codes._version, self._source.scale_codes._version)
            else:
                require(self._source_key == key, 'Master gewijzigd tussen beide precisiepasses.')
                require(self._source_version == (self._source.codes._version, self._source.scale_codes._version),
                        'Gecachte hoge codes of schalen gewijzigd.')
            h = self._source
            self._used_modes.append(bit)
        else:
            # Standalone diagnostic/inference call: still high encoding first.
            h = high.encode_tensor(effective)
        encoded = h if bit == 6 else truncate_codes(h)
        return _CachedCodeSTE.apply(effective, encoded.codes, encoded.scale_codes, bit)


def select_mode(model, bit):
    codec(bit)
    require(not hasattr(model, '_frozen_weight_bit') or model._frozen_weight_bit == bit,
            'Bevroren emulator: reconstrueer de gewenste modus opnieuw uit dezelfde hoge bron.')
    for m in model.modules():
        if isinstance(m, (DualMXQuantizer, WeightModeBatchNorm)):
            m.weight_mode = bit
    model.weight_mode = bit


def adapt_copy(reference):
    # Reuse exactly the established BN mapping and A32 initialization.
    model = base.adapt_copy(reference)
    for name in TARGET_LAYERS:
        model.get_submodule(name).quantize_fn = DualMXQuantizer()
    select_mode(model, 6)
    return model


@contextmanager
def shared_source_pairs(model):
    """Wrap an unchanged author forward loop. Training 6->4; evaluation 4->6.

    Exactly six high encodings per minibatch for six MX layers. Codes remain
    unchanged across both passes and are cleared before the optimizer update.
    Nothing in the cache is registered as a parameter or checkpoint buffer.
    """
    quantizers = [model.get_submodule(n).quantize_fn for n in TARGET_LAYERS]
    require(all(isinstance(q, DualMXQuantizer) and not q._pair_active and q._source is None
                for q in quantizers), 'Oude of geneste broncache.')
    position = [0]
    training = model.training
    order = (6, 4) if training else (4, 6)
    audit = {'source_pairs': 0, 'high_encodings': 0, 'low_from_same_source': 0,
             'scale_copies_unchanged': 0, 'source_cache_cleared': False,
             'no_direct_low_quantizer': True}

    def pre(module, inputs):
        require(module.training == training, 'Train/eval gewijzigd binnen broncyclus.')
        require(module.weight_mode == order[position[0]], 'Verkeerde bronpassvolgorde.')
        if position[0] == 0:
            for q in quantizers:
                require(q._source is None and not q._pair_active, 'Onverwacht oude bron.')
                q._pair_active = True

    def post(module, inputs, output):
        if position[0] == 0:
            require(all(q._used_modes == [order[0]] and q._source is not None for q in quantizers),
                    'Niet alle hoge bronnen zijn gevormd.')
            position[0] = 1
        else:
            for q in quantizers:
                require(q._used_modes == list(order), 'Modi gebruiken niet dezelfde laagbron.')
                require(q._source_version == (q._source.codes._version, q._source.scale_codes._version),
                        'Broncodes werden overschreven.')
                q._reset()
            position[0] = 0
            audit['source_pairs'] += 1
            audit['high_encodings'] += 6
            audit['low_from_same_source'] += 6
            audit['scale_copies_unchanged'] += 6
    handles = [model.register_forward_pre_hook(pre), model.register_forward_hook(post)]
    try:
        yield audit
        require(position[0] == 0 and all(q._source is None for q in quantizers), 'Onvolledig paar.')
        audit['source_cache_cleared'] = True
    finally:
        for h in handles:
            h.remove()
        for q in quantizers:
            q._reset()


def source_payload(model, provenance):
    encoded, rows = {}, []
    for name in TARGET_LAYERS:
        effective = effective_reference_weight(model.get_submodule(name).weight.detach())
        h = high.encode_tensor(effective)
        encoded[name] = h.payload()
        for bit, value in ((6, h), (4, truncate_codes(h))):
            rows.append({'weight_bits': bit, 'weight_format': FORMATS[bit], 'layer': name,
                         'derivation': 'high_RNE' if bit == 6 else 'high_codes_shift_2',
                         **codec(bit).distortion(effective, value)})
    remove = {n+'.weight' for n in TARGET_LAYERS}
    auxiliary = {k: v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in remove}
    p = {'schema': SOURCE_SCHEMA, 'model': 'svhnq', 'expand': 2, 'activation_bits': 32,
         'weight_bits': 6, 'weight_format': FORMATS[6], 'bn_bank_key': '32',
         'supported_weight_modes': [6,4], 'bank_mapping': {'6':'32','4':'4'},
         'weight_preprocess': 'already_G(W); do_not_repeat',
         'training_policy': 'joint_high_RNE_low_code_truncation_identity_STE',
         'mx_layers': encoded, 'auxiliary_fp32_state': auxiliary, 'provenance': provenance}
    return p, rows


def derive_payload(source: dict, source_sha256: str):
    validate_source(source)
    require(len(source_sha256) == 64 and set(source_sha256) <= set('0123456789abcdef'), 'Ongeldige bronhash.')
    out = copy.deepcopy(source)
    out.update(schema=DERIVED_SCHEMA, weight_bits=4, weight_format=FORMATS[4], bn_bank_key='4')
    out['mx_layers'] = {n: truncate_codes(high.MXTensor.from_payload(v)).payload()
                       for n,v in source['mx_layers'].items()}
    out['derivation'] = {'method':'source_codes_shift_right_2', 'source_sha256':source_sha256,
                         'scales_unchanged':True, 'masters_required':False}
    return out


def validate_source(p):
    require(type(p) is dict and set(p) == SOURCE_KEYS, 'Onverwachte bronvelden.')
    require(plain_metadata(p['provenance']), 'Verborgen toestand in bronmetadata.')
    require(p.get('schema') == SOURCE_SCHEMA and p.get('weight_bits') == 6, 'Geen hoge MX-bron.')
    require(p.get('weight_format') == FORMATS[6] and p.get('bn_bank_key') == '32', 'Verkeerd bronformaat.')
    require(p.get('supported_weight_modes') == [6,4] and p.get('bank_mapping') == {'6':'32','4':'4'},
            'Modi/BN onjuist.')
    require(p.get('model') == 'svhnq' and p.get('expand') == 2 and p.get('activation_bits') == 32,
            'Model/A32 gewijzigd.')
    require(p.get('weight_preprocess') == 'already_G(W); do_not_repeat' and
            p.get('training_policy') == 'joint_high_RNE_low_code_truncation_identity_STE', 'Methode gewijzigd.')
    require(tuple(p['mx_layers']) == TARGET_LAYERS, 'Verkeerde lagen.')
    for v in p['mx_layers'].values(): high.MXTensor.from_payload(v)


def export_payload(model, bit, provenance):
    """Compatibility helper for synthetic tests. Production derives low from disk."""
    h, rows = source_payload(model, provenance)
    return (h if bit == 6 else derive_payload(h, '0'*64)), [r for r in rows if r['weight_bits'] == bit]


def load_exported_model(models, payload, device, bit=None):
    is_high = payload.get('schema') == SOURCE_SCHEMA
    if is_high:
        validate_source(payload)
        bit = 6 if bit is None else bit
        codec(bit)
    else:
        require(type(payload) is dict and set(payload) == SOURCE_KEYS | {'derivation'}, 'Onverwachte exportvelden.')
        require(payload.get('schema') == DERIVED_SCHEMA, 'Onbekende export.')
        require(plain_metadata(payload['provenance']), 'Verborgen toestand in metadata.')
        require(payload.get('weight_preprocess') == 'already_G(W); do_not_repeat' and
                payload.get('training_policy') == 'joint_high_RNE_low_code_truncation_identity_STE' and
                payload.get('bank_mapping') == {'6':'32','4':'4'} and
                payload.get('supported_weight_modes') == [6,4], 'Afwijkend afleidingsbeleid.')
        bit = 4 if bit is None else bit
        require(bit == 4 and payload.get('weight_bits') == 4 and payload.get('weight_format') == FORMATS[4]
                and payload.get('bn_bank_key') == '4', 'Afwijkend laag formaat.')
        require(payload.get('activation_bits') == 32 and payload.get('model') == 'svhnq'
                and payload.get('expand') == 2, 'Afwijkend model.')
        require(payload['derivation']['method'] == 'source_codes_shift_right_2'
                and payload['derivation']['scales_unchanged'] is True
                and payload['derivation']['masters_required'] is False, 'Afwijkende afleiding.')
    model = adapt_copy(models.svhnq([1,2,4,8,32], expand=2))
    expected = set(model.state_dict()) - {n+'.weight' for n in TARGET_LAYERS}
    require(set(payload['auxiliary_fp32_state']) == expected, 'Verborgen master of ontbrekende hulpstaat.')
    require(tuple(payload['mx_layers']) == TARGET_LAYERS, 'Onjuiste laagselectie.')
    for name in TARGET_LAYERS:
        layer = model.get_submodule(name)
        mx = (high if is_high else low).MXTensor.from_payload(payload['mx_layers'][name])
        if is_high and bit == 4: mx = truncate_codes(mx)
        require(mx.shape == tuple(layer.weight.shape), 'Onjuiste gewichtsvorm.')
        parent, child = name.rsplit('.',1)
        setattr(model.get_submodule(parent), child, base.original_adapter.FrozenMXConv(layer, mx))
    missing, extra = model.load_state_dict(payload['auxiliary_fp32_state'], strict=False)
    require(set(missing) == {n+'.decoded_weight' for n in TARGET_LAYERS} and not extra, 'Onvolledige reconstructie.')
    select_mode(model, bit)
    model._frozen_weight_bit = bit
    return model.to(device).eval().requires_grad_(False)


def load_verified_payload(path):
    """Hash-verified MX payload, distinct from the FP32 resume-checkpoint schema."""
    path = Path(path)
    marker = path.with_suffix(path.suffix + '.sha256')
    require(path.is_file() and marker.is_file(), 'MX-bestand of checksum ontbreekt.')
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    require(digest == marker.read_text(encoding='ascii').strip(), 'MX-checksum wijkt af.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(isinstance(payload, dict), 'Geen MX-payload.')
    return payload
