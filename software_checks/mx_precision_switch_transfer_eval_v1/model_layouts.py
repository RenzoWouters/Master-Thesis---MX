"""Explicit prospective inference layouts, not PyTorch tensor/DRAM measurements.

Only immutable inference parameters are represented. FP32 master/optimizer
states and inactive BN banks remain in the original checkpoint, not this
operational image. Every mode retains its own active BN values. Direct E2M3
and E2M1 share one scale array: their existing codecs use identical groups,
FP32 preprocessing, emax=2, zero and clipping policies. Their codes remain
independent. The GitHub low modes share one FP32 E per convolution.
"""
from __future__ import annotations

import math

LAYER_SHAPES = [
    ('layers.3', (16, 12, 3, 3)), ('layers.6', (16, 16, 3, 3)),
    ('layers.10', (32, 16, 3, 3)), ('layers.13', (32, 32, 3, 3)),
    ('layers.16', (32, 32, 3, 3)), ('layers.20', (128, 32, 5, 5)),
]
MODELS = [
    dict(id='github5', label='GitHub 5 precisies', modes=[32,8,4,2,1], group_size=None,
         family='github', activation_policy='paired', route='direct'),
    dict(id='dual_direct', label='2 MX direct · B32', modes=[6,4], group_size=32,
         family='mx', activation_policy='a32', route='direct'),
    dict(id='dual_mantissa', label='2 MX mantissatruncatie · B32', modes=[6,4], group_size=32,
         family='mx', activation_policy='a32', route='mantissa'),
    dict(id='triple_direct', label='3 MX direct · B32', modes=[8,6,4], group_size=32,
         family='mx', activation_policy='a32', route='direct'),
    dict(id='triple_scale64_b32', label='3 MX truncatie ×64 · B32', modes=[8,6,4], group_size=32,
         family='mx', activation_policy='a32', route='exponent_mantissa'),
    dict(id='triple_scale64_b64', label='3 MX truncatie ×64 · B64', modes=[8,6,4], group_size=64,
         family='mx', activation_policy='a32', route='exponent_mantissa'),
]
MODEL_BY_ID = {m['id']: m for m in MODELS}


def variants(model_id):
    return ['direct'] if MODEL_BY_ID[model_id]['route'] == 'direct' else ['selective', 'full_source']


def build_layout(model_id, variant=None):
    """Each region is one dense contiguous field array for a whole layer.

    The simulator aligns regions to its transfer size. Such alignment is
    counted separately from useful bytes. Headers/descriptors are excluded
    (fixed known CNN and statically configured addresses).
    """
    model = dict(MODEL_BY_ID[model_id])
    variant = variant or variants(model_id)[0]
    if variant not in variants(model_id):
        raise ValueError(f'Unsupported layout {variant} for {model_id}')
    regions, requests, layer_rows = [], {str(m): [] for m in model['modes']}, []

    def region(rid, size, kind, layer, **extra):
        if any(r['id'] == rid for r in regions):
            raise ValueError('Duplicate region: ' + rid)
        regions.append(dict(id=rid, size_bytes=int(size), kind=kind, layer=layer, **extra))
        return rid

    def use(mode, rid, layer):
        requests[str(mode)].append(dict(region=rid, layer=layer))

    first = region('shared.first_fp32', (12*3*5*5+12)*4, 'aux', 'layers.0')
    for mode in model['modes']:
        use(mode, first, 'layers.0')

    for index, (layer, shape) in enumerate(LAYER_SHAPES):
        cout, k = shape[0], math.prod(shape[1:])
        n = cout*k
        b = model['group_size']
        groups = cout*math.ceil(k/b) if b else 0
        slots = groups*b if b else n
        layer_rows.append(dict(layer=layer, shape=list(shape), n_weights=n,
                               groups=groups, padded_slots=slots-n, stored_slots=slots))
        code = {}
        scales = {}
        if model['family'] == 'github':
            for mode in model['modes']:
                code[mode] = [region(f'{layer}.w{mode}', math.ceil(n*mode/8), 'codes', layer)]
            scalar = region(f'{layer}.E_fp32', 4, 'scales', layer)
            scales = {mode: ([] if mode == 32 else [scalar]) for mode in model['modes']}
        elif variant == 'direct':
            for mode in model['modes']:
                code[mode] = [region(f'{layer}.w{mode}', math.ceil(slots*mode/8), 'codes', layer)]
            e2 = region(f'{layer}.e2_shared_scales', groups, 'scales', layer)
            scales = {mode: [e2] for mode in model['modes'] if mode in (4,6)}
            if 8 in model['modes']:
                scales[8] = [region(f'{layer}.e4_scales', groups, 'scales', layer)]
        else:
            high = max(model['modes'])
            if variant == 'full_source':
                src = region(f'{layer}.source{high}', slots*high//8, 'codes', layer)
                code = {mode: [src] for mode in model['modes']}
            else:
                base = region(f'{layer}.base4', slots*4//8, 'codes', layer)
                mant = region(f'{layer}.extra_mantissa2', slots*2//8, 'codes', layer)
                code = {4:[base], 6:[base,mant]}
                if high == 8:
                    exp = region(f'{layer}.extra_exponent2', slots*2//8, 'codes', layer)
                    code[8] = [base,mant,exp]
            scale = region(f'{layer}.source_scales', groups, 'scales', layer)
            scales = {mode:[scale] for mode in model['modes']}
        for mode in model['modes']:
            for rid in code[mode] + scales[mode]:
                use(mode, rid, layer)
        # Running counters are training bookkeeping, not inference operands.
        if index < 5:
            for mode in model['modes']:
                bn = region(f'{layer}.bn_w{mode}', cout*4*4, 'aux', layer)
                use(mode, bn, layer)

    last = region('shared.fc_fp32', (10*128+10)*4, 'aux', 'fc')
    for mode in model['modes']:
        use(mode, last, 'fc')
    by_id = {r['id']:r for r in regions}
    mode_payloads = {}
    for mode in model['modes']:
        totals = {kind:0 for kind in ('codes','scales','aux')}
        for rid in dict.fromkeys(q['region'] for q in requests[str(mode)]):
            r=by_id[rid]; totals[r['kind']] += r['size_bytes']
        mode_payloads[str(mode)] = dict(totals, total_bytes=sum(totals.values()))
    storage = {kind:sum(r['size_bytes'] for r in regions if r['kind']==kind) for kind in ('codes','scales','aux')}
    return dict(model=model, variant=variant, regions=regions, requests_by_mode=requests,
                layer_rows=layer_rows, mode_payloads=mode_payloads,
                storage=dict(storage, total_bytes=sum(storage.values())),
                layout_status='proposed_packed_inference_layout',
                scale_sharing='direct_E2M3_E2M1_common_scale; source_scale_plus6_computed_for_triple_truncation',
                metadata_policy='fixed geometry/address descriptors omitted equally; no file/container overhead',
                original_checkpoint_aux_bytes=19248,
                original_checkpoint_aux_note='All five BN banks and counters in original checkpoint; image contains only supported active BN banks and no counters.')


def pack_unsigned(values, bits):
    """Little-endian dense bit packing; used only for exhaustive route checks."""
    if not 1 <= bits <= 8: raise ValueError('bits')
    out=bytearray(); acc=0; count=0
    for v in values:
        if not isinstance(v,int) or not 0<=v<(1<<bits): raise ValueError('code')
        acc |= v << count; count += bits
        while count >= 8:
            out.append(acc & 255); acc >>= 8; count -= 8
    if count: out.append(acc)
    return bytes(out)


def unpack_unsigned(data, bits, n):
    if len(data) != math.ceil(n*bits/8): raise ValueError('length')
    out=[]; acc=0; count=0; it=iter(data)
    for _ in range(n):
        while count<bits:
            acc |= next(it) << count; count += 8
        out.append(acc & ((1<<bits)-1)); acc >>= bits; count -= bits
    return out


def route_self_check():
    """Check all source codes against the established bit formulas, no FP32 recoding."""
    n=0
    for source in range(256):
        if source in (127,255): continue  # E4M3 NaNs, forbidden by original codec.
        old6=((source>>7)<<5) | (((source>>3 & 15)>>2)<<3) | (source & 7)
        old4=old6>>2
        base=((source>>7)<<3) | (((source>>5)&3)<<1) | ((source>>2)&1)
        mant=source&3; exp=(source>>3)&3
        restored=((base>>3)<<7) | (((base>>1)&3)<<5) | (exp<<3) | ((base&1)<<2) | mant
        if (base, (base<<2)|mant, restored) != (old4, old6, source):
            raise AssertionError('E4 component reconstruction failed')
        n+=1
    for source in range(64):
        base, mant=source>>2, source&3
        if (base<<2)|mant != source: raise AssertionError('E2 component reconstruction failed')
    for bits in (1,2,4,6,8):
        for length in (0,1,7,31,32,63,64,129):
            values=[(i*37+3)% (1<<bits) for i in range(length)]
            if unpack_unsigned(pack_unsigned(values,bits),bits,length) != values:
                raise AssertionError('Packed code round-trip failed')
    for scale_code in range(249):
        # Every allowable source E8M0 code for the existing exact +6 policy.
        if scale_code+6 > 254: raise AssertionError('scale overflow')
    return dict(status='PASS', finite_e4_codes=n, e2m3_codes=64,
                checked='component reconstruction, dense packing, existing truncation formulas',
                limitation='universal code mapping check; numerical checkpoint tensors are not deserialized')
