"""Exact layout costs for the six MX layers; no torch, data or model loading.

Groups flatten Cin*Kh*Kw separately within each Cout. Every output row is
zero-padded to the next group. One E8M0 byte accompanies each group. Reported
payloads describe a single exported precision, not three simultaneously stored
networks. They exclude FP32 auxiliary state, container metadata and GPU buffers.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


LAYER_SHAPES = {
    'layers.3': (16, 12, 3, 3),
    'layers.6': (16, 16, 3, 3),
    'layers.10': (32, 16, 3, 3),
    'layers.13': (32, 32, 3, 3),
    'layers.16': (32, 32, 3, 3),
    'layers.20': (128, 32, 5, 5),
}
FORMATS = {8: 'E4M3', 6: 'E2M3', 4: 'E2M1'}
SCHEMA = 'mx-group-size-costs-v1'
SCOPE = ('Logical packed codes plus one 8-bit shared scale per group for the six MX layers. '
         'Each precision is a separate possible export. Excludes FP32 exceptions/BatchNorm, '
         'optimizer state, container metadata, GPU buffers, hardware energy and measured execution time.')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    """Allow CSV decimal integers; reject bools, NaN and fractional counts."""
    _require(not isinstance(value, bool), name + ' must be an integer count.')
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be an integer count.') from exc
    _require(math.isfinite(number) and number == result, name + ' must be an integer count.')
    return result


def _shapes(layer_shapes=None):
    shapes = LAYER_SHAPES if layer_shapes is None else layer_shapes
    _require(isinstance(shapes, dict) and bool(shapes), 'A nonempty layer-shape mapping is required.')
    cleaned = {}
    for name, shape in shapes.items():
        _require(type(name) is str and name, 'Each layer must have a name.')
        _require(isinstance(shape, (list, tuple)) and len(shape) >= 2, 'Weight tensor must have an output row dimension.')
        dims = tuple(_integer(x, name + '/shape') for x in shape)
        _require(all(x > 0 for x in dims), 'Tensor dimensions must be positive.')
        cleaned[name] = dims
    return cleaned


def layer_cost(layer, shape, group_size, weight_bits):
    group_size = _integer(group_size, 'group_size')
    weight_bits = _integer(weight_bits, 'weight_bits')
    _require(group_size > 0 and weight_bits in FORMATS, 'Unsupported group size or weight precision.')
    # The active codecs pack complete bytes per group; these costs use the same rule.
    _require(group_size * weight_bits % 8 == 0, 'This packing requires byte-aligned groups.')
    shape = _shapes({layer: shape})[layer]
    rows, row_values = shape[0], math.prod(shape[1:])
    blocks_per_row = (row_values + group_size - 1) // group_size
    blocks = rows * blocks_per_row
    stored = blocks * group_size
    real = math.prod(shape)
    code_bytes = stored * weight_bits // 8
    scale_bytes = blocks
    payload_bytes = code_bytes + scale_bytes
    return {
        'layer': layer, 'shape': list(shape), 'group_size': group_size,
        'weight_bits': weight_bits, 'weight_format': FORMATS[weight_bits],
        'output_rows': rows, 'weights_per_row': row_values,
        'blocks_per_row': blocks_per_row, 'padded_weights_per_row': blocks_per_row * group_size,
        'n_weights': real, 'blocks': blocks, 'stored_weight_slots': stored,
        'padded_slots': stored - real, 'packed_weight_bytes': code_bytes,
        'scale_bytes': scale_bytes, 'packed_weight_and_scale_bytes': payload_bytes,
        'fp32_weight_bytes': real * 4,
        'ideal_scale_bits_per_weight_without_padding': 8 / group_size,
        'effective_bits_per_real_weight': payload_bytes * 8 / real,
    }


def build_cost_report(layer_shapes=None, group_sizes=(16, 32, 64), active_group_size=64):
    """Return reproducible analytical costs; this does not assert any accuracy."""
    shapes = _shapes(layer_shapes)
    sizes = tuple(_integer(x, 'group_size') for x in group_sizes)
    active = _integer(active_group_size, 'active_group_size')
    _require(len(sizes) == len(set(sizes)) and bool(sizes), 'Duplicate or missing group sizes.')
    _require(active in sizes, 'The active group size must be reported.')
    rows = [layer_cost(name, shape, size, bits)
            for size in sizes for name, shape in shapes.items() for bits in FORMATS]
    summary = []
    for size in sizes:
        source = [r for r in rows if r['group_size'] == size and r['weight_bits'] == 8]
        total = {field: sum(r[field] for r in source) for field in
                 ('n_weights', 'blocks', 'stored_weight_slots', 'padded_slots', 'scale_bytes', 'fp32_weight_bytes')}
        total.update(group_size=size, active_group_size=size == active,
                     accuracy_evidence='not asserted by this analytical layout report',
                     ideal_scale_bits_per_weight_without_padding=8 / size)
        for bit, label in FORMATS.items():
            selected = [r for r in rows if r['group_size'] == size and r['weight_bits'] == bit]
            total['packed_' + label.lower() + '_bytes'] = sum(r['packed_weight_and_scale_bytes'] for r in selected)
            total['effective_bits_' + label.lower()] = total['packed_' + label.lower() + '_bytes'] * 8 / total['n_weights']
        summary.append(total)
    return {
        'schema': SCHEMA, 'analytical_layout_only': True, 'scope': SCOPE,
        'group_axis': 'flatten Cin*Kh*Kw within each Cout, zero-pad tail',
        'scale_bits_per_group': 8, 'fixed_lower_scale_factor': 64,
        'source_precision': 'E4M3', 'active_group_size': active,
        'group_sizes': list(sizes), 'layer_shapes': {k: list(v) for k, v in shapes.items()},
        'excluded': ['FP32 exceptions and BatchNorm', 'optimizer', 'file/container metadata',
                     'PyTorch buffers and GPU allocator', 'measured time or hardware energy'],
        'precision_sizes_are_alternative_exports_not_a_sum': True,
        'summary': summary, 'layers': rows,
    }


def write_cost_report(out, layer_shapes=None, group_sizes=(16, 32, 64), active_group_size=64):
    """Write three fresh files and return the report. Never overwrite a result."""
    out = Path(out)
    _require(out.is_dir(), 'Output directory must already exist.')
    report = build_cost_report(layer_shapes, group_sizes, active_group_size)
    names = ('group_size_costs.json', 'group_size_costs_summary.csv', 'group_size_costs_layers.csv')
    _require(all(not (out / name).exists() for name in names), 'Existing group-size cost report must be preserved.')
    with (out / names[0]).open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    for name, rows in ((names[1], report['summary']), (names[2], report['layers'])):
        with (out / name).open('x', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def validate_cost_report(report, expected_layer_shapes=None, expected_group_size=64):
    """Reject changed totals, unsupported shape provenance and mislabeled costs."""
    expected = build_cost_report(expected_layer_shapes, (16, 32, 64), expected_group_size)
    _require(report == expected, 'Saved layout report differs from its recomputed layer geometry or scope.')
    return {'status': 'PASS', 'verified': True, 'active_group_size': expected_group_size,
            'six_layer_geometry': len(expected['layer_shapes']) == 6}


def validate_export_layers(rows, weight_bits, expected_group_size=64, layer_shapes=None):
    """Check saved export diagnostics against the actual intended row grouping.

    Accept CSV dictionaries as well as in-memory scalar rows. This verifies
    reported layout counts; it does not replace codec-byte roundtrip checks.
    """
    shapes = _shapes(layer_shapes)
    bit = _integer(weight_bits, 'weight_bits')
    _require(isinstance(rows, list) and len(rows) == len(shapes), 'Export must contain exactly the expected MX layers.')
    _require(all(isinstance(r, dict) for r in rows), 'Export rows must be dictionaries.')
    _require(len({r.get('layer') for r in rows}) == len(rows)
             and {r.get('layer') for r in rows} == set(shapes), 'Export layer names are missing, duplicated or unexpected.')
    totals = {key: 0 for key in ('n_weights', 'blocks', 'padded_slots', 'packed_weight_and_scale_bytes')}
    for row in rows:
        name = row['layer']
        expected = layer_cost(name, shapes[name], expected_group_size, bit)
        _require(_integer(row.get('weight_bits'), name + '/weight_bits') == bit, 'Wrong export precision: ' + name)
        if 'block_size' in row:
            _require(_integer(row['block_size'], name + '/block_size') == expected_group_size, 'Wrong export group size: ' + name)
        if 'mx_block_size' in row:
            _require(_integer(row['mx_block_size'], name + '/mx_block_size') == expected_group_size, 'Wrong export group size: ' + name)
        for key in totals:
            value = _integer(row.get(key), name + '/' + key)
            _require(value == expected[key], f'{name}/{key}: {value} differs from B={expected_group_size} expectation {expected[key]}.')
            totals[key] += value
    return {'status': 'PASS', 'reported_layout_verified': True, 'group_size': expected_group_size,
            'weight_bits': bit, 'layers': len(shapes), **totals,
            'scope': 'Reported export-layer counts, not a byte-level revalidation or hardware measurement.'}
