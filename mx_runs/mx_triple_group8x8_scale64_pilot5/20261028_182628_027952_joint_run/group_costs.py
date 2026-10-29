"""Exact 1x32, 1x64 and 8x8 costs; no torch, data or model loading.

The matrix is Cout by flattened Cin*Kh*Kw. Row groups pad each output row;
8x8 groups pad both matrix axes to a multiple of eight. One E8M0 byte accompanies each group. Reported
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
SCHEMA = 'mx-group-layout-costs-v2'
LAYOUTS = {'row32': (32, 'row', '1×32'), 'row64': (64, 'row', '1×64'),
           'tile8x8': (64, 'tile8x8', '8×8')}
TILE_AXIS = '8 Cout rows x 8 flattened Cin*Kh*Kw columns; zero-pad both matrix tails'
ROW_AXIS = 'flatten Cin*Kh*Kw within each Cout, zero-pad tail'
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


def layer_cost(layer, shape, group_size, weight_bits, layout='row'):
    group_size = _integer(group_size, 'group_size')
    weight_bits = _integer(weight_bits, 'weight_bits')
    _require(group_size > 0 and weight_bits in FORMATS, 'Unsupported group size or weight precision.')
    # The active codecs pack complete bytes per group; these costs use the same rule.
    _require(group_size * weight_bits % 8 == 0, 'This packing requires byte-aligned groups.')
    shape = _shapes({layer: shape})[layer]
    rows, row_values = shape[0], math.prod(shape[1:])
    _require(layout in ('row', 'tile8x8'), 'Unsupported group layout.')
    if layout == 'tile8x8':
        _require(group_size == 64, '8x8 tiles contain exactly 64 weights.')
        tile_rows, blocks_per_row = (rows + 7) // 8, (row_values + 7) // 8
        padded_rows, padded_cols = tile_rows * 8, blocks_per_row * 8
        blocks = tile_rows * blocks_per_row
        layout_id, label = 'tile8x8', '8×8'
    else:
        tile_rows, blocks_per_row = rows, (row_values + group_size - 1) // group_size
        padded_rows, padded_cols = rows, blocks_per_row * group_size
        blocks = rows * blocks_per_row
        layout_id, label = 'row' + str(group_size), '1×' + str(group_size)
    stored = blocks * group_size
    real = math.prod(shape)
    code_bytes = stored * weight_bits // 8
    scale_bytes = blocks
    payload_bytes = code_bytes + scale_bytes
    return {
        'layer': layer, 'shape': list(shape), 'group_size': group_size,
        'layout_id': layout_id, 'layout_label': label,
        'group_shape': [8, 8] if layout == 'tile8x8' else [1, group_size],
        'group_axis': TILE_AXIS if layout == 'tile8x8' else ROW_AXIS,
        'weight_bits': weight_bits, 'weight_format': FORMATS[weight_bits],
        'output_rows': rows, 'weights_per_row': row_values,
        'block_rows': tile_rows, 'blocks_per_row': blocks_per_row,
        'padded_output_rows': padded_rows, 'padded_weights_per_row': padded_cols,
        'n_weights': real, 'blocks': blocks, 'stored_weight_slots': stored,
        'padded_slots': stored - real, 'packed_weight_bytes': code_bytes,
        'scale_bytes': scale_bytes, 'packed_weight_and_scale_bytes': payload_bytes,
        'fp32_weight_bytes': real * 4,
        'ideal_scale_bits_per_weight_without_padding': 8 / group_size,
        'effective_bits_per_real_weight': payload_bytes * 8 / real,
    }


def build_cost_report(layer_shapes=None, group_sizes=(32, 64), active_group_size=64, active_layout='tile8x8'):
    """Return reproducible analytical costs; this does not assert any accuracy."""
    shapes = _shapes(layer_shapes)
    sizes = tuple(_integer(x, 'group_size') for x in group_sizes)
    active = _integer(active_group_size, 'active_group_size')
    _require(len(sizes) == len(set(sizes)) and bool(sizes), 'Duplicate or missing group sizes.')
    _require(sizes == (32, 64), 'This comparison fixes row32, row64 and tile8x8.')
    _require(active == 64 and active_layout == 'tile8x8', 'This pilot uses the 8x8 layout.')
    rows = [layer_cost(name, shape, size, bits, layout)
            for size, layout, _ in LAYOUTS.values() for name, shape in shapes.items() for bits in FORMATS]
    summary = []
    for layout_id, (size, layout, label) in LAYOUTS.items():
        source = [r for r in rows if r['layout_id'] == layout_id and r['weight_bits'] == 8]
        total = {field: sum(r[field] for r in source) for field in
                 ('n_weights', 'blocks', 'stored_weight_slots', 'padded_slots', 'scale_bytes', 'fp32_weight_bytes')}
        total.update(group_size=size, layout_id=layout_id, layout_label=label,
                     active_group_size=layout_id == active_layout, active_layout=layout_id == active_layout,
                     accuracy_evidence='not asserted by this analytical layout report',
                     ideal_scale_bits_per_weight_without_padding=8 / size)
        for bit, label in FORMATS.items():
            selected = [r for r in rows if r['layout_id'] == layout_id and r['weight_bits'] == bit]
            total['packed_' + label.lower() + '_bytes'] = sum(r['packed_weight_and_scale_bytes'] for r in selected)
            total['effective_bits_' + label.lower()] = total['packed_' + label.lower() + '_bytes'] * 8 / total['n_weights']
        summary.append(total)
    return {
        'schema': SCHEMA, 'analytical_layout_only': True, 'scope': SCOPE,
        'group_axis': TILE_AXIS,
        'scale_bits_per_group': 8, 'fixed_lower_scale_factor': 64,
        'source_precision': 'E4M3', 'active_group_size': active, 'active_layout': active_layout,
        'group_sizes': list(sizes), 'group_layouts': list(LAYOUTS),
        'layer_shapes': {k: list(v) for k, v in shapes.items()},
        'excluded': ['FP32 exceptions and BatchNorm', 'optimizer', 'file/container metadata',
                     'PyTorch buffers and GPU allocator', 'measured time or hardware energy'],
        'precision_sizes_are_alternative_exports_not_a_sum': True,
        'summary': summary, 'layers': rows,
    }


def write_cost_report(out, layer_shapes=None, group_sizes=(32, 64), active_group_size=64, active_layout='tile8x8'):
    """Write three fresh files and return the report. Never overwrite a result."""
    out = Path(out)
    _require(out.is_dir(), 'Output directory must already exist.')
    report = build_cost_report(layer_shapes, group_sizes, active_group_size, active_layout)
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


def validate_cost_report(report, expected_layer_shapes=None, expected_group_size=64, expected_layout='tile8x8'):
    """Reject changed totals, unsupported shape provenance and mislabeled costs."""
    expected = build_cost_report(expected_layer_shapes, (32, 64), expected_group_size, expected_layout)
    _require(report == expected, 'Saved layout report differs from its recomputed layer geometry or scope.')
    return {'status': 'PASS', 'verified': True, 'active_group_size': expected_group_size,
            'active_layout': expected_layout,
            'six_layer_geometry': len(expected['layer_shapes']) == 6}


def validate_export_layers(rows, weight_bits, expected_group_size=64, layer_shapes=None, expected_layout='tile8x8'):
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
        expected = layer_cost(name, shapes[name], expected_group_size, bit, expected_layout)
        group_shape = row.get('group_shape')
        if isinstance(group_shape, str):
            try:
                group_shape = json.loads(group_shape)
            except (TypeError, ValueError) as exc:
                raise ValueError(name + '/group_shape: invalid CSV list.') from exc
        _require(isinstance(group_shape, (list, tuple)) and len(group_shape) == 2
                 and all(type(value) is int for value in group_shape)
                 and list(group_shape) == expected['group_shape'],
                 name + '/group_shape: missing or wrong group geometry.')
        _require(row.get('group_axis') == expected['group_axis'],
                 name + '/group_axis: missing or wrong grouping axes.')
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
            'layout': expected_layout,
            'weight_bits': bit, 'layers': len(shapes), **totals,
            'scope': 'Reported export geometry and counts, not a byte-level revalidation or hardware measurement.'}
