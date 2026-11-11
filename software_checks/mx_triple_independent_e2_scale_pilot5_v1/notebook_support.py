"""Read-only notebook support for the independent absolute E2-scale pilot.

Only JSON/CSV evidence is read. No torch, checkpoint, model or dataset is loaded.
The optional save_figures function creates a fresh directory and never rewrites data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEMA = 'mx-triple-independent-e2-scale-pilot5-report-v1'
RUN_STEM = 'mx_triple_independent_e2_scale_pilot5'
PACKAGE = RUN_STEM + '_v1'
BITS = (8, 6, 4)
FORMATS = {8: 'MXFP8_E4M3', 6: 'MXFP6_E2M3', 4: 'MXFP4_E2M1'}
LABELS = {8: 'E4M3', 6: 'E2M3', 4: 'E2M1'}
BANKS = {8: '8', 6: '32', 4: '4'}
COLORS = {8: '#245A81', 6: '#33846D', 4: '#C98220'}
LAYER_SHAPES = {'layers.3': (16, 4), 'layers.6': (16, 5), 'layers.10': (32, 5),
                'layers.13': (32, 9), 'layers.16': (32, 9), 'layers.20': (128, 25)}
LAYER_COLORS = ['#245A81', '#33846D', '#C98220', '#8657A6', '#B44040', '#5D8790']
TOTAL_BLOCKS = sum(math.prod(v) for v in LAYER_SHAPES.values())
REFERENCE_MARKERS = {
    'relative_scale10_reference': 'e2df45acecccb8e56115b4d01f90b11a5f43bb4c9bfa0291a4000401046f866e',
    'fixed_scale64_reference': 'a323762894db83b15af10e79a8aff0128c50f7946805a04517e17a9f627377db',
}
VARIANT_LABELS = {'independent': 'Onafhankelijke E2-schaal',
                  'relative_scale10_reference': '10 relatieve keuzes · start ×64',
                  'fixed_scale64_reference': 'Vaste verhouding ×64'}
VARIANT_STYLES = {'independent': ('-', 'o', 1.0),
                  'relative_scale10_reference': ('--', 's', .78),
                  'fixed_scale64_reference': (':', '^', .75)}
STYLE = {'axes.spines.top': False, 'axes.spines.right': False,
         'axes.titleweight': 'bold', 'font.size': 10, 'figure.facecolor': 'white',
         'axes.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .16,
         'savefig.facecolor': 'white'}


@dataclass
class RunAnalysis:
    path: Path
    report: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    scale_summary: dict = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    times: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected: pd.DataFrame = field(default_factory=pd.DataFrame)
    scale_blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    scale_epochs: pd.DataFrame = field(default_factory=pd.DataFrame)
    references: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    input_files: list = field(default_factory=list)
    metrics_safe: bool = True
    selection_safe: bool = True
    scales_safe: bool = True
    reference_safe: bool = True

    @property
    def config(self):
        return self.report.get('config', {})

    @property
    def synthetic(self):
        return (self.report.get('synthetic_fixture_only') is True
                or self.report.get('synthetic_checks_only') is True
                or self.summary.get('synthetic_fixture_only') is True)

    @property
    def test(self):
        if not self.metrics_safe or self.metrics.empty:
            return pd.DataFrame()
        return self.metrics[self.metrics.phase == 'test'].sort_values(['epoch', 'weight_bits'])

    @property
    def last_epoch(self):
        return int(self.test.epoch.max()) if not self.test.empty else None

    @property
    def best_epoch(self):
        if self.test.empty:
            return None
        high = self.test[self.test.weight_bits == 8].sort_values('epoch')
        return int(high.loc[high.top1_percent.idxmax(), 'epoch'])

    @property
    def pilot_complete(self):
        return (self.report.get('status') == 'PASS' and self.last_epoch == 5
                and self.summary.get('pilot_completed') is True
                and self.metrics_safe and self.selection_safe and self.scales_safe
                and not any(i['Niveau'] in ('FOUT', 'RUNFOUT') for i in self.issues))

    def issue(self, level, message):
        self.issues.append({'Niveau': level, 'Melding': str(message)})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _read_json(path):
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Verwacht een JSON-object: ' + path.name)
    return value


def resolve_run_dir(run_dir=None):
    """Explicit path wins; otherwise find only this package's run family."""
    if run_dir:
        path = Path(run_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError('RUN_DIR bestaat niet: ' + str(path))
        return path
    starts = list(dict.fromkeys((Path.cwd().resolve(), Path(__file__).resolve().parent)))
    for start in starts:
        if (start / 'run_report.json').is_file():
            return start
    roots = list(dict.fromkeys(p for start in starts for p in [start, *list(start.parents)[:4]]))
    candidates = []
    for base in roots:
        runs = base / 'mx_runs' / RUN_STEM
        if runs.is_dir():
            for report_file in runs.glob('*/run_report.json'):
                try:
                    report = _read_json(report_file)
                    if report.get('schema') == SCHEMA:
                        candidates.append((report.get('command') == 'run', report_file.parent.name,
                                           report_file.parent))
                except (OSError, ValueError):
                    continue
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return Path(__file__).resolve().parent


def _history_frame(history):
    records = history.get('epochs', [])
    if not isinstance(records, list):
        raise ValueError('history.epochs is geen lijst.')
    rows = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Ongeldige epochrij in history.json.')
        for phase in ('train', 'test'):
            values = record.get(phase, [])
            if not isinstance(values, list) or not all(isinstance(v, dict) for v in values):
                raise ValueError('Ongeldige metrics per fase in history.json.')
            for value in values:
                if value.get('epoch') != record.get('epoch') or value.get('phase') != phase:
                    raise ValueError('Epoch of fase binnen history komt niet overeen.')
                rows.append(dict(value))
    return pd.DataFrame(rows), records


def _same_frame(expected, actual, keys, fields, label):
    if not set(keys + fields) <= set(actual):
        raise ValueError(label + ': vereiste kolommen ontbreken.')
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(label + ': dubbele sleutels.')
    left, right = expected.set_index(keys).sort_index(), actual.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError(label + ': rijen verschillen van history.json.')
    for field in fields:
        a, b = left[field], right[field]
        if pd.api.types.is_numeric_dtype(a):
            converted = pd.to_numeric(b, errors='coerce')
            if not np.allclose(a.to_numpy(float), converted.to_numpy(float), atol=1e-7, rtol=1e-12,
                               equal_nan=False):
                raise ValueError(label + ': veld ' + field + ' verschilt van history.json.')
        elif not a.astype(str).equals(b.astype(str)):
            raise ValueError(label + ': veld ' + field + ' verschilt van history.json.')


def _validate_contract(run):
    if not run.report:
        return
    if run.report.get('schema') != SCHEMA:
        raise ValueError('Deze notebook accepteert uitsluitend de onafhankelijke E2-schaalpilot.')
    expected = {'activation_bits': 32, 'master_dtype': 'float32', 'mx_block_size': 32,
                'mx_scale_format': 'E8M0', 'source_weight_bits': 8, 'truncation': True,
                'exponent_truncation': True, 'mantissa_truncation': True,
                'direct_lower_quantization': False, 'source_scale_learned': False,
                'learned_shared_scale': True, 'learned_scale_parameters': 4080,
                'base_parameters': 132954, 'parameters': 137034,
                'planned_epochs': 100, 'pilot_stop_epoch': 5, 'reference_end_epoch': 100,
                'scale_proxy_dtype': 'float32', 'scale_proxy_bounds': [-127.0, 127.0],
                'scale_gradient_normalization': 'none', 'scale_lr': .001,
                'independent_E2_shared_scale': True, 'scale_parameterization': 'independent_absolute_log2_E2_scale',
                'initial_relative_offset': 6, 'initial_relative_factor': 64,
                'relative_factor_constrained_after_initialization': False,
                'observed_relative_offset_is_diagnostic_only': True,
                'scheduler_milestones': [50, 75, 90], 'scale_freeze_epoch': None}
    wrong = [key for key, value in expected.items() if run.config.get(key) != value]
    if wrong:
        raise ValueError('Protocolvelden ontbreken of verschillen: ' + ', '.join(wrong))
    if run.config.get('weight_formats') != {str(k): v for k, v in FORMATS.items()}:
        raise ValueError('Niet de drie afgesproken gewichtsformaten.')
    if {str(k): str(v) for k, v in run.config.get('bn_bank_mapping', {}).items()} != {str(k): v for k, v in BANKS.items()}:
        raise ValueError('Onjuiste BN-bankmapping.')


def _validate_metrics(run, records, csv):
    if not records:
        if not csv.empty:
            raise ValueError('Metrics zonder bijbehorende history.json; niet als resultaat gebruikt.')
        return
    if [r.get('epoch') for r in records] != list(range(1, len(records) + 1)) or len(records) > 5:
        raise ValueError('Pilotgeschiedenis moet aaneengesloten epochs 1 t/m maximaal 5 bevatten.')
    required = {'epoch', 'phase', 'weight_bits', 'weight_format', 'activation_bits', 'bn_bank_key',
                'n', 'correct1', 'correct5', 'top1_percent', 'top5_percent', 'objective_loss'}
    if not required <= set(run.metrics):
        raise ValueError('Essentiële metricvelden ontbreken.')
    frame = run.metrics
    if frame.duplicated(['epoch', 'phase', 'weight_bits']).any():
        raise ValueError('Dubbele epoch/fase/precisie.')
    for record in records:
        for phase in ('train', 'test'):
            part = frame[(frame.epoch == record['epoch']) & (frame.phase == phase)]
            if len(part) != 3 or set(part.weight_bits) != set(BITS):
                raise ValueError('Een epoch/fase bevat niet alle drie precisies.')
            for row in part.to_dict('records'):
                bit = row['weight_bits']
                if (row['weight_format'] != FORMATS[bit] or row['activation_bits'] != 32
                        or str(row['bn_bank_key']) != BANKS[bit]):
                    raise ValueError('Verkeerd gewichtsformaat, activatieformaat of BN-bank.')
                n = _number(row['n'])
                expected_n = run.config.get('train_examples' if phase == 'train' else 'test_examples')
                if n is None or n <= 0 or n != int(n) or n != expected_n:
                    raise ValueError('Aantal geëvalueerde voorbeelden wijkt af van het runcontract.')
                for top in (1, 5):
                    correct, percent = _number(row['correct' + str(top)]), _number(row['top' + str(top) + '_percent'])
                    if correct is None or percent is None or correct != int(correct) or not 0 <= correct <= n or not 0 <= percent <= 100:
                        raise ValueError('Ongeldig aantal correcte voorspellingen of percentage.')
                    if abs(percent - 100 * correct / n) >= 1e-4:
                        raise ValueError('Accuracy strookt niet met aantallen correcte voorspellingen.')
                if _number(row['objective_loss']) is None:
                    raise ValueError('Niet-eindige loss.')
    if not csv.empty:
        _same_frame(frame, csv, ['epoch', 'phase', 'weight_bits'],
                    ['weight_format', 'activation_bits', 'n', 'correct1', 'correct5', 'top1_percent', 'top5_percent'],
                    'epoch_metrics.csv')
    else:
        run.issue('ONTBREEKT', 'epoch_metrics.csv ontbreekt; grafieken gebruiken gecontroleerde history.json.')
    for container, key in ((run.report, 'completed_epoch'), (run.summary, 'completed_epochs'),
                           (run.summary, 'last_epoch')):
        if key in container and container[key] != len(records):
            raise ValueError('Afgerond epochaantal verschilt tussen rapporten en history.json.')
    history_base = run.history.get('initial_base_model_sha256')
    if history_base != run.report.get('initial_base_model_sha256'):
        raise ValueError('Hash van de oorspronkelijke basisgewichten verschilt tussen rapport en history.')
    if run.summary.get('full_training_completed') or run.summary.get('full_100_epoch_training_completed'):
        raise ValueError('Een vijfepochpilot kan geen voltooide 100-epochtraining zijn.')


def _validate_selection(run):
    if run.test.empty:
        return
    if not run.selection:
        run.selection_safe = False
        run.issue('ONTBREEKT', 'Checkpointselectie ontbreekt; geen claim over opgeslagen best/last-checkpoints.')
        return
    for role, expected_epoch in [('best', run.best_epoch), ('last', run.last_epoch)]:
        detail = run.selection.get(role, {})
        if detail.get('epoch') != expected_epoch:
            raise ValueError('Checkpointselectie is niet het eerste E4M3-maximum of de laatste epoch.')
        high = run.test[(run.test.epoch == expected_epoch) & (run.test.weight_bits == 8)].iloc[0]
        if not math.isclose(detail.get('selection_score_E4M3', float('nan')), high.top1_percent,
                            rel_tol=0, abs_tol=1e-7):
            raise ValueError('Selectiescore verschilt van de E4M3-testscore.')
    if (run.selection.get('selected_on_official_test') is not True
            or run.selection.get('independent_test_estimate') is not False
            or run.selection.get('all_modes_from_same_selected_checkpoint') is not True):
        raise ValueError('Checkpointselectie heeft een onjuist evaluatiecontract.')
    expected = pd.concat([run.test[run.test.epoch == e].assign(role=r)
                          for r, e in [('best_E4M3_test', run.best_epoch), ('last_epoch', run.last_epoch)]])
    if not run.selected.empty:
        _same_frame(expected, run.selected, ['role', 'weight_bits'],
                    ['epoch', 'weight_format', 'top1_percent', 'correct1', 'n'], 'selected_metrics.csv')
    else:
        run.issue('ONTBREEKT', 'selected_metrics.csv ontbreekt; controle gebruikt history en checkpoint_selection.')


def _hard(phi):
    floor = np.floor(phi)
    return (floor + (phi - floor >= .5)).astype(np.int64)


def _scale_rows(run, records):
    states = [(0, run.history.get('initial_scale_state'), run.history.get('initial_source_scale_state'))]
    states += [(r['epoch'], r.get('scale_state'), r.get('source_scale_state')) for r in records]
    rows = []
    for epoch, phis, sources in states:
        if not isinstance(phis, dict) or not isinstance(sources, dict) or set(phis) != set(LAYER_SHAPES) or set(sources) != set(LAYER_SHAPES):
            raise ValueError('Absolute schaalstaat of E4-bronschaal ontbreekt bij epoch ' + str(epoch))
        for layer, shape in LAYER_SHAPES.items():
            phi, code = np.asarray(phis[layer], dtype=np.float64), np.asarray(sources[layer], dtype=np.float64)
            if (phi.shape != shape or code.shape != shape or not np.isfinite(phi).all()
                    or not np.isfinite(code).all() or not ((phi >= -127) & (phi <= 127)).all()
                    or not ((code >= 0) & (code <= 254)).all() or not (code == np.floor(code)).all()):
                raise ValueError('Ongeldige phi/E8M0-broncode: epoch ' + str(epoch) + ', ' + layer)
            e2, e4 = _hard(phi), code.astype(np.int64) - 127
            if epoch == 0 and not np.array_equal(phi, e4 + 6):
                raise ValueError('Init is niet exact éénmalig E4-exponent +6 voor alle groepen.')
            for channel, block in np.ndindex(shape):
                delta = int(e2[channel, block] - e4[channel, block])
                rows.append({'epoch': epoch, 'layer': layer, 'output_channel': channel, 'block_index': block,
                             'phi': phi[channel, block], 'e2_exponent': int(e2[channel, block]),
                             'e2_scale_code': int(e2[channel, block] + 127),
                             'source_exponent': int(e4[channel, block]), 'source_scale_code': int(code[channel, block]),
                             'relative_offset': delta, 'relative_factor': math.ldexp(1., delta)})
    return pd.DataFrame(rows)


def _validate_scales(run, records, blocks_csv, epochs_csv):
    if not records:
        if not blocks_csv.empty or not epochs_csv.empty:
            raise ValueError('Schaaltabellen zonder afgeronde geschiedenis.')
        return
    run.scale_blocks = _scale_rows(run, records)
    if not blocks_csv.empty:
        _same_frame(run.scale_blocks, blocks_csv, ['epoch', 'layer', 'output_channel', 'block_index'],
                    ['phi', 'e2_exponent', 'e2_scale_code', 'source_exponent', 'source_scale_code', 'relative_offset', 'relative_factor'],
                    'scale_block_state.csv')
    else:
        run.issue('ONTBREEKT', 'scale_block_state.csv ontbreekt; schaaltabel in geheugen uit history gereconstrueerd.')
    rows = []
    for record in records:
        audit = record.get('scale_epoch_metrics')
        if not isinstance(audit, list) or len(audit) != len(LAYER_SHAPES) or {r.get('layer') for r in audit} != set(LAYER_SHAPES):
            raise ValueError('Schaaldiagnostiek mist of dupliceert lagen.')
        for row in audit:
            epoch, layer = record['epoch'], row['layer']
            now = run.scale_blocks[(run.scale_blocks.epoch == epoch) & (run.scale_blocks.layer == layer)]
            before = run.scale_blocks[(run.scale_blocks.epoch == epoch - 1) & (run.scale_blocks.layer == layer)]
            phi, old = now.phi.to_numpy(), before.phi.to_numpy()
            net = int((now.e2_exponent.to_numpy() != before.e2_exponent.to_numpy()).sum())
            if row.get('epoch') != epoch or row.get('blocks') != len(now):
                raise ValueError('Onjuiste epoch of groepsaantal in schaaldiagnostiek.')
            expected = {'phi_min': phi.min(), 'phi_max': phi.max(), 'phi_mean': phi.mean(),
                        'phi_std': phi.std(), 'phi_update_l2': np.linalg.norm(phi - old)}
            for key, value in expected.items():
                if not math.isclose(float(row.get(key, float('nan'))), float(value), rel_tol=1e-10, abs_tol=1e-11):
                    raise ValueError('Schaaldiagnostiek wijkt af van ruwe staat: ' + key)
            if row.get('phi_changed_blocks') != int((phi != old).sum()) or row.get('net_exponent_changes') != net:
                raise ValueError('Schaalveranderingen stroken niet met de opeenvolgende opgeslagen staten.')
            seen, flips = row.get('unique_blocks_switched_during_epoch', -1), row.get('exponent_flips_during_epoch', -1)
            if not net <= seen <= len(now) or flips < seen:
                raise ValueError('Ongeldige aantallen exponentwissels.')
            for key, column in [('e2_exponent_counts', 'e2_exponent'), ('observed_relative_offset_counts', 'relative_offset')]:
                actual = row.get(key)
                if not isinstance(actual, dict):
                    raise ValueError('Histogram ontbreekt: ' + key)
                actual = {str(k): int(v) for k, v in actual.items() if int(v) != 0}
                expected_counts = {str(k): int(v) for k, v in now[column].value_counts().items()}
                if actual != expected_counts:
                    raise ValueError('Histogram strookt niet met ruwe groepstaten: ' + key)
            if not 0 <= row.get('scale_grad_l2_min', -1) <= row.get('scale_grad_l2_max', -1) < float('inf'):
                raise ValueError('Ongeldige schaalgradientnorm.')
            rows.append(row)
    run.scale_epochs = pd.DataFrame(rows)
    if not epochs_csv.empty:
        fields = ['blocks', 'phi_min', 'phi_max', 'phi_mean', 'phi_std', 'phi_changed_blocks',
                  'phi_update_l2', 'net_exponent_changes', 'exponent_flips_during_epoch',
                  'unique_blocks_switched_during_epoch', 'scale_grad_l2_min', 'scale_grad_l2_max',
                  'projected_values_during_epoch']
        _same_frame(run.scale_epochs, epochs_csv, ['epoch', 'layer'], fields, 'scale_epoch_metrics.csv')


def _validate_timing(run, records):
    rows = []
    for record in records:
        keys = ('train_seconds', 'test_seconds', 'checkpoint_seconds', 'other_seconds',
                'total_epoch_seconds', 'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes')
        row = {'epoch': record['epoch'], **{k: record.get(k) for k in keys}}
        if any(_number(row[k]) is None or row[k] < 0 for k in keys[:5]):
            run.issue('FOUT', 'Niet-eindige of ontbrekende epochtijd; tijdgrafiek weggelaten.')
            run.times = pd.DataFrame()
            return
        rows.append(row)
    run.times = pd.DataFrame(rows)
    if rows:
        total = sum(r['total_epoch_seconds'] for r in rows)
        if not math.isclose(run.timing.get('measured_completed_epoch_seconds', total), total, abs_tol=1e-6, rel_tol=1e-10):
            run.issue('FOUT', 'Gemeten totaaltijd wijkt af van som van epochtijden.')
        if run.timing.get('measured_100_epoch_seconds') is not None:
            run.issue('FOUT', '100 epochtijden kunnen niet in deze vijfepochpilot gemeten zijn.')


def _load_reference(run, name):
    directory = run.path / name
    if not directory.is_dir():
        directory = Path(__file__).resolve().parent / name
    if not directory.is_dir():
        run.issue('ONTBREEKT', 'Historische referentie ontbreekt: ' + name)
        return
    marker = directory / 'REFERENCE.json'
    if name not in REFERENCE_MARKERS or _sha(marker) != REFERENCE_MARKERS[name]:
        raise ValueError('Referentiemarker is gewijzigd: ' + name)
    meta = _read_json(marker)
    for file, digest in meta.get('files_sha256', {}).items():
        path = (directory / file).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or _sha(path) != digest:
            raise ValueError('Historisch referentiebestand gewijzigd: ' + name + '/' + file)
    report = _read_json(directory / 'run_report.json')
    c = report.get('config', {})
    if report.get('status') != 'PASS' or report.get('synthetic_fixture_only') is not False:
        raise ValueError('Historische referentie is geen geslaagde echte run.')
    frame = pd.read_csv(directory / 'epoch_metrics.csv', float_precision='round_trip')
    frame = frame[(frame.phase == 'test') & frame.epoch.between(1, 5)].copy()
    if frame.duplicated(['epoch', 'weight_bits']).any() or len(frame) != 15:
        raise ValueError('Historische referentie bevat niet alle eerste vijf epochs × drie precisies.')
    for epoch in range(1, 6):
        if set(frame[frame.epoch == epoch].weight_bits) != set(BITS):
            raise ValueError('Historische precisies onvolledig.')
    for row in frame.to_dict('records'):
        if row['weight_format'] != FORMATS[row['weight_bits']] or row['activation_bits'] != 32 or row['n'] != 26032:
            raise ValueError('Historisch precisie- of datasetcontract wijkt af.')
        if abs(row['top1_percent'] - 100 * row['correct1'] / row['n']) >= 1e-4:
            raise ValueError('Historische accuracy en correcte aantallen verschillen.')
    if name == 'relative_scale10_reference':
        if c.get('scale_proxy_init') != 6.0 or c.get('scale_exponent_offset_choices') != list(range(10)):
            raise ValueError('De relatieve referentie moet tien keuzes met start ×64 hebben.')
    elif c.get('scale_factor') != 64 and c.get('scale_exponent_offset') != 6:
        # Older fixed protocols use a descriptive name; the pinned marker/report
        # together are the authoritative fixed-factor identity.
        if meta.get('factor') != 64 or report.get('schema') != 'mx-triple-scale64-reference100-report-v1':
            raise ValueError('Vaste referentie is niet de geverifieerde ×64-run.')
    comparable = False
    if not run.test.empty and not run.synthetic:
        old_base = report.get('initial_base_model_sha256', report.get('initial_model_sha256'))
        if old_base != run.history.get('initial_base_model_sha256'):
            raise ValueError('Basisinitialisatie verschilt van historische referentie: ' + name)
        for key in ('activation_bits', 'master_dtype', 'mx_block_size', 'batch_size', 'seed', 'lr',
                    'weight_decay', 'optimizer', 'scheduler_milestones', 'scheduler_gamma',
                    'train_examples', 'test_examples', 'weight_preprocess', 'bn_bank_mapping'):
            if run.config.get(key) != c.get(key):
                raise ValueError('Vergelijkingsprotocol verschilt in ' + key + ': ' + name)
        if name == 'relative_scale10_reference':
            old_history = _read_json(directory / 'history.json')
            for record in run.history['epochs']:
                old = old_history['epochs'][record['epoch'] - 1]
                for phase in ('train', 'test'):
                    if record.get('coverage', {}).get(phase, {}).get('order_sha256') != old.get('coverage', {}).get(phase, {}).get('order_sha256'):
                        raise ValueError('Samplevolgorde verschilt van relatieve referentie.')
        comparable = True
    run.references[name] = {'path': directory, 'report': report, 'metadata': meta,
                            'test': frame, 'comparable': comparable}
    run.input_files.append(marker)


def load_run(run_dir=None):
    run = RunAnalysis(resolve_run_dir(run_dir))
    json_files = {'run_report.json': 'report', 'history.json': 'history',
                  'training_summary.json': 'summary', 'checkpoint_selection.json': 'selection',
                  'timing_estimate.json': 'timing', 'scale_learning_summary.json': 'scale_summary'}
    for name, attr in json_files.items():
        path = run.path / name
        if not path.is_file():
            continue
        try:
            setattr(run, attr, _read_json(path)); run.input_files.append(path)
        except (OSError, ValueError) as exc:
            run.issue('FOUT', name + ': ' + str(exc))
    csvs = {}
    for name in ('epoch_metrics.csv', 'selected_metrics.csv', 'scale_block_state.csv', 'scale_epoch_metrics.csv'):
        path = run.path / name
        csvs[name] = pd.DataFrame()
        if path.is_file():
            try:
                csvs[name] = pd.read_csv(path, float_precision='round_trip')
                run.input_files.append(path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                run.issue('FOUT', name + ': ' + str(exc))
                if name == 'epoch_metrics.csv': run.metrics_safe = False
                elif name == 'selected_metrics.csv': run.selection_safe = False
                else: run.scales_safe = False
    run.selected = csvs['selected_metrics.csv']
    records = []
    try:
        _validate_contract(run)
        run.metrics, records = _history_frame(run.history)
        _validate_metrics(run, records, csvs['epoch_metrics.csv'])
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        run.metrics_safe = False; run.scales_safe = False; run.issue('FOUT', str(exc))
    try:
        _validate_selection(run)
    except (ValueError, KeyError, TypeError) as exc:
        run.selection_safe = False; run.issue('FOUT', str(exc))
    try:
        _validate_scales(run, records, csvs['scale_block_state.csv'], csvs['scale_epoch_metrics.csv'])
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        run.scales_safe = False; run.issue('FOUT', str(exc))
    _validate_timing(run, records)
    for name in ('relative_scale10_reference', 'fixed_scale64_reference'):
        try:
            _load_reference(run, name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            run.reference_safe = False; run.issue('FOUT', str(exc))
    if run.report.get('error'):
        run.issue('RUNFOUT', run.report['error'])
    for warning in run.report.get('warnings', []):
        run.issue('RUNWAARSCHUWING', warning.get('message', warning) if isinstance(warning, dict) else warning)
    if not run.report:
        run.issue('INFO', 'Er is nog geen runrapport. Deze notebook kan na check/run opnieuw worden uitgevoerd.')
    elif run.test.empty:
        run.issue('INFO', 'Geen bruikbare afgeronde SVHN-epochresultaten; er worden geen accuracies ingevuld.')
    if records and run.report.get('synthetic_fixture_only') is None:
        run.issue('LET OP', 'De herkomst is niet expliciet als echte of synthetische run gemarkeerd.')
    return run


def diagnostics_table(run):
    return pd.DataFrame(run.issues or [{'Niveau': 'OK', 'Melding': 'Geen inconsistenties of runwaarschuwingen gevonden.'}])


def status_markdown(run):
    status = run.report.get('status', 'NOG NIET UITGEVOERD')
    tag = '**SYNTHETISCHE SOFTWARETEST — geen onderzoeksresultaat.**\n\n' if run.synthetic else ''
    completed = 'Nog geen gecontroleerde afgeronde epoch.' if run.last_epoch is None else f'{run.last_epoch} van maximaal 5 epochs beschikbaar.'
    quality = 'Pilot volledig en consistent.' if run.pilot_complete else 'Pilot is nog niet volledig bevestigd; raadpleeg de controles hieronder.'
    return (f'{tag}**Status: {status}.** {completed} {quality}\n\n'
            f'**Gelezen resultatenmap:** `{run.path}`\n\n'
            'E4M3 behoudt zijn actuele amax-schaal. E2M3 en E2M1 delen één absolute geleerde E8M0-schaal '
            'per groep. Alleen bij initialisatie is deze precies ×64 ten opzichte van E4. '
            'Daarna wordt de lagere schaal uitsluitend uit de eigen parameter φ bepaald.')


def protocol_table(run):
    return pd.DataFrame([
        ('Model / dataset', 'CNN-8 (svhnq, expand=2) / SVHN train + test; geen extra-split'),
        ('Precisies', 'E4M3 → E2M3 → E2M1; alle activaties A32'),
        ('Masters / optimizer', 'FP32; Adam; drie recursieve distillatiepasses en één update per minibatch'),
        ('Groepen', 'B32; per outputkanaal; 4.080 groepen in zes MX-lagen'),
        ('E4-schaal', 'Bestaande amax-regel; verandert met gewichten; geen nieuwe E4-schaalparameter'),
        ('E2-schaal', 'Absoluut S₂=2^round_half_up(φ); E2M3 en E2M1 delen dezelfde schaal'),
        ('Initialisatie', 'Eenmalig φ₀=e₄,₀+6; daarna geen vaste of beperkte verhouding tot actuele E4-schaal'),
        ('Exponentcodes', 'E4M3: s e3 e2 e1 e0 m2 m1 m0 → E2M3: s e3 e2 m2 m1 m0'),
        ('Mantissacodes', 'E2M1-code = E2M3-code >> 2; dezelfde onafhankelijke E2-schaal'),
        ('Schaalgradient', 'ln(2)·Σ(gradient_q · q) per echte groep; som van beide E2-passes; geen normalisatie'),
        ('Grenzen', 'φ ∈ [−127,127]; projectie na Adam; E8M0 code255 ongeldig; FP32-decode-overflow geeft fout'),
        ('Leerplan', 'Origineel 100-epoch-LR-schema (50/75/90), maar dit pakket stopt uiterlijk op epoch5'),
        ('Checkpointselectie', 'Laatste afgeronde epoch én eerste maximale E4M3-testaccuracy; steeds alle drie modi samen'),
        ('Uitvoering', 'FP32 convoluties met gesimuleerde quantisatie; geen gemeten native low-bit versnelling')],
        columns=['Onderdeel', 'Afspraak'])


def _variants(run):
    if run.test.empty:
        return {}
    result = {}
    if not run.synthetic and run.reference_safe:
        for name in ('fixed_scale64_reference', 'relative_scale10_reference'):
            reference = run.references.get(name)
            if reference and reference['comparable']:
                result[name] = reference['test'][reference['test'].epoch <= run.last_epoch].copy()
    result['independent'] = run.test
    return result


def comparison_markdown(run):
    if run.test.empty:
        return 'Vergelijkingen verschijnen zodra de nieuwe pilot afgeronde, consistente epochs bevat. Ontbrekende resultaten worden niet als 0% weergegeven.'
    if run.synthetic:
        return '**Synthetische fixture:** historische SVHN-accuracies worden niet als vergelijkingsgroep gebruikt.'
    names = [VARIANT_LABELS[key] for key in _variants(run) if key != 'independent']
    if not names:
        return 'Er is nog geen gevalideerde historische vergelijking. Controleer de referentiemappen en meldingen.'
    return (f'Vergeleken worden exact epochs **1–{run.last_epoch}** van de nieuwe pilot, '
            + ' en '.join(names) + '. De vaste ×64-reeks komt uit de **eerste vijf epochs van een eerdere 100-epochrun**; '
            'de relatieve tienkeuzereeks uit de eerdere vijfepochpilot. De baselinebestanden zijn ongewijzigd en op hash gecontroleerd.\n\n'
            'De basisinitialisatie en genoemde trainingsinstellingen zijn gecontroleerd; voor de tienkeuzereferentie is ook de '
            'opgeslagen samplevolgorde gecontroleerd. Voor de vaste referentie is geen volledige history in dit pakket aanwezig. '
            'Dit zijn historische individuele runs, met de oorspronkelijke niet-bitexacte CUDA-backend. Kleine verschillen bewijzen '
            'daarom geen statistisch significant voordeel. De onafhankelijke parametrisatie verandert ook hoe de lagere schaal '
            'op veranderingen in de E4-bronschaal reageert.')


def checkpoint_table(run, role='last'):
    if not run.selection_safe:
        return pd.DataFrame({'Melding': ['Checkpointselectie niet bevestigd; raadpleeg de controles.']})
    rows = []
    for name, frame in _variants(run).items():
        if role == 'best':
            high = frame[frame.weight_bits == 8].sort_values('epoch')
            epoch = int(high.loc[high.top1_percent.idxmax(), 'epoch'])
        else:
            epoch = int(frame.epoch.max())
        record = {'Variant': VARIANT_LABELS[name], 'Epoch': epoch,
                  'Selectie': 'eerste beste E4 in beschikbaar venster' if role == 'best' else 'gemeenschappelijk eindpunt'}
        for bit in BITS:
            row = frame[(frame.epoch == epoch) & (frame.weight_bits == bit)].iloc[0]
            record[LABELS[bit] + ' testaccuracy [%]'] = row.top1_percent
            record[LABELS[bit] + ' correct'] = int(row.correct1)
        rows.append(record)
    return pd.DataFrame(rows) if rows else pd.DataFrame({'Melding': ['Nog geen vergelijkbare checkpointresultaten.']})


def difference_table(run):
    variants = _variants(run)
    if 'independent' not in variants:
        return pd.DataFrame({'Melding': ['Nog geen gecontroleerde eindpuntvergelijking.']})
    new = variants['independent'].query('epoch == @run.last_epoch').set_index('weight_bits')
    rows = []
    for name, frame in variants.items():
        if name == 'independent':
            continue
        old = frame[frame.epoch == run.last_epoch].set_index('weight_bits')
        for bit in BITS:
            rows.append({'Epoch': run.last_epoch, 'Referentie': VARIANT_LABELS[name], 'Precisie': LABELS[bit],
                         'Onafhankelijk [%]': new.loc[bit, 'top1_percent'],
                         'Referentie [%]': old.loc[bit, 'top1_percent'],
                         'Verschil [procentpunt]': new.loc[bit, 'top1_percent'] - old.loc[bit, 'top1_percent'],
                         'Verschil correct [beelden]': int(new.loc[bit, 'correct1'] - old.loc[bit, 'correct1'])})
    return pd.DataFrame(rows) if rows else pd.DataFrame({'Melding': ['Geen gevalideerde historische comparator.']})


def reference_table(run):
    return pd.DataFrame([{'Referentie': VARIANT_LABELS[name], 'Bronrun': r['metadata'].get('source_run', r['metadata'].get('source_run_directory')),
                          'Gebruikt venster': 'epochs 1–5 (of dezelfde kortere nieuwe run)',
                          'Bestand': str(r['path'] / 'REFERENCE.json'), 'Marker SHA256': _sha(r['path'] / 'REFERENCE.json'),
                          'Vergelijking met nieuwe run gevalideerd': r['comparable']}
                         for name, r in run.references.items()])


def _limits(values, span=1.0):
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return (0, 100)
    lo, hi = array.min(), array.max()
    width = max(float(hi - lo), span)
    mid = float((hi + lo) / 2)
    return max(0, min(float(lo) - .15 * width, mid - .65 * width)), min(100, max(float(hi) + .15 * width, mid + .65 * width))


def _title(run, text):
    return ('SYNTHETISCHE CONTROLE · ' if run.synthetic else '') + text


def _empty(ax, message='Nog geen gecontroleerde resultaten'):
    ax.text(.5, .5, message, ha='center', va='center', transform=ax.transAxes, wrap=True)
    ax.set_axis_off()


def plot_learning(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Testaccuracy gedurende de pilot · dezelfde eerste epochs'), fontsize=15, fontweight='bold')
        variants = _variants(run)
        if not variants:
            for ax in axes: _empty(ax)
            return fig
        limits = _limits(v for f in variants.values() for v in f.top1_percent)
        for bit, ax in zip(BITS, axes):
            for name, frame in variants.items():
                r = frame[frame.weight_bits == bit].sort_values('epoch')
                style, marker, alpha = VARIANT_STYLES[name]
                ax.plot(r.epoch, r.top1_percent, linestyle=style, marker=marker, color=COLORS[bit], alpha=alpha,
                        label=VARIANT_LABELS[name], linewidth=2.4 if name == 'independent' else 1.7)
            ax.set(title=LABELS[bit] + ' / A32', xlabel='Epoch', ylabel='Testaccuracy [%] · ingezoomd',
                   xticks=range(1, run.last_epoch + 1), ylim=limits)
            ax.legend(fontsize=8, frameon=False, loc='lower right')
        return fig


def plot_checkpoints(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        fig.suptitle(_title(run, 'Drie precisies uit elk gezamenlijk checkpoint'), fontsize=15, fontweight='bold')
        variants = _variants(run)
        if not variants or not run.selection_safe:
            for ax in axes: _empty(ax, 'Geen bevestigde checkpointselectie')
            return fig
        tables = [checkpoint_table(run, role) for role in ('last', 'best')]
        columns = [LABELS[b] + ' testaccuracy [%]' for b in BITS]
        limits = _limits(v for table in tables for column in columns for v in table[column])
        for ax, table, role in zip(axes, tables, ('last', 'best')):
            names = list(variants)
            offsets = np.linspace(-.2, .2, len(names)) if len(names) > 1 else [0.]
            for offset, name, (_, row) in zip(offsets, names, table.iterrows()):
                _, marker, alpha = VARIANT_STYLES[name]
                for x, bit in enumerate(BITS):
                    value = row[LABELS[bit] + ' testaccuracy [%]']
                    ax.scatter(x + offset, value, color=COLORS[bit], marker=marker, alpha=alpha, s=65,
                               label=VARIANT_LABELS[name] + f' · ep{row.Epoch}' if x == 0 else None)
            ax.set(xticks=range(3), xticklabels=[LABELS[b] for b in BITS], ylim=limits,
                   ylabel='Testaccuracy [%] · ingezoomd', xlabel='Gewichtsformaat · alle activaties A32',
                   title=(f'Laatste gemeenschappelijke epoch {run.last_epoch}' if role == 'last'
                          else f'Beste E4M3 binnen epochs 1–{run.last_epoch}'))
            ax.legend(fontsize=8, frameon=False, loc='lower right')
        return fig


def scale_distribution_table(run, epoch=None):
    if not run.scales_safe or run.scale_blocks.empty:
        return pd.DataFrame({'Melding': ['Geen gecontroleerde absolute schaalstaten.']})
    epoch = run.last_epoch if epoch is None else epoch
    part = run.scale_blocks[run.scale_blocks.epoch == epoch]
    rows = []
    for delta, count in part.relative_offset.value_counts().sort_index().items():
        rows.append({'Epoch': epoch, 'Geobserveerde Δ=e₂−e₄': int(delta), 'Verhouding S₂/S₄': _ratio_label(delta),
                     'Groepen': int(count), 'Aandeel [%]': 100 * count / len(part)})
    return pd.DataFrame(rows)


def _ratio_label(delta):
    delta = int(delta)
    if 0 <= delta <= 15:
        return '×' + str(1 << delta)
    if -15 <= delta < 0:
        return '×1/' + str(1 << -delta)
    return '×2^' + str(delta)


def _scale_available(run):
    return run.scales_safe and not run.scale_blocks.empty


def plot_scale_distributions(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
        fig.suptitle(_title(run, 'Eigen E2-schaal en de verhouding die daaruit ontstaat'), fontsize=15, fontweight='bold')
        if not _scale_available(run):
            for ax in axes: _empty(ax, 'Geen gecontroleerde schaalstaten')
            return fig
        first = run.scale_blocks[run.scale_blocks.epoch == 0]
        last_epoch = int(run.scale_blocks.epoch.max())
        last = run.scale_blocks[run.scale_blocks.epoch == last_epoch]
        for ax, column, title in zip(axes, ('e2_exponent', 'relative_offset'),
                                     ('Absolute E2-exponent e₂ · S₂=2^e₂', 'Geobserveerde verhouding · geen keuzemenu')):
            keys = sorted(set(first[column]) | set(last[column]))
            x = np.arange(len(keys))
            for offset, part, color, label in [(-.2, first, '#A5B0B9', 'Initialisatie'),
                                               (.2, last, '#245A81', f'Epoch {last_epoch}')]:
                counts = part[column].value_counts()
                ax.bar(x + offset, [100 * counts.get(k, 0) / len(part) for k in keys], width=.38, color=color, label=label)
            labels = [str(k) if column == 'e2_exponent' else _ratio_label(k) for k in keys]
            ax.set(title=title, ylabel='Aandeel van alle 4.080 groepen [%]', xticks=x, xticklabels=labels, ylim=(0, 105))
            ax.tick_params(axis='x', labelrotation=45 if len(keys) > 7 else 0)
            ax.legend(frameon=False)
        return fig


def plot_scale_evolution(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle(_title(run, 'Verdeling per epoch · initialisatie is epoch 0'), fontsize=15, fontweight='bold')
        if not _scale_available(run):
            for ax in axes: _empty(ax, 'Geen gecontroleerde schaalstaten')
            return fig
        for ax, column, title in zip(axes, ('e2_exponent', 'relative_offset'),
                                     ('Absolute E2-exponent', 'Geobserveerde verhouding S₂/S₄')):
            table = pd.crosstab(run.scale_blocks[column], run.scale_blocks.epoch).sort_index()
            values = 100 * table.to_numpy() / TOTAL_BLOCKS
            im = ax.imshow(values, origin='lower', aspect='auto', vmin=0, vmax=100, cmap='Blues')
            labels = [str(k) if column == 'e2_exponent' else _ratio_label(k) for k in table.index]
            ax.set(title=title, xlabel='Epoch', xticks=range(len(table.columns)), xticklabels=table.columns,
                   yticks=range(len(labels)), yticklabels=labels)
            ax.grid(False)
            if values.size <= 160:
                for (y, x), value in np.ndenumerate(values):
                    if value > 0:
                        ax.text(x, y, f'{value:.1f}', ha='center', va='center', fontsize=8,
                                color='white' if value > 50 else '#263B4A')
            fig.colorbar(im, ax=ax, label='Aandeel groepen [%]', shrink=.8)
        return fig


def scale_layer_table(run):
    if not _scale_available(run):
        return pd.DataFrame({'Melding': ['Geen gecontroleerde schaalstaten.']})
    last = run.scale_blocks[run.scale_blocks.epoch == run.scale_blocks.epoch.max()]
    return last.groupby('layer', sort=False).agg(
        Groepen=('phi', 'size'), phi_min=('phi', 'min'), phi_gemiddeld=('phi', 'mean'), phi_max=('phi', 'max'),
        E2_exponent_min=('e2_exponent', 'min'), E2_exponent_max=('e2_exponent', 'max'),
        geobserveerde_delta_min=('relative_offset', 'min'), geobserveerde_delta_max=('relative_offset', 'max')).reset_index()


def plot_scale_by_layer(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        fig.suptitle(_title(run, 'Schaalontwikkeling per laag · gemiddelden over groepen'), fontsize=15, fontweight='bold')
        if not _scale_available(run):
            for ax in axes: _empty(ax, 'Geen gecontroleerde schaalstaten')
            return fig
        frame = run.scale_blocks.groupby(['epoch', 'layer'], sort=False)[['phi', 'source_exponent', 'relative_offset']].mean().reset_index()
        for layer, color in zip(LAYER_SHAPES, LAYER_COLORS):
            r = frame[frame.layer == layer].sort_values('epoch')
            axes[0].plot(r.epoch, r.phi, marker='o', color=color, label=layer)
            axes[1].plot(r.epoch, r.relative_offset, marker='o', color=color, label=layer)
        axes[0].set(title='Geleerde absolute log₂-proxy φ', ylabel='Gemiddelde φ')
        axes[1].set(title='Afgeleide Δ=e₂−e₄', ylabel='Gemiddelde geobserveerde exponentafstand')
        for ax in axes:
            ax.set(xlabel='Epoch', xticks=range(int(frame.epoch.max()) + 1))
            ax.legend(fontsize=8, ncol=2, frameon=False)
        return fig


def source_change_table(run):
    if not _scale_available(run):
        return pd.DataFrame()
    frame, rows = run.scale_blocks, []
    keys = ['layer', 'output_channel', 'block_index']
    for epoch in range(1, int(frame.epoch.max()) + 1):
        before = frame[frame.epoch == epoch - 1].set_index(keys).sort_index()
        now = frame[frame.epoch == epoch].set_index(keys).sort_index()
        rows.append({'Epoch': epoch,
                     'E4-bronschaal gewijzigd': int((before.source_exponent != now.source_exponent).sum()),
                     'Absolute E2-schaal gewijzigd': int((before.e2_exponent != now.e2_exponent).sum()),
                     'Geobserveerde verhouding gewijzigd': int((before.relative_offset != now.relative_offset).sum())})
    return pd.DataFrame(rows)


def plot_scale_changes(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        fig.suptitle(_title(run, 'Bronbeweging en leerbare schaalbeweging apart'), fontsize=15, fontweight='bold')
        table = source_change_table(run)
        if table.empty:
            for ax in axes: _empty(ax, 'Geen gecontroleerde schaalstaten')
            return fig
        for i, (column, color) in enumerate(zip(table.columns[1:], ['#98A5B0', '#245A81', '#C98220'])):
            axes[0].bar(table.Epoch + (i - 1) * .24, table[column], width=.23, label=column, color=color)
        axes[0].set(title='Verschil tussen opeenvolgende epochs', ylabel='Aantal groepen met andere harde waarde', xlabel='Epoch', xticks=table.Epoch)
        axes[0].legend(frameon=False, fontsize=8)
        if not run.scale_epochs.empty:
            audit = run.scale_epochs.groupby('epoch')[['exponent_flips_during_epoch', 'unique_blocks_switched_during_epoch']].sum()
            axes[1].plot(audit.index, audit.exponent_flips_during_epoch, marker='o', color='#245A81', label='Alle E2-wissels, inclusief terugwissels')
            axes[1].plot(audit.index, audit.unique_blocks_switched_during_epoch, marker='s', color='#C98220', label='Unieke groepen met ≥1 E2-wissel')
            axes[1].set(title='E2-schaalwissels tijdens de updates', xlabel='Epoch', ylabel='Aantal', xticks=audit.index)
            axes[1].legend(frameon=False, fontsize=8)
        else:
            _empty(axes[1], 'Geen diagnostiek per update beschikbaar')
        return fig


def plot_scale_gradients(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Gradienten en projecties van de absolute schaalparameter'), fontsize=15, fontweight='bold')
        if not run.scales_safe or run.scale_epochs.empty:
            for ax in axes: _empty(ax, 'Geen gecontroleerde schaalgradientdiagnostiek')
            return fig
        for layer, color in zip(LAYER_SHAPES, LAYER_COLORS):
            r = run.scale_epochs[run.scale_epochs.layer == layer].sort_values('epoch')
            axes[0].plot(r.epoch, r.scale_grad_l2_max, color=color, marker='o', label=layer)
            axes[1].plot(r.epoch, r.projected_values_during_epoch, color=color, marker='o', label=layer)
        axes[0].set(title='Maximale L2-norm per epoch', ylabel='Schaalgradientnorm per laag')
        axes[0].set_yscale('symlog', linthresh=1e-8)
        axes[1].set(title='Projectie naar [−127,127] na Adam', ylabel='Geprojecteerde parameterwaarden (herhalingen tellen mee)', ylim=(0, None))
        for ax in axes:
            ax.set(xlabel='Epoch', xticks=sorted(run.scale_epochs.epoch.unique()))
            ax.legend(fontsize=8, ncol=2, frameon=False)
        return fig


def storage_cost_table(run=None):
    rows = []
    for name, extra_bits, learned in [('Vaste ×64', 0, False), ('10 relatieve keuzes', 4, True), ('Onafhankelijke E2-schaal', 8, True)]:
        extra = TOTAL_BLOCKS * extra_bits // 8
        rows.append({'Variant': name, 'Groepen': TOTAL_BLOCKS,
                     'E4-schaalbytes': TOTAL_BLOCKS, 'Extra schaal/keuzebytes': extra,
                     'Totaal schaalinformatie [bytes]': TOTAL_BLOCKS + extra,
                     'Schaalinformatie [bit per B32-positie]': (8 + extra_bits) / 32,
                     'E4-codes + schaalinformatie, zes MX-lagen [bytes]': TOTAL_BLOCKS * 32 + TOTAL_BLOCKS + extra,
                     'Extra training: proxy+grad+Adam [bytes]': TOTAL_BLOCKS * 16 if learned else 0})
    return pd.DataFrame(rows)


def plot_storage_cost(run):
    table = storage_cost_table(run)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        fig.suptitle('Analytische opslag · dezelfde 4.080 B32-groepen', fontsize=15, fontweight='bold')
        x = np.arange(len(table))
        axes[0].bar(x, table['E4-schaalbytes'], color='#A7B2BB', label='E4-bronschaal: 1 byte/groep')
        axes[0].bar(x, table['Extra schaal/keuzebytes'], bottom=table['E4-schaalbytes'], color='#245A81', label='Extra keuze of absolute E2-schaal')
        for i, value in enumerate(table['Totaal schaalinformatie [bytes]']):
            axes[0].text(i, value + 100, f'{value:,}'.replace(',', '.'), ha='center')
        axes[0].set(title='Alleen schaalmetadata', ylabel='Bytes', ylim=(0, 9600))
        axes[0].legend(frameon=False, fontsize=8)
        values = table['E4-codes + schaalinformatie, zes MX-lagen [bytes]'].to_numpy() / 1024
        axes[1].bar(x, values, color=['#A7B2BB', '#C98220', '#245A81'])
        for i, value in enumerate(values):
            axes[1].text(i, value + 1.1, f'{value:.2f}', ha='center')
        axes[1].set(title='E4-broncodes + metadata voor alle modi', ylabel='KiB · zes MX-lagen, inclusief padding', ylim=(0, max(values) * 1.15))
        for ax in axes:
            ax.set(xticks=x, xticklabels=['Vaste ×64', '10 keuzes', 'Onafhankelijk'])
        return fig


def timing_markdown(run):
    if run.times.empty:
        return 'Nog geen gemeten epochtijden. Een check zonder SVHN-training levert geen trainingstijdmeting op.'
    measured = float(run.times.total_epoch_seconds.sum())
    estimate = _number(run.timing.get('estimated_100_epoch_seconds'))
    text = f'**Gemeten:** {measured / 60:.2f} minuten voor {len(run.times)} complete epochs, inclusief train, test en checkpoints.'
    if estimate is not None and run.timing.get('available') is True:
        text += f' **Projectie voor een nieuwe 100-epochrun:** {estimate / 3600:.2f} uur; dit is geen gemeten 100-epochtijd en geen resterende looptijd.'
    text += ('\n\nVoorbereiding van data, installatie en uiteindelijke rapportexport vallen buiten de epochtijden. '
             'De pilot meet de latere LR-overgangen en langdurige thermische belasting niet. '
             'Een kleinere of grotere geobserveerde tijd wordt daarom niet meteen aan schaalparametrisatie toegeschreven.')
    return text


def plot_timing(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Gemeten tijd en GPU-geheugen'), fontsize=15, fontweight='bold')
        if run.times.empty:
            for ax in axes: _empty(ax, 'Nog geen epochtijden')
            return fig
        r, bottom = run.times, np.zeros(len(run.times))
        for column, label, color in [('train_seconds', 'Training (3 precisies)', '#245A81'), ('test_seconds', 'Test (3 precisies)', '#33846D'),
                                     ('checkpoint_seconds', 'Checkpoint', '#C98220'), ('other_seconds', 'Overige epochdiagnostiek', '#A7B2BB')]:
            axes[0].bar(r.epoch, r[column], bottom=bottom, color=color, label=label)
            bottom += r[column].to_numpy()
        axes[0].set(xlabel='Epoch', ylabel='Seconden', title='Werkelijk gemeten per epoch', xticks=r.epoch)
        axes[0].legend(frameon=False, fontsize=8)
        for column, label, color in [('peak_gpu_allocated_bytes', 'Piek allocated', '#245A81'), ('peak_gpu_reserved_bytes', 'Piek reserved', '#A7B2BB')]:
            values = pd.to_numeric(r[column], errors='coerce') / 2**30
            if values.notna().any():
                axes[1].plot(r.epoch, values, marker='o', color=color, label=label)
        axes[1].set(xlabel='Epoch', ylabel='GiB', title='GPU-allocator · geen busoverdrachtmeting', xticks=r.epoch, ylim=(0, None))
        if axes[1].lines: axes[1].legend(frameon=False)
        else: _empty(axes[1], 'Geen GPU-geheugenmeting beschikbaar')
        return fig


def audit_table(run):
    records = run.history.get('epochs', [])
    items = [('Rapportstatus', run.report.get('status')),
             ('Metrics consistent met geschiedenis', run.metrics_safe if records else None),
             ('Schaalstaten en bitexponenten consistent', run.scales_safe if records else None),
             ('Checkpointselectie gecontroleerd', run.selection_safe if records else None),
             ('Historische referenties geverifieerd', run.reference_safe if run.references else None),
             ('FP32-masters en Adam-momenten', run.summary.get('fp32_masters_and_moments_verified')),
             ('Eén E4-bron per laag en minibatch', run.summary.get('one_E4_source_per_layer_per_minibatch_verified')),
             ('E2M1-schaal identiek aan E2M3', run.summary.get('all_mantissa_scale_bytes_equal_E2M3')),
             ('Cache leeg vóór optimizerupdates', run.summary.get('source_cache_cleared_before_every_optimizer_update')),
             ('Evaluatie zonder optimizerupdates', all(r.get('test_optimizer_steps') == 0 for r in records) if records else None),
             ('Bronbestanden ongewijzigd', run.report.get('source_unchanged')),
             ('Datasetbestanden ongewijzigd', run.report.get('dataset_files_unchanged')),
             ('Optimizerupdates', run.summary.get('optimizer_steps'))]
    return pd.DataFrame([(key, 'niet gerapporteerd' if value is None else value) for key, value in items],
                        columns=['Controle', 'Opgeslagen resultaat / analyse'])


def export_table(run):
    rows = []
    for role, modes in run.report.get('mx_exports', {}).items():
        for bit, detail in modes.items():
            rows.append({'Rol': role, 'Epoch': detail.get('epoch'), 'Precisie': LABELS.get(int(bit), bit),
                         'Bestand': detail.get('file'), 'Containerbytes': detail.get('bytes'),
                         'Gewichten exact gereconstrueerd': detail.get('decoded_weights_exact'),
                         'MX-masters afwezig': detail.get('six_fp32_masters_absent'),
                         'FP32-schaalproxy afwezig': detail.get('fp32_scale_proxies_absent'),
                         'Afgeleid uit opgeslagen bron': detail.get('derived_from_saved_source'),
                         'Masters gelezen bij afleiding': detail.get('master_weights_read_to_derive'),
                         'Bron-SHA256': detail.get('source_file_sha256')})
    return pd.DataFrame(rows) if rows else pd.DataFrame({'Melding': ['Nog geen exportcontrole beschikbaar.']})


def save_figures(figures, run):
    """Opt-in only: write a new folder, never overwrite raw evidence or old figures."""
    destination = run.path / ('notebook_figures_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(exist_ok=False)
    for name, fig in figures.items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Onveilige figuurnaam.')
        fig.savefig(destination / (name + '.png'), dpi=160, bbox_inches='tight')
        fig.savefig(destination / (name + '.pdf'), bbox_inches='tight')
    provenance = {'purpose': 'derived notebook figures; original evidence left unchanged',
                  'run_directory': str(run.path), 'synthetic_fixture_only': run.synthetic,
                  'inputs_sha256': {str(p): _sha(p) for p in run.input_files},
                  'figures': list(figures)}
    (destination / 'FIGURE_PROVENANCE.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return destination
