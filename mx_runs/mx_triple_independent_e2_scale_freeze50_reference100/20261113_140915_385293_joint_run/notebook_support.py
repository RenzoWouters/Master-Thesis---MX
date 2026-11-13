"""Read-only notebook support for the independent absolute E2-scale freeze-after-50 run through epoch 100.

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

SCHEMA = 'mx-triple-independent-e2-scale-freeze50-reference100-report-v1'
RUN_STEM = 'mx_triple_independent_e2_scale_freeze50_reference100'
FREEZE_EPOCH = 50
BRANCH_END = 100
PARENT_SHA = 'd6c6d7281f39f818ae7f7a61324362858c518ac5e9ebeb37def9b66ea0e1654a'
PARENT_CONTRACT_SHA = '557dfcb42bcaf19be3739690a0980364fda8ac7e4122ed97f33109943c0aa767'
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
    'independent_full100_reference': 'd36e9f83c7d9de07ff17ef1fbcfd893f57f8a6daf95d05ae0cf6e1256bc8bed9',
    'relative_scale10_full100_reference': '1198b78a9ce247fb0ae5c712f0ae63a7975a555e95dffc95bfa772ce531b0f4e',
    'independent_pilot_reference': 'a0d1ea4e5b3fce20ead37b9ed476a050360f966b4055a77cc2d391188e513a5a',
    'fixed_scale64_reference': 'a323762894db83b15af10e79a8aff0128c50f7946805a04517e17a9f627377db',
}
VARIANT_LABELS = {'independent': 'E2-schaal vast na epoch 50',
                  'independent_full100_reference': 'E2-schaal blijft leren',
                  'relative_scale10_full100_reference': '10 relatieve keuzes',
                  'independent_pilot_reference': 'Eerdere onafhankelijke vijfepochpilot',
                  'fixed_scale64_reference': 'Vaste verhouding ×64'}
VARIANT_COLORS = {'independent': '#C98220', 'independent_full100_reference': '#245A81',
                  'relative_scale10_full100_reference': '#8657A6', 'fixed_scale64_reference': '#6D7C86'}
VARIANT_STYLES = {'independent': '-', 'independent_full100_reference': '--',
                  'relative_scale10_full100_reference': ':', 'fixed_scale64_reference': '-.'}
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
    epoch50: dict = field(default_factory=dict)
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
    def reference_complete(self):
        return (self.report.get('status') == 'PASS' and self.last_epoch == BRANCH_END
                and self.report.get('branch_training_completed') is True
                and self.report.get('full_training_completed') is True
                and self.report.get('full_100_epoch_training_completed') is True
                and self.summary.get('branch_training_completed') is True
                and self.summary.get('full_training_completed') is True
                and self.summary.get('full_100_epoch_training_completed') is True
                and self.summary.get('branch_completed_epochs') == BRANCH_END - FREEZE_EPOCH
                and self.metrics_safe and self.selection_safe and self.scales_safe
                and not any(i['Niveau'] in ('FOUT', 'RUNFOUT') for i in self.issues))

    @property
    def branch_best_epoch(self):
        if self.test.empty: return None
        high = self.test[(self.test.weight_bits == 8) & (self.test.epoch > FREEZE_EPOCH)].sort_values('epoch')
        return int(high.loc[high.top1_percent.idxmax(), 'epoch']) if not high.empty else None

    @property
    def branch_epochs(self):
        return max(0, (self.last_epoch or FREEZE_EPOCH) - FREEZE_EPOCH)

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
                        candidates.append((report.get('command') in ('run', 'resume'), report_file.parent.name,
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
    if not run.report: return
    if run.report.get('schema') != SCHEMA:
        raise ValueError('Deze notebook accepteert uitsluitend de freeze-na-50-run tot epoch 100; oude pilotrapporten zijn een ander experiment.')
    expected = {'activation_bits':32, 'master_dtype':'float32', 'mx_block_size':32,
        'mx_scale_format':'E8M0', 'source_weight_bits':8, 'truncation':True,
        'exponent_truncation':True, 'mantissa_truncation':True, 'direct_lower_quantization':False,
        'source_scale_learned':False, 'learned_scale_parameters':4080,
        'base_parameters':132954, 'parameters':137034, 'planned_epochs':100,
        'reference_end_epoch':100, 'branch_stop_epoch':100, 'branch_new_epochs':50,
        'inherited_epochs':50, 'branch_start_epoch':51,
        'scale_learning_end_epoch':50, 'scale_freeze_epoch':50, 'frozen_scale_lr':0.,
        'freeze_absolute_E2_scale':True, 'scale_proxy_dtype':'float32',
        'scale_proxy_bounds':[-127.,127.], 'scale_gradient_normalization':'none', 'scale_lr':.001,
        'independent_E2_shared_scale':True, 'scale_parameterization':'independent_absolute_log2_E2_scale',
        'initial_relative_offset':6, 'initial_relative_factor':64,
        'relative_factor_constrained_after_initialization':False,
        'observed_relative_offset_is_diagnostic_only':True, 'scheduler_milestones':[50,75,90]}
    wrong=[k for k,v in expected.items() if run.config.get(k)!=v]
    if wrong: raise ValueError('Protocolvelden ontbreken of verschillen: '+', '.join(wrong))
    if run.config.get('weight_formats')!={str(k):v for k,v in FORMATS.items()}:
        raise ValueError('Niet de drie afgesproken gewichtsformaten.')
    if {str(k):str(v) for k,v in run.config.get('bn_bank_mapping',{}).items()}!={str(k):v for k,v in BANKS.items()}:
        raise ValueError('Onjuiste BN-bankmapping.')



def _validate_metrics(run, records, csv):
    if not records:
        if not csv.empty:
            raise ValueError('Metrics zonder bijbehorende history.json; niet als resultaat gebruikt.')
        return
    if [r.get('epoch') for r in records] != list(range(1, len(records) + 1)) or not FREEZE_EPOCH <= len(records) <= BRANCH_END:
        raise ValueError('Geschiedenis moet epochs 1–50 uit de ouder en maximaal epochs 51–100 bevatten.')
    required = {'epoch', 'phase', 'weight_bits', 'weight_format', 'activation_bits', 'bn_bank_key',
                'n', 'correct1', 'correct5', 'top1_percent', 'top5_percent', 'objective_loss'}
    if not required <= set(run.metrics):
        raise ValueError('Essentiële metricvelden ontbreken.')
    frame = run.metrics
    if frame.duplicated(['epoch', 'phase', 'weight_bits']).any():
        raise ValueError('Dubbele epoch/fase/precisie.')
    for record in records:
        epoch = record['epoch']
        for key, initial, after in [('lr_used', run.config['lr'], False), ('lr_next', run.config['lr'], True),
                                    ('scale_lr_used', run.config['scale_lr'], False), ('scale_lr_next', run.config['scale_lr'], True)]:
            expected = initial
            for milestone in run.config['scheduler_milestones']:
                if milestone < epoch or (after and milestone == epoch): expected *= run.config['scheduler_gamma']
            if key.startswith('scale_lr') and epoch > FREEZE_EPOCH: expected = 0.0
            if record.get(key) != expected:
                raise ValueError('Opgeslagen leerfactor wijkt af van het ongewijzigde LR-schema: ' + key)
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
    completion_keys = ('full_training_completed', 'full_100_epoch_training_completed', 'branch_training_completed')
    for key in completion_keys:
        values = [container[key] for container in (run.report, run.summary) if key in container]
        if len(values) == 2 and values[0] != values[1]:
            raise ValueError('Voltooiingsstatus verschilt tussen runrapport en samenvatting: ' + key)
    for container in (run.report, run.summary):
        flags = [container[key] for key in completion_keys if key in container]
        if flags and any(value is not flags[0] for value in flags):
            raise ValueError('Voltooiingsvelden binnen één rapport spreken elkaar tegen.')
        for key in completion_keys:
            if container.get(key) is True and len(records) != BRANCH_END:
                raise ValueError('Voltooide training tot epoch 100 geclaimd zonder alle 50 nieuwe epochs: ' + key)
        if 'inherited_epochs' in container and container['inherited_epochs'] != FREEZE_EPOCH:
            raise ValueError('Het aantal geërfde epochs is niet 50.')
        if 'branch_completed_epochs' in container and container['branch_completed_epochs'] != len(records)-FREEZE_EPOCH:
            raise ValueError('Aantal nieuwe freeze-epochs strookt niet met history.')
    if run.report.get('command') != 'check' and run.report.get('status') == 'PASS' and len(records) != BRANCH_END:
        raise ValueError('Training-PASS vereist precies 50 nieuwe epochs tot en met epoch 100.')


def _validate_selection(run):
    if run.test.empty: return
    if not run.selection:
        run.selection_safe=False;run.issue('ONTBREEKT','Checkpointselectie ontbreekt.');return
    roles=[('best',run.best_epoch,'best_E4M3_test'),('last',run.last_epoch,'last_epoch')]
    if run.branch_best_epoch is not None:
        roles.append(('branch_best',run.branch_best_epoch,'branch_best_E4M3_test'))
    for role,epoch,_ in roles:
        detail=run.selection.get(role,{})
        if detail.get('epoch')!=epoch:
            raise ValueError('Checkpointselectie wijkt af van het eerste E4-maximum in het juiste venster: '+role)
        high=run.test[(run.test.epoch==epoch)&(run.test.weight_bits==8)].iloc[0]
        if not math.isclose(detail.get('selection_score_E4M3',float('nan')),high.top1_percent,rel_tol=0,abs_tol=1e-7):
            raise ValueError('Checkpointselectiescore verschilt van de E4-testscore.')
    if (run.selection.get('selected_on_official_test') is not True
        or run.selection.get('independent_test_estimate') is not False
        or run.selection.get('all_modes_from_same_selected_checkpoint') is not True):
        raise ValueError('Checkpointselectie heeft een onjuist evaluatiecontract.')
    expected=pd.concat([run.test[run.test.epoch==e].assign(role=r) for _,e,r in roles])
    if not run.selected.empty:
        _same_frame(expected,run.selected,['role','weight_bits'],['epoch','weight_format','top1_percent','correct1','n'],'selected_metrics.csv')
    else:run.issue('ONTBREEKT','selected_metrics.csv ontbreekt; gecontroleerde history wordt gebruikt.')



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
    baseline=run.scale_blocks[run.scale_blocks.epoch==FREEZE_EPOCH].set_index(['layer','output_channel','block_index']).sort_index()
    for epoch in range(FREEZE_EPOCH+1,len(records)+1):
        now=run.scale_blocks[run.scale_blocks.epoch==epoch].set_index(['layer','output_channel','block_index']).sort_index()
        for column in ('phi','e2_exponent','e2_scale_code'):
            if not np.array_equal(now[column].to_numpy(),baseline[column].to_numpy()):
                raise ValueError('Bevroren absolute E2-schaal gewijzigd na epoch50: '+column)
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
            if epoch > FREEZE_EPOCH:
                if row.get('scale_gradients_enabled') is not False:
                    raise ValueError('Schaalgradienten moeten na50 expliciet als uitgeschakeld zijn gemarkeerd.')
                for key in ('phi_changed_blocks','phi_update_l2','net_exponent_changes','exponent_flips_during_epoch',
                            'exponent_jump_distance_during_epoch','max_exponent_jump','unique_blocks_switched_during_epoch',
                            'projected_values_during_epoch','scale_grad_l2_min','scale_grad_l2_max','scale_grad_nonzero_blocks_max','scale_lr'):
                    if row.get(key) != 0:
                        raise ValueError('Niet-nul freeze-diagnostiek na epoch50: '+key)
            rows.append(row)
    run.scale_epochs = pd.DataFrame(rows)
    if not epochs_csv.empty:
        fields = ['blocks', 'phi_min', 'phi_max', 'phi_mean', 'phi_std', 'phi_changed_blocks',
                  'phi_update_l2', 'net_exponent_changes', 'exponent_flips_during_epoch',
                  'unique_blocks_switched_during_epoch', 'scale_grad_l2_min', 'scale_grad_l2_max',
                  'projected_values_during_epoch']
        _same_frame(run.scale_epochs, epochs_csv, ['epoch', 'layer'], fields, 'scale_epoch_metrics.csv')


def _validate_timing(run,records):
    if not run.metrics_safe:
        run.times=pd.DataFrame()
        return
    rows=[]
    for record in records:
        keys=('train_seconds','test_seconds','checkpoint_seconds','other_seconds','total_epoch_seconds','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')
        row={'epoch':record['epoch'],**{k:record.get(k) for k in keys}}
        if any(_number(row[k]) is None or row[k]<0 for k in keys[:5]):
            run.issue('FOUT','Ontbrekende of ongeldige epochtijd; tijdgrafiek weggelaten.');run.times=pd.DataFrame();return
        row['segment']='overgenomen' if record['epoch']<=FREEZE_EPOCH else 'nieuwe freeze-epoch'
        rows.append(row)
    run.times=pd.DataFrame(rows)
    if run.timing.get('measured_100_epoch_seconds') is not None:
        run.issue('FOUT','Deze run voert maximaal 50 nieuwe epochs uit; 100 nieuw gemeten epochs mogen niet worden geclaimd.')
    if rows:
        combined=sum(r['total_epoch_seconds'] for r in rows)
        inherited=sum(r['total_epoch_seconds'] for r in rows if r['epoch']<=FREEZE_EPOCH)
        branch=sum(r['total_epoch_seconds'] for r in rows if r['epoch']>FREEZE_EPOCH)
        expected_fields=[('measured_completed_epoch_seconds',branch),
            ('measured_new_branch_epoch_seconds',branch), ('measured_branch_epoch_seconds',branch),
            ('measured_inherited_epoch_seconds',inherited),
            ('combined_inherited_and_branch_epoch_seconds',combined)]
        if len(records)==BRANCH_END:
            expected_fields += [('measured_50_epoch_branch_seconds',branch),
                ('remaining_epochs_to100',0),('estimated_remaining_to100_seconds',0)]
        elif run.timing.get('measured_50_epoch_branch_seconds') is not None:
            run.issue('FOUT','Gemeten tijd voor 50 nieuwe epochs geclaimd voordat epoch 100 is voltooid.')
        for key,expected in expected_fields:
            if key in run.timing and (run.timing[key] is None or not math.isclose(float(run.timing[key]),expected,abs_tol=1e-6,rel_tol=1e-10)):
                run.issue('FOUT','Tijdrapport strookt niet met het juiste epochvenster: '+key)



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
    for file, digest in meta.get('files_sha256', meta.get('files', {})).items():
        path = (directory / file).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or _sha(path) != digest:
            raise ValueError('Historisch referentiebestand gewijzigd: ' + name + '/' + file)
    report = _read_json(directory / 'run_report.json')
    c = report.get('config', {})
    if report.get('status') != 'PASS' or report.get('synthetic_fixture_only') is not False:
        raise ValueError('Historische referentie is geen geslaagde echte run.')
    pilot = name == 'independent_pilot_reference'
    epochs = 5 if pilot else 100
    frame = pd.read_csv(directory / 'epoch_metrics.csv', float_precision='round_trip')
    frame = frame[(frame.phase == 'test') & frame.epoch.between(1, epochs)].copy()
    if frame.duplicated(['epoch', 'weight_bits']).any() or len(frame) != epochs * 3:
        raise ValueError(f'Historische referentie bevat niet alle {epochs} epochs × drie precisies.')
    for epoch in range(1, epochs + 1):
        if set(frame[frame.epoch == epoch].weight_bits) != set(BITS):
            raise ValueError('Historische precisies onvolledig.')
    for row in frame.to_dict('records'):
        if row['weight_format'] != FORMATS[row['weight_bits']] or row['activation_bits'] != 32 or row['n'] != 26032:
            raise ValueError('Historisch precisie- of datasetcontract wijkt af.')
        if abs(row['top1_percent'] - 100 * row['correct1'] / row['n']) >= 1e-4:
            raise ValueError('Historische accuracy en correcte aantallen verschillen.')
    history = _read_json(directory / 'history.json') if (directory / 'history.json').is_file() else None
    if history is not None:
        history_frame, records = _history_frame(history)
        history_frame = history_frame[history_frame.phase == 'test']
        _same_frame(history_frame, frame, ['epoch', 'phase', 'weight_bits'],
                    ['weight_format', 'activation_bits', 'n', 'correct1', 'top1_percent'], name + '/epoch_metrics.csv')
    if name == 'independent_full100_reference':
        if (report.get('schema') != 'mx-triple-independent-e2-scale-reference100-report-v1'
                or c.get('scale_learning_end_epoch') != 100 or c.get('scale_freeze_epoch') is not None
                or report.get('full_training_completed') is not True
                or meta.get('parent_checkpoint_sha256') != PARENT_SHA
                or meta.get('parent_contract_sha256') != PARENT_CONTRACT_SHA):
            raise ValueError('De ouderreferentie is niet de vastgelegde onafhankelijke E2-run.')
    elif name == 'relative_scale10_full100_reference':
        if (c.get('scale_proxy_init') != 6.0 or c.get('scale_exponent_offset_choices') != list(range(10))
                or c.get('scale_learning_end_epoch') != 100 or c.get('scale_freeze_epoch') is not None
                or report.get('full_training_completed') is not True):
            raise ValueError('De relatieve referentie moet alle 100 epochs met tien keuzes en start ×64 leren.')
    elif pilot:
        if (report.get('schema') != 'mx-triple-independent-e2-scale-pilot5-report-v1'
                or report.get('pilot_completed') is not True or c.get('independent_E2_shared_scale') is not True
                or c.get('initial_relative_factor') != 64):
            raise ValueError('De pilotherkomst is niet de geslaagde onafhankelijke E2-proef.')
    elif meta.get('factor') != 64 or report.get('schema') != 'mx-triple-scale64-reference100-report-v1':
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
        if history is not None:
            for record in run.history['epochs'][:epochs]:
                old = history['epochs'][record['epoch'] - 1]
                for phase in ('train', 'test'):
                    if record.get('coverage', {}).get(phase, {}).get('order_sha256') != old.get('coverage', {}).get(phase, {}).get('order_sha256'):
                        raise ValueError('Samplevolgorde verschilt van historische referentie: ' + name)
        comparable = True
    run.references[name] = {'path': directory, 'report': report, 'metadata': meta,
                            'test': frame, 'history': history, 'comparable': comparable}
    run.input_files.append(marker)


def load_run(run_dir=None):
    run = RunAnalysis(resolve_run_dir(run_dir))
    json_files = {'run_report.json': 'report', 'history.json': 'history',
                  'training_summary.json': 'summary', 'checkpoint_selection.json': 'selection',
                  'timing_estimate.json': 'timing', 'scale_learning_summary.json': 'scale_summary',
                  'epoch50_checkpoint.json': 'epoch50'}
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
    for name in ('independent_full100_reference', 'relative_scale10_full100_reference', 'fixed_scale64_reference'):
        try:
            _load_reference(run, name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            run.reference_safe = False; run.issue('FOUT', str(exc))
    try:
        _validate_freeze_link(run)
    except (ValueError, KeyError, TypeError) as exc:
        run.metrics_safe=False;run.scales_safe=False;run.reference_safe=False;run.issue('FOUT',str(exc))
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
    status=run.report.get('status','NOG NIET UITGEVOERD')
    tag='**SYNTHETISCHE SOFTWARETEST — geen onderzoeksresultaat.**\n\n' if run.synthetic else ''
    if run.last_epoch is None:
        completed='Nog geen gecontroleerde epochresultaten van deze run beschikbaar.'
    else:
        completed=(f'Epochs 1–50 zijn overgenomen uit de ouder; **{run.branch_epochs} van de 50 nieuwe epochs** '
            f'zijn uitgevoerd tot en met epoch {run.last_epoch}.')
    if run.reference_complete:
        quality='De voortzetting tot epoch 100 is consistent afgerond: 50 geërfde en 50 nieuw uitgevoerde epochs.'
    elif run.last_epoch is None:
        quality='Het geplande eindpunt is epoch 100. Er worden nog geen accuracies ingevuld.'
    elif run.last_epoch < BRANCH_END:
        quality='Dit zijn gedeeltelijke resultaten; de voortzetting tot epoch 100 is nog niet voltooid.'
    else:
        quality='Epoch 100 is aanwezig, maar de eindcontrole is nog niet bevestigd; raadpleeg de controles.'
    return (f'{tag}**Status: {status}.** {completed} {quality}\n\n'
        f'**Gelezen resultatenmap:** `{run.path}`\n\n'
        'Deze proef herstelt het oorspronkelijke checkpoint van epoch 50. Vanaf epoch 51 blijven φ en de absolute E2-schaal '
        'vast; gewichten en BN leren verder. De E4-bronschaal blijft met de gewichten veranderen. '
        'De bestaande doorleerrun is de controle en wordt niet opnieuw getraind. '
        'De vijf eerdere freeze-pilotepochs worden niet overgenomen: deze run berekent epochs 51–100 opnieuw.')



def protocol_table(run):
    return pd.DataFrame([
        ('Model / dataset','CNN-8 (svhnq, expand=2) / SVHN train + test; geen extra-split'),
        ('Precisies','E4M3 → E2M3 → E2M1; alle activaties A32; FP32-masters'),
        ('Groepen','B32; 4.080 groepen in zes MX-lagen'),
        ('Start','Volledig herstel van oudercheckpoint50: gewichten, φ, Adam, scheduler en RNG'),
        ('Nieuwe training','50 nieuwe epochs: 51–100; drie recursieve passes en één Adam-update per minibatch'),
        ('E4-schaal','Dynamische amax-berekening uit actuele gewichten; geen aparte geleerde parameter'),
        ('E2-schaal','De absolute schaal uit epoch50 blijft vast; gedeeld door E2M3 en E2M1'),
        ('Freeze','φ.requires_grad=False; gradient afwezig; schaal-LR0; schaal-Adam-momenten en staptellers behouden'),
        ('Gewichten / BN','Leren verder; weight-LR 0,0001 voor 51–75, 0,00001 voor 76–90 en 0,000001 voor 91–100'),
        ('Exponentcodes','E4M3 s e3 e2 e1 e0 m2 m1 m0 → E2M3 s e3 e2 m2 m1 m0'),
        ('Mantissacodes','E2M1-code = E2M3-code >>2; dezelfde absolute E2-schaal'),
        ('Verhouding E2/E4','Diagnose achteraf; kan nog veranderen door beweging van E4'),
        ('Checkpointselectie','Gemeenschappelijk eindpunt; beste E4 binnen1..laatste; aanvullend beste E4 alleen51..laatste'),
        ('Uitvoering','FP32 convoluties met gesimuleerde quantisatie; geen native low-bit snelheidsclaim')],columns=['Onderdeel','Afspraak'])



def _variants(run,context=True):
    if run.test.empty:return {}
    result={}
    if not run.synthetic and run.reference_safe:
        names=['independent_full100_reference']
        if context:names+=['fixed_scale64_reference','relative_scale10_full100_reference']
        for name in names:
            r=run.references.get(name)
            if r and r['comparable']:result[name]=r['test'][r['test'].epoch<=run.last_epoch].copy()
    result['independent']=run.test
    return result



def comparison_markdown(run):
    if run.test.empty:return 'Vergelijkingen verschijnen na consistente nieuwe freeze-epochs. Ontbrekende resultaten zijn geen 0%-scores.'
    if run.synthetic:return '**Synthetische fixture:** er worden geen historische SVHN-accuracies naast de testwaarden gezet.'
    return (f'De hoofdvergelijking is **E2 vast na50 tegenover E2 doorleren**, met exact hetzelfde checkpoint50 als vertrekpunt. '
        f'Alle varianten worden getoond binnen epochs **1–{run.last_epoch}**. De freezevariant en haar doorleercontrole delen de eerste 50 epochs als voorgeschiedenis. '
        f'Het eindpunt is **epoch {run.last_epoch} tegenover epoch {run.last_epoch}**; historische latere epochs worden niet in een gedeeltelijke vergelijking gebruikt. '
        'Vaste ×64 en tien relatieve keuzes staan erbij als historische context.\n\n'
        'Het globale beste E4-checkpoint kan vóór de freeze liggen en daardoor geen effect van de nieuwe ingreep tonen. '
        'Daarom wordt daarnaast het beste E4-checkpoint binnen de nieuwe epochs51–laatste getoond. Per gekozen checkpoint '
        'komen alle drie de precisies uit dezelfde epoch.\n\n'
        'Basisinitialisatie, protocol en beschikbare samplevolgordes zijn gecontroleerd. De oorspronkelijke CUDA-backend '
        'is niet bitexact; één vertakking bewijst geen statistisch significant voordeel. De officiële testset wordt ook '
        'voor checkpointselectie gebruikt, zodat de geselecteerde score geen onafhankelijke testschatting is.')



def checkpoint_table(run,role='last',context=True):
    if not run.selection_safe:return pd.DataFrame({'Melding':['Checkpointselectie niet bevestigd.']})
    if role not in ('last','best','branch_best'):raise ValueError('Onbekende checkpointrol.')
    rows=[]
    for name,frame in _variants(run,context).items():
        if role=='branch_best':frame=frame[frame.epoch>FREEZE_EPOCH]
        if frame.empty:continue
        if role in ('best','branch_best'):
            high=frame[frame.weight_bits==8].sort_values('epoch');epoch=int(high.loc[high.top1_percent.idxmax(),'epoch'])
        else:epoch=int(frame.epoch.max())
        label={'last':'gemeenschappelijk eindpunt','best':'eerste beste E4 in1..laatste','branch_best':'eerste beste E4 in51..laatste'}[role]
        record={'Variant':VARIANT_LABELS[name],'Variant_key':name,'Epoch':epoch,'Selectie':label}
        for bit in BITS:
            row=frame[(frame.epoch==epoch)&(frame.weight_bits==bit)].iloc[0]
            record[LABELS[bit]+' testaccuracy [%]']=row.top1_percent
            record[LABELS[bit]+' correct']=int(row.correct1)
        rows.append(record)
    return pd.DataFrame(rows) if rows else pd.DataFrame({'Melding':['Nog geen vergelijkbare checkpointresultaten.']})



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
                         'Freeze na50 [%]': new.loc[bit, 'top1_percent'],
                         'Referentie [%]': old.loc[bit, 'top1_percent'],
                         'Verschil [procentpunt]': new.loc[bit, 'top1_percent'] - old.loc[bit, 'top1_percent'],
                         'Verschil correct [beelden]': int(new.loc[bit, 'correct1'] - old.loc[bit, 'correct1'])})
    return pd.DataFrame(rows) if rows else pd.DataFrame({'Melding': ['Geen gevalideerde historische comparator.']})


def reference_table(run):
    roles={'independent_full100_reference':'hoofdcontrole; zelfde oudercheckpoint50',
        'fixed_scale64_reference':'historische context op hetzelfde epochvenster',
        'relative_scale10_full100_reference':'historische context op hetzelfde epochvenster'}
    return pd.DataFrame([{'Referentie':VARIANT_LABELS[name],
        'Bronrun':r['metadata'].get('source_run',r['metadata'].get('source_run_directory')),
        'Beschikbare epochs':100,'Rol':roles.get(name,'herkomst'),
        'Marker SHA256':_sha(r['path']/'REFERENCE.json'),'Vergelijking gevalideerd':r['comparable']}
        for name,r in run.references.items()])



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
        fig,axes=plt.subplots(1,3,figsize=(15,5.1),constrained_layout=True)
        fig.suptitle(_title(run,'Volledig accuracyverloop · eerste50 epochs gedeeld met de ouder'),fontsize=15,fontweight='bold')
        variants=_variants(run)
        if not variants:
            for ax in axes:_empty(ax)
            return fig
        for bit,ax in zip(BITS,axes):
            values=[]
            for name,frame in variants.items():
                r=frame[frame.weight_bits==bit].sort_values('epoch')
                if name=='independent' and not run.synthetic:r=r[r.epoch>=FREEZE_EPOCH]
                values.extend(r.top1_percent)
                ax.plot(r.epoch,r.top1_percent,linestyle=VARIANT_STYLES[name],color=VARIANT_COLORS[name],
                    linewidth=2.5 if name=='independent' else 1.6,label=VARIANT_LABELS[name])
            ax.axvspan(.5,FREEZE_EPOCH+.5,color='#DDE3E8',alpha=.28,zorder=-1)
            ax.set(title=LABELS[bit]+' / A32',xlabel='Epoch',ylabel='Testaccuracy [%] · ingezoomd',
                xticks=_epoch_ticks(run.last_epoch),ylim=_limits(values))
            _mark_schedule(ax,run.last_epoch)
            ax.legend(fontsize=8,frameon=False,loc='lower right')
        return fig



def plot_checkpoints(run,role='last'):
    """One precision per panel; categorical variant axis and an annotated circle per score."""
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(17,5.2),constrained_layout=True)
        title={'last':f'Accuracy bij gemeenschappelijke epoch {run.last_epoch or "…"}',
            'best':f'Alle modi op het beste E4-checkpoint binnen epochs1–{run.last_epoch or "…"}',
            'branch_best':f'Alle modi op het beste E4-checkpoint binnen epochs51–{run.last_epoch or "…"}'}[role]
        fig.suptitle(_title(run,title),fontsize=15,fontweight='bold')
        table=checkpoint_table(run,role)
        if 'Variant' not in table:
            for ax in axes:_empty(ax,'Nog geen bevestigde checkpointvergelijking')
            return fig
        labels=[row.Variant+f' · ep{row.Epoch}' for row in table.itertuples()]
        y=np.arange(len(table))
        for bit,ax in zip(BITS,axes):
            values=table[LABELS[bit]+' testaccuracy [%]'].to_numpy(float)
            lo,hi=_limits(values,span=.65)
            for i,(value,key) in enumerate(zip(values,table.Variant_key)):
                ax.axhspan(i-.42,i+.42,color='#F1F4F7' if i%2==0 else 'white',zorder=0)
                ax.scatter(value,i,marker='o',s=72 if key=='independent' else 54,color=COLORS[bit],zorder=3)
                label_left=value>lo+.75*(hi-lo)
                ax.annotate(f'{value:.3f}%'.replace('.',','),(value,i),xytext=(-7 if label_left else 7,0),textcoords='offset points',
                    va='center',ha='right' if label_left else 'left',fontsize=10,fontweight='bold' if key=='independent' else 'normal',
                    color='#243A49',clip_on=False)
            # Leave data-coordinate space for value labels; all actual scores remain visible.
            ax.set(title=LABELS[bit]+' / A32',yticks=y,yticklabels=labels,xlabel='Testaccuracy [%] · ingezoomd',
                xlim=(lo,min(100,max(hi,float(values.max())+.40*(hi-lo)))),ylim=(len(table)-.5,-.5))
            ax.grid(axis='y',visible=False)
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
        fig,axes=plt.subplots(1,2,figsize=(14,5.4),constrained_layout=True)
        fig.suptitle(_title(run,'Schaalverdeling · dezelfde vertrekstaat op epoch50'),fontsize=15,fontweight='bold')
        if not _scale_available(run):
            for ax in axes:_empty(ax,'Geen gecontroleerde schaalstaten')
            return fig
        parts=[('Checkpoint50',run.scale_blocks[run.scale_blocks.epoch==FREEZE_EPOCH],'#A7B2BB'),
            (f'Freeze · epoch{run.last_epoch}',run.scale_blocks[run.scale_blocks.epoch==run.last_epoch],'#C98220')]
        control=_control_scale_states(run,run.last_epoch)
        if control is not None:parts.append((f'Doorleren · epoch{run.last_epoch}',control,'#245A81'))
        for ax,column,title in zip(axes,['e2_exponent','relative_offset'],
            ['Absolute E2-exponent · blijft vast in freeze','Geobserveerde verhouding S₂/S₄ · mag veranderen']):
            keys=sorted(set(k for _,part,_ in parts for k in part[column]))
            x=np.arange(len(keys));width=.78/len(parts)
            for i,(label,part,color) in enumerate(parts):
                counts=part[column].value_counts();values=[100*counts.get(k,0)/len(part) for k in keys]
                ax.bar(x+(i-(len(parts)-1)/2)*width,values,width=width*.94,color=color,label=label)
            labels=[str(k) if column=='e2_exponent' else _ratio_label(k) for k in keys]
            ax.set(title=title,ylabel='Aandeel van4.080 groepen [%]',xticks=x,xticklabels=labels,ylim=(0,105))
            ax.legend(frameon=False,fontsize=9)
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
            ticks = _epoch_ticks(int(table.columns.max()), zero=True)
            positions = [list(table.columns).index(tick) for tick in ticks if tick in table.columns]
            ax.set(title=title, xlabel='Epoch', xticks=positions, xticklabels=[int(table.columns[pos]) for pos in positions],
                   yticks=range(len(labels)), yticklabels=labels)
            ax.grid(False)
            if values.size <= 160:
                for (y, x), value in np.ndenumerate(values):
                    if value > 0:
                        ax.text(x, y, f'{value:.1f}', ha='center', va='center', fontsize=8,
                                color='white' if value > 50 else '#263B4A')
            for milestone in (50, 75, 90):
                if milestone in table.columns:
                    ax.axvline(list(table.columns).index(milestone) + .5, color='#A7B2BB', linestyle='--', linewidth=.7)
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
            axes[0].plot(r.epoch, r.phi, marker='o', markevery=max(1, len(r) // 10), color=color, label=layer)
            axes[1].plot(r.epoch, r.relative_offset, marker='o', markevery=max(1, len(r) // 10), color=color, label=layer)
        axes[0].set(title='Absolute log₂-proxy φ · vast vanaf51', ylabel='Gemiddelde φ')
        axes[1].set(title='Afgeleide Δ=e₂−e₄', ylabel='Gemiddelde geobserveerde exponentafstand')
        for ax in axes:
            ax.set(xlabel='Epoch', xticks=_epoch_ticks(int(frame.epoch.max()), zero=True))
            _mark_schedule(ax, int(frame.epoch.max()))
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
        long = len(table) > 15
        for i, (column, color) in enumerate(zip(table.columns[1:], ['#98A5B0', '#245A81', '#C98220'])):
            if long:
                axes[0].plot(table.Epoch, table[column], label=column, color=color)
            else:
                axes[0].bar(table.Epoch + (i - 1) * .24, table[column], width=.23, label=column, color=color)
        axes[0].set(title='Verschil tussen opeenvolgende epochs', ylabel='Aantal groepen met andere harde waarde', xlabel='Epoch',
                    xticks=_epoch_ticks(int(table.Epoch.max())), ylim=(0,None))
        axes[0].legend(frameon=False, fontsize=8)
        _mark_schedule(axes[0], int(table.Epoch.max()))
        if not run.scale_epochs.empty:
            audit = run.scale_epochs.groupby('epoch')[['exponent_flips_during_epoch', 'unique_blocks_switched_during_epoch']].sum()
            axes[1].plot(audit.index, audit.exponent_flips_during_epoch, marker='o', markevery=max(1,len(audit)//10), color='#245A81', label='Alle E2-wissels, inclusief terugwissels')
            axes[1].plot(audit.index, audit.unique_blocks_switched_during_epoch, marker='s', markevery=max(1,len(audit)//10), color='#C98220', label='Unieke groepen met ≥1 E2-wissel')
            axes[1].set(title='E2-schaalwissels tijdens de updates', xlabel='Epoch', ylabel='Aantal', xticks=_epoch_ticks(int(audit.index.max())), ylim=(0,None))
            axes[1].legend(frameon=False, fontsize=8)
            _mark_schedule(axes[1], int(audit.index.max()))
        else:
            _empty(axes[1], 'Geen diagnostiek per update beschikbaar')
        return fig


def plot_scale_gradients(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(14,4.9),constrained_layout=True)
        fig.suptitle(_title(run,'Schaaloptimalisatie stopt na50 · geen nieuwe φ-gradient opgeslagen'),fontsize=15,fontweight='bold')
        if not run.scales_safe or run.scale_epochs.empty:
            for ax in axes:_empty(ax,'Geen gecontroleerde schaaldiagnostiek')
            return fig
        for layer,color in zip(LAYER_SHAPES,LAYER_COLORS):
            r=run.scale_epochs[(run.scale_epochs.layer==layer)&(run.scale_epochs.epoch<=FREEZE_EPOCH)].sort_values('epoch')
            axes[0].plot(r.epoch,r.scale_grad_l2_max,color=color,label=layer)
            axes[1].plot(r.epoch,r.projected_values_during_epoch,color=color,label=layer)
        axes[0].set(title='Maximale opgeslagen L2-gradientnorm · alleen leerfase',ylabel='Schaalgradientnorm per laag')
        axes[0].set_yscale('symlog',linthresh=1e-8)
        axes[1].set(title='Grensprojecties · alleen leerfase',ylabel='Geprojecteerde parameterwaarden',ylim=(0,None))
        for ax in axes:
            ax.axvspan(FREEZE_EPOCH+.5,(run.last_epoch or FREEZE_EPOCH)+.5,color='#F4E4CF',alpha=.7,zorder=-1)
            ax.text(.98,.94,'Na50 uitgeschakeld\nφ.grad = None',transform=ax.transAxes,ha='right',va='top',fontsize=9,
                bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
            ax.set(xlabel='Epoch',xticks=_epoch_ticks(run.last_epoch),xlim=(.5,run.last_epoch+.5))
            _mark_schedule(ax,run.last_epoch)
            ax.legend(fontsize=8,ncol=2,frameon=False,loc='center left')
        return fig



def storage_cost_table(run=None):
    rows=[]
    for name,extra_bits,payload in [('Vaste ×64',0,0),('10 relatieve keuzes',4,TOTAL_BLOCKS*16),
        ('E2 blijft leren',8,TOTAL_BLOCKS*16),('E2 vast na50',8,TOTAL_BLOCKS*12)]:
        extra=TOTAL_BLOCKS*extra_bits//8
        rows.append({'Variant':name,'Groepen':TOTAL_BLOCKS,'E4-schaalbytes':TOTAL_BLOCKS,
            'Extra schaal/keuzebytes':extra,'Totaal schaalinformatie [bytes]':TOTAL_BLOCKS+extra,
            'Schaalinformatie [bit per B32-positie]':(8+extra_bits)/32,
            'E4-codes + schaalinformatie, zes MX-lagen [bytes]':TOTAL_BLOCKS*32+TOTAL_BLOCKS+extra,
            'Nominale bewaarde FP32-schaalhulppayload [bytes]':payload})
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
        axes[1].bar(x, values, color=['#A7B2BB', '#8657A6', '#245A81', '#C98220'])
        for i, value in enumerate(values):
            axes[1].text(i, value + 1.1, f'{value:.2f}', ha='center')
        axes[1].set(title='E4-broncodes + metadata voor alle modi', ylabel='KiB · zes MX-lagen, inclusief padding', ylim=(0, max(values) * 1.15))
        for ax in axes:
            ax.set(xticks=x, xticklabels=['Vaste ×64', '10 keuzes', 'E2 leert', 'E2 vast na50'])
        return fig


def timing_markdown(run):
    if not run.metrics_safe or (not run.times.empty and run.last_epoch is None):
        return 'Geen betrouwbare epochtijdweergave: de epochgeschiedenis is niet consistent. Raadpleeg de controles.'
    if run.times.empty:return 'Nog geen gemeten freeze-epochtijden. De softwarecheck is geen SVHN-tijdmeting.'
    branch=run.times[run.times.epoch>FREEZE_EPOCH]
    inherited=run.times[run.times.epoch<=FREEZE_EPOCH]
    text=(f'**Nieuw gemeten:** {branch.total_epoch_seconds.sum()/60:.2f} minuten voor {len(branch)} freeze-epochs. '
        f'De {inherited.total_epoch_seconds.sum()/60:.2f} minuten van epochs 1–50 zijn overgenomen uit de ouder '
        'en zijn hier niet opnieuw uitgevoerd.')
    if run.last_epoch == BRANCH_END:
        text+=(' **Alle 50 nieuwe epochs tot en met epoch 100 zijn gemeten; er is geen resterende tijdprojectie.** '
            f'De historische en nieuwe epochtijden tellen samen op tot {run.times.total_epoch_seconds.sum()/60:.2f} minuten. '
            'Dat gecombineerde getal beschrijft het volledige traject met hergebruikte voorgeschiedenis, niet 100 opnieuw uitgevoerde epochs.')
    elif not branch.empty:
        remaining=BRANCH_END-run.last_epoch
        stored_median=_number(run.timing.get('median_branch_epoch_seconds'))
        median=stored_median if stored_median is not None else float(branch.total_epoch_seconds.median())
        projection=median*remaining/60
        text+=(f' Op basis van de mediane freeze-epochtijd is de resterende duur voor **{remaining} epochs tot 100** '
            f'naar schatting **{projection:.1f} minuten**. Dit is een extrapolatie, geen gemeten eindduur.')
    else:
        text+=' Er zijn nog geen nieuwe freeze-epochtijden voor een resterende-duurschatting.'
    return text+' Voorbereiding, gegevens laden en laatste rapportexport vallen buiten de epochtijden.'



def plot_timing(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        fig.suptitle(_title(run, 'Gemeten tijd en GPU-geheugen'), fontsize=15, fontweight='bold')
        if not run.metrics_safe or run.times.empty:
            for ax in axes: _empty(ax, 'Nog geen epochtijden')
            return fig
        r = run.times[run.times.epoch > FREEZE_EPOCH]
        if r.empty:
            for ax in axes: _empty(ax, 'Nog geen nieuwe freeze-epochtijden')
            return fig
        bottom = np.zeros(len(r))
        for column, label, color in [('train_seconds', 'Training (3 precisies)', '#245A81'), ('test_seconds', 'Test (3 precisies)', '#33846D'),
                                     ('checkpoint_seconds', 'Checkpoint', '#C98220'), ('other_seconds', 'Overige epochdiagnostiek', '#A7B2BB')]:
            axes[0].bar(r.epoch, r[column], bottom=bottom, color=color, label=label)
            bottom += r[column].to_numpy()
        axes[0].set(xlabel='Epoch', ylabel='Seconden', title='Alleen nieuw gemeten freeze-epochs', xticks=_branch_ticks(int(r.epoch.max())))
        axes[0].legend(frameon=False, fontsize=8)
        for column, label, color in [('peak_gpu_allocated_bytes', 'Piek allocated', '#245A81'), ('peak_gpu_reserved_bytes', 'Piek reserved', '#A7B2BB')]:
            values = pd.to_numeric(r[column], errors='coerce') / 2**30
            if values.notna().any():
                axes[1].plot(r.epoch, values, marker='o', color=color, label=label)
        axes[1].set(xlabel='Epoch', ylabel='GiB', title='GPU-allocator · geen busoverdrachtmeting', xticks=_branch_ticks(int(r.epoch.max())), ylim=(0, None))
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
             ('Optimizerupdates', run.summary.get('optimizer_steps')),
             ('Epoch50-hervatcheckpoint bewaard', run.epoch50.get('available')),
             ('Schaalfreeze uitgevoerd', run.config.get('scale_freeze_epoch') is not None if run.config else None)]
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



def _epoch_ticks(last, zero=False):
    last = int(last)
    first = 0 if zero else 1
    if last <= 10:
        return list(range(first, last + 1))
    return sorted({first, last, *range(10, last + 1, 10), *[v for v in (50,75,90) if v <= last]})


def _branch_ticks(last, include_start=False):
    first = FREEZE_EPOCH if include_start else FREEZE_EPOCH + 1
    if last - first <= 10:
        return list(range(first, last + 1))
    return sorted({first, last, *range(60, last + 1, 10),
                   *[value for value in (75, 90) if first <= value <= last]})


def _mark_schedule(ax,last,first=0):
    if first <= FREEZE_EPOCH <= last:
        ax.axvline(FREEZE_EPOCH+.5,color='#9B5E15',linewidth=1.1,linestyle='--',zorder=1)



def plot_final_window(run,window=5):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(15,5.1),constrained_layout=True)
        fig.suptitle(_title(run,f'Vertakking vanaf hetzelfde checkpoint 50 · tot epoch {run.last_epoch or "…"}'),fontsize=15,fontweight='bold')
        variants={name:frame[frame.epoch>=FREEZE_EPOCH] for name,frame in _variants(run,context=False).items()}
        if not variants:
            for ax in axes:_empty(ax)
            return fig
        for bit,ax in zip(BITS,axes):
            values=[]
            for name,frame in variants.items():
                r=frame[frame.weight_bits==bit].sort_values('epoch');values.extend(r.top1_percent)
                ax.plot(r.epoch,r.top1_percent,linestyle=VARIANT_STYLES[name],color=VARIANT_COLORS[name],
                    marker='o',markersize=4,linewidth=2,label=VARIANT_LABELS[name])
            ax.set(title=LABELS[bit]+' / A32',xlabel='Epoch',ylabel='Testaccuracy [%] · extra zoom',
                xticks=_branch_ticks(run.last_epoch,include_start=True),ylim=_limits(values,span=.6))
            _mark_schedule(ax,run.last_epoch,FREEZE_EPOCH)
            ax.legend(frameon=False,fontsize=9,loc='lower right')
        return fig



def pilot_evidence_table(run):
    return pd.DataFrame({'Melding':['Deze freezeproef gebruikt het volledige oudercheckpoint50; de eerdere pilot is geen startpunt.']})



def pilot_comparison_table(run):
    return pd.DataFrame({'Melding':['Deze freezeproef gebruikt het volledige oudercheckpoint50; de eerdere pilot is geen startpunt.']})



def _validate_freeze_link(run):
    if not run.history.get('epochs'):return
    records=run.history['epochs']
    parent=run.history.get('parent_checkpoint',{})
    if parent.get('epoch')!=FREEZE_EPOCH:
        raise ValueError('Trace is niet gekoppeld aan oudercheckpoint50.')
    if not run.synthetic:
        for key,expected in [('sha256',PARENT_SHA),('contract_sha256',PARENT_CONTRACT_SHA),
                             ('reference_sha256',REFERENCE_MARKERS['independent_full100_reference'])]:
            if parent.get(key)!=expected:
                raise ValueError('Ouderverwijzing wijkt af van de vastgelegde bron: '+key)
        reference=run.references.get('independent_full100_reference')
        if not reference or not reference['comparable']:
            raise ValueError('De eigen doorleercontrole is niet beschikbaar of niet vergelijkbaar.')
        old=reference['history']['epochs']
        for index,record in enumerate(records[:FREEZE_EPOCH]):
            if record!=old[index]:
                raise ValueError('Geërfde epoch1..50 is gewijzigd ten opzichte van de bron: '+str(index+1))
    for record in records[FREEZE_EPOCH:]:
        audit=record.get('freeze_checks',{})
        for key in ('phi_unchanged','hard_E2_scales_unchanged','scale_optimizer_state_unchanged','scale_gradients_absent'):
            if audit.get(key) is not True:
                raise ValueError('Freezecontrole ontbreekt of is niet geslaagd: '+key)
        gradient=record.get('gradient_checks',{})
        expected_steps=math.ceil(run.config['train_examples']/run.config['batch_size'])
        if (gradient.get('scale_gradients_enabled') is not False
                or gradient.get('scale_optimizer_steps') != 0
                or gradient.get('frozen_scale_checks') != expected_steps):
            raise ValueError('Nieuwe epochs vereisen expliciet uitgeschakelde schaalgradients en freezecontrole bij iedere update.')


def freeze_evidence_table(run):
    if not _scale_available(run) or run.last_epoch is None:
        return pd.DataFrame({'Melding':['Nog geen gecontroleerde freeze-schaalstaat.']})
    keys=['layer','output_channel','block_index']
    start=run.scale_blocks[run.scale_blocks.epoch==FREEZE_EPOCH].set_index(keys).sort_index()
    end=run.scale_blocks[run.scale_blocks.epoch==run.last_epoch].set_index(keys).sort_index()
    audit=run.scale_epochs[run.scale_epochs.epoch>FREEZE_EPOCH]
    rows=[('Nieuwe freeze-epochs',run.branch_epochs),
        ('Groepen met gewijzigde continue φ sinds50',int((end.phi!=start.phi).sum())),
        ('Groepen met andere absolute E2-exponent sinds50',int((end.e2_exponent!=start.e2_exponent).sum())),
        ('E2-wisselgebeurtenissen tijdens nieuwe epochs',int(audit.exponent_flips_during_epoch.sum())),
        ('Groepen met andere E4-bronexponent sinds50',int((end.source_exponent!=start.source_exponent).sum())),
        ('Groepen met andere verhouding E2/E4 sinds50',int((end.relative_offset!=start.relative_offset).sum()))]
    return pd.DataFrame(rows,columns=['Freezecontrole','Waarde'])


def _control_scale_states(run,epoch):
    reference=run.references.get('independent_full100_reference')
    if run.synthetic or not run.reference_safe or not reference or not reference['comparable']:return None
    record=reference['history']['epochs'][epoch-1]
    rows=[]
    for layer,shape in LAYER_SHAPES.items():
        phi=np.asarray(record['scale_state'][layer],float)
        e2=_hard(phi);e4=np.asarray(record['source_scale_state'][layer],int)-127
        for ch,block in np.ndindex(shape):
            rows.append({'layer':layer,'output_channel':ch,'block_index':block,'phi':phi[ch,block],
                'e2_exponent':e2[ch,block],'source_exponent':e4[ch,block],
                'relative_offset':e2[ch,block]-e4[ch,block]})
    return pd.DataFrame(rows)


def plot_freeze_comparison(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        fig.suptitle(_title(run,'Schaalwissels na50 · absolute E2-schaal en E4-bron apart'),fontsize=15,fontweight='bold')
        if not _scale_available(run) or run.branch_epochs==0:
            for ax in axes:_empty(ax,'Nog geen nieuwe freeze-epochs')
            return fig
        changed=source_change_table(run);changed=changed[changed.Epoch>FREEZE_EPOCH]
        axes[0].plot(changed.Epoch,changed['E4-bronschaal gewijzigd'],color='#6D7C86',marker='o',label='Freeze: E4-bron verandert')
        axes[0].plot(changed.Epoch,changed['Absolute E2-schaal gewijzigd'],color='#C98220',marker='o',label='Freeze: absolute E2-schaal')
        axes[0].set(title='Netto wijziging ten opzichte van vorige epoch',ylabel='Aantal groepen',xlabel='Epoch',
            xticks=_branch_ticks(run.last_epoch),ylim=(0,None))
        mine=run.scale_epochs[run.scale_epochs.epoch>FREEZE_EPOCH].groupby('epoch').exponent_flips_during_epoch.sum()
        axes[1].plot(mine.index,mine.values,color='#C98220',marker='o',label='E2 vast na50')
        reference=run.references.get('independent_full100_reference')
        if not run.synthetic and run.reference_safe and reference and reference['comparable']:
            epochs=range(FREEZE_EPOCH+1,run.last_epoch+1)
            flips=[sum(r['exponent_flips_during_epoch'] for r in reference['history']['epochs'][e-1]['scale_epoch_metrics']) for e in epochs]
            axes[1].plot(list(epochs),flips,color='#245A81',marker='o',label='E2 blijft leren')
        axes[1].set(title='Alle harde E2-wisselgebeurtenissen tijdens updates',ylabel='Aantal wissels, inclusief terugwissels',
            xlabel='Epoch',xticks=_branch_ticks(run.last_epoch),ylim=(0,None))
        for ax in axes:ax.legend(frameon=False,fontsize=9)
        return fig
