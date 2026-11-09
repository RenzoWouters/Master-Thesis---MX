"""Read-only analysis of the epoch 50 -> 100 frozen-offset continuation.

The existing dynamic-offset trajectory is a historical comparator. This module
does not load checkpoints, import torch, download data, or start training.
All three precision scores in a selected group come from one E4-selected epoch.
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

START, RESUME, END = 50, 55, 100
EPOCHS = tuple(range(51, 101))
BITS = (8, 6, 4)
FORMATS = {8: 'MXFP8_E4M3', 6: 'MXFP6_E2M3', 4: 'MXFP4_E2M1'}
LABELS = {8: 'E4M3', 6: 'E2M3', 4: 'E2M1'}
BANKS = {8: '8', 6: '32', 4: '4'}
LAYER_BLOCKS = {'layers.3': 64, 'layers.6': 80, 'layers.10': 160,
                'layers.13': 288, 'layers.16': 288, 'layers.20': 3200}
TOTAL_BLOCKS = sum(LAYER_BLOCKS.values())
OFFSET_CHOICES = tuple(range(10))
SCALE_FACTORS = tuple(2 ** offset for offset in OFFSET_CHOICES)
FACTOR_COLORS = ['#173F5F', '#20639B', '#258EA6', '#38A18C', '#75AF62',
                 '#D2A94F', '#BD7B3D', '#AC5757', '#855F9B', '#596387']
PILOT_REFERENCE_SHA256 = '4e3aecbddf78845dd012e9c2411768472cf921d10adbe5004c74c70c67890a35'
DYNAMIC_REFERENCE_SHA256 = '94f8bcc189ab2c3aaf34a95618b1bf5b32c0b5d581393b9563c6af92372cd24b'
BLUE, GREY, GREEN, AMBER = '#245A81', '#7A858E', '#33846D', '#C98220'
MODE_COLORS = {8: BLUE, 6: GREEN, 4: AMBER}
STYLE = {'axes.spines.top': False, 'axes.spines.right': False,
         'axes.titleweight': 'bold', 'axes.labelsize': 10, 'font.size': 10,
         'figure.facecolor': 'white', 'axes.facecolor': 'white',
         'savefig.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .16}
NEW_LABEL = 'Offset vast na epoch 50'
OLD_LABEL = 'Offset verder geleerd (bestaand)'


@dataclass
class Analysis:
    path: Path
    report: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    selected: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_manifest: dict = field(default_factory=dict)
    historical_records: dict = field(default_factory=dict)
    pilot_report: dict = field(default_factory=dict)
    pilot_history: dict = field(default_factory=dict)
    pilot_manifest: dict = field(default_factory=dict)
    parent_report: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference: pd.DataFrame = field(default_factory=pd.DataFrame)
    times: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_times: pd.DataFrame = field(default_factory=pd.DataFrame)
    scales: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_scales: pd.DataFrame = field(default_factory=pd.DataFrame)
    blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list[dict] = field(default_factory=list)
    inputs: list[Path] = field(default_factory=list)
    metrics_safe: bool = True
    reference_safe: bool = True
    freeze_safe: bool = True

    def issue(self, level, message):
        self.issues.append({'Niveau': level, 'Melding': message})

    @property
    def synthetic(self):
        return self.report.get('synthetic_fixture_only') is True or self.summary.get('synthetic_fixture_only') is True

    @property
    def test(self):
        if not self.metrics_safe or self.metrics.empty:
            return pd.DataFrame()
        return self.metrics.loc[self.metrics.phase.eq('test')].sort_values(['epoch', 'weight_bits'])

    @property
    def historical_test(self):
        if not self.reference_safe or self.reference.empty:
            return pd.DataFrame()
        return self.reference.loc[self.reference.phase.eq('test')].sort_values(['epoch', 'weight_bits'])

    @property
    def observed_epochs(self):
        return sorted(int(x) for x in self.test.epoch.unique()) if not self.test.empty else []

    @property
    def last_epoch(self):
        return max(self.observed_epochs) if self.observed_epochs else None

    @property
    def complete(self):
        return (self.report.get('status') == 'PASS' and self.observed_epochs == list(EPOCHS)
                and self.report.get('continuation_completed') is True
                and self.freeze_safe and self.reference_safe
                and not any(x['Niveau'] in ('FOUT', 'RUNFOUT') for x in self.issues))

    def best_epoch(self, historical=False, through=None, global_selection=False):
        frame = self.historical_test if historical else (self.global_test if global_selection else self.test)
        if frame.empty:
            return None
        through = self.last_epoch if through is None else through
        if through is None:
            return None
        frame = frame.loc[frame.epoch.between(1 if global_selection else 51, through) & frame.weight_bits.eq(8)].sort_values('epoch')
        return int(frame.loc[frame.top1_percent.idxmax(), 'epoch']) if not frame.empty else None


    @property
    def global_test(self):
        """One logical trajectory: shared epochs 1-50 plus the frozen51..N branch."""
        if self.test.empty or self.historical_test.empty:
            return pd.DataFrame()
        shared = self.historical_test.loc[self.historical_test.epoch.le(50)]
        return pd.concat([shared, self.test], ignore_index=True).sort_values(['epoch', 'weight_bits'])

    @property
    def new_epochs(self):
        return [epoch for epoch in self.observed_epochs if epoch > RESUME]


def _hard_offsets(theta):
    values = np.asarray(theta, dtype=np.float64)
    lower = np.floor(values)
    return (lower + (values - lower >= .5)).astype(np.int64)

def _state_counts(state):
    if not isinstance(state, dict) or set(state) != set(LAYER_BLOCKS):
        return {}
    counts = {factor: 0 for factor in SCALE_FACTORS}
    try:
        for layer, count in LAYER_BLOCKS.items():
            values = np.asarray(state[layer], dtype=np.float64)
            if values.ndim != 2 or values.size != count or not np.isfinite(values).all() or not ((0 <= values) & (values <= 9)).all():
                return {}
            offsets = _hard_offsets(values)
            for offset, factor in zip(OFFSET_CHOICES, SCALE_FACTORS):
                counts[factor] += int((offsets == offset).sum())
    except (ValueError, TypeError):
        return {}
    return counts

def _format_count(value):
    return f'{int(value):,}'.replace(',', '.')

def _format_percent(value):
    return f'{float(value):.2f}'.replace('.', ',') + '%'

def _expected_lr(epoch, after=False):
    value = .001
    for milestone in (50, 75, 90):
        if (milestone <= epoch) if after else (milestone < epoch):
            value *= .1
    return value


def _ticks(start, end):
    candidates = (25,50,75,100) if start<50 else (50,55,60,70,75,80,90,100)
    return sorted({start,end,*[x for x in candidates if start<=x<=end]})


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same(left, right, atol=1e-7):
    return _finite(left) and _finite(right) and math.isclose(float(left), float(right), rel_tol=0, abs_tol=atol)


def _sizes(run, historical=False):
    """Small counts are accepted only for explicitly labelled software fixtures."""
    report = run.parent_report if historical else run.report
    config = report.get('config', {})
    permitted_fixture = run.synthetic and report.get('synthetic_fixture_only') is True
    defaults = {'train_examples': 73257, 'test_examples': 26032, 'batch_size': 128}
    if not permitted_fixture:
        return defaults
    values = {key: config.get(key, value) for key, value in defaults.items()}
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError('Synthetische fixture heeft ongeldige dataset- of batchaantallen.')
    return values


def _json(run, path, required=False):
    if not path.is_file():
        if required:
            run.issue('FOUT', f'Ontbrekend bestand: {path.name}.')
        return {}
    run.inputs.append(path)
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(result, dict):
            raise ValueError('JSON-object verwacht')
        return result
    except (ValueError, OSError) as exc:
        run.issue('FOUT', f'{path.name} kan niet veilig worden gelezen: {exc}.')
        return {}


def _csv(run, path):
    if not path.is_file():
        return pd.DataFrame()
    run.inputs.append(path)
    try:
        return pd.read_csv(path, float_precision="round_trip")
    except (ValueError, OSError, pd.errors.EmptyDataError) as exc:
        run.issue('FOUT', f'{path.name} kan niet veilig worden gelezen: {exc}.')
        return pd.DataFrame()


def _records(run):
    records = run.history.get('epochs', [])
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        run.issue('FOUT', 'history.json bevat geen geldige lijst met epochrecords.')
        return []
    return records


def _metrics_from_history(records):
    rows = []
    for record in records:
        for phase in ('train', 'test'):
            values = record.get(phase)
            if isinstance(values, list):
                rows.extend(x for x in values if isinstance(x, dict))
    return pd.DataFrame(rows)


def _validate_metrics(run, historical=False):
    field = 'reference' if historical else 'metrics'
    safety = 'reference_safe' if historical else 'metrics_safe'
    frame = getattr(run, field)
    label = 'Historische referentie' if historical else 'Nieuwe voortzetting'
    failures = []
    required = {'epoch', 'phase', 'mode', 'weight_bits', 'weight_format', 'activation_bits',
                'bn_bank_key', 'n', 'objective_loss', 'top1_percent', 'correct1', 'lr'}
    if frame.empty or not required <= set(frame):
        failures.append('volledige train-/testmetriek ontbreekt.')
    else:
        # Historical input spans1–100; paired figures use the same observed window.
        numeric = ['epoch', 'weight_bits', 'activation_bits', 'n', 'objective_loss',
                   'top1_percent', 'correct1', 'lr']
        for key in numeric:
            frame[key] = pd.to_numeric(frame[key], errors='coerce')
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            failures.append('ontbrekende of niet-eindige numerieke velden.')
        if not failures:
            minimum = 1 if historical else START + 1
            frame = frame.loc[frame.epoch.between(minimum, END)].copy() if historical else frame
            setattr(run, field, frame)
            epochs = sorted(int(e) for e in frame.epoch.unique())
            if historical and epochs != list(range(1, 101)):
                failures.append('de historische epochs 1–100 zijn niet volledig aanwezig.')
            if not historical and (not epochs or epochs != list(range(51, max(epochs) + 1)) or max(epochs) > END or max(epochs) < RESUME):
                failures.append('de bevroren geschiedenis moet aaneengesloten 51–N zijn en minstens de geërfde 51–55 bevatten.')
            if not frame.epoch.mod(1).eq(0).all():
                failures.append('epochnummers zijn geen gehele getallen.')
            if frame.duplicated(['epoch', 'phase', 'weight_bits']).any():
                failures.append('dubbele epoch-/fase-/precisierijen.')
            for epoch in epochs:
                group = frame.loc[frame.epoch.eq(epoch)]
                if set(zip(group.phase, group.weight_bits)) != {(p, b) for p in ('train', 'test') for b in BITS} or len(group) != 6:
                    failures.append(f'epoch {epoch} heeft niet precies zes train-/testmetriekrijen.')
            for row in frame.itertuples(index=False):
                if row.weight_bits not in BITS or row.phase not in ('train', 'test'):
                    failures.append('onbekende modus of fase.'); break
                bit = int(row.weight_bits)
                bank = str(row.bn_bank_key).removesuffix('.0')
                if row.weight_format != FORMATS[bit] or row.activation_bits != 32 or bank != BANKS[bit] or row.mode != 'joint':
                    failures.append(f'epoch {row.epoch}/{bit}: formaat, A32 of BN-bank wijkt af.'); break
                count = _sizes(run, historical)[row.phase + '_examples']
                if row.n != count or row.correct1 % 1 != 0 or not 0 <= row.correct1 <= row.n:
                    failures.append(f'epoch {row.epoch}/{bit}: ongeldige aantallen.'); break
                if row.objective_loss < 0 or not 0 <= row.top1_percent <= 100 or not _same(row.top1_percent, 100 * row.correct1 / row.n, 2e-6):
                    failures.append(f'epoch {row.epoch}/{bit}: accuracy/loss strookt niet met aantallen.'); break
                expected_lr = _expected_lr(row.epoch)
                if not _same(row.lr, expected_lr, 1e-14):
                    failures.append(f'epoch {row.epoch}: gewichts-LR is niet {expected_lr:g}.'); break
    if failures:
        setattr(run, safety, False)
        for failure in failures:
            run.issue('FOUT', label + ': ' + failure)


def _compare_history(run, records):
    if not run.metrics_safe:
        return
    from_history = _metrics_from_history(records)
    if from_history.empty:
        run.metrics_safe = False
        run.issue('FOUT', 'Train-/testmetriek ontbreekt in de opgeslagen epochgeschiedenis.')
        return
    keys = ['epoch', 'phase', 'weight_bits']
    try:
        if from_history.duplicated(keys).any():
            raise ValueError('dubbele rijen in history.json')
        observed = run.metrics.set_index(keys)
        saved = from_history.set_index(keys)
        if set(observed.index) != set(saved.index):
            raise ValueError('CSV en geschiedenis bevatten verschillende epochs/modi')
        for key in observed.index:
            for column in ['n', 'correct1', 'top1_percent', 'objective_loss', 'lr']:
                if not _same(observed.loc[key, column], saved.loc[key, column]):
                    raise ValueError(f'{key}/{column} verschilt tussen CSV en geschiedenis')
    except (KeyError, ValueError, TypeError) as exc:
        run.metrics_safe = False
        run.issue('FOUT', 'Metriekcontrole: ' + str(exc) + '.')


def _validate_reference(run):
    if not run.synthetic:
        for dirname, expected in [('pilot_reference', PILOT_REFERENCE_SHA256),
                                   ('dynamic_reference', DYNAMIC_REFERENCE_SHA256)]:
            marker = run.path / dirname / 'REFERENCE.json'
            if not marker.is_file() or hashlib.sha256(marker.read_bytes()).hexdigest() != expected:
                run.reference_safe = False
                run.issue('FOUT', dirname + ': de vaste referentie-identiteit wijkt af.')
    for dirname, manifest in [('dynamic_reference', run.reference_manifest), ('pilot_reference', run.pilot_manifest)]:
        mapping = manifest.get('files_sha256', manifest.get('sha256', {}))
        if not isinstance(mapping, dict) or not mapping:
            run.reference_safe = False
            run.issue('FOUT', dirname + ': manifest met bestandschecksums ontbreekt.')
            continue
        for name, expected in mapping.items():
            path = run.path / dirname / name
            if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file():
                run.reference_safe = False
                run.issue('FOUT', dirname + ': ongeldig of ontbrekend bestand ' + name)
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                run.reference_safe = False
                run.issue('FOUT', dirname + ': checksum wijkt af voor ' + name)
            if path not in run.inputs:
                run.inputs.append(path)
    for label, report in [('Dynamische referentie', run.parent_report), ('Pilot', run.pilot_report)]:
        c = report.get('config', {})
        if c.get('scale_exponent_offset_choices') != list(OFFSET_CHOICES) or c.get('scale_factor_choices') != list(SCALE_FACTORS):
            run.reference_safe = False
            run.issue('FOUT', label + ': dit is niet de tienkeuzeroute.')
    if run.parent_report.get('status') != 'PASS' or run.parent_report.get('completed_epoch') != 100:
        run.reference_safe = False
        run.issue('FOUT', 'De dynamische referentie bevestigt geen volledig afgeronde 100-epochrun met PASS.')
    if (run.pilot_report.get('status') != 'PASS' or run.pilot_report.get('completed_epoch') != 55
            or run.pilot_report.get('continuation_completed') is not True):
        run.reference_safe = False
        run.issue('FOUT', 'De bevroren pilot bevestigt geen volledig afgeronde voortzetting 51–55 met PASS.')
    if run.synthetic is not (run.pilot_report.get('synthetic_fixture_only') is True):
        run.reference_safe = False
        run.issue('FOUT', 'Echte data en synthetische pilotdata worden vermengd.')
    if run.synthetic is not (run.parent_report.get('synthetic_fixture_only') is True):
        run.reference_safe = False
        run.issue('FOUT', 'Echte data en synthetische dynamische referentiedata worden vermengd.')
    historical = run.historical_records.get('epochs', [])
    if not isinstance(historical, list) or [r.get('epoch') for r in historical] != list(range(1, 101)):
        run.reference_safe = False
        run.issue('FOUT', 'De volledige historische epochgeschiedenis 1–100 ontbreekt.')
        return
    run.baseline = historical[49]
    if run.reference_safe:
        saved = _metrics_from_history(historical)
        try:
            observed = run.reference.set_index(['epoch', 'phase', 'weight_bits'])
            expected = saved.set_index(['epoch', 'phase', 'weight_bits'])
            if set(observed.index) != set(expected.index):
                raise ValueError('CSV en geschiedenis hebben verschillende epochs/modi')
            for key in observed.index:
                for col in ('n', 'correct1', 'top1_percent', 'objective_loss', 'lr'):
                    if not _same(observed.loc[key,col], expected.loc[key,col]):
                        raise ValueError(str(key) + '/' + col)
        except (ValueError, KeyError, TypeError) as exc:
            run.reference_safe = False
            run.issue('FOUT', 'Historische CSV/geschiedenis mismatch: ' + str(exc))
    pilot_records = run.pilot_history.get('epochs', [])
    if not isinstance(pilot_records, list) or [r.get('epoch') for r in pilot_records] != list(range(51, 56)):
        run.reference_safe = False
        run.issue('FOUT', 'De vijf geërfde pilotepochs 51–55 ontbreken.')
        return
    prefix = _records(run)[:5]
    if len(prefix) != 5:
        run.reference_safe = False
        run.issue('FOUT', 'De nieuwe geschiedenis bevat niet alle geërfde pilotepochs.')
    else:
        for original, inherited in zip(pilot_records, prefix):
            # File paths can be relinked after copying a run. Scientific state,
            # metrics, counters and RNG/order evidence must remain identical.
            for key in ('epoch','train','test','lr_used','lr_next','scale_lr_used','scale_lr_next',
                        'scale_state','scale_epoch_metrics','optimizer_steps','model_sha256',
                        'gradient_checks','freeze_checks','coverage_historical_match'):
                if original.get(key) != inherited.get(key):
                    run.reference_safe = False
                    run.issue('FOUT', f'Geërfde epoch {original.get("epoch")}/{key} is gewijzigd.')
                    break


def _validate_branch_contract(run, records):
    config = run.report.get('config', {})
    expected = {'planned_new_epochs':45, 'start_after_epoch':55, 'end_epoch':100,
                'selection_epoch_range':[51,100], 'activation_bits':32, 'master_dtype':'float32',
                'weight_formats':{str(k):v for k,v in FORMATS.items()},
                'mx_block_size':32, 'mx_scale_format':'E8M0',
                'scale_exponent_offset_choices':list(OFFSET_CHOICES),
                'scale_factor_choices':list(SCALE_FACTORS),
                'scale_proxy_bounds':[0.0,9.0], 'scale_learning_end_epoch':50,
                'scale_freeze_epoch':50, 'scale_lr':0.0, 'offset_updates_enabled':False}
    for key, value in expected.items():
        if config.get(key) != value:
            run.issue('FOUT', f'Voortzettingscontract {key}: {config.get(key)!r}; verwacht {value!r}.')
    if not run.synthetic:
        for key, value in _sizes(run).items():
            if config.get(key) != value:
                run.issue('FOUT', f'Echte SVHN-run vereist {key}={value}.')
    if run.history.get('parent_epoch') != 50 or run.history.get('continuation_parent_epoch') != 55:
        run.issue('FOUT', 'Geschiedenis moet onderscheid maken tussen freeze-anker 50 en hervatcheckpoint 55.')
    anchor_model = run.baseline.get('model_sha256')
    if not anchor_model or run.history.get('initial_model_sha256') != anchor_model:
        run.issue('FOUT', 'De modelhash van het freeze-anker 50 wijkt af.')
    verification = run.report.get('parent55_verification', {})
    pilot_records = run.pilot_history.get('epochs', [])
    last_pilot = pilot_records[-1] if pilot_records else {}
    if (verification.get('parent_epoch') != 55
            or verification.get('parent_model_sha256') != last_pilot.get('model_sha256')
            or verification.get('parent_contract_sha256') != run.pilot_report.get('contract_sha256')
            or verification.get('parent_checkpoint_verified') is not True
            or verification.get('parent_model_matches_reference') is not True
            or verification.get('parent_contract_matches_reference') is not True
            or verification.get('parent_trace_matches_reference') is not True):
        run.issue('FOUT', 'De identiteit van het hervatte pilotcheckpoint 55 is niet volledig bevestigd.')
    checkpoint_hash = verification.get('parent_checkpoint_sha256', '')
    if checkpoint_hash != run.pilot_manifest.get('parent_checkpoint_sha256'):
        run.issue('FOUT', 'Checkpoint55-bestandshash wijkt af van de vastgelegde pilotreferentie.')
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_hash):
        run.issue('FOUT', 'SHA256 van het hervatte checkpoint 55 ontbreekt.')
    restore = verification.get('strict_restore_checks', {})
    if any(restore.get(k) is not True for k in ('model_restored','optimizer_restored','scheduler_restored','rng_restored','lr_matches_scheduler')):
        run.issue('FOUT', 'Strikt herstel van model, Adam, scheduler en RNG na 55 is niet volledig bevestigd.')
    old = {r.get('epoch'):r for r in run.historical_records.get('epochs',[]) if isinstance(r,dict)}
    steps = 0
    sizes = _sizes(run)
    batches = math.ceil(sizes['train_examples']/sizes['batch_size'])
    for record in records:
        epoch = record.get('epoch')
        freeze = record.get('freeze_checks', {})
        checks = {'all_theta_unchanged':True,'all_hard_offsets_unchanged':True,
                  'checked_optimizer_steps':batches,'theta_lr_zero':True}
        if any(freeze.get(k) != v for k,v in checks.items()):
            run.freeze_safe = False
            run.issue('FOUT', f'Epoch {epoch}: niet elke optimizerstap is op vaste theta/keuzes gecontroleerd.')
        steps += int(freeze.get('checked_optimizer_steps',0))
        for phase in ('train','test'):
            observed = record.get('coverage',{}).get(phase,{})
            historical = old.get(epoch,{}).get('coverage',{}).get(phase,{})
            if (record.get('coverage_historical_match',{}).get(phase) is not True
                    or observed.get('complete_once') is not True or not observed.get('order_sha256')
                    or observed.get('order_sha256') != historical.get('order_sha256')):
                run.issue('FOUT', f'Epoch {epoch}/{phase}: historische voorbeeldvolgorde komt niet overeen.')
        if record.get('optimizer_steps') != epoch*batches:
            run.issue('FOUT', f'Epoch {epoch}: absolute Adamstapteller wijkt af.')
        if not _same(record.get('lr_used'),_expected_lr(epoch),1e-14) or not _same(record.get('lr_next'),_expected_lr(epoch,True),1e-14):
            run.issue('FOUT', f'Epoch {epoch}: oorspronkelijke LR-dalingen 50/75/90 niet behouden.')
    freeze_summary = run.report.get('freeze_summary',{})
    counts = _state_counts(run.baseline.get('scale_state', {}))
    expected_summary = {'all_theta_unchanged':True,'all_hard_offsets_unchanged':True,
        'per_update_checks':steps,'theta_lr':0.0,'theta_gradients_computed':True,
        'theta_adam_moments_updated':True,'total_choice_flips':0,'total_theta_changed_blocks':0,
        'blocks': TOTAL_BLOCKS, 'packed_choice_bytes': 2040,
        **{f'count_x{factor}': counts.get(factor) for factor in SCALE_FACTORS}}
    if records and any(freeze_summary.get(k)!=v for k,v in expected_summary.items()):
        run.freeze_safe = False
        run.issue('FOUT', 'freeze_summary bevestigt niet alle geërfde en nieuwe vaste-offsetepochs 51–N.')
    if records:
        expected_new_steps=max(0,records[-1]['epoch']-55)*batches
        if (run.report.get('new_optimizer_steps')!=expected_new_steps
                or run.report.get('new_training_forward_passes')!=3*expected_new_steps):
            run.issue('FOUT','Nieuwe updates/passes tellen niet uitsluitend de epochs vanaf 56.')
    if run.report.get('status')=='PASS' and (run.report.get('continuation_completed') is not True or run.last_epoch !=100):
        run.issue('FOUT', 'PASS zonder werkelijk voltooide epoch 100.')


def _validate_training_audits(run, records):
    sizes = _sizes(run)
    batches = math.ceil(sizes['train_examples'] / sizes['batch_size'])
    for record in records:
        epoch = record.get('epoch')
        audit = record.get('gradient_checks', {})
        expected = {'optimizer_steps_in_phase': batches, 'finite_gradient_checks': batches,
                    'detached_teacher_checks': 2*batches, 'forward_calls_E4M3': batches,
                    'forward_calls_E2M3': batches, 'forward_calls_E2M1': batches,
                    'cache_empty_before_all_optimizer_steps': True,
                    'scale_projection_after_every_update': True, 'scale_optimizer_steps': batches}
        if any(audit.get(key) != value for key, value in expected.items()):
            run.issue('FOUT', f'Epoch {epoch}: trainingspasses, gradients of updatecontroles wijken af.')
        adam = record.get('adam_checks', {})
        if any(adam.get(key) is not True for key in ('all_active_adam_steps_correct', 'moments_fp32_finite', 'unused_bn_has_no_optimizer_state')):
            run.issue('FOUT', f'Epoch {epoch}: de Adamcontrole is niet volledig bevestigd.')
        if record.get('batchnorm_counters_correct') is not True or record.get('test_optimizer_steps') != 0:
            run.issue('FOUT', f'Epoch {epoch}: BatchNorm-tellers of evaluatie-updates wijken af.')
        for phase in ('train', 'test'):
            phase_batches = math.ceil(sizes[phase + '_examples'] / sizes['batch_size'])
            source = audit.get('shared_source_checks', {}) if phase == 'train' else record.get('test_source_checks', {})
            expected = {'source_triplets': phase_batches, 'high_encodings': phase_batches*6,
                        'exponent_derivations': phase_batches*6, 'mantissa_derivations': phase_batches*6,
                        'scale_offsets_learned': phase_batches*12, 'mantissa_scale_copies_unchanged': phase_batches*6,
                        'same_hard_offsets_across_lower_modes': True, 'source_cache_cleared': True,
                        'no_direct_lower_quantizer': True, 'cache_cleared_before_adam': True}
            if any(source.get(key) != value for key, value in expected.items()):
                run.issue('FOUT', f'Epoch {epoch}/{phase}: gezamenlijke E4M3-bron en truncatieroute niet volledig bevestigd.')
        label_counts = record.get('test_label_counts', [])
        predicted = record.get('test_prediction_counts', {})
        def valid_counts(values):
            return (isinstance(values, list) and len(values) == 10
                    and all(type(n) is int and n >= 0 for n in values)
                    and sum(values) == sizes['test_examples'])
        if not valid_counts(label_counts) or set(predicted) != {'8', '6', '4'} or any(not valid_counts(v) for v in predicted.values()):
            run.issue('FOUT', f'Epoch {epoch}: de volledige label-/voorspellingstellingen ontbreken.')
    if not run.synthetic:
        for key in ('source_unchanged', 'dataset_files_unchanged', 'parent_checkpoint_unchanged'):
            if run.report.get(key) is not True:
                run.issue('FOUT', f'Eindcontrole {key} is niet bevestigd.')


def _validate_reported_selection(run):
    if not run.metrics_safe or run.test.empty or not run.reference_safe:
        return
    if run.synthetic and not run.selection:
        run.issue('SYNTHETISCH','Analysefixture zonder binair exportbewijs; geen echte exportclaim.')
        return
    last = run.last_epoch
    postbest = run.best_epoch()
    globalbest = run.best_epoch(global_selection=True)
    expected = {'selection_epoch_range':[51,last], 'planned_selection_epoch_range':[51,100],
                'primary_endpoint':100, 'all_modes_from_same_selected_checkpoint':True,
                'global_selection_epoch_range':[1,last], 'planned_global_selection_epoch_range':[1,100]}
    if any(run.selection.get(k)!=v for k,v in expected.items()):
        run.issue('FOUT','Opgeslagen selectie volgt niet het afgesproken venster 51–100.')
    # The canonical post-freeze role preserves the previous pilot naming.
    roles = [('last_epoch','last',last), ('best_E4M3_postfreeze','best_postfreeze',postbest),
             ('best_E4M3_global','best_global',globalbest)]
    for role,key,epoch in roles:
        detail=run.selection.get(key,{})
        if detail.get('epoch') != epoch:
            run.issue('FOUT',f'Selectie {key} wijkt af van de opnieuw bepaalde E4-epoch.')
        if role=='best_E4M3_global':
            expected_stage='shared_before_freeze' if epoch<=50 else ('frozen_pilot51_55' if epoch<=55 else 'frozen_continuation56_100')
            if detail.get('source_stage') != expected_stage:
                run.issue('FOUT','Globaal beste checkpoint is niet correct als vóór/na freeze gelabeld.')
    for key,expected_value in [('selected_best_epoch',postbest),('selected_global_best_epoch',globalbest),('selected_last_epoch',last)]:
        if run.report.get(key)!=expected_value:
            run.issue('FOUT',f'Rapportveld {key} wijkt af van de gemeten selectie.')
    exports=run.report.get('mx_exports',{})
    for role,key,epoch in roles:
        values=exports.get(role,{})
        if set(values)!={'8','6','4'}:
            run.issue('FOUT',f'Exportcontrole {role}: drie precisies ontbreken.')
            continue
        for bit in BITS:
            item=values[str(bit)]
            expected={'epoch':epoch,'weight_bits':bit,'decoded_weights_exact':True,
                'six_fp32_masters_absent':True,'fp32_scale_proxies_absent':True,
                'saved_and_reloaded':True,'full_test_forward_rerun':False,
                'is_source_export':bit==8,'derived_from_saved_source':bit!=8,
                'packed_scale_choices_bytes':2040}
            stage='shared_before_freeze' if epoch<=50 else ('frozen_pilot51_55' if epoch<=55 else 'frozen_continuation56_100')
            expected.update(offsets_frozen_at_epoch=50 if epoch>50 else None,
                            offset_updates_enabled=epoch<=50,source_stage=stage)
            if any(item.get(k)!=v for k,v in expected.items()):
                run.issue('FOUT',f'Exportcontrole {role}/{LABELS[bit]} wijkt af; binaire bestanden worden hier niet opnieuw ingeladen.')
    selected=run.selected
    if selected.empty or not {'role','epoch','weight_bits','top1_percent'}<=set(selected) or len(selected)!=9:
        run.issue('FOUT','selected_metrics.csv moet drie modi voor elk van drie selectierollen bevatten.')
    else:
        for role,key,epoch in roles:
            frame=run.global_test if role=='best_E4M3_global' else run.test
            for bit in BITS:
                row=selected.loc[selected.role.eq(role)&selected.weight_bits.eq(bit)]
                metric=frame.loc[frame.epoch.eq(epoch)&frame.weight_bits.eq(bit)]
                if len(row)!=1 or len(metric)!=1 or row.iloc[0].epoch!=epoch or not _same(row.iloc[0].top1_percent,metric.iloc[0].top1_percent):
                    run.issue('FOUT',f'Geselecteerde metriek {role}/{LABELS[bit]} gebruikt niet hetzelfde E4-checkpoint.')


def _validate_freeze(run, records):
    baseline = run.baseline.get('scale_state', {})
    failures = []
    arrays = {}
    if set(baseline) != set(LAYER_BLOCKS):
        failures.append('de zes oorspronkelijke schaalproxy-tensors ontbreken.')
    else:
        for layer, count in LAYER_BLOCKS.items():
            try:
                values = np.asarray(baseline[layer], dtype=float)
                if values.ndim != 2 or values.size != count or not np.isfinite(values).all() or not ((values >= 0) & (values <= 9)).all():
                    raise ValueError('ongeldige vorm of waarden')
                arrays[layer] = values
            except (ValueError, TypeError):
                failures.append(f'{layer}: ongeldige beginstaat.')
    if not failures:
        initial = run.history.get('initial_scale_state', {})
        if set(initial) != set(arrays) or any(not np.array_equal(np.asarray(initial[k]), v) for k, v in arrays.items()):
            failures.append('de beginproxy’s van de voortzetting verschillen van het checkpoint na epoch 50.')
        for record in records:
            epoch = record.get('epoch')
            state = record.get('scale_state', {})
            if set(state) != set(arrays) or any(not np.array_equal(np.asarray(state[k]), v) for k, v in arrays.items()):
                failures.append(f'epoch {epoch}: minstens één proxy is veranderd.')
            if not _same(record.get('scale_lr_used'), 0, 0) or not _same(record.get('scale_lr_next'), 0, 0):
                failures.append(f'epoch {epoch}: de schaal-LR is niet exact nul.')
            rows = record.get('scale_epoch_metrics', [])
            if not isinstance(rows, list) or len(rows) != len(LAYER_BLOCKS) or {r.get('layer') for r in rows} != set(LAYER_BLOCKS):
                failures.append(f'epoch {epoch}: schaalstatistieken per laag ontbreken.')
                continue
            for row in rows:
                layer = row['layer']; values = arrays[layer]
                offsets = _hard_offsets(values)
                counts = {factor: int((offsets == offset).sum()) for offset, factor in zip(OFFSET_CHOICES, SCALE_FACTORS)}
                expected = {'blocks': values.size,
                            **{f'count_x{factor}': count for factor, count in counts.items()},
                            **{f'fraction_x{factor}': count / values.size for factor, count in counts.items()},
                            'choice_jump_distance_during_epoch': 0, 'max_choice_jump': 0,
                            'choice_min_blocks': counts[1], 'choice_max_blocks': counts[512],
                            'theta_at_min_blocks': int((values == 0).sum()),
                            'theta_at_max_blocks': int((values == 9).sum()),
                            'theta_changed_blocks': 0,
                            'theta_update_l2': 0, 'net_choice_changes': 0,
                            'choice_flips_during_epoch': 0, 'unique_blocks_switched_during_epoch': 0,
                            'projected_values_during_epoch': 0, 'scale_lr': 0,
                            'theta_min': float(values.min()), 'theta_max': float(values.max()),
                            'theta_mean': float(values.mean()), 'theta_std': float(values.std(ddof=0))}
                if any(not _same(row.get(key), value, 1e-12) for key, value in expected.items()):
                    failures.append(f'epoch {epoch}/{layer}: statistieken bevestigen geen onveranderde offsetkeuze.')
    if not records:
        failures.append('er zijn nog geen afgeronde nieuwe epochs voor de bevriezingscontrole.')
    if run.scales.empty:
        rows = [row for record in records for row in record.get('scale_epoch_metrics', []) if isinstance(row, dict)]
        run.scales = pd.DataFrame(rows)
        if rows:
            run.issue('INFO', 'Schaalstatistieken uit history.json geladen; scale_epoch_metrics.csv ontbreekt.')
    elif records:
        try:
            from_history = pd.DataFrame([row for record in records for row in record.get('scale_epoch_metrics', [])])
            keys = ['epoch', 'layer']
            if run.scales.duplicated(keys).any() or set(map(tuple, run.scales[keys].to_numpy())) != set(map(tuple, from_history[keys].to_numpy())):
                raise ValueError('andere schaalrijen')
            observed, saved = run.scales.set_index(keys), from_history.set_index(keys)
            for key in observed.index:
                for column in ([f'count_x{factor}' for factor in SCALE_FACTORS]
                               + [f'fraction_x{factor}' for factor in SCALE_FACTORS]
                               + ['choice_flips_during_epoch', 'theta_changed_blocks', 'theta_update_l2',
                                  'scale_lr', 'choice_jump_distance_during_epoch', 'max_choice_jump']):
                    if not _same(observed.loc[key, column], saved.loc[key, column], 1e-12):
                        raise ValueError(f'{key}/{column}')
        except (KeyError, ValueError, TypeError) as exc:
            failures.append('schaal-CSV verschilt van de geschiedenis: ' + str(exc) + '.')
    if records:
        required = {'epoch', 'layer', 'output_channel', 'block_index', 'theta', 'offset', 'factor'}
        if run.blocks.empty or not required <= set(run.blocks):
            failures.append('scale_block_state.csv ontbreekt of heeft onvolledige kolommen.')
        else:
            try:
                block_keys = ['epoch', 'layer', 'output_channel', 'block_index']
                epochs = [50] + [record['epoch'] for record in records]
                if run.blocks.duplicated(block_keys).any() or sorted(run.blocks.epoch.unique()) != epochs:
                    raise ValueError('dubbele blokken of onvolledige epochnummers')
                if set(run.blocks.layer) != set(LAYER_BLOCKS):
                    raise ValueError('andere lagen')
                for epoch in epochs:
                    for layer, values in arrays.items():
                        rows = run.blocks.loc[run.blocks.epoch.eq(epoch) & run.blocks.layer.eq(layer)].sort_values(['output_channel', 'block_index'])
                        coordinates = np.indices(values.shape).reshape(2, -1).T
                        if (len(rows) != values.size or not np.array_equal(rows[['output_channel', 'block_index']].to_numpy(), coordinates)
                                or not np.array_equal(rows.theta.to_numpy(), values.reshape(-1))
                                or not np.array_equal(rows.offset.to_numpy(), _hard_offsets(values).reshape(-1))
                                or not np.array_equal(rows.factor.to_numpy(), np.exp2(_hard_offsets(values)).reshape(-1))):
                            raise ValueError(f'{layer}/epoch{epoch}: proxy, harde keuze of coördinaten verschillen')
            except (ValueError, TypeError, KeyError) as exc:
                failures.append('blok-CSV stemt niet overeen met het startcheckpoint: ' + str(exc) + '.')
    if failures:
        run.freeze_safe = False
        for message in failures:
            run.issue('FOUT', 'Bevriezingscontrole: ' + message)


def load_run(run_dir) -> Analysis:
    run = Analysis(Path(run_dir).expanduser().resolve())
    if not run.path.is_dir():
        raise FileNotFoundError(f'RUN_DIR bestaat niet: {run.path}')
    run.report = _json(run, run.path / 'run_report.json', required=True)
    run.history = _json(run, run.path / 'history.json')
    run.summary = _json(run, run.path / 'training_summary.json')
    run.selection = _json(run, run.path / 'checkpoint_selection.json')
    run.selected = _csv(run, run.path / 'selected_metrics.csv')
    run.metrics = _csv(run, run.path / 'epoch_metrics.csv')
    run.times = _csv(run, run.path / 'epoch_times.csv')
    run.scales = _csv(run, run.path / 'scale_epoch_metrics.csv')
    run.blocks = _csv(run, run.path / 'scale_block_state.csv')
    dynamic, pilot = run.path / 'dynamic_reference', run.path / 'pilot_reference'
    run.parent_report = _json(run, dynamic / 'run_report.json')
    run.reference_manifest = _json(run, dynamic / 'REFERENCE.json')
    run.historical_records = _json(run, dynamic / 'history.json')
    run.reference = _csv(run, dynamic / 'epoch_metrics.csv')
    run.reference_times = _csv(run, dynamic / 'epoch_times.csv')
    run.reference_scales = _csv(run, dynamic / 'scale_epoch_metrics.csv')
    run.pilot_report = _json(run, pilot / 'run_report.json')
    run.pilot_history = _json(run, pilot / 'history.json')
    run.pilot_manifest = _json(run, pilot / 'REFERENCE.json')
    records = _records(run)
    if run.metrics.empty and records:
        run.metrics = _metrics_from_history(records)
        run.issue('INFO', 'Metriek uit history.json geladen; epoch_metrics.csv ontbreekt.')
    if run.reference.empty and run.historical_records.get('epochs'):
        run.reference = _metrics_from_history(run.historical_records['epochs'])
        run.issue('INFO', 'Historische metriek uit history.json geladen.')
    if run.report.get('schema') != 'mx-triple-freeze-scale10-offsets-e50-reference100-report-v1':
        run.metrics_safe = False
        run.issue('FOUT', 'Onverwacht schema: dit is niet de tienkeuze-freezevoortzetting tot 100.')
    check_only = run.report.get('command') == 'check'
    if run.report.get('status') not in ('PASS', 'STOPPED_AT_EPOCH_BOUNDARY'):
        run.issue('RUNFOUT', f'Rapportstatus: {run.report.get("status", "ontbreekt")}; afgeronde meetrijen maken dit geen PASS.')
    if run.synthetic:
        run.issue('SYNTHETISCH', 'Softwarefixture: getallen zijn geen SVHN-onderzoeksresultaten.')
    elif run.report.get('synthetic_fixture_only') is not False and records:
        run.issue('FOUT', 'Herkomst als echte training is niet expliciet bevestigd.')
    if not records and run.metrics.empty:
        run.metrics_safe = run.freeze_safe = run.reference_safe = False
        run.issue('INFO', 'Technische controle; er zijn geen nieuwe trainingsepochs uitgevoerd.' if check_only else
                  'Er zijn nog geen volledig geëxporteerde trainingsepochs beschikbaar.')
        return run
    _validate_metrics(run, False)
    _validate_metrics(run, True)
    _compare_history(run, records)
    _validate_reference(run)
    _validate_freeze(run, records)
    _validate_branch_contract(run, records)
    _validate_training_audits(run, records)
    _validate_reported_selection(run)
    history_path = run.path / 'history.json'
    guard = run.report.get('history_export_validation', {})
    if history_path.is_file() and records:
        raw = history_path.read_bytes()
        expected = {'file': 'history.json', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'epochs': len(records), 'exact_trace_match': True}
        if any(guard.get(key) != value for key, value in expected.items()):
            run.issue('FOUT', 'De volledige history-exportcontrole ontbreekt of wijkt af van het opgeslagen bestand.')
    if run.observed_epochs and run.observed_epochs != list(EPOCHS):
        run.issue('VOORLOPIG', f'Laatste gemeten epoch is {run.last_epoch}; epoch 100 is nog geen eindresultaat. Alle selecties gebruiken hetzelfde gemeten venster.')
    if run.report.get('completed_epoch') != run.last_epoch:
        run.issue('FOUT', 'completed_epoch wijkt af van de geldige meetrijen.')
    run.issue('INTERPRETATIE', 'Historische dynamische controle; geen nieuwe controlerun. CUDA is niet bitexact reproduceerbaar en deze voortzetting meet geen run-tot-runspreiding.')
    return run


def status_markdown(run):
    if run.synthetic:
        heading='**SYNTHETISCHE SOFTWARECONTROLE — geen onderzoeksresultaat.**'
    elif run.complete:
        heading='**Voortzetting tot epoch 100 voltooid — rapportstatus PASS.**'
    else:
        heading='**Voorlopige of onvolledig bevestigde analyse.**'
    last=run.last_epoch if run.last_epoch is not None else 'geen'
    text=f'Laatste gemeten epoch: {last}. '
    if run.test.empty:
        text+='Nog geen trainingsepochs geëxporteerd. '
    else:
        inherited=int(run.test.loc[run.test.epoch.between(51,55), 'epoch'].nunique())
        text+=f'De geschiedenis bevat {inherited} geërfde bevroren pilotepochs uit 51–55 en {len(run.new_epochs)} nieuw gemeten epochs vanaf 56. '
    text+=f'Historische referentie: {"gecontroleerd" if run.reference_safe else "niet bevestigd"}. '
    text+=f'Vaste offsetkeuzes: {"bevestigd" if run.freeze_safe else "niet bevestigd"}.'
    return heading+'\n\n'+text+'\n\n**Hoofdvergelijking:** alle drie precisies op epoch 100. **Aanvullend:** het E4M3-beste checkpoint na de freeze (51–100) en het globale E4M3-beste checkpoint (1–100). Iedere selectie gebruikt voor alle drie precisies dezelfde epoch. Een globaal checkpoint vóór epoch 51 hoort bij de gemeenschappelijke voorfase.'


def diagnostics_table(run):
    return pd.DataFrame(run.issues, columns=['Niveau', 'Melding'])


def protocol_table(run):
    counts = _state_counts(run.baseline.get('scale_state', {}))
    distribution = '; '.join(f'×{factor}: {_format_count(count)}' for factor, count in counts.items()) if counts else 'Nog niet bevestigd'
    return pd.DataFrame([
        ('Hervatpunt', 'Bevroren checkpoint na epoch 55; model, BN, Adam, scheduler en RNG hersteld'),
        ('Nieuwe training', 'Alleen 56–100; bevroren pilot 51–55 en gemeenschappelijke fase 1–50 behouden'),
        ('Freeze-anker', 'Extra exponentoffset per B32-groep blijft die na epoch 50'),
        ('Tien factoren', '×1, ×2, ×4, ×8, ×16, ×32, ×64, ×128, ×256, ×512'),
        ('Groepsverdeling bij50', distribution),
        ('Historische vergelijking', 'Bestaande doorlerende training 1–100; geen nieuwe controletraining'),
        ('E4M3-bronschaal', 'Wordt telkens uit actuele gewichten bepaald; alleen de offset is vast'),
        ('Gewichten / activaties', 'FP32-master; E4M3 → E2M3 → E2M1; A32'),
        ('Gewichts-LR', '56–75: 0,0001; 76–90: 0,00001; 91–100: 0,000001'),
        ('Schaal-LR', '0; theta exact vast; gradients en Adam-momenten blijven berekend'),
        ('Selecties', 'Epoch100; E4M3-best na freeze51–100; globale E4M3-best1–100 met faseaanduiding'),
        ('Accuracyfiguren', 'Ingezoomde assen, gelijke schalen voor vergelijkbare panelen; geen meetpunten weggelaten'),
        ('Interpretatie', 'Eén historische vergelijking bepaalt geen run-tot-runspreiding of optimaal freezemoment')
    ], columns=['Onderdeel', 'Afspraak'])


def checkpoint_table(run,role='endpoint'):
    if role=='best':role='postfreeze'
    if role not in ('endpoint','postfreeze','global'):
        raise ValueError('Onbekende selectierol.')
    columns=['Variant','Selectie','Epoch','Fase','Precisie','weight_bits','Testaccuracy [%]','Correct','Testbeelden']
    if not run.reference_safe or not run.metrics_safe or run.last_epoch is None:
        return pd.DataFrame(columns=columns)
    if role=='endpoint' and run.last_epoch!=100:
        return pd.DataFrame(columns=columns)
    rows=[]
    for historical,label in [(True,OLD_LABEL),(False,NEW_LABEL)]:
        frame=run.historical_test if historical else (run.global_test if role=='global' else run.test)
        epoch=100 if role=='endpoint' else run.best_epoch(historical=historical,global_selection=role=='global')
        selection=('Primair: epoch 100' if role=='endpoint' else
                   f'E4M3-best na freeze: 51–{run.last_epoch}' if role=='postfreeze' else
                   f'Globaal E4M3-best: 1–{run.last_epoch}')
        if epoch is None:return pd.DataFrame(columns=columns)
        phase=('Gemeenschappelijke voorfase — vóór freeze' if epoch<=50 else
               'Verder geleerd — historische run' if historical else
               'Bevroren pilot 51–55 — geërfd' if epoch<=55 else 'Bevroren voortzetting 56–100')
        for bit in BITS:
            found=frame.loc[frame.epoch.eq(epoch)&frame.weight_bits.eq(bit)]
            if len(found)!=1:return pd.DataFrame(columns=columns)
            row=found.iloc[0]
            rows.append({'Variant':label,'Selectie':selection,'Epoch':epoch,'Fase':phase,
                         'Precisie':LABELS[bit]+'/A32','weight_bits':bit,
                         'Testaccuracy [%]':float(row.top1_percent),'Correct':int(row.correct1),'Testbeelden':int(row.n)})
    return pd.DataFrame(rows,columns=columns)


def delta_table(run):
    if run.test.empty or run.historical_test.empty:
        return pd.DataFrame()
    columns = ['epoch', 'weight_bits', 'top1_percent', 'correct1']
    merged = run.test[columns].merge(run.historical_test[columns], on=['epoch', 'weight_bits'], suffixes=('_new', '_old'), validate='one_to_one')
    merged['Precisie'] = merged.weight_bits.map(LABELS)
    merged['Verschil [procentpunt]'] = merged.top1_percent_new - merged.top1_percent_old
    merged['Verschil correct voorspeld'] = merged.correct1_new - merged.correct1_old
    return merged.rename(columns={'epoch': 'Epoch', 'top1_percent_new': 'Vaste offset [%]',
                                  'top1_percent_old': 'Verder geleerd [%]'})[
        ['Epoch', 'Precisie', 'weight_bits', 'Vaste offset [%]', 'Verder geleerd [%]', 'Verschil [procentpunt]', 'Verschil correct voorspeld']]


def freeze_table(run):
    if run.scales.empty:
        return pd.DataFrame()
    columns = ['epoch', 'blocks', *[f'count_x{factor}' for factor in SCALE_FACTORS],
               'theta_changed_blocks', 'choice_flips_during_epoch', 'unique_blocks_switched_during_epoch']
    if not set(columns) <= set(run.scales):
        return pd.DataFrame()
    grouped = run.scales[columns].groupby('epoch', as_index=False).sum()
    return grouped.rename(columns={'epoch': 'Epoch', 'blocks': 'Groepen',
        **{f'count_x{factor}': '×' + str(factor) for factor in SCALE_FACTORS},
        'theta_changed_blocks': 'Gewijzigde proxy’s', 'choice_flips_during_epoch': 'Offsetwissels tijdens epoch',
        'unique_blocks_switched_during_epoch': 'Groepen met offsetwissel'})


def _empty(ax, message):
    ax.axis('off')
    ax.text(.5, .5, message, transform=ax.transAxes, ha='center', va='center', color='#525D65', wrap=True)


def _title(run, text):
    prefix = 'SYNTHETISCH · ' if run.synthetic else ('CONTROLE NIET GESLAAGD · ' if any(x['Niveau'] in ('FOUT', 'RUNFOUT') for x in run.issues) else ('VOORLOPIG · ' if not run.complete else ''))
    return prefix + text


def plot_accuracy_curves(run, full_history=False):
    """Matched precision panels use one shared, explicit data-derived percent scale."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.1), constrained_layout=True, sharey=True)
        start = 1 if full_history else 50
        last = run.last_epoch or END
        title = ('Volledige accuracygeschiedenis · datagestuurde as' if full_history else
                 'Accuracy vanaf het freeze-anker · ingezoomde as')
        fig.suptitle(_title(run, title), fontsize=14, fontweight='bold')
        if run.test.empty or run.historical_test.empty:
            for ax in axes:
                _empty(ax, 'Vergelijkbare meetrijen ontbreken')
            return fig
        old_all = run.historical_test.loc[run.historical_test.epoch.between(start, last)]
        new_all = run.global_test.loc[run.global_test.epoch.between(start, last)]
        limits = _accuracy_limits([*old_all.top1_percent, *new_all.top1_percent])
        for ax, bit in zip(axes, BITS):
            old = old_all.loc[old_all.weight_bits.eq(bit)].sort_values('epoch')
            new = new_all.loc[new_all.weight_bits.eq(bit)].sort_values('epoch')
            if full_history:
                shared = new.loc[new.epoch.le(50)]
                ax.plot(shared.epoch, shared.top1_percent, color='#334956', linewidth=1.5,
                        label='Gedeeld · epochs 1–50')
            dynamic = old.loc[old.epoch.ge(50)]
            fixed = new.loc[new.epoch.ge(50)]
            ax.plot(dynamic.epoch, dynamic.top1_percent, '--', color=GREY, linewidth=1.5,
                    label='Verder geleerd · bestaand')
            ax.plot(fixed.epoch, fixed.top1_percent, color=MODE_COLORS[bit], linewidth=1.8,
                    label='Vaste offset na50')
            anchor = old.loc[old.epoch.eq(50)]
            ax.scatter(anchor.epoch, anchor.top1_percent, s=30, marker='D', color='#24333E',
                       zorder=5, label='Freeze-anker50')
            for milestone in (50, 55, 75, 90):
                if start <= milestone <= last:
                    ax.axvline(milestone, color='#BAC2C7', linewidth=.8, alpha=.7)
            ax.set(title=LABELS[bit] + '/A32', xlabel='Epoch',
                   ylabel='Testaccuracy [%] · ingezoomde as', ylim=limits,
                   xlim=(start, max(start + 1, last)), xticks=_ticks(start, last))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.legend(fontsize=7.7, frameon=False, loc='best')
        note = ('Alle epochs en meetwaarden blijven zichtbaar; dezelfde accuracy-as in alle drie panelen. ' if full_history else
                'Ingezoomd op het gemeten bereik; dezelfde accuracy-as in alle drie panelen. ')
        note += '1–50: gedeeld; 51–55: geërfde pilot; 56–100: nieuwe voortzetting.'
        fig.supxlabel(note, fontsize=9, color='#525D65')
        return fig


def plot_checkpoint_accuracy(run, role='endpoint'):
    """Paired points keep the zoom visible without assigning misleading truncated bar lengths."""
    if role == 'best':
        role = 'postfreeze'
    table = checkpoint_table(run, role)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 4.9), constrained_layout=True)
        title = {'endpoint': 'Primair eindpunt · accuracy op epoch 100',
                 'postfreeze': 'Gemeenschappelijk E4M3-beste checkpoint na freeze',
                 'global': 'Globaal E4M3-beste checkpoint over de volledige training'}[role]
        fig.suptitle(_title(run, title + ' · ingezoomde as'), fontsize=14, fontweight='bold')
        if table.empty:
            _empty(ax, 'Epoch 100 is nog niet gemeten.' if role == 'endpoint' else
                   'Geen geldige gemeenschappelijke checkpointvergelijking beschikbaar.')
            return fig
        xlim = _checkpoint_limits(run)
        y = np.arange(len(BITS))
        scores = {}
        for label in (OLD_LABEL, NEW_LABEL):
            scores[label] = table.loc[table.Variant.eq(label)].set_index('weight_bits').loc[list(BITS)]
        for i, bit in enumerate(BITS):
            left = float(scores[OLD_LABEL].loc[bit, 'Testaccuracy [%]'])
            right = float(scores[NEW_LABEL].loc[bit, 'Testaccuracy [%]'])
            ax.plot([left, right], [i, i], color='#B6C0C6', linewidth=2, zorder=2)
        for label, color, marker, dy in [(OLD_LABEL, GREY, 'o', -12), (NEW_LABEL, BLUE, 'D', 12)]:
            rows = scores[label]
            epoch = int(rows.Epoch.iloc[0])
            stage = ' · vóór freeze' if epoch <= 50 else (' · geërfde pilot' if label == NEW_LABEL and epoch <= 55 else '')
            values = rows['Testaccuracy [%]'].to_numpy()
            ax.scatter(values, y, s=58, marker=marker, color=color,
                       label=f'{label} · epoch {epoch}' + stage, zorder=3)
            for pos, value in zip(y, values):
                ax.annotate(f'{value:.3f}%'.replace('.', ','), (value, pos), xytext=(0, dy),
                            textcoords='offset points', ha='center', va='center',
                            fontsize=10, color=color)
        ax.set(yticks=y, yticklabels=[LABELS[bit] + '/A32' for bit in BITS],
               xlabel='Testaccuracy [%] · ingezoomde as', xlim=xlim, ylim=(2.6, -.6))
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.grid(axis='y', visible=False)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=2, frameon=False, fontsize=9)
        if role == 'global' and table.Epoch.le(50).any():
            note = 'Een checkpoint vóór epoch51 hoort bij de gedeelde voorfase; zijn score is geen resultaat van de freeze.'
        elif role == 'global':
            note = f'Globale selectie 1–{run.last_epoch}; alle drie precisies per variant uit hetzelfde E4M3-checkpoint.'
        elif role == 'postfreeze':
            note = f'Selectie uitsluitend 51–{run.last_epoch}; alle drie precisies per variant uit hetzelfde E4M3-checkpoint.'
        else:
            note = 'Dezelfde epoch100 voor beide varianten. Vergelijk de exacte verschillen in procentpunt in de tabel.'
        note += '\nDe drie checkpointfiguren gebruiken dezelfde ingezoomde accuracyschaal.'
        if role != 'endpoint' and not run.complete:
            note = 'VOORLOPIG. ' + note
        fig.supxlabel(note, fontsize=9, color='#525D65')
        return fig


def plot_accuracy_delta(run):
    table=delta_table(run)
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(11.5,4.2),constrained_layout=True)
        fig.suptitle(_title(run,'Vaste offset minus verder geleerde offset'),fontsize=14,fontweight='bold')
        if table.empty:
            _empty(ax,'Geen vergelijkbare epochs');return fig
        for bit in BITS:
            rows=table.loc[table.weight_bits.eq(bit)].sort_values('Epoch')
            ax.plot([50]+rows.Epoch.tolist(),[0]+rows['Verschil [procentpunt]'].tolist(),color=MODE_COLORS[bit],linewidth=1.7,label=LABELS[bit]+'/A32')
        ax.axhline(0,color=GREY,linewidth=1)
        for milestone in (55,75,90):
            if milestone<=run.last_epoch:ax.axvline(milestone,color='#BAC2C7',linewidth=.8)
        ax.set(xlabel='Epoch',ylabel='Accuracyverschil [procentpunt]',xticks=_ticks(50,run.last_epoch))
        ax.legend(frameon=False,ncol=3)
        fig.supxlabel('Positief betekent hogere accuracy met vaste offsets. Opeenvolgende epochs zijn geen onafhankelijke herhalingen.',fontsize=9,color='#525D65')
        return fig


def plot_scale_choices(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True,
                                 gridspec_kw={'width_ratios': [1, 1, 1.2]})
        fig.suptitle(_title(run, 'Schaalfactoren per groep en wijzigingen tijdens training'), fontsize=14, fontweight='bold')
        columns = ['blocks', *[f'count_x{factor}' for factor in SCALE_FACTORS], 'choice_flips_during_epoch']
        if (run.scales.empty or run.reference_scales.empty or not run.freeze_safe or not run.reference_safe
                or any(not set(columns) <= set(frame) for frame in (run.scales, run.reference_scales))):
            for ax in axes:
                _empty(ax, 'Schaalgegevens zijn niet volledig bevestigd')
            return fig
        last = run.last_epoch
        initial = _state_counts(run.baseline.get('scale_state', {}))
        for ax, frame, label, historical in [(axes[0], run.reference_scales, 'Verder geleerd · bestaand', True),
                                            (axes[1], run.scales, 'Offset vast na 50', False)]:
            data = frame.loc[frame.epoch.between(50 if historical else 51, last)].groupby('epoch', as_index=False)[columns].sum()
            if not historical:
                anchor = {'epoch': 50, 'blocks': TOTAL_BLOCKS, 'choice_flips_during_epoch': 0,
                          **{f'count_x{factor}': initial[factor] for factor in SCALE_FACTORS}}
                data = pd.concat([pd.DataFrame([anchor]), data], ignore_index=True)
            shares = [100 * data[f'count_x{factor}'].to_numpy() / data.blocks.to_numpy() for factor in SCALE_FACTORS]
            ax.stackplot(data.epoch.to_numpy(), *shares, colors=FACTOR_COLORS,
                         labels=['×' + str(factor) for factor in SCALE_FACTORS], alpha=.95)
            ax.set(title=label, xlabel='Epoch', ylabel='Groepen [%]', xticks=_ticks(50, last), ylim=(0, 100))
            ax.legend(ncol=2, fontsize=7, loc='lower left', frameon=True, facecolor='white')
            branch = data.loc[data.epoch.ge(51)]
            axes[2].plot(branch.epoch, branch.choice_flips_during_epoch,
                         '--' if historical else '-', color=GREY if historical else BLUE, label=label)
        axes[2].set(title='Gebeurtenissen per epoch', xlabel='Epoch', ylabel='Offsetwissels', xticks=_ticks(51, last), ylim=(0, None))
        axes[2].legend(frameon=False, fontsize=8)
        fig.supxlabel('Een groep kan meermaals per epoch wisselen. Vastzetten betreft alleen de offset; de E4M3-bronschaal blijft dynamisch.', fontsize=9, color='#525D65')
        return fig


def plot_objectives(run):
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,3,figsize=(13,4.7),constrained_layout=True)
        fig.suptitle(_title(run,'Trainingsobjectief per precisie na de freeze'),fontsize=14,fontweight='bold')
        for ax,bit in zip(axes,BITS):
            if not run.metrics_safe or not run.reference_safe or run.last_epoch is None:
                _empty(ax,'Geen bevestigde metrieken');continue
            for frame,color,style,label in [(run.reference,GREY,'--','Verder geleerd · bestaand'),(run.metrics,MODE_COLORS[bit],'-','Offset vast na 50')]:
                rows=frame.loc[frame.phase.eq('train')&frame.weight_bits.eq(bit)&frame.epoch.between(51,run.last_epoch)]
                ax.plot(rows.epoch,rows.objective_loss,linestyle=style,color=color,label=label)
            objective='harde CE' if bit==8 else 'zachte CE van leraar'
            ax.set(title=LABELS[bit]+' · '+objective,xlabel='Epoch',ylabel='Gemiddeld trainingsobjectief',xticks=_ticks(51,run.last_epoch))
            ax.legend(frameon=False,fontsize=8)
        fig.supxlabel('E2M3/E2M1 gebruiken recursive distillation. Vergelijk per precisie; de objectieven zijn onderling verschillend.',fontsize=9,color='#525D65')
        return fig


def timing_table(run):
    rows=[]
    last=run.last_epoch or 55
    for frame,label in [(run.reference_times,OLD_LABEL),(run.times,NEW_LABEL)]:
        if frame.empty or not {'epoch','total_epoch_seconds'}<=set(frame):continue
        for record in frame.loc[frame.epoch.between(51,last)].itertuples(index=False):
            if _finite(record.total_epoch_seconds) and record.total_epoch_seconds>=0:
                stage='Historisch' if label==OLD_LABEL else ('Geërfde pilot' if record.epoch<=55 else 'Nieuwe voortzetting')
                rows.append({'Variant':label,'Epoch':int(record.epoch),'Fase':stage,'Gemeten epochtijd [s]':float(record.total_epoch_seconds)})
    return pd.DataFrame(rows,columns=['Variant','Epoch','Fase','Gemeten epochtijd [s]'])


def plot_timing(run):
    table=timing_table(run)
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(11.5,4.3),constrained_layout=True)
        fig.suptitle(_title(run,'Gemeten duur per epoch'),fontsize=14,fontweight='bold')
        if table.empty:
            _empty(ax,'Geen bruikbare epochtijden opgeslagen');return fig
        for label,color,style in [(OLD_LABEL,GREY,'--'),(NEW_LABEL,BLUE,'-')]:
            data=table.loc[table.Variant.eq(label)].sort_values('Epoch')
            ax.plot(data.Epoch,data['Gemeten epochtijd [s]'],linestyle=style,color=color,label=label)
        if run.last_epoch and run.last_epoch>=55:ax.axvline(55,color='#BAC2C7',linewidth=1)
        ax.set(xlabel='Epoch',ylabel='Gemeten epochtijd [s]',xticks=_ticks(51,run.last_epoch or 55),ylim=(0,None))
        ax.legend(frameon=False,fontsize=9)
        fig.supxlabel('51–55 is geërfd; vanaf 56 is nieuw uitgevoerd. Hervatkosten en uitvoeringsomstandigheden kunnen de tijd beïnvloeden.',fontsize=9,color='#525D65')
        return fig


def save_figures(figures, run):
    """Explicit opt-in; writes only a fresh subdirectory, never source results."""
    destination = run.path / ('analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(exist_ok=False)
    for name, figure in figures.items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Ongeldige figuurnaam.')
        figure.savefig(destination / (name + '.png'), dpi=160, bbox_inches='tight')
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'analysis_only': True,
                'training_started': False, 'checkpoint_loaded': False,
                'synthetic_fixture_only': run.synthetic,
                'input_sha256': {str(p.relative_to(run.path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.inputs},
                'figures': [name + '.png' for name in figures], 'issues': run.issues}
    (destination / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return destination

def _accuracy_limits(values):
    """Data-derived percent limits with margin; every supplied observation stays visible."""
    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (0.0, 100.0)
    lo, hi = float(finite.min()), float(finite.max())
    if lo < 0 or hi > 100:
        raise ValueError('Accuracywaarden buiten 0–100%.')
    pad = max(.08, (hi - lo) * .13)
    step = .05 if hi - lo < 1 else .1 if hi - lo < 5 else .5
    lower = max(0.0, math.floor((lo - pad) / step) * step)
    upper = min(100.0, math.ceil((hi + pad) / step) * step)
    if upper <= lower:
        upper = min(100.0, lower + .2)
        lower = max(0.0, upper - .2)
    return (lower, upper)


def _checkpoint_limits(run):
    values = []
    for role in ('endpoint', 'postfreeze', 'global'):
        frame = checkpoint_table(run, role)
        if not frame.empty:
            values.extend(frame['Testaccuracy [%]'].tolist())
    return _accuracy_limits(values)

def factor_distribution_table(run, role='last'):
    """Counts/shares at actual selected checkpoints, including a possible pre-freeze global best."""
    aliases = {'best': 'postfreeze', 'best_postfreeze': 'postfreeze', 'best_global': 'global'}
    role = aliases.get(role, role)
    columns = ['Checkpoint', 'Epoch', 'Fase', 'Factor', 'Offset', 'Groepen', 'Aandeel [%]']
    if role not in ('last', 'postfreeze', 'global', 'parent'):
        raise ValueError('role moet last, postfreeze, global of parent zijn.')
    if not run.reference_safe or (role != 'parent' and (not run.freeze_safe or not run.metrics_safe)):
        return pd.DataFrame(columns=columns)
    epoch = (50 if role == 'parent' else run.last_epoch if role == 'last' else
             run.best_epoch(global_selection=role == 'global'))
    if epoch is None:
        return pd.DataFrame(columns=columns)
    if epoch <= 50:
        record = next((r for r in run.historical_records.get('epochs', []) if r.get('epoch') == epoch), {})
        stage = 'Gedeelde voorfase · vóór freeze'
    else:
        record = next((r for r in _records(run) if r.get('epoch') == epoch), {})
        stage = 'Geërfde bevroren pilot51–55' if epoch <= 55 else 'Nieuwe bevroren voortzetting56–100'
    if role == 'parent':
        record = run.baseline
        stage = 'Freeze-anker na epoch50'
    counts = _state_counts(record.get('scale_state', {}))
    if not counts or sum(counts.values()) != TOTAL_BLOCKS:
        return pd.DataFrame(columns=columns)
    label = {'last': 'Laatste checkpoint', 'postfreeze': f'E4M3-best na freeze51–{run.last_epoch}',
             'global': f'Globaal E4M3-best1–{run.last_epoch}', 'parent': 'Freeze-anker'}[role]
    return pd.DataFrame([{'Checkpoint': label, 'Epoch': epoch, 'Fase': stage,
        'Factor': '×' + str(factor), 'Offset': offset, 'Groepen': counts[factor],
        'Aandeel [%]': 100 * counts[factor] / TOTAL_BLOCKS}
        for offset, factor in zip(OFFSET_CHOICES, SCALE_FACTORS)], columns=columns)


def plot_factor_distribution(run):
    """All ten choices remain visible; zero counts are not omitted or renormalized."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 6.8), constrained_layout=True, sharex=True)
        fig.suptitle(_title(run, 'Gekozen schaalfactoren · aantal groepen én aandeel'), fontsize=14, fontweight='bold')
        roles = ('last', 'postfreeze', 'global')
        tables = [factor_distribution_table(run, role) for role in roles]
        maximum = max((int(table.Groepen.max()) for table in tables if not table.empty), default=TOTAL_BLOCKS)
        for ax, table, role in zip(axes, tables, roles):
            if table.empty:
                _empty(ax, 'Nog geen bevestigd checkpoint voor deze selectie')
                continue
            epoch = int(table.Epoch.iloc[0])
            label = {'last': 'Laatste checkpoint', 'postfreeze': 'E4M3-best na freeze', 'global': 'Globaal E4M3-best'}[role]
            stage = 'vóór freeze' if epoch <= 50 else 'geërfde pilot' if epoch <= 55 else 'na freeze'
            title = f'{label}\nEpoch {epoch} · {stage}'
            if role == 'last' and epoch != END:
                title += ' (voorlopig)'
            y = np.arange(len(SCALE_FACTORS))
            ax.barh(y, table.Groepen.to_numpy(), height=.65, color=FACTOR_COLORS, zorder=3)
            for position, row in enumerate(table.itertuples(index=False)):
                count = int(row.Groepen)
                if count == 0:
                    ax.scatter([0], [position], s=12, color='#89949B', zorder=4, clip_on=False)
                ax.text(count + maximum*.025, position,
                        f'{_format_count(count)} | {_format_percent(100*count/TOTAL_BLOCKS)}',
                        va='center', fontsize=9)
            ax.set(title=title, yticks=y, yticklabels=table.Factor.tolist(),
                   xlabel='Aantal groepen (totaal 4.080)', xlim=(0, maximum*1.42),
                   ylim=(len(SCALE_FACTORS)-.4, -.6))
            ax.grid(axis='y', visible=False)
        fig.supxlabel('Alle tien factoren blijven zichtbaar, ook bij nul. Na freeze is de verdeling die van epoch50; een globaal checkpoint vóór freeze kan afwijken.',
                      fontsize=9, color='#525D65')
        return fig

def storage_cost_table(run=None):
    """Analytical additional payload/state, excluding container overhead and the rest of the network."""
    return pd.DataFrame([
        {'Variant': 'Vaste factor', 'Keuzebits per groep': 0, 'Groepen': TOTAL_BLOCKS,
         'Keuzemetadata [bytes]': 0, 'Extra FP32 theta+gradient+Adam [bytes]': 0},
        {'Variant': 'Twee keuzes · doorleren', 'Keuzebits per groep': 1, 'Groepen': TOTAL_BLOCKS,
         'Keuzemetadata [bytes]': 510, 'Extra FP32 theta+gradient+Adam [bytes]': 65280},
        {'Variant': 'Tien keuzes · doorleren', 'Keuzebits per groep': 4, 'Groepen': TOTAL_BLOCKS,
         'Keuzemetadata [bytes]': 2040, 'Extra FP32 theta+gradient+Adam [bytes]': 65280},
        {'Variant': 'Tien keuzes · vast na 50', 'Keuzebits per groep': 4, 'Groepen': TOTAL_BLOCKS,
         'Keuzemetadata [bytes]': 2040, 'Extra FP32 theta+gradient+Adam [bytes]': 65280},
    ])

def plot_storage_cost(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        fig.suptitle(_title(run, 'Analytische extra opslag voor de schaalkeuze'), fontsize=14, fontweight='bold')
        table = storage_cost_table(run)
        labels = ['Vaste\nfactor', '2 keuzes\ndoorleren', '10 keuzes\ndoorleren', '10 keuzes\nvast na 50']
        colors = [GREY, GREEN, BLUE, AMBER]
        for ax, key, divisor, ylabel in [
            (axes[0], 'Keuzemetadata [bytes]', 1, 'Keuzemetadata in flexibele bron [bytes]'),
            (axes[1], 'Extra FP32 theta+gradient+Adam [bytes]', 1024, 'Extra trainingstensors [KiB]')]:
            values = table[key].to_numpy() / divisor
            ax.bar(np.arange(len(values)), values, color=colors, width=.6, zorder=3)
            for position, value in enumerate(values):
                label = _format_count(value) if divisor == 1 else f'{value:g}'.replace('.', ',')
                ax.text(position, value + max(values) * .025, label, ha='center', va='bottom')
            ax.set(xticks=np.arange(len(values)), xticklabels=labels, ylabel=ylabel, ylim=(0, max(values) * 1.2))
            ax.grid(axis='x', visible=False)
        fig.supxlabel('Deze diagnostische freeze berekent schaalgradients en Adam-momenten verder; geen besparing door het verwijderen van tensors.',
                      fontsize=9, color='#525D65')
        return fig
