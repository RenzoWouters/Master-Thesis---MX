"""Read-only analysis of the epoch 50 -> 55 frozen-offset continuation.

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

START, END = 50, 55
EPOCHS = tuple(range(51, 56))
BITS = (8, 6, 4)
FORMATS = {8: 'MXFP8_E4M3', 6: 'MXFP6_E2M3', 4: 'MXFP4_E2M1'}
LABELS = {8: 'E4M3', 6: 'E2M3', 4: 'E2M1'}
BANKS = {8: '8', 6: '32', 4: '4'}
LAYER_BLOCKS = {'layers.3': 64, 'layers.6': 80, 'layers.10': 160,
                'layers.13': 288, 'layers.16': 288, 'layers.20': 3200}
TOTAL_BLOCKS = sum(LAYER_BLOCKS.values())
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

    def best_epoch(self, historical=False, through=None):
        frame = self.historical_test if historical else self.test
        if frame.empty:
            return None
        through = self.last_epoch if through is None else through
        if through is None:
            return None
        frame = frame.loc[frame.epoch.between(51, through) & frame.weight_bits.eq(8)].sort_values('epoch')
        return int(frame.loc[frame.top1_percent.idxmax(), 'epoch']) if not frame.empty else None


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
        return pd.read_csv(path)
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
        # Historical raw input remains intact; all comparisons use only50..55.
        numeric = ['epoch', 'weight_bits', 'activation_bits', 'n', 'objective_loss',
                   'top1_percent', 'correct1', 'lr']
        for key in numeric:
            frame[key] = pd.to_numeric(frame[key], errors='coerce')
        if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
            failures.append('ontbrekende of niet-eindige numerieke velden.')
        if not failures:
            minimum = START if historical else START + 1
            frame = frame.loc[frame.epoch.between(minimum, END)].copy() if historical else frame
            setattr(run, field, frame)
            epochs = sorted(int(e) for e in frame.epoch.unique())
            if historical and epochs != list(range(50, 56)):
                failures.append('de historische epochs 50–55 zijn niet volledig aanwezig.')
            if not historical and (not epochs or epochs != list(range(51, max(epochs) + 1)) or max(epochs) > END):
                failures.append('de voortzetting moet een aaneengesloten deel van 51–55 zijn.')
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
                expected_lr = .001 if historical and row.epoch == 50 else .0001
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
    manifest = run.reference_manifest
    mapping = manifest.get('files_sha256', manifest.get('sha256', {}))
    if not isinstance(mapping, dict) or not mapping:
        run.reference_safe = False
        run.issue('FOUT', 'De referentiemanifest mist de SHA256-hashes van de bijgevoegde historische bestanden.')
    else:
        for name, expected in mapping.items():
            path = run.path / 'parent_reference' / name
            if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_file():
                run.reference_safe = False
                run.issue('FOUT', f'Referentiebestand ontbreekt of heeft een ongeldig pad: {name}.')
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                run.reference_safe = False
                run.issue('FOUT', f'Referentiehash wijkt af: {name}.')
            if path not in run.inputs:
                run.inputs.append(path)
    synthetic_parent = run.synthetic and run.parent_report.get('synthetic_fixture_only') is True
    parent_done = run.parent_report.get('completed_epoch')
    valid_parent_epoch = isinstance(parent_done, int) and parent_done >= 55 if synthetic_parent else parent_done == 100
    if run.parent_report.get('status') != 'PASS' or not valid_parent_epoch:
        run.reference_safe = False
        run.issue('FOUT', 'Het historische rapport bevestigt geen volledige brontraining (echt:100 epochs; expliciete fixture:minstens55) met PASS.')
    if run.baseline.get('epoch') != 50 or not isinstance(run.baseline.get('scale_state'), dict):
        run.reference_safe = False
        run.issue('FOUT', 'Het startrecord na epoch 50 of de oorspronkelijke blokkeuzes ontbreken.')
    if run.reference_safe:
        saved = _metrics_from_history([run.baseline])
        try:
            for row in saved.itertuples(index=False):
                selected = run.reference.loc[run.reference.epoch.eq(50) & run.reference.phase.eq(row.phase) & run.reference.weight_bits.eq(row.weight_bits)]
                if len(selected) != 1 or not _same(selected.iloc[0].top1_percent, row.top1_percent):
                    raise ValueError('epoch 50-accuracy wijkt af van het vastgelegde startrecord')
        except (AttributeError, KeyError, ValueError) as exc:
            run.reference_safe = False
            run.issue('FOUT', 'Startreferentie: ' + str(exc) + '.')


def _validate_branch_contract(run, records):
    config = run.report.get('config', {})
    expected = {'planned_new_epochs': 5, 'start_after_epoch': 50, 'end_epoch': 55,
                'selection_epoch_range': [51, 55], 'activation_bits': 32,
                'master_dtype': 'float32', 'weight_formats': {str(k): v for k, v in FORMATS.items()}}
    for key, value in expected.items():
        if config.get(key) != value:
            run.issue('FOUT', f'Voortzettingscontract {key}: {config.get(key)!r}; verwacht {value!r}.')
    if not run.synthetic:
        for key, value in _sizes(run).items():
            if config.get(key) != value:
                run.issue('FOUT', f'Echte SVHN-run vereist {key}={value}; gerapporteerd {config.get(key)!r}.')
    if run.history.get('parent_epoch') != 50:
        run.issue('FOUT', 'De voortzettingsgeschiedenis verwijst niet naar parent_epoch50.')
    parent_model = run.reference_manifest.get('parent_model_sha256')
    if (not parent_model or run.history.get('initial_model_sha256') != parent_model
            or run.baseline.get('model_sha256') != parent_model):
        run.issue('FOUT', 'De beginmodelhash strookt niet met het historische checkpoint na epoch 50.')
    verification = run.report.get('parent_verification', {})
    checks = {'parent_epoch': 50, 'parent_model_sha256': parent_model,
              'parent_contract_sha256': run.reference_manifest.get('source_contract_sha256'),
              'parent_checkpoint_trace_sha256': run.reference_manifest.get('parent_checkpoint_trace_sha256'),
              'parent_checkpoint_verified': True, 'parent_model_matches_reference': True,
              'parent_contract_matches_reference': True, 'parent_trace_matches_reference': True}
    if any(verification.get(key) != value for key, value in checks.items()):
        run.issue('FOUT', 'De vastgelegde verificatie van het oorspronkelijke checkpoint is onvolledig of wijkt af.')
    checkpoint_hash = verification.get('parent_checkpoint_sha256', '')
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_hash):
        run.issue('FOUT', 'SHA256 van het lokaal geladen oorspronkelijke checkpoint ontbreekt.')
    restore = verification.get('strict_restore_checks', {})
    if any(restore.get(key) is not True for key in ('model_restored', 'optimizer_restored', 'scheduler_restored', 'rng_restored', 'lr_matches_scheduler')):
        run.issue('FOUT', 'Strikt herstel van model, Adam, scheduler en toevalstoestand is niet volledig bevestigd.')
    old = run.historical_records.get('epochs', [])
    old = {record.get('epoch'): record for record in old if isinstance(record, dict)}
    if set(old) != set(EPOCHS):
        run.reference_safe = False
        run.issue('FOUT', 'De historische vergelijkingsrecords51–55 ontbreken.')
    steps = 0
    sizes = _sizes(run)
    batches = math.ceil(sizes['train_examples'] / sizes['batch_size'])
    for record in records:
        epoch = record.get('epoch')
        freeze = record.get('freeze_checks', {})
        expected_checks = {'all_theta_unchanged': True, 'all_hard_offsets_unchanged': True,
                           'checked_optimizer_steps': batches, 'theta_lr_zero': True}
        if any(freeze.get(k) != v for k, v in expected_checks.items()):
            run.freeze_safe = False
            run.issue('FOUT', f'Epoch {epoch}: bevriezing is niet na alle {batches} optimizerstappen bevestigd.')
        steps += int(freeze.get('checked_optimizer_steps', 0))
        coverage = record.get('coverage', {})
        matches = record.get('coverage_historical_match', {})
        for phase in ('train', 'test'):
            observed = coverage.get(phase, {})
            historical = old.get(epoch, {}).get('coverage', {}).get(phase, {})
            if (matches.get(phase) is not True or observed.get('complete_once') is not True
                    or not observed.get('order_sha256')
                    or observed.get('order_sha256') != historical.get('order_sha256')):
                run.issue('FOUT', f'Epoch {epoch}/{phase}: dezelfde volledige voorbeeldvolgorde als de historische run is niet bevestigd.')
        if record.get('optimizer_steps') != epoch * batches:
            run.issue('FOUT', f'Epoch {epoch}: het totale aantal geërfde optimizerstappen wijkt af.')
    freeze_summary = run.report.get('freeze_summary', {})
    n32 = sum(int((np.asarray(v, dtype=float) < 5.5).sum()) for v in run.baseline.get('scale_state', {}).values())
    expected_summary = {'all_theta_unchanged': True, 'all_hard_offsets_unchanged': True,
                        'per_update_checks': steps, 'theta_lr': 0.0,
                        'theta_gradients_computed': True, 'theta_adam_moments_updated': True,
                        'total_choice_flips': 0, 'total_theta_changed_blocks': 0,
                        'blocks': TOTAL_BLOCKS, 'count_x32': n32, 'count_x64': TOTAL_BLOCKS - n32}
    if records and any(freeze_summary.get(k) != v for k, v in expected_summary.items()):
        run.freeze_safe = False
        run.issue('FOUT', 'freeze_summary bevestigt niet alle constante proxy’s, offsets en controles per update.')
    if run.report.get('status') == 'PASS' and run.report.get('continuation_completed') is not True:
        run.issue('FOUT', 'PASS zonder bevestiging dat de volledige voortzetting51–55 is afgerond.')


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
    if not run.metrics_safe or run.test.empty:
        return
    # A deliberately small analysis fixture may omit binary-export evidence.
    # Genuine research reports must include all selected/exported checkpoints.
    if run.synthetic and not run.selection:
        run.issue('SYNTHETISCH', 'Deze analysefixture bevat geen checkpoint-/exportbewijs; er wordt geen echte exportclaim gedaan.')
        return
    last, best = run.last_epoch, run.best_epoch()
    expected = {'selection_epoch_range': [51, last], 'planned_selection_epoch_range': [51, 55],
                'primary_endpoint': 55, 'all_modes_from_same_selected_checkpoint': True}
    if any(run.selection.get(key) != value for key, value in expected.items()):
        run.issue('FOUT', 'De opgeslagen checkpointselectie volgt niet het nieuwe venster51–55.')
    for key, epoch in [('best', best), ('last', last)]:
        if run.selection.get(key, {}).get('epoch') != epoch:
            run.issue('FOUT', f'Opgeslagen {key}-checkpointepoch wijkt af van de opnieuw bepaalde selectie.')
    for key, expected_value in [('selected_best_epoch', best), ('selected_last_epoch', last)]:
        if run.report.get(key) != expected_value:
            run.issue('FOUT', f'Rapportveld {key} wijkt af van de gemeten selectie.')
    exports = run.report.get('mx_exports', {})
    for role, epoch in [('best_E4M3_test', best), ('last_epoch', last)]:
        values = exports.get(role, {})
        if set(values) != {'8', '6', '4'}:
            run.issue('FOUT', f'Exportcontrole {role}: drie precisies ontbreken.')
            continue
        for bit in BITS:
            item = values[str(bit)]
            expected = {'epoch': epoch, 'weight_bits': bit, 'decoded_weights_exact': True,
                        'six_fp32_masters_absent': True, 'fp32_scale_proxies_absent': True,
                        'saved_and_reloaded': True, 'offsets_frozen_at_epoch': 50,
                        'offset_updates_enabled': False, 'full_test_forward_rerun': False,
                        'is_source_export': bit == 8, 'derived_from_saved_source': bit != 8,
                        'packed_scale_choices_bytes': 510}
            if any(item.get(key) != value for key, value in expected.items()):
                run.issue('FOUT', f'Exportcontrole {role}/{LABELS[bit]} wijkt af. De notebook leest alleen de exportcontrole, niet de binaire export zelf.')
    selected = run.selected
    if selected.empty or not {'role', 'epoch', 'weight_bits', 'top1_percent'} <= set(selected) or len(selected) != 6:
        run.issue('FOUT', 'selected_metrics.csv bevat niet alle drie modi voor beide checkpointrollen.')
    else:
        for role, epoch in [('best_E4M3_test', best), ('last_epoch', last)]:
            for bit in BITS:
                rows = selected.loc[selected.role.eq(role) & selected.weight_bits.eq(bit)]
                metric = run.test.loc[run.test.epoch.eq(epoch) & run.test.weight_bits.eq(bit)]
                if len(rows) != 1 or len(metric) != 1 or rows.iloc[0].epoch != epoch or not _same(rows.iloc[0].top1_percent, metric.iloc[0].top1_percent):
                    run.issue('FOUT', f'Geselecteerde metriek {role}/{LABELS[bit]} strookt niet met de gedeelde checkpointselectie.')


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
                if values.ndim != 2 or values.size != count or not np.isfinite(values).all() or not ((values >= 5) & (values <= 6)).all():
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
                n32 = int((values < 5.5).sum())
                expected = {'blocks': values.size, 'count_x32': n32, 'count_x64': values.size - n32,
                            'fraction_x32': n32 / values.size, 'theta_changed_blocks': 0,
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
                for column in ['count_x32', 'count_x64', 'choice_flips_during_epoch', 'theta_changed_blocks', 'theta_update_l2', 'scale_lr']:
                    if not _same(observed.loc[key, column], saved.loc[key, column], 1e-12):
                        raise ValueError(f'{key}/{column}')
        except (KeyError, ValueError, TypeError) as exc:
            failures.append('schaal-CSV verschilt van de geschiedenis: ' + str(exc) + '.')
    if failures:
        run.freeze_safe = False
        for message in failures:
            run.issue('FOUT', 'Bevriezingscontrole: ' + message)


def load_run(run_dir) -> Analysis:
    run = Analysis(Path(run_dir).expanduser().resolve())
    if not run.path.is_dir():
        raise FileNotFoundError(f'RUN_DIR bestaat niet: {run.path}')
    run.report = _json(run, run.path / 'run_report.json', required=True)
    run.history = _json(run, run.path / 'history.json', required=True)
    run.summary = _json(run, run.path / 'training_summary.json')
    run.selection = _json(run, run.path / 'checkpoint_selection.json')
    run.selected = _csv(run, run.path / 'selected_metrics.csv')
    run.metrics = _csv(run, run.path / 'epoch_metrics.csv')
    run.times = _csv(run, run.path / 'epoch_times.csv')
    run.scales = _csv(run, run.path / 'scale_epoch_metrics.csv')
    run.blocks = _csv(run, run.path / 'scale_block_state.csv')
    root = run.path / 'parent_reference'
    run.parent_report = _json(run, root / 'run_report.json', required=True)
    run.reference_manifest = _json(run, root / 'REFERENCE.json', required=True)
    run.historical_records = _json(run, root / 'comparison_epochs51_55.json', required=True)
    run.baseline = _json(run, root / 'baseline_epoch50.json', required=True)
    run.reference = _csv(run, root / 'epoch_metrics.csv')
    run.reference_times = _csv(run, root / 'epoch_times.csv')
    run.reference_scales = _csv(run, root / 'scale_epoch_metrics.csv')
    records = _records(run)
    if run.metrics.empty and records:
        run.metrics = _metrics_from_history(records)
        run.issue('INFO', 'Metriek uit history.json geladen; epoch_metrics.csv ontbreekt.')
    schema = run.report.get('schema')
    if schema != 'mx-triple-freeze-offsets-e50-pilot5-report-v1':
        run.issue('FOUT', f'Onverwacht runrapport: {schema!r}.')
        run.metrics_safe = False
    if run.report.get('status') not in ('PASS', 'STOPPED_AT_EPOCH_BOUNDARY'):
        run.issue('RUNFOUT', f'Rapportstatus: {run.report.get("status", "ontbreekt")}. Eventueel getoonde afgeronde epochs maken deze run geen PASS.')
    if run.synthetic:
        run.issue('SYNTHETISCH', 'Softwarefixture: getallen zijn geen SVHN-onderzoeksresultaten.')
    elif run.report.get('synthetic_fixture_only') is not False:
        run.issue('FOUT', 'Herkomst als echte training is niet expliciet bevestigd.')
    _validate_metrics(run, False)
    _validate_metrics(run, True)
    _compare_history(run, records)
    _validate_reference(run)
    _validate_freeze(run, records)
    _validate_branch_contract(run, records)
    _validate_training_audits(run, records)
    _validate_reported_selection(run)
    if run.observed_epochs and run.observed_epochs != list(EPOCHS):
        run.issue('VOORLOPIG', f'Alleen nieuwe epochs {run.observed_epochs} aanwezig; epoch 55 is nog geen eindresultaat.')
    if run.report.get('completed_epoch') != run.last_epoch:
        run.issue('FOUT', 'completed_epoch in het rapport wijkt af van de geldige meetrijen.')
    run.issue('INTERPRETATIE', 'De dynamische variant is de bestaande training. Er is geen nieuwe dynamische controlerun. CUDA is niet bitexact reproduceerbaar; één voortzetting bepaalt geen run-tot-runspreiding.')
    return run


def status_markdown(run):
    if run.synthetic:
        heading = '**SYNTHETISCHE SOFTWARECONTROLE — geen onderzoeksresultaat.**'
    elif run.complete:
        heading = '**Voortzetting voltooid: vijf nieuwe epochs (51–55), rapportstatus PASS.**'
    else:
        heading = '**Voorlopige of onvolledig bevestigde analyse.**'
    last = run.last_epoch if run.last_epoch is not None else 'geen'
    checks = f'Geldige nieuwe meetepochs: {run.observed_epochs}; laatste: {last}. '
    checks += f'Historische referentie: {"gecontroleerd" if run.reference_safe else "niet bevestigd"}. '
    checks += f'Vaste offsetkeuzes: {"bevestigd in de opgeslagen controles" if run.freeze_safe else "niet bevestigd"}.'
    return heading + '\n\n' + checks + '\n\nHet primaire vergelijkingspunt is epoch 55. Het aanvullende beste checkpoint wordt per variant door E4M3 gekozen, uitsluitend binnen 51–55. Alle drie de precisies gebruiken binnen zo’n groep dezelfde epoch.'


def diagnostics_table(run):
    return pd.DataFrame(run.issues, columns=['Niveau', 'Melding'])


def protocol_table(run):
    state = run.baseline.get('scale_state', {})
    n32 = sum(int((np.asarray(v, dtype=float) < 5.5).sum()) for v in state.values()) if set(state) == set(LAYER_BLOCKS) else None
    return pd.DataFrame([
        ('Start', 'Het bestaande checkpoint na epoch 50; geen nieuwe initialisatie'),
        ('Nieuwe training', 'Alleen de variant met vastgezette offsets: epochs 51–55'),
        ('Historische vergelijking', 'Bestaande dynamische epochs 51–55 uit de afgeronde 100-epochtraining'),
        ('Offsetkeuze per blok', f'Na epoch 50 behouden: {n32 if n32 is not None else "onbekend"}×32; {TOTAL_BLOCKS-n32 if n32 is not None else "onbekend"}×64'),
        ('E4M3-bronschaal', 'Blijft opnieuw bepaald uit de actuele gewichten'),
        ('Gewichten / activaties', 'FP32-master; E4M3 → E2M3 → E2M1; alle activaties A32'),
        ('Gewichts-LR51–55', '0,0001; schema en optimizerstatus geërfd van het checkpoint'),
        ('Schaal-LR51–55', '0; opgeslagen proxywaarden en harde keuzes blijven vast'),
        ('Primair eindpunt', 'Alle drie testaccuracies op epoch 55'),
        ('Aanvullende selectie', 'Eerste hoogste E4M3-testaccuracy binnen 51–55; drie modi op dezelfde epoch'),
        ('Interpretatie', 'Historische controle; geen herhalingsspreiding en geen bewezen optimaal stopmoment'),
    ], columns=['Onderdeel', 'Afspraak'])


def checkpoint_table(run, role='endpoint'):
    columns = ['Variant', 'Selectie', 'Epoch', 'Precisie', 'weight_bits', 'Testaccuracy [%]', 'Correct', 'Testbeelden']
    if not run.reference_safe or not run.metrics_safe or run.last_epoch is None:
        return pd.DataFrame(columns=columns)
    if role == 'endpoint' and run.last_epoch != END:
        return pd.DataFrame(columns=columns)
    rows = []
    for historical, label in [(True, OLD_LABEL), (False, NEW_LABEL)]:
        frame = run.historical_test if historical else run.test
        epoch = END if role == 'endpoint' else run.best_epoch(historical=historical)
        selection = 'Primair: epoch 55' if role == 'endpoint' else f'E4M3-best binnen 51–{run.last_epoch}'
        for bit in BITS:
            found = frame.loc[frame.epoch.eq(epoch) & frame.weight_bits.eq(bit)]
            if len(found) != 1:
                return pd.DataFrame(columns=columns)
            row = found.iloc[0]
            rows.append({'Variant': label, 'Selectie': selection, 'Epoch': epoch,
                         'Precisie': LABELS[bit] + '/A32', 'weight_bits': bit,
                         'Testaccuracy [%]': float(row.top1_percent), 'Correct': int(row.correct1),
                         'Testbeelden': int(row.n)})
    return pd.DataFrame(rows, columns=columns)


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
    columns = ['epoch', 'blocks', 'count_x32', 'count_x64', 'theta_changed_blocks', 'choice_flips_during_epoch', 'unique_blocks_switched_during_epoch']
    if not set(columns) <= set(run.scales):
        return pd.DataFrame()
    grouped = run.scales[columns].groupby('epoch', as_index=False).sum()
    grouped['Aandeel ×32 [%]'] = 100 * grouped.count_x32 / grouped.blocks
    return grouped.rename(columns={'epoch': 'Epoch', 'blocks': 'Blokken', 'count_x32': '×32', 'count_x64': '×64',
                                   'theta_changed_blocks': 'Gewijzigde proxy’s', 'choice_flips_during_epoch': 'Offsetwissels tijdens epoch',
                                   'unique_blocks_switched_during_epoch': 'Blokken met offsetwissel'})


def _empty(ax, message):
    ax.axis('off')
    ax.text(.5, .5, message, transform=ax.transAxes, ha='center', va='center', color='#525D65', wrap=True)


def _title(run, text):
    prefix = 'SYNTHETISCH · ' if run.synthetic else ('CONTROLE NIET GESLAAGD · ' if any(x['Niveau'] in ('FOUT', 'RUNFOUT') for x in run.issues) else ('VOORLOPIG · ' if not run.complete else ''))
    return prefix + text


def plot_accuracy_curves(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout=True)
        fig.suptitle(_title(run, 'Testaccuracy vanaf hetzelfde checkpoint na epoch 50'), fontsize=14, fontweight='bold')
        for ax, bit in zip(axes, BITS):
            if run.test.empty or run.historical_test.empty:
                _empty(ax, 'Vergelijkbare meetrijen ontbreken'); continue
            old = run.historical_test.loc[run.historical_test.weight_bits.eq(bit)]
            new = run.test.loc[run.test.weight_bits.eq(bit)]
            baseline = old.loc[old.epoch.eq(50)]
            joined = pd.concat([baseline, new]).sort_values('epoch')
            ax.plot(old.epoch, old.top1_percent, '--o', color=GREY, markersize=4, label='Verder geleerd · bestaand')
            ax.plot(joined.epoch, joined.top1_percent, '-o', color=MODE_COLORS[bit], markersize=4, label='Offset vast na 50')
            ax.scatter(baseline.epoch, baseline.top1_percent, s=46, marker='D', color='#24333E', zorder=5, label='Gemeenschappelijk startpunt')
            ax.set(title=LABELS[bit] + '/A32', xlabel='Epoch', ylabel='Testaccuracy [%]', xticks=range(50, 56))
            ax.legend(fontsize=8, frameon=False, loc='best')
        fig.supxlabel('Epoch50 is het bestaande startcheckpoint. Alleen de vaste-offsetvariant is opnieuw voortgezet.', fontsize=9, color='#525D65')
        return fig


def plot_checkpoint_accuracy(run, role='endpoint'):
    table = checkpoint_table(run, role)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.7, 5), constrained_layout=True)
        title = 'Primair eindpunt: accuracy op epoch 55' if role == 'endpoint' else 'Beste E4M3-checkpoint binnen de voortzetting'
        fig.suptitle(_title(run, title), fontsize=14, fontweight='bold')
        if table.empty:
            _empty(ax, 'Epoch55 is nog niet volledig gemeten.' if role == 'endpoint' else 'Geen geldige gezamenlijke checkpointvergelijking beschikbaar.')
            return fig
        x = np.arange(3)
        for shift, label, color in [(-.19, OLD_LABEL, GREY), (.19, NEW_LABEL, BLUE)]:
            selected = table.loc[table.Variant.eq(label)].set_index('weight_bits').loc[list(BITS)]
            values = selected['Testaccuracy [%]'].to_numpy()
            epoch = int(selected.Epoch.iloc[0])
            ax.bar(x + shift, values, width=.35, color=color, label=f'{label} · epoch {epoch}', zorder=3)
            for position, value in zip(x + shift, values):
                ax.text(position, value + .9, f'{value:.3f}%', ha='center', va='bottom', fontsize=9)
        ax.set(xticks=x, xticklabels=[LABELS[b] + '/A32' for b in BITS], ylabel='Testaccuracy [%]', ylim=(0, 108), yticks=range(0, 101, 20))
        ax.legend(loc='lower center', frameon=True, facecolor='white', edgecolor='none', fontsize=9)
        note = ('Alle drie precisies per variant gebruiken dezelfde epoch. Selectie op E4M3-testaccuracy; geen onafhankelijke validatieset.'
                if role != 'endpoint' else 'Zelfde epoch 55 voor beide varianten; verschillen in procentpunt staan in de tabel.')
        if role != 'endpoint' and run.last_epoch != END:
            note = f'VOORLOPIG: selectie alleen binnen 51–{run.last_epoch}. ' + note
        fig.supxlabel(note, fontsize=9, color='#525D65')
        return fig


def plot_accuracy_delta(run):
    table = delta_table(run)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.7, 4), constrained_layout=True)
        fig.suptitle(_title(run, 'Vaste offset minus verder geleerde offset'), fontsize=14, fontweight='bold')
        if table.empty:
            _empty(ax, 'Geen vergelijkbare epochs'); return fig
        for bit in BITS:
            rows = table.loc[table.weight_bits.eq(bit)].sort_values('Epoch')
            ax.plot([50] + rows.Epoch.tolist(), [0] + rows['Verschil [procentpunt]'].tolist(), marker='o', color=MODE_COLORS[bit], label=LABELS[bit] + '/A32')
        ax.axhline(0, color=GREY, linewidth=1)
        ax.set(xlabel='Epoch', ylabel='Accuracyverschil [procentpunt]', xticks=range(50, 56))
        ax.legend(frameon=False, ncol=3)
        fig.supxlabel('Positief: hogere accuracy met vaste offsets. Geen betrouwbaarheidsinterval uit één voortzetting.', fontsize=9, color='#525D65')
        return fig


def plot_scale_choices(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
        fig.suptitle(_title(run, 'Vaste blokkeuzes en offsetwissels tijdens training'), fontsize=14, fontweight='bold')
        if run.scales.empty or run.reference_scales.empty or not run.freeze_safe or not run.reference_safe:
            for ax in axes:
                _empty(ax, 'Schaalgegevens zijn niet volledig bevestigd')
            return fig
        for frame, label, color, linestyle in [(run.reference_scales, 'Verder geleerd · bestaand', GREY, '--'), (run.scales, 'Offset vast na 50', BLUE, '-')]:
            data = frame.loc[frame.epoch.between(50 if linestyle == '--' else 51, 55)].groupby('epoch', as_index=False)[['blocks', 'count_x32', 'choice_flips_during_epoch']].sum()
            epochs = data.epoch.tolist(); shares = (100 * data.count_x32 / data.blocks).tolist()
            if linestyle == '-':
                values = run.baseline.get('scale_state', {})
                initial = sum(int((np.asarray(v) < 5.5).sum()) for v in values.values())
                epochs.insert(0, 50); shares.insert(0, 100 * initial / TOTAL_BLOCKS)
            axes[0].plot(epochs, shares, linestyle=linestyle, marker='o', color=color, label=label)
            branch = data.loc[data.epoch.ge(51)]
            axes[1].plot(branch.epoch, branch.choice_flips_during_epoch, linestyle=linestyle, marker='o', color=color, label=label)
        axes[0].set(xlabel='Epoch', ylabel='Blokken met ×32 [%]', xticks=range(50, 56), ylim=(0, 100))
        axes[1].set(xlabel='Epoch', ylabel='Offsetwissels tijdens epoch', xticks=EPOCHS, ylim=(0, None))
        for ax in axes:
            ax.legend(frameon=False, fontsize=8)
        fig.supxlabel('Een offsetwissel is een gebeurtenis; hetzelfde blok kan meerdere keren wisselen. De E4M3-bronschaal blijft dynamisch.', fontsize=9, color='#525D65')
        return fig


def plot_objectives(run):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)
        fig.suptitle(_title(run, 'Trainingsobjectief per precisie'), fontsize=14, fontweight='bold')
        for ax, bit in zip(axes, BITS):
            if not run.metrics_safe or not run.reference_safe:
                _empty(ax, 'Geen bevestigde metrieken'); continue
            for frame, color, linestyle, label in [(run.reference, GREY, '--', 'Verder geleerd · bestaand'), (run.metrics, MODE_COLORS[bit], '-', 'Offset vast na 50')]:
                rows = frame.loc[frame.phase.eq('train') & frame.weight_bits.eq(bit) & frame.epoch.between(51, 55)]
                ax.plot(rows.epoch, rows.objective_loss, linestyle=linestyle, marker='o', color=color, label=label)
            objective = 'harde CE' if bit == 8 else 'zachte CE van leraar'
            ax.set(title=LABELS[bit] + ' · ' + objective, xlabel='Epoch', ylabel='Gemiddeld trainingsobjectief', xticks=EPOCHS)
            ax.legend(frameon=False, fontsize=8)
        fig.supxlabel('E2M3 en E2M1 gebruiken recursive distillation. Hun trainingsobjectief is niet hetzelfde als harde test-CE.', fontsize=9, color='#525D65')
        return fig


def timing_table(run):
    rows = []
    for frame, label in [(run.reference_times, OLD_LABEL), (run.times, NEW_LABEL)]:
        if frame.empty or not {'epoch', 'total_epoch_seconds'} <= set(frame):
            continue
        for record in frame.loc[frame.epoch.between(51, 55)].itertuples(index=False):
            if _finite(record.total_epoch_seconds) and record.total_epoch_seconds >= 0:
                rows.append({'Variant': label, 'Epoch': int(record.epoch), 'Gemeten epochtijd [s]': float(record.total_epoch_seconds)})
    return pd.DataFrame(rows, columns=['Variant', 'Epoch', 'Gemeten epochtijd [s]'])


def plot_timing(run):
    table = timing_table(run)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.7, 4.3), constrained_layout=True)
        fig.suptitle(_title(run, 'Gemeten duur per epoch'), fontsize=14, fontweight='bold')
        if table.empty:
            _empty(ax, 'Geen bruikbare epochtijden opgeslagen'); return fig
        for shift, label, color in [(-.19, OLD_LABEL, GREY), (.19, NEW_LABEL, BLUE)]:
            data = table.loc[table.Variant.eq(label)]
            ax.bar(data.Epoch + shift, data['Gemeten epochtijd [s]'], width=.35, color=color, label=label, zorder=3)
        ax.set(xlabel='Epoch', ylabel='Seconden (train + test + controles + checkpoint)', xticks=EPOCHS, ylim=(0, None))
        ax.legend(frameon=False, fontsize=9)
        fig.supxlabel('Andere uitvoering en hervatkosten kunnen de tijd beïnvloeden; dit meet geen native MX-snelheid of energie.', fontsize=9, color='#525D65')
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
