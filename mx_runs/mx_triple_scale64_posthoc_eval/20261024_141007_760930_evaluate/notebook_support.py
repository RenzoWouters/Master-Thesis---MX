"""Read-only analysis of the two-cohort frozen MX scale64 experiment.

Reads JSON/CSV/NPZ only. Never imports torch, a model, dataset or checkpoint.
Figures may be saved explicitly to a fresh analysis directory.
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

SCHEMA = 'mx-triple-scale64-posthoc-report-v1'
TRAININGS = ('direct', 'truncation')
ROLES = ('last_epoch', 'best_E4M3_test')
PATHS = ('source8', 'trained6', 'trained4', 'A6', 'A4', 'B6', 'B4', 'C6', 'C4')
DERIVED = PATHS[3:]
BITS = {'source8': 8, 'trained6': 6, 'trained4': 4, **{f'{f}{b}': b for f in 'ABC' for b in (6, 4)}}
BANKS = {8: 8, 6: 32, 4: 4}
BASELINES = {p: ('source8' if BITS[p] == 8 else 'trained' + str(BITS[p])) for p in PATHS}
LAYERS = ('layers.3', 'layers.6', 'layers.10', 'layers.13', 'layers.16', 'layers.20')
OFFSET = {p: (6 if p[0] in 'BC' else 0) for p in PATHS}
COMPARISONS = tuple((f'trained{b}', f'{f}{b}') for f in 'ABC' for b in (6, 4)) + (
    ('A6', 'B6'), ('A4', 'B4'), ('B6', 'C6'), ('B4', 'C4'), ('A6', 'C6'), ('A4', 'C4'))
PRIMARY_PAIRS = (('A6', 'B6'), ('B6', 'C6'), ('A4', 'B4'), ('B4', 'C4'))
TRAINING_LABELS = {'direct': 'Direct getraind', 'truncation': 'Met naïeve truncatie getraind'}
LABELS = {'source8': 'E4M3-bron, ongewijzigd', 'trained6': 'E2M3 zoals getraind', 'trained4': 'E2M1 zoals getraind',
          'A6': 'A · naïeve exponenttruncatie', 'A4': 'A · exponent + mantissa',
          'B6': 'B · naïeve exponenttruncatie, schaal ×64', 'B4': 'B · exponent + mantissa, schaal ×64',
          'C6': 'C · afronden vanuit E4M3, schaal ×64', 'C4': 'C · afgeronde E2M3 → mantissa, schaal ×64'}
FAMILY_LABELS = {'trained': 'Zoals getraind', 'A': 'A · naïef, S', 'B': 'B · naïef, 64S', 'C': 'C · afronden, 64S'}
COLORS = {'trained': '#335c81', 'A': '#278477', 'B': '#d49228', 'C': '#8466a8'}
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 10,
         'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.spines.top': False,
         'axes.spines.right': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
         'savefig.facecolor': 'white', 'axes.axisbelow': True, 'grid.alpha': .15}


def _bool(value):
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, str) and value.lower() in ('true', 'false'): return value.lower() == 'true'
    return None


def _finite(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe_child(root, name):
    if not isinstance(name, str): raise ValueError('Ongeldig bestandspad.')
    path = (root / name).resolve()
    if path == root or not path.is_relative_to(root): raise ValueError('Bestandspad verlaat de resultaatmap.')
    return path


@dataclass
class RunAnalysis:
    path: Path
    report: dict = field(default_factory=dict)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    codes: pd.DataFrame = field(default_factory=pd.DataFrame)
    reproduction: pd.DataFrame = field(default_factory=pd.DataFrame)
    issues: list = field(default_factory=list)
    output_checks: list = field(default_factory=list)
    prediction_rows: list = field(default_factory=list)
    input_files: list = field(default_factory=list)
    metrics_safe: bool = True

    @property
    def synthetic(self):
        return self.report.get('synthetic_fixture_only') is True or self.report.get('synthetic_checks_only') is True

    @property
    def confirmed(self):
        return (self.report.get('status') == 'PASS' and self.report.get('command') == 'evaluate'
                and self.metrics_safe and len(self.metrics) == 36 and len(self.output_checks) == 4
                and not any(x['Niveau'] in ('FOUT', 'RUNFOUT', 'ONTBREEKT') for x in self.issues))

    def issue(self, level, message):
        self.issues.append({'Niveau': level, 'Melding': message})

    def frame(self, training, role):
        if self.metrics.empty or not self.metrics_safe: return pd.DataFrame()
        return self.metrics.loc[(self.metrics.training == training) & (self.metrics.role == role)]

    def epoch(self, training, role):
        f = self.frame(training, role)
        values = f.epoch.unique() if not f.empty else []
        return int(values[0]) if len(values) == 1 else None


def load_run(run_dir, verify_per_image=True):
    run = RunAnalysis(Path(run_dir).expanduser().resolve())
    for name, attr in [('run_report.json', 'report'), ('metrics.csv', 'metrics'),
                       ('paired_metrics.csv', 'pairs'), ('weight_pair_metrics.csv', 'weights'),
                       ('layer_code_metrics.csv', 'codes'), ('historical_reproduction.csv', 'reproduction')]:
        path = run.path / name
        if not path.is_file():
            run.issue('ONTBREEKT', name); continue
        run.input_files.append(path)
        try:
            value = json.loads(path.read_text(encoding='utf-8-sig')) if name.endswith('.json') else pd.read_csv(path)
            if attr == 'report' and not isinstance(value, dict): raise ValueError('Rapport is geen object.')
            setattr(run, attr, value)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            run.issue('FOUT', f'{name}: {exc}')
    if run.report.get('schema') != SCHEMA:
        run.issue('FOUT', 'Dit rapport hoort niet bij de vastgelegde schaal×64-proef.')
    if run.report.get('error'): run.issue('RUNFOUT', str(run.report['error']))
    if run.report.get('command') == 'check':
        run.issues = [x for x in run.issues if x['Niveau'] != 'ONTBREEKT']
        run.issue('INFO', 'Softwarecheck; er zijn geen post-hoctestaccuracies.')
        return run
    expected = {'training_kinds': list(TRAININGS), 'scale_exponent_offset': 6,
                'optimizer_steps': 0, 'batchnorm_recalibration': False,
                'new_checkpoint_selection': False, 'learned_shared_scale': False,
                'shared_scale_compensation': True}
    for key, value in expected.items():
        if run.report.get(key) != value: run.issue('FOUT', f'Afwijkend of ontbrekend protocolveld: {key}.')
    if run.report.get('synthetic_fixture_only') is not False and not run.synthetic:
        run.issue('FOUT', 'Herkomst ontbreekt; geen echte SVHN-proef bevestigen.')
    for key in ('input_files_unchanged', 'test_file_unchanged', 'original_checkpoint_selection_unchanged',
                'derived_models_all_evaluated_from_saved_and_reloaded_codes'):
        if run.report.get(key) is not True: run.issue('FOUT', 'Uitvoeringscontrole niet bevestigd: ' + key)
    if run.report.get('train_and_extra_read') is not False:
        run.issue('FOUT', 'Uitsluitend de testset gebruiken is niet bevestigd.')
    _validate_metrics(run)
    _validate_pairs(run)
    _validate_roles(run)
    _validate_weights(run)
    _validate_reproduction(run)
    if verify_per_image and run.metrics_safe and not run.metrics.empty:
        _verify_outputs(run)
    elif not verify_per_image:
        run.issue('LET OP', 'Per-beeldcontrole uitgeschakeld; geen volledig bevestigde evaluatie.')
    for item in run.report.get('warnings', []):
        run.issue('RUNWAARSCHUWING', item.get('message', str(item)) if isinstance(item, dict) else str(item))
    return run


def _validate_metrics(run):
    required = {'training', 'role', 'epoch', 'path', 'weight_bits', 'activation_bits', 'bn_bank_key',
                'n', 'correct1', 'correct5', 'top1_percent', 'top5_percent', 'hard_cross_entropy',
                'optimizer_steps', 'batchnorm_recalibrated'}
    if run.metrics.empty or not required <= set(run.metrics):
        run.metrics_safe = False; run.issue('FOUT', 'Essentiële metriektabel ontbreekt of is onvolledig.'); return
    d = run.metrics
    numeric = ('epoch', 'weight_bits', 'activation_bits', 'bn_bank_key', 'n', 'correct1', 'correct5',
               'top1_percent', 'top5_percent', 'hard_cross_entropy', 'optimizer_steps')
    for name in numeric: d[name] = pd.to_numeric(d[name], errors='coerce')
    if not np.isfinite(d[list(numeric)].to_numpy(dtype=float)).all():
        run.metrics_safe = False; run.issue('FOUT', 'Niet-eindige of ontbrekende metrics.'); return
    expected = {(t, r, p) for t in TRAININGS for r in ROLES for p in PATHS}
    actual = set(zip(d.training, d.role, d.path))
    if len(d) != 36 or actual != expected or d.duplicated(['training', 'role', 'path']).any():
        run.metrics_safe = False; run.issue('FOUT', 'Verwacht 36 unieke routes: twee trainingen × twee checkpoints × negen routes.')
    for t in TRAININGS:
        for r in ROLES:
            f = d.loc[(d.training == t) & (d.role == r)]
            if f.epoch.nunique() != 1 or f.n.nunique() != 1:
                run.metrics_safe = False; run.issue('FOUT', f'{t}/{r}: gemengde checkpoint- of testsetidentiteit.')
            if r == 'last_epoch' and not f.epoch.eq(100).all():
                run.issue('FOUT', f'{t}: vast eindpunt is niet epoch 100.')
    for name in ('epoch', 'weight_bits', 'activation_bits', 'bn_bank_key', 'n', 'correct1', 'correct5', 'optimizer_steps'):
        if (d[name] % 1 != 0).any(): run.issue('FOUT', 'Niet-gehele telling: ' + name)
    if ((d.epoch < 1) | (d.epoch > 100) | (d.n <= 0) | (d.correct1 < 0)
            | (d.correct5 < d.correct1) | (d.correct5 > d.n) | (d.hard_cross_entropy < 0)).any():
        run.issue('FOUT', 'Ongeldige epochs, aantallen of cross-entropy.')
    if not d.activation_bits.eq(32).all() or not d.optimizer_steps.eq(0).all():
        run.issue('FOUT', 'Verwacht A32 en nul optimizerupdates.')
    if not d.batchnorm_recalibrated.map(_bool).eq(False).all(): run.issue('FOUT', 'BN-herkalibratie staat niet overal uit.')
    for p in PATHS:
        f = d.loc[d.path == p]
        if not f.weight_bits.eq(BITS[p]).all() or not f.bn_bank_key.eq(BANKS[BITS[p]]).all():
            run.issue('FOUT', p + ': verkeerde doelprecisie of BN-bank.')
    for metric, count in [('top1_percent', 'correct1'), ('top5_percent', 'correct5')]:
        if not np.allclose(d[metric], 100 * d[count] / d.n, rtol=0, atol=1e-8):
            run.issue('FOUT', metric + ': percentage strookt niet met correcte aantallen.')
    if not run.synthetic and not d.n.eq(26032).all(): run.issue('FOUT', 'De echte evaluatie moet alle 26.032 testbeelden bevatten.')
    recorded = run.report.get('metrics')
    if not isinstance(recorded, list) or len(recorded) != 36:
        run.issue('FOUT', 'De metrieken ontbreken in het runrapport.')
    elif run.metrics_safe:
        by_key = {(x.get('training'), x.get('role'), x.get('path')): x for x in recorded}
        if set(by_key) != expected: run.issue('FOUT', 'Rapport en CSV gebruiken verschillende routes.')
        else:
            for _, row in d.iterrows():
                other = by_key[(row.training, row.role, row.path)]
                if any(_finite(other.get(k)) is None or not math.isclose(float(row[k]), float(other[k]), rel_tol=0, abs_tol=1e-8) for k in numeric):
                    run.issue('FOUT', 'Rapport en CSV bevatten verschillende metrieken.'); break


def _validate_pairs(run):
    required = {'training', 'role', 'epoch', 'reference_path', 'other_path', 'n', 'accuracy_difference_pp',
                'reference_correct1', 'other_correct1', 'prediction_changes', 'wrong_to_correct', 'correct_to_wrong',
                'both_correct', 'both_wrong', 'wrong_to_other_wrong', 'net_correct_change'}
    if run.pairs.empty or not required <= set(run.pairs):
        run.issue('FOUT', 'Gepaarde vergelijkingen ontbreken of zijn onvolledig.'); return
    d = run.pairs
    expected = {(t, r, a, b) for t in TRAININGS for r in ROLES for a, b in COMPARISONS}
    actual = set(zip(d.training, d.role, d.reference_path, d.other_path))
    if len(d) != 48 or actual != expected or d.duplicated(['training', 'role', 'reference_path', 'other_path']).any():
        run.issue('FOUT', 'Verwacht de 48 vastgelegde gepaarde vergelijkingen zonder dubbele rijen.')
    for _, row in d.iterrows():
        keys = ('n', 'accuracy_difference_pp', 'prediction_changes', 'wrong_to_correct', 'correct_to_wrong',
                'both_correct', 'both_wrong', 'wrong_to_other_wrong', 'net_correct_change', 'reference_correct1', 'other_correct1')
        if any(_finite(row[k]) is None for k in keys) or row.n <= 0:
            run.issue('FOUT', 'Niet-eindige gepaarde telling.'); continue
        if row.net_correct_change != row.wrong_to_correct - row.correct_to_wrong or row.net_correct_change != row.other_correct1 - row.reference_correct1:
            run.issue('FOUT', 'Netto verandering klopt niet met gepaarde aantallen.')
        if row.prediction_changes != row.wrong_to_correct + row.correct_to_wrong + row.wrong_to_other_wrong:
            run.issue('FOUT', 'Aantal gewijzigde voorspellingen klopt niet.')
        if row.both_correct + row.both_wrong + row.wrong_to_correct + row.correct_to_wrong != row.n:
            run.issue('FOUT', 'Gepaarde categorieën dekken de testset niet.')
        if not math.isclose(row.accuracy_difference_pp, 100 * row.net_correct_change / row.n, abs_tol=1e-8):
            run.issue('FOUT', 'Gepaarde accuracydelta klopt niet.')
        f = run.frame(row.training, row.role)
        if not f.empty and row.reference_path in set(f.path) and row.other_path in set(f.path):
            f = f.set_index('path');a,b = f.loc[row.reference_path],f.loc[row.other_path]
            if row.epoch != a.epoch or row.n != a.n or row.reference_correct1 != a.correct1 or row.other_correct1 != b.correct1:
                run.issue('FOUT', 'Gepaarde tabel gebruikt andere metrics of checkpointidentiteit.')


def _validate_roles(run):
    roles, refs = run.report.get('roles'), run.report.get('references')
    if not isinstance(roles, dict) or set(roles) != set(TRAININGS) or not isinstance(refs, dict) or set(refs) != set(TRAININGS):
        run.issue('FOUT', 'De twee afzonderlijke trainingreferenties ontbreken.'); return
    for t in TRAININGS:
        if not isinstance(roles[t], dict) or set(roles[t]) != set(ROLES):
            run.issue('FOUT', t + ': beide vaste checkpointrollen zijn vereist.'); continue
        ref = refs[t]
        if ref.get('checkpoint_files_opened') is not False or ref.get('FP32_master_weights_opened') is not False or ref.get('contract_matches') is not True:
            run.issue('FOUT', t + ': broncontrole of afwezigheid van mastertoegang ontbreekt.')
        selection = ref.get('checkpoint_selection', {})
        for r in ROLES:
            d = roles[t][r]; original = selection.get('last' if r == 'last_epoch' else 'best', {})
            if d.get('epoch') != original.get('epoch') or d.get('checkpoint_sha256') != original.get('sha256'):
                run.issue('FOUT', f'{t}/{r}: oorspronkelijke checkpointselectie gewijzigd.')
            if run.metrics_safe and d.get('epoch') != run.epoch(t, r): run.issue('FOUT', f'{t}/{r}: rapport en metrics gebruiken andere epochs.')
            for key in ('all_model_and_bn_states_unchanged', 'all_models_no_grad', 'all_paths_use_frozen_target_precision_bn',
                        'six_FP32_master_parameters_absent_all_paths'):
                if d.get(key) is not True: run.issue('FOUT', f'{t}/{r}: audit niet bevestigd: {key}.')
            if d.get('optimizer_steps') != 0 or d.get('batchnorm_recalibrated') is not False:
                run.issue('FOUT', f'{t}/{r}: updates of BN-herkalibratie niet uitgesloten.')


def _validate_weights(run):
    required = {'training', 'role', 'epoch', 'layer', 'path', 'target_bits', 'bn_bank_key', 'n_weights',
                'blocks', 'mean_abs', 'rms', 'zero_count', 'scales_equal_source', 'scale_exponent_offset',
                'scale_exp_min', 'scale_exp_max', 'source_scale_delta_counts'}
    d = run.codes
    if d.empty or not required <= set(d):
        run.issue('FOUT', 'Code- en schaaldiagnostiek ontbreekt of is onvolledig.'); return
    expected = {(t, r, layer, p) for t in TRAININGS for r in ROLES for layer in LAYERS for p in PATHS}
    if len(d) != 216 or set(zip(d.training, d.role, d.layer, d.path)) != expected or d.duplicated(['training','role','layer','path']).any():
        run.issue('FOUT', 'Verwacht negen routes voor elk van de zes MX-lagen en vier checkpointgroepen.')
    for _, row in d.iterrows():
        if row.path not in PATHS: continue
        numeric = ('target_bits','bn_bank_key','n_weights','blocks','mean_abs','rms','zero_count','scale_exp_min','scale_exp_max')
        if any(_finite(row[k]) is None for k in numeric):
            run.issue('FOUT', 'Niet-eindige laagdiagnostiek.'); continue
        if row.target_bits != BITS[row.path] or row.bn_bank_key != BANKS[BITS[row.path]]:
            run.issue('FOUT', 'Codecontrole gebruikt verkeerde doelprecisie of BN-bank.')
        if row.scale_exponent_offset != OFFSET[row.path] or _bool(row.scales_equal_source) is not (OFFSET[row.path] == 0):
            run.issue('FOUT', row.path + ': schaalbeleid is niet A=S, B/C=64S, baseline ongewijzigd.')
        try:
            delta = json.loads(row.source_scale_delta_counts)
            if delta != {str(OFFSET[row.path]): int(row.blocks)}: raise ValueError('Onjuiste schaaloffset per blok.')
        except (ValueError, TypeError): run.issue('FOUT', row.path + ': onjuiste getelde schaaloffsets.')
        if any(_finite(row[k]) is None for k in ('n_weights','blocks','mean_abs','rms','zero_count','scale_exp_min','scale_exp_max')):
            run.issue('FOUT', 'Niet-eindige laagdiagnostiek.'); continue
        if row.n_weights <= 0 or row.blocks <= 0 or min(row.mean_abs,row.rms,row.zero_count) < 0 or row.zero_count > row.n_weights:
            run.issue('FOUT', 'Ongeldige gewichtsamplitude of nultelling.')
        if row.scale_exp_min < -127 or row.scale_exp_max > 127 or row.scale_exp_min > row.scale_exp_max:
            run.issue('FOUT', 'Schaalexponent buiten eindig E8M0-bereik.')
        if run.metrics_safe and row.epoch != run.epoch(row.training, row.role): run.issue('FOUT', 'Laagdiagnostiek gebruikt andere epoch.')
    for t in TRAININGS:
        for r in ROLES:
            for layer in LAYERS:
                frame = d.loc[(d.training == t) & (d.role == r) & (d.layer == layer)].set_index('path')
                if frame.index.has_duplicates: continue
                for bit in (6,4):
                    if {f'A{bit}',f'B{bit}'} <= set(frame.index):
                        a,b = frame.loc[f'A{bit}'],frame.loc[f'B{bit}']
                        if any(not math.isclose(float(b[k]),64*float(a[k]),rel_tol=2e-12,abs_tol=0) for k in ('mean_abs','rms')) or a.zero_count != b.zero_count:
                            run.issue('FOUT', f'{t}/{r}/{layer}: B is numeriek niet precies 64×A.')
    needed = {'training','role','epoch','layer','reference_path','other_path','mae','rmse','max_abs_difference',
              'reference_l2_norm','difference_l2_norm'}
    if run.weights.empty or not needed <= set(run.weights):
        run.issue('FOUT', 'Gewichtsvergelijkingen ontbreken of zijn onvolledig.'); return
    expected_pairs = COMPARISONS + tuple(('source8', p) for p in DERIVED)
    expected_weights = {(t,r,layer,a,b) for t in TRAININGS for r in ROLES for layer in LAYERS for a,b in expected_pairs}
    actual_weights = set(zip(run.weights.training, run.weights.role, run.weights.layer, run.weights.reference_path, run.weights.other_path))
    if len(run.weights) != 432 or actual_weights != expected_weights:
        run.issue('FOUT', 'Verwacht 432 gewichtsvergelijkingen: achttien paren × zes lagen × vier groepen.')
    if run.weights.duplicated(['training','role','layer','reference_path','other_path']).any(): run.issue('FOUT', 'Dubbele gewichtsvergelijkingen.')
    for key in ('mae','rmse','max_abs_difference','reference_l2_norm','difference_l2_norm'):
        vals=pd.to_numeric(run.weights[key],errors='coerce')
        if not np.isfinite(vals).all() or (vals < 0).any(): run.issue('FOUT','Ongeldige gewichtsafwijking: '+key)
    for _,row in run.weights.iterrows():
        if row.reference_path in PATHS and row.other_path in PATHS:
            same_format = BITS[row.reference_path] == BITS[row.other_path]
            if 'code_comparison_same_format' in row and _bool(row.code_comparison_same_format) != same_format:
                run.issue('FOUT', 'Codevergelijking gebruikt een onjuiste formaatidentiteit.')
            if not same_format and 'code_differences' in row and pd.notna(row.code_differences):
                run.issue('FOUT', 'Cross-formaatcodebytes zijn ten onrechte als vergelijkbaar geteld.')
        if row.reference_path not in PATHS or row.other_path not in PATHS or row.layer not in LAYERS:
            run.issue('FOUT','Onbekende route/laag bij gewichtsvergelijking.')
        if run.metrics_safe and row.epoch != run.epoch(row.training,row.role): run.issue('FOUT','Gewichtsvergelijking gebruikt andere epoch.')


def _validate_reproduction(run):
    d=run.reproduction
    needed={'training','role','epoch','path','historical_correct1','reevaluated_correct1','difference_correct1','accuracy_difference_pp','correct1_matches_history'}
    if d.empty or not needed <= set(d): run.issue('FOUT','Historische herhalingscontroles ontbreken.');return
    expected={(t,r,p) for t in TRAININGS for r in ROLES for p in PATHS[:3]}
    if len(d)!=12 or set(zip(d.training,d.role,d.path))!=expected or d.duplicated(['training','role','path']).any():
        run.issue('FOUT','Verwacht twaalf historische herhalingscontroles.')
    for _,row in d.iterrows():
        f=run.frame(row.training,row.role)
        if f.empty or row.path not in set(f.path):continue
        metric=f.set_index('path').loc[row.path]
        if (row.epoch!=metric.epoch or row.reevaluated_correct1!=metric.correct1
                or row.difference_correct1!=row.reevaluated_correct1-row.historical_correct1
                or not math.isclose(float(row.accuracy_difference_pp),100*float(row.difference_correct1)/metric.n,abs_tol=1e-8)
                or _bool(row.correct1_matches_history)!=(row.difference_correct1==0)):
            run.issue('FOUT','Historische herhalingscontrole is intern inconsistent.')
        elif row.difference_correct1:
            run.issue('LET OP', f'{row.training}/{row.role}/{row.path}: herhaling verschilt {int(row.difference_correct1):+d} correcte beelden; gebruik de huidige gepaarde referentie.')


def _verify_outputs(run):
    label_reference=None
    for t in TRAININGS:
        for r in ROLES:
            d=run.report.get('roles',{}).get(t,{}).get(r,{})
            try:
                path=_safe_child(run.path,d['per_image_file'])
                if _hash(path)!=d.get('per_image_sha256'):raise ValueError('Per-beeldchecksum wijkt af.')
                run.input_files.append(path)
                with np.load(path,allow_pickle=False) as z:
                    expected={'labels','indices'}|{p+'__'+k for p in PATHS for k in ('logits','predictions','per_image_ce')}
                    if set(z.files)!=expected:raise ValueError('Onvolledige of onverwachte per-beeldvelden.')
                    y,indices=z['labels'],z['indices'];n=len(y)
                    if y.shape!=(n,) or indices.shape!=(n,) or not np.array_equal(indices,np.arange(n)):
                        raise ValueError('Testindices niet exact eenmaal in oorspronkelijke volgorde.')
                    if not np.issubdtype(y.dtype,np.integer) or ((y<0)|(y>9)).any():raise ValueError('Ongeldige labels.')
                    if n!=d.get('n_test') or hashlib.sha256(y.tobytes()).hexdigest()!=d.get('label_sha256'):
                        raise ValueError('Labels of aantal beelden wijken af.')
                    if label_reference is None:label_reference=y.copy()
                    elif not np.array_equal(y,label_reference):raise ValueError('Checkpointgroepen gebruiken verschillende testlabels.')
                    outputs={}
                    for p in PATHS:
                        pred,logits,ce=z[p+'__predictions'],z[p+'__logits'],z[p+'__per_image_ce']
                        if pred.shape!=(n,) or logits.shape!=(n,10) or ce.shape!=(n,):raise ValueError(p+': onjuiste arrayvorm.')
                        if not np.issubdtype(pred.dtype,np.integer) or ((pred<0)|(pred>9)).any():raise ValueError(p+': ongeldige voorspelling.')
                        if not np.isfinite(logits).all() or not np.isfinite(ce).all() or (ce<0).any():raise ValueError(p+': niet-eindige logits/loss.')
                        # A saved topk winner may differ from NumPy argmax under ties.
                        if not np.array_equal(logits[np.arange(n),pred],logits.max(axis=1)):raise ValueError(p+': voorspelling is geen logitmaximum.')
                        f=run.frame(t,r).set_index('path');row=f.loc[p]
                        if row.n!=n or int((pred==y).sum())!=row.correct1:raise ValueError(p+': incorrecte aantallen.')
                        if not math.isclose(float(ce.astype(np.float64).mean()),float(row.hard_cross_entropy),rel_tol=0,abs_tol=1e-9):raise ValueError(p+': CE-gemiddelde wijkt af.')
                        ld=logits.astype(np.float64);m=ld.max(axis=1)
                        recomputed=np.log(np.exp(ld-m[:,None]).sum(axis=1))+m-ld[np.arange(n),y]
                        if not np.allclose(ce,recomputed,rtol=3e-5,atol=2e-5):raise ValueError(p+': CE strookt niet met logits en labels.')
                        kth=np.sort(logits,axis=1)[:,-5];true_logit=logits[np.arange(n),y]
                        lower=int((true_logit>kth).sum());upper=int((true_logit>=kth).sum())
                        if not lower<=row.correct5<=upper:raise ValueError(p+': top5-telling strijdig met logits, ook rekening houdend met ties.')
                        counts=np.bincount(pred,minlength=10);dominant=int(counts.argmax())
                        for key, actual in [('classes_predicted',int((counts>0).sum())), ('dominant_predicted_class',dominant),
                                            ('dominant_prediction_fraction',float(counts[dominant]/n)), ('scale_exponent_offset',OFFSET[p])]:
                            if key in row and (_finite(row[key]) is None or not math.isclose(float(row[key]),actual,rel_tol=0,abs_tol=1e-12)):
                                raise ValueError(p+': voorspeldiagnostiek of schaaloffset strookt niet met de route.')
                        run.prediction_rows.append({'training':t,'role':r,'epoch':run.epoch(t,r),'path':p,'n':n,
                            'classes_predicted':int((counts>0).sum()),'dominant_class':dominant,'dominant_fraction':float(counts[dominant]/n),
                            **{f'class_{i}':int(v) for i,v in enumerate(counts)}})
                        outputs[p]={'pred':pred,'logits':logits}
                    if not {'training','role','reference_path','other_path','wrong_to_correct','correct_to_wrong','wrong_to_other_wrong','prediction_changes','both_correct','both_wrong'} <= set(run.pairs):
                        raise ValueError('Gepaarde tabel is onvolledig.')
                    for _,row in run.pairs.loc[(run.pairs.training==t)&(run.pairs.role==r)].iterrows():
                        a,b=outputs[row.reference_path],outputs[row.other_path];ac,bc=a['pred']==y,b['pred']==y;change=a['pred']!=b['pred']
                        actual={'wrong_to_correct':int((~ac&bc).sum()),'correct_to_wrong':int((ac&~bc).sum()),
                            'wrong_to_other_wrong':int((~ac&~bc&change).sum()),'prediction_changes':int(change.sum()),
                            'both_correct':int((ac&bc).sum()),'both_wrong':int((~ac&~bc).sum())}
                        if any(row[k]!=v for k,v in actual.items()):raise ValueError('Gepaarde tellingen wijken af van voorspellingen.')
                        error=b['logits'].astype(np.float64)-a['logits'].astype(np.float64)
                        for k,v in [('mean_abs_logit_difference',float(np.abs(error).mean())),('max_abs_logit_difference',float(np.abs(error).max()))]:
                            if k in row and not math.isclose(float(row[k]),v,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Gepaard logitverschil wijkt af.')
                run.output_checks.append({'Training':TRAINING_LABELS[t],'Rol':r,'Epoch':run.epoch(t,r),'Testbeelden':n,
                    'Checksum':'PASS','Negen routes':'PASS','Gepaarde tellingen':'PASS'})
            except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
                run.issue('FOUT',f'{t}/{r}: {exc}')


def diagnostics_table(run):
    return pd.DataFrame(run.issues or [{'Niveau':'OK','Melding':'Alle ingelezen consistentiecontroles geslaagd.'}])


def status_markdown(run):
    prefix='**SYNTHETISCHE SOFTWARECONTROLE — GEEN SVHN-RESULTATEN.**\n\n' if run.synthetic else ''
    status=('**Post-hocevaluatie bevestigd.** Twee trainingen, twee vaste checkpointrollen en negen routes per groep zijn gecontroleerd.'
            if run.confirmed else '**Geen volledig bevestigde evaluatie.** Bekijk eerst de meldingen; ontbrekende of ongeldige scores worden niet als eindresultaat voorgesteld.')
    return prefix+status+'\n\nDeze notebook leest opgeslagen resultaten. Zij voert geen modelevaluatie, training of BN-herkalibratie uit. De schaaloffset +6 is vooraf vastgelegd; er wordt geen schaal, route of checkpoint op deze resultaten geselecteerd.'


def route_table():
    rows=[]
    for p in PATHS:
        if p in PATHS[:3]:scale='Zoals opgeslagen bij deze training';method='Herhaling opgeslagen gewichtsmodus'
        else:
            scale='E4M3-bronschaal S' if p[0]=='A' else '64 × E4M3-bronschaal S'
            method='Exponent-MSB-behoud' if p[0] in 'AB' else 'RNE_E2M3(decoded_E4M3 / 64)'
            if BITS[p]==4:method+='; daarna dezelfde mantissatruncatie E2M3 → E2M1'
        rows.append({'Route':p,'Beschrijving':LABELS[p],'Doelbits':BITS[p],'Activatiebits':32,'BN-bank':BANKS[BITS[p]],'Bewerking':method,'Schaal':scale})
    return pd.DataFrame(rows)


def accuracy_table(run):
    if run.metrics.empty or not run.metrics_safe:return pd.DataFrame()
    d=run.metrics.copy();d['Beschrijving']=d.path.map(LABELS)
    d['_t']=d.training.map({v:i for i,v in enumerate(TRAININGS)});d['_r']=d.role.map({v:i for i,v in enumerate(ROLES)});d['_p']=d.path.map({v:i for i,v in enumerate(PATHS)})
    return d.sort_values(['_t','_r','_p'])[['training','role','epoch','path','Beschrijving','weight_bits','correct1','n','top1_percent','hard_cross_entropy']]


def delta_table(run):
    rows=[]
    for t in TRAININGS:
        for r in ROLES:
            f=run.frame(t,r)
            if f.empty:continue
            f=f.set_index('path')
            for a,b in COMPARISONS:
                rows.append({'training':t,'role':r,'epoch':run.epoch(t,r),'Vergelijking':b+' − '+a,
                    'Delta accuracy [pp]':float(f.loc[b,'top1_percent']-f.loc[a,'top1_percent']),
                    'Netto correcte beelden':int(f.loc[b,'correct1']-f.loc[a,'correct1'])})
    return pd.DataFrame(rows)


def prediction_table(run):
    return pd.DataFrame(run.prediction_rows)


def audit_table(run):
    rows=[]
    for t in TRAININGS:
        for r in ROLES:
            d=run.report.get('roles',{}).get(t,{}).get(r,{})
            rows.append({'Training':TRAINING_LABELS[t],'Rol':r,'Epoch':d.get('epoch'),'Model/BN ongewijzigd':d.get('all_model_and_bn_states_unchanged'),
                'Geen gradients':d.get('all_models_no_grad'),'Optimizerupdates':d.get('optimizer_steps'),'BN-herkalibratie':d.get('batchnorm_recalibrated'),
                'Zes masters afwezig':d.get('six_FP32_master_parameters_absent_all_paths'),'Checkpoint-SHA256':d.get('checkpoint_sha256')})
    return pd.DataFrame(rows)


def timing_markdown(run):
    s=_finite(run.report.get('session_wall_seconds'))
    measured=f'**Gemeten evaluatiesessie: {s/60:.2f} minuten.** ' if s is not None else 'Geen gemeten sessieduur beschikbaar. '
    return measured+'Deze tijd omvat alle routes en administratie. Het is geen native-MX-snelheids- of energiebenchmark.'


def _title(run,text):
    return ('SYNTHETISCHE CONTROLE · ' if run.synthetic else '')+text


def _role_title(run,t,r):
    e=run.epoch(t,r)
    return ('Epoch 100' if r=='last_epoch' else 'E4M3-best')+(f' · epoch {e}' if e and r!='last_epoch' else '')


def _empty(ax,text='Geen volledige, veilige metriektabel beschikbaar'):
    ax.set_axis_off();ax.text(.5,.5,text,transform=ax.transAxes,ha='center',va='center',wrap=True,color='#59616b')


def plot_checkpoint_accuracy(run,training):
    if training not in TRAININGS:raise ValueError('Onbekende training.')
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(13.8,5.4));fig.subplots_adjust(left=.06,right=.985,top=.80,bottom=.22,wspace=.19)
        fig.suptitle(_title(run,TRAINING_LABELS[training]+' — testaccuracy'),fontsize=15,y=.97)
        fig.text(.5,.895,'De getrainde modi worden herhaald; A, B en C worden uit dezelfde opgeslagen E4M3-bron afgeleid.',ha='center',fontsize=10,color='#59616b')
        for ax,r in zip(axes,ROLES):
            ax.set_title(_role_title(run,training,r))
            f=run.frame(training,r)
            if f.empty:_empty(ax);continue
            f=f.set_index('path');positions=np.arange(3);width=.19
            for j,family in enumerate(('trained','A','B','C')):
                xs=[];ys=[]
                for i,bit in enumerate((8,6,4)):
                    if bit==8 and family!='trained':continue
                    p='source8' if bit==8 else ('trained'+str(bit) if family=='trained' else family+str(bit))
                    xs.append(positions[i] if bit==8 else positions[i]+(j-1.5)*width);ys.append(f.loc[p,'top1_percent'])
                bars=ax.bar(xs,ys,width,color=COLORS[family],label=FAMILY_LABELS[family])
                ax.bar_label(bars,labels=[f'{v:.2f}' for v in ys],fontsize=7.2,padding=3,rotation=0)
            ax.set(xticks=positions,xticklabels=['E4M3','E2M3','E2M1'],ylim=(0,108),ylabel='Testaccuracy [%]');ax.grid(axis='y')
        handles,labels=axes[0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=4,frameon=False,fontsize=9.5)
        fig.text(.5,.025,'E4M3 blijft ongewijzigd. Alle routes gebruiken A32 en de opgeslagen BN-bank van hun doelprecisie.',ha='center',fontsize=9,color='#59616b')
        return fig


def _grid(run,title):
    fig,axes=plt.subplots(2,2,figsize=(13,7.8));fig.subplots_adjust(left=.075,right=.98,top=.84,bottom=.14,hspace=.43,wspace=.25)
    fig.suptitle(_title(run,title),fontsize=15,y=.98)
    for i,r in enumerate(ROLES):
        for j,t in enumerate(TRAININGS):axes[i,j].set_title(TRAINING_LABELS[t]+'\n'+_role_title(run,t,r))
    return fig,axes


def plot_accuracy_deltas(run):
    with plt.rc_context(STYLE):
        fig,axes=_grid(run,'Schaalwijziging en afrondingscontrole afzonderlijk vergelijken')
        allvalues=[]
        for i,r in enumerate(ROLES):
            for j,t in enumerate(TRAININGS):
                ax=axes[i,j];f=run.frame(t,r)
                if f.empty:_empty(ax);continue
                f=f.set_index('path')
                for k,(a,b,color,label) in enumerate([('A','B',COLORS['B'],'B − A: schaal ×64'),('B','C',COLORS['C'],'C − B: afronding bij 64S')]):
                    vals=[f.loc[b+str(bit),'top1_percent']-f.loc[a+str(bit),'top1_percent'] for bit in (6,4)];allvalues+=vals
                    bars=ax.bar(np.arange(2)+(k-.5)*.32,vals,.32,color=color,label=label)
                    ax.bar_label(bars,labels=[f'{v:+.2f}' for v in vals],fontsize=9,padding=3)
                ax.axhline(0,color='#4b5563',lw=.8);ax.set(xticks=[0,1],xticklabels=['E2M3','E2M1'],ylabel='Verschil in accuracy [procentpunt]');ax.grid(axis='y')
        if allvalues:
            lo,hi=min(0,min(allvalues)),max(0,max(allvalues));pad=max(3,(hi-lo)*.16)
            for ax in axes.flat:ax.set_ylim(lo-pad,hi+pad)
        h,l=axes[0,0].get_legend_handles_labels()
        if h:fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
        fig.text(.5,.014,'Dit zijn verschillen op vaste checkpoints en vaste BN; geen onafhankelijke causale optelsom of nieuwe modelselectie.',ha='center',fontsize=9,color='#59616b')
        return fig


def plot_paired_changes(run):
    with plt.rc_context(STYLE):
        fig,axes=_grid(run,'Welke testbeelden winnen of verliezen?')
        extent=1
        for i,r in enumerate(ROLES):
            for j,t in enumerate(TRAININGS):
                ax=axes[i,j]
                if run.pairs.empty or not {'training','role','reference_path','other_path'}<=set(run.pairs):_empty(ax);continue
                f=run.pairs.loc[(run.pairs.training==t)&(run.pairs.role==r)]
                up=[];down=[]
                for a,b in PRIMARY_PAIRS:
                    row=f.loc[(f.reference_path==a)&(f.other_path==b)]
                    if len(row)!=1:break
                    row=row.iloc[0];up.append(100*row.wrong_to_correct/row.n);down.append(-100*row.correct_to_wrong/row.n)
                if len(up)!=4:_empty(ax);continue
                ax.bar(np.arange(4)-.16,up,.32,color='#278477',label='Fout → correct')
                ax.bar(np.arange(4)+.16,down,.32,color='#b65e57',label='Correct → fout')
                extent=max(extent,max(up),-min(down));ax.axhline(0,color='#4b5563',lw=.8)
                ax.set(xticks=np.arange(4),xticklabels=['A → B\nE2M3','B → C\nE2M3','A → B\nE2M1','B → C\nE2M1'],ylabel='Aandeel testbeelden [%]');ax.grid(axis='y')
        for ax in axes.flat:ax.set_ylim(-extent*1.15,extent*1.15)
        h,l=axes[0,0].get_legend_handles_labels()
        if h:fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
        return fig


def plot_weight_magnitudes(run):
    with plt.rc_context(STYLE):
        fig,axes=_grid(run,'Grootte van de afgeleide gewichten tegenover de E4M3-bron')
        for i,r in enumerate(ROLES):
            for j,t in enumerate(TRAININGS):
                ax=axes[i,j]
                if run.codes.empty or not {'training','role','path','layer','rms','rms_ratio_to_source'}<=set(run.codes):_empty(ax,'Geen volledige RMS-diagnostiek');continue
                f=run.codes.loc[(run.codes.training==t)&(run.codes.role==r)];plotted=False
                for p in DERIVED:
                    z=f.loc[f.path==p].set_index('layer').reindex(LAYERS);v=pd.to_numeric(z.rms_ratio_to_source,errors='coerce')
                    if v.notna().any():
                        ax.plot(np.arange(6),v,color=COLORS[p[0]],ls='-' if BITS[p]==6 else '--',marker='o' if BITS[p]==6 else 's',ms=3,lw=1.7,label=p);plotted=True
                if not plotted:_empty(ax);continue
                ax.axhline(1,color='#4b5563',lw=.8,ls=':');ax.set(xticks=np.arange(6),xticklabels=[s.split('.')[1] for s in LAYERS],xlabel='MX-convolutielaag',ylabel='RMS route / RMS E4M3-bron',ylim=(0,None));ax.grid(axis='y')
        h,l=axes[0,0].get_legend_handles_labels()
        if h:fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.025),ncol=6,frameon=False)
        return fig


def plot_weight_differences(run):
    with plt.rc_context(STYLE):
        fig,axes=_grid(run,'Numerieke afwijking tegenover dezelfde E4M3-bron')
        for i,r in enumerate(ROLES):
            for j,t in enumerate(TRAININGS):
                ax=axes[i,j]
                needed={'training','role','reference_path','other_path','layer','reference_l2_norm','difference_l2_norm'}
                if run.weights.empty or not needed<=set(run.weights):_empty(ax,'Geen volledige L2-diagnostiek');continue
                f=run.weights.loc[(run.weights.training==t)&(run.weights.role==r)&(run.weights.reference_path=='source8')];plotted=False
                for p in DERIVED:
                    z=f.loc[f.other_path==p].set_index('layer').reindex(LAYERS)
                    ref=pd.to_numeric(z.reference_l2_norm,errors='coerce');err=pd.to_numeric(z.difference_l2_norm,errors='coerce')
                    v=err/ref.where(ref!=0)
                    if v.notna().any():
                        ax.plot(np.arange(6),v,color=COLORS[p[0]],ls='-' if BITS[p]==6 else '--',marker='o' if BITS[p]==6 else 's',ms=3,lw=1.7,label=p);plotted=True
                if not plotted:_empty(ax,'Geen vergelijkbare E4M3-bronrijen');continue
                ax.set(xticks=np.arange(6),xticklabels=[s.split('.')[1] for s in LAYERS],xlabel='MX-convolutielaag',ylabel='Relatieve L2-afwijking\ntegenover E4M3',ylim=(0,None));ax.grid(axis='y')
        h,l=axes[0,0].get_legend_handles_labels()
        if h:fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.025),ncol=6,frameon=False)
        return fig


def plot_prediction_diversity(run):
    with plt.rc_context(STYLE):
        fig,axes=_grid(run,'Verdeling van de voorspelde cijfers')
        d=prediction_table(run)
        for i,r in enumerate(ROLES):
            for j,t in enumerate(TRAININGS):
                ax=axes[i,j]
                if d.empty:_empty(ax,'Geen geverifieerde per-beeldvoorspellingen');continue
                f=d.loc[(d.training==t)&(d.role==r)].set_index('path').reindex(PATHS)
                bars=ax.bar(np.arange(9),100*f.dominant_fraction,color=[COLORS[p[0]] if p in DERIVED else COLORS['trained'] for p in PATHS])
                ax.bar_label(bars,labels=[f'{int(v)}/10' if pd.notna(v) else '?' for v in f.classes_predicted],fontsize=8,padding=3)
                ax.set(xticks=np.arange(9),xticklabels=['Bron8','Train6','Train4',*DERIVED],ylim=(0,110),ylabel='Aandeel dominante klasse [%]');ax.tick_params(axis='x',rotation=30);ax.grid(axis='y')
        fig.text(.5,.035,'Labels boven de balken: aantal voorspelde klassen / 10. Eén klasse en 100% concentratie betekent een constante voorspelling.',ha='center',fontsize=9,color='#59616b')
        return fig


def save_figures(run, figures):
    """Explicitly save PNG/PDF in a fresh directory; never save an executed notebook."""
    for name in figures:
        if not name or not all(c.isalnum() or c in '_-' for c in name):raise ValueError('Ongeldige figuurnaam.')
    before={str(p):_hash(p) for p in run.input_files}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');target=run.path/('analysis_'+stamp);target.mkdir(exist_ok=False)
    for name,figure in figures.items():
        figure.savefig(target/(name+'.png'),dpi=180,bbox_inches='tight');figure.savefig(target/(name+'.pdf'),bbox_inches='tight')
    if before!={str(p):_hash(p) for p in run.input_files}:raise RuntimeError('Een ruwe invoer is tijdens analyse gewijzigd.')
    meta={'schema':'mx-triple-scale64-posthoc-analysis-v1','synthetic_fixture_only':run.synthetic,'source_directory':str(run.path),
          'confirmed_evaluation':run.confirmed,'input_sha256':before,'figures':list(figures),'raw_inputs_unchanged':True,
          'training_or_model_evaluation_performed':False,'executed_notebook_saved':False}
    (target/'analysis_provenance.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return target
