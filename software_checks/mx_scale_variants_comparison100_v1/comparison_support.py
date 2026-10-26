"""Alleen-lezen vergelijking van vier bestaande, echte MX-trainingsresultaten.

Geen Torch, checkpoints, downloads of trainingsaanroepen. De vijf-epochpilot en
de voortzetting vormen samen het bevroren traject vanaf epoch 51; epochs 1–50
komen uit de gemeenschappelijke geleerde voorgeschiedenis.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ORDER = ('fixed64', 'fixed32', 'learned', 'frozen')
BITS = (8, 6, 4)
LABELS = {'fixed64': 'Vast ×64', 'fixed32': 'Vast ×32',
          'learned': 'Geleerd tot 100', 'frozen': 'Geleerd; vast na 50'}
FORMATS = {8: 'E4M3', 6: 'E2M3', 4: 'E2M1'}
COLORS = {'fixed64': '#315F8C', 'fixed32': '#218A7E',
          'learned': '#C18A28', 'frozen': '#8258A4'}
MODES = {8: 'MXFP8_E4M3', 6: 'MXFP6_E2M3', 4: 'MXFP4_E2M1'}
BANKS = {8: '8', 6: '32', 4: '4'}
ROLES = ('epoch100', 'global_best', 'window51_best')
ROLE_LABELS = {'epoch100': 'Epoch 100', 'global_best': 'Beste E4M3 binnen 1–100',
               'window51_best': 'Beste E4M3 binnen 51–100'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _equal(a, b, tolerance=1e-7):
    return bool(np.isclose(float(a), float(b), rtol=0, atol=tolerance))


def _lr(epoch):
    return .001 * (.1 ** sum(m < int(epoch) for m in (50, 75, 90)))


@dataclass
class Comparison:
    root: Path
    manifest: dict
    reports: dict
    selections: dict
    summaries: dict
    raw: dict
    test: dict
    chosen_epochs: dict
    checks: list

    def table(self, role):
        _require(role in ROLES, 'Onbekende checkpointselectie: ' + str(role))
        rows = []
        for variant in ORDER:
            epoch = self.chosen_epochs[role][variant]
            group = self.test[variant].loc[self.test[variant].epoch.eq(epoch)].set_index('weight_bits')
            row = {'Variant': LABELS[variant], 'Epoch': epoch}
            for bit in BITS:
                row[FORMATS[bit] + ' (%)'] = float(group.loc[bit, 'top1_percent'])
            row['Herkomst'] = ('Gedeelde voorfase' if variant == 'frozen' and epoch <= 50
                               else 'Bevroren fase' if variant == 'frozen' else 'Eigen training')
            rows.append(row)
        return pd.DataFrame(rows)

    def differences(self, reference='fixed64', role='epoch100'):
        _require(reference in ORDER, 'Onbekende referentievariant.')
        table = self.table(role).set_index('Variant')
        columns = [FORMATS[b] + ' (%)' for b in BITS]
        result = table[columns].sub(table.loc[LABELS[reference], columns], axis='columns')
        result.columns = [FORMATS[b] + ' (procentpunt)' for b in BITS]
        return result.reset_index()

    def availability(self):
        return pd.DataFrame([
            {'Variant': LABELS[v], 'Voltooide eindepoch': 100,
             'Eigen gemeten epochs': '51–100; 1–50 gedeeld' if v == 'frozen' else '1–100',
             'Globaal E4M3-best': self.chosen_epochs['global_best'][v],
             'E4M3-best 51–100': self.chosen_epochs['window51_best'][v],
             'Nieuwe training nodig': 'Nee'} for v in ORDER])


def load_comparison(root):
    root = Path(root).expanduser().resolve()
    manifest = _json(root / 'SOURCE_MANIFEST.json')
    hashes = manifest.get('source_files_sha256')
    _require(isinstance(hashes, dict) and hashes, 'Bronmanifest mist source_files_sha256.')
    checks = []
    for name, expected in hashes.items():
        path = (root / name).resolve()
        _require(path.is_relative_to(root) and path.is_file(), 'Bronbestand ontbreekt: ' + name)
        _require(_hash(path) == expected, 'Bronbestand gewijzigd: ' + name)
    checks.append(f'{len(hashes)} bronbestanden gecontroleerd op SHA256')

    reports, selections, summaries, raw, selected = {}, {}, {}, {}, {}
    for variant in ORDER:
        folder = root / 'data' / variant
        reports[variant] = report = _json(folder / 'run_report.json')
        selections[variant] = _json(folder / 'checkpoint_selection.json')
        summaries[variant] = summary = _json(folder / 'training_summary.json')
        config = report['config']
        _require(report.get('status') == 'PASS' and report.get('completed_epoch') == 100
                 and report.get('full_training_completed') is True
                 and report.get('synthetic_fixture_only') is False,
                 LABELS[variant] + ': geen voltooide echte 100-epochrun.')
        _require(summary.get('full_training_completed') is True and summary.get('synthetic_fixture_only') is False,
                 LABELS[variant] + ': samenvatting is niet compleet of is synthetisch.')
        _require(config.get('activation_bits') == 32 and config.get('master_dtype') == 'float32'
                 and config.get('gradient_dtype') == 'float32'
                 and config.get('adam_moments_dtype') == 'float32',
                 LABELS[variant] + ': verwacht A32 en FP32-training.')
        _require(config.get('train_examples') == 73257 and config.get('test_examples') == 26032
                 and config.get('extra_used') is False,
                 LABELS[variant] + ': verkeerde datasetomvang.')
        _require(config.get('exponent_truncation') is True and config.get('mantissa_truncation') is True,
                 LABELS[variant] + ': verwacht dezelfde beide truncatieroutes.')
        frame = pd.read_csv(folder / 'epoch_metrics.csv', float_precision='round_trip')
        selected[variant] = pd.read_csv(folder / 'selected_metrics.csv', float_precision='round_trip')
        fields = ['epoch', 'weight_bits', 'activation_bits', 'n', 'correct1', 'correct5',
                  'objective_loss', 'top1_percent', 'top5_percent', 'lr']
        _require(set(fields + ['phase', 'weight_format', 'bn_bank_key']) <= set(frame),
                 LABELS[variant] + ': onvolledige metriek.')
        _require(np.isfinite(frame[fields].to_numpy(float)).all(), LABELS[variant] + ': niet-eindige metriek.')
        start = 51 if variant == 'frozen' else 1
        expected_keys = {(e, phase, bit) for e in range(start, 101) for phase in ('train', 'test') for bit in BITS}
        observed_keys = set(zip(frame.epoch.astype(int), frame.phase, frame.weight_bits.astype(int)))
        _require(observed_keys == expected_keys and len(frame) == len(expected_keys),
                 LABELS[variant] + ': ontbrekende of dubbele epoch-/fase-/precisierij.')
        for row in frame.itertuples():
            bit = int(row.weight_bits)
            _require(row.activation_bits == 32 and row.n == (73257 if row.phase == 'train' else 26032)
                     and row.weight_format == MODES[bit] and str(row.bn_bank_key) == BANKS[bit],
                     LABELS[variant] + ': modus- of datasetveld klopt niet.')
            _require(0 <= row.correct1 <= row.correct5 <= row.n
                     and _equal(row.top1_percent, 100 * row.correct1 / row.n, 2e-6)
                     and _equal(row.top5_percent, 100 * row.correct5 / row.n, 2e-6),
                     LABELS[variant] + ': accuracy en gehele aantallen verschillen.')
            _require(_equal(row.lr, _lr(row.epoch), 1e-12), LABELS[variant] + ': verkeerd gewichts-LR-schema.')
        raw[variant] = frame
        checks.append(LABELS[variant] + ': compleet, echt, correcte A32/FP32-metrics en LR')

    common_fields = ('dataset', 'model', 'expand', 'train_split', 'test_split', 'batch_size', 'seed',
                     'mx_block_size', 'optimizer', 'lr', 'weight_decay', 'scheduler_milestones',
                     'scheduler_gamma', 'loss', 'training_mode_order', 'passes_per_training_batch',
                     'optimizer_steps_per_training_batch', 'bn_bank_mapping', 'source_weight_bits',
                     'weight_preprocess', 'exponent_mapping')
    reference_config = reports['fixed64']['config']
    for field in common_fields:
        _require(all(reports[v]['config'].get(field) == reference_config.get(field) for v in ORDER),
                 'Vergelijkingsprotocol verschilt bij ' + field)
    _require(reports['fixed64']['initial_model_sha256'] == reports['fixed32']['initial_model_sha256']
             == reports['learned']['initial_base_model_sha256'], 'De basisinitialisatie verschilt.')
    checks.append('Gemeenschappelijk protocol en basisinitialisatie gecontroleerd')

    for variant, factor, offset in (('fixed32', 32, 5), ('fixed64', 64, 6)):
        folder = root / 'data' / variant
        config = reports[variant]['config']
        _require(config.get('scale_factor') == factor and config.get('scale_exponent_offset') == offset
                 and config.get('learned_shared_scale') is False, 'Verkeerde vaste factor: ' + variant)
        marker = _json(folder / 'REFERENCE.json')
        _require(marker.get('factor') == factor and marker.get('completed_epochs') == 100
                 and marker.get('synthetic_fixture_only') is False, 'Verkeerde vaste referentie: ' + variant)
        for name, expected in marker['files_sha256'].items():
            _require(_hash(folder / name) == expected, 'Vaste referentiehash verschilt: ' + variant + '/' + name)
    for variant in ('learned', 'frozen'):
        config = reports[variant]['config']
        _require(config.get('scale_exponent_offset_choices') == [5, 6]
                 and config.get('learned_scale_parameters') == 4080
                 and config.get('source_scale_learned') is False, 'Verkeerde geleerde route: ' + variant)

    lineage = _json(root / 'data/frozen/lineage.json')
    parent = lineage['parent_verification']
    _require(parent == reports['frozen']['parent_verification']
             and parent['parent_epoch'] == 50
             and parent['parent_contract_sha256'] == reports['learned']['contract_sha256']
             and parent['parent_model_sha256'] == reports['frozen']['initial_model_sha256']
             and parent['parent_checkpoint_verified'] is True,
             'De freeze is niet aantoonbaar afkomstig uit de geleerde training na epoch 50.')
    freeze = _json(root / 'data/frozen/freeze_summary.json')
    _require(freeze['all_theta_unchanged'] is True and freeze['all_hard_offsets_unchanged'] is True
             and freeze['blocks'] == 4080 and freeze['count_x32'] == 1654 and freeze['count_x64'] == 2426
             and freeze['total_choice_flips'] == 0 and freeze['total_theta_changed_blocks'] == 0
             and freeze['per_update_checks'] == 28650 and freeze['source_scale_dynamic'] is True,
             'Freeze-evidence klopt niet.')
    checks.append('Freeze-afstamming, vaste blokkeuzes en dynamische bronschaal gecontroleerd')

    test = {v: raw[v].loc[raw[v].phase.eq('test')].sort_values(['epoch', 'weight_bits']).copy() for v in ORDER}
    shared = test['learned'].loc[test['learned'].epoch.le(50)].copy()
    test['frozen'] = pd.concat([shared, test['frozen']], ignore_index=True).sort_values(['epoch', 'weight_bits'])
    _require(test['frozen'].loc[test['frozen'].epoch.le(50)].reset_index(drop=True).equals(shared.reset_index(drop=True)),
             'De gedeelde eerste 50 epochs zijn niet exact overgenomen.')
    chosen = {role: {} for role in ROLES}
    for variant in ORDER:
        high = test[variant].loc[test[variant].weight_bits.eq(8)].sort_values('epoch')
        for role, start in (('global_best', 1), ('window51_best', 51)):
            candidates = high.loc[high.epoch.between(start, 100)]
            chosen[role][variant] = int(candidates.loc[candidates.top1_percent.idxmax(), 'epoch'])
        chosen['epoch100'][variant] = 100
        selection = selections[variant]
        expected_roles = {'last_epoch': ('epoch100', 'last')}
        if variant == 'frozen':
            expected_roles.update(best_E4M3_global=('global_best', 'best_global'),
                                  best_E4M3_postfreeze=('window51_best', 'best_postfreeze'))
        else:
            expected_roles['best_E4M3_test'] = ('global_best', 'best')
        _require(set(selected[variant].role) == set(expected_roles)
                 and len(selected[variant]) == len(expected_roles) * 3,
                 LABELS[variant] + ': geselecteerde rollen ontbreken of zijn dubbel.')
        for csv_role, (role, selection_key) in expected_roles.items():
            epoch = chosen[role][variant]
            _require(selection[selection_key]['epoch'] == epoch, LABELS[variant] + ': gerapporteerde selectie verschilt.')
            group = selected[variant].loc[selected[variant].role.eq(csv_role)]
            _require(set(group.weight_bits) == set(BITS) and group.epoch.eq(epoch).all(),
                     LABELS[variant] + ': modi gebruiken niet hetzelfde geselecteerde checkpoint.')
            expected = test[variant].loc[test[variant].epoch.eq(epoch)].set_index('weight_bits')
            for row in group.itertuples():
                source = expected.loc[int(row.weight_bits)]
                _require(row.correct1 == source.correct1 and row.n == source.n
                         and _equal(row.top1_percent, source.top1_percent, 1e-10)
                         and _equal(row.objective_loss, source.objective_loss, 1e-12),
                         LABELS[variant] + ': geselecteerde metriek verschilt van de geschiedenis.')
            _require(_equal(selection[selection_key]['selection_score_E4M3'], expected.loc[8, 'top1_percent'], 1e-10),
                     LABELS[variant] + ': E4M3-selectiescore verschilt.')
    if chosen['global_best']['frozen'] == chosen['global_best']['learned'] <= 50:
        _require(selections['frozen']['best_global']['sha256'] == selections['learned']['best']['sha256'],
                 'De gedeelde globale selectie heeft een afwijkende checkpointidentiteit.')
    checks.append('Alle drie selecties herberekend; alle opgeslagen geselecteerde scores sluiten aan')
    return Comparison(root, manifest, reports, selections, summaries, raw, test, chosen, checks)


def styled_table(table):
    """Presentatiekopie zonder de optionele Jinja2-afhankelijkheid van Styler."""
    result = table.copy()
    for column in result.columns:
        if '(%)' in column or '(procentpunt)' in column:
            result[column] = result[column].map(lambda value: f'{float(value):.4f}'.replace('.', ','))
    return result


def selection_figure(comparison, role):
    table = comparison.table(role)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(12.0, 5.7))
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.76)
    width = .19
    x = np.arange(len(BITS))
    for i, variant in enumerate(ORDER):
        values = [float(table.loc[i, FORMATS[bit] + ' (%)']) for bit in BITS]
        epoch = comparison.chosen_epochs[role][variant]
        label = LABELS[variant] if role == 'epoch100' else f'{LABELS[variant]} · epoch {epoch}'
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=COLORS[variant], zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .055,
                    f'{value:.3f}'.replace('.', ','), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, [FORMATS[b] for b in BITS])
    ax.set_ylim(92.5, 95.5)
    ax.set_yticks(np.arange(92.5, 95.51, .5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'.replace('.', ',')))
    ax.set_ylabel('Testaccuracy (%) — uitvergrote y-as')
    ax.grid(axis='y', alpha=.18, zorder=0)
    fig.suptitle(ROLE_LABELS[role], x=.075, ha='left', y=.975, fontsize=15, fontweight='bold')
    subtitle = ('Alle vier varianten op hetzelfde vooraf afgesproken eindpunt.' if role == 'epoch100'
                else 'Per variant één E4M3-selectie; alle drie precisies uit datzelfde checkpoint.')
    fig.text(.075, .91, subtitle, fontsize=10.5, color='#4D5963')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.068, .885), ncol=2, frameon=False, fontsize=9.5)
    if role == 'global_best':
        foot = 'Geleerd en vast-na-50 selecteren allebei epoch 34: hetzelfde model uit de gemeenschappelijke fase vóór de freeze.'
    elif role == 'window51_best':
        foot = 'Aanvullend venster 51–100 voor alle vier varianten; dit vervangt de globale selectie binnen 1–100 niet.'
    else:
        foot = 'SVHN-test: 26.032 beelden · A32 · FP32-master · één seed · uitsluitend reeds uitgevoerde trainingen.'
    fig.text(.075, .065, foot, fontsize=9.5, color='#4D5963', wrap=True)
    return fig


def curves_figure(comparison):
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 10.5), sharex=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.13, top=.865, hspace=.33)
    all_values = pd.concat([comparison.test[v].top1_percent for v in ORDER])
    lower = max(0, np.floor(all_values.min()) - .5)
    upper = min(100, np.ceil(all_values.max() * 2) / 2 + .2)
    for ax, bit in zip(axes, BITS):
        for variant in ORDER:
            values = comparison.test[variant].loc[comparison.test[variant].weight_bits.eq(bit)].sort_values('epoch')
            # De identieke voorfase is één gemeten lijn; na 50 verschijnt de vertakking.
            if variant == 'frozen':
                values = values.loc[values.epoch.ge(50)]
            ax.plot(values.epoch, values.top1_percent, label=LABELS[variant], color=COLORS[variant],
                    linewidth=1.7 if variant == 'frozen' else 1.35, linestyle='--' if variant == 'frozen' else '-')
        ax.axvline(50, color=COLORS['frozen'], linestyle=':', linewidth=1.25)
        ax.set_title(FORMATS[bit], loc='left', fontsize=11, fontweight='bold')
        ax.set_ylabel('Testaccuracy (%)')
        ax.set_xlim(1, 100)
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}'))
        ax.grid(alpha=.16)
    axes[-1].set_xlabel('Epoch')
    axes[-1].set_xticks([1, 10, 25, 50, 75, 90, 100])
    fig.suptitle('Het volledige gemeten traject', x=.075, ha='left', y=.975, fontsize=15, fontweight='bold')
    fig.text(.075, .937, 'Vast-na-50 deelt exact epochs 1–50 met de geleerde training; vanaf epoch 51 worden de offsets vastgehouden.', fontsize=10.5, color='#4D5963')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.068, .915), ncol=4, frameon=False, fontsize=9.5)
    fig.text(.075, .045, 'De paarse stippellijn markeert de freeze na epoch 50. Deze grafiek bevat geen nieuwe evaluaties of gladgestreken waarden.', fontsize=9.5, color='#4D5963')
    return fig


def cost_table():
    return pd.DataFrame([
        {'Variant': LABELS['fixed64'], 'Extra FP32-theta’s tijdens training': 0,
         'Extra blokkeuzes in compacte export': '0; factor is globaal vast', 'Offsetbeleid': 'Altijd +6 (×64)'},
        {'Variant': LABELS['fixed32'], 'Extra FP32-theta’s tijdens training': 0,
         'Extra blokkeuzes in compacte export': '0; factor is globaal vast', 'Offsetbeleid': 'Altijd +5 (×32)'},
        {'Variant': LABELS['learned'], 'Extra FP32-theta’s tijdens training': 4080,
         'Extra blokkeuzes in compacte export': '4080 bits = 510 bytes', 'Offsetbeleid': '+5/+6 per blok; leren tot 100'},
        {'Variant': LABELS['frozen'], 'Extra FP32-theta’s tijdens training': 4080,
         'Extra blokkeuzes in compacte export': '4080 bits = 510 bytes', 'Offsetbeleid': '+5/+6 leren tot 50, daarna vast'},
    ])
