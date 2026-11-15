"""Standalone, unexecuted result notebook for the five-format direct reference100.

No new training results are embedded. Current metrics and the verified historical
three-format direct reference are read;
figure exports use a new analysis directory and never overwrite raw results.
"""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

NOTEBOOK_NAME = 'MX_five_direct_reference100_resultaten.ipynb'
ROOT = Path(__file__).resolve().parent
FORMATS = ('E4M3', 'E5M2', 'E2M3', 'E3M2', 'E2M1')


def _cell(kind: str, source: str) -> dict:
    result = {'cell_type': kind, 'id': kind[:1] + '-' + hashlib.sha256(source.encode()).hexdigest()[:12], 'metadata': {}, 'source': source.strip().splitlines(keepends=True)}
    # splitlines preserves internal newlines; a final newline is not required.
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook(run_dir: Path | None = None) -> dict:
    configured_path = json.dumps(str(run_dir.resolve()) if run_dir is not None else '.', ensure_ascii=False)
    cells = [
        _cell('markdown', r'''
# Vijf directe MX-gewichtsformaten — honderd-epochreferentie

**Onderzoeksvraag:** welke nauwkeurigheid bereikt één gezamenlijk netwerk met E4M3, E5M2, E2M3, E3M2 en E2M1 wanneer ieder formaat rechtstreeks uit dezelfde FP32-mastergewichten wordt gekwantiseerd?

SVHN · CNN-8 (`svhnq`, expand=2) · A32 · MX-blokken van 32 · Adam, lr 0,001, weight decay 0 · batch 128 · seed 20261009 · nieuwe initialisatie.

**Primair eindpunt:** alle vijf formaten op hetzelfde checkpoint na epoch 100. Er wordt geen beste epoch op de testset geselecteerd. Een onvoltooide run wordt niet als een voltooide honderd-epochproef gepresenteerd.

De aanvullende historische vergelijking gebruikt de eerdere **directe gezamenlijke training met drie MX-formaten**. Alleen de gedeelde formaten E4M3, E2M3 en E2M1 worden daarbij vergeleken, op dezelfde epoch.

Dit is een directe referentietraining met één vooraf vastgelegde seed. Een technische PASS bewijst geen accuracyverbetering of statistische superioriteit. Deze notebook start geen training en wijzigt geen ruwe onderzoeksdata.
        '''),
        _cell('code', r'''
from pathlib import Path
from datetime import datetime
import json
import math
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = str

RUN_DIR = Path(''' + configured_path + ''').resolve()
FORMATS = ['E4M3', 'E5M2', 'E2M3', 'E3M2', 'E2M1']
BITS = {'E4M3': 8, 'E5M2': 8, 'E2M3': 6, 'E3M2': 6, 'E2M1': 4}
report_path = RUN_DIR / 'report.json'
if not report_path.is_file():
    raise FileNotFoundError(f'Geen report.json in {RUN_DIR}. Vul RUN_DIR in met de nieuwe resultaatmap.')
report = json.loads(report_path.read_text(encoding='utf-8'))
config = report.get('config', {})
metrics_path = RUN_DIR / 'metrics.csv'
metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
print('Resultaatmap:', RUN_DIR)
print('Rapportstatus:', report.get('status', 'ONTBREEKT'))
print('Voltooide epochs volgens rapport:', report.get('completed_epochs', 'niet opgegeven'))
print('Nieuwe resultaten komen uit deze run; de historische referentie wordt afzonderlijk geïdentificeerd.')
if report.get('synthetic_fixture_only'):
    print('ONTWIKKELFIXTURE: de nieuwe meetwaarden zijn synthetisch en vormen geen onderzoeksresultaat.')
        '''),
        _cell('markdown', r'''
## Controle van de meettabel

Accuracy wordt opnieuw berekend uit het aantal correcte voorspellingen en het aantal beelden. De controle hieronder weigert dubbele rijen, onbekende formaten, ongeldige aantallen en afwijkingen tussen aantallen en gerapporteerde accuracy. Ontbrekende epochs worden niet aangevuld.

Bij een technische `check` zijn er geen SVHN-metingen. Dan blijven de accuracytabellen en -grafieken leeg.
        '''),
        _cell('code', r'''
validation_notes = []
if metrics.empty:
    print('Geen epochmetingen beschikbaar. Er wordt geen SVHN-accuracy geconstrueerd.')
else:
    required = {'epoch', 'phase', 'weight_format', 'weight_bits', 'activation_bits',
                'n', 'correct1', 'correct5', 'top1_percent', 'top5_percent',
                'objective_loss', 'hard_ce', 'objective_type', 'teacher_formats', 'lr'}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError('Ontbrekende metricvelden: ' + ', '.join(sorted(missing)))
    metrics = metrics.copy()
    metrics['weight_format'] = metrics['weight_format'].astype(str).map(lambda value: value.rsplit('_', 1)[-1])
    if not set(metrics['weight_format']).issubset(FORMATS):
        raise ValueError('Onbekend gewichtsformaat in metrics.csv.')
    if not set(metrics['phase']).issubset({'train', 'test'}):
        raise ValueError('Onbekende meetfase in metrics.csv.')
    if metrics.duplicated(['epoch', 'phase', 'weight_format']).any():
        raise ValueError('Dubbele epoch/fase/formaatrijen in metrics.csv.')
    numeric = ['epoch', 'weight_bits', 'activation_bits', 'n', 'correct1', 'correct5',
               'top1_percent', 'top5_percent', 'objective_loss', 'hard_ce', 'lr']
    for name in numeric:
        metrics[name] = pd.to_numeric(metrics[name], errors='raise')
        if not metrics[name].map(math.isfinite).all():
            raise ValueError(f'Niet-eindige waarden in {name}.')
    for name in ['epoch', 'weight_bits', 'activation_bits', 'n', 'correct1', 'correct5']:
        if not (metrics[name] == metrics[name].round()).all():
            raise ValueError(f'Niet-gehele waarden in {name}.')
        metrics[name] = metrics[name].astype('int64')
    if not metrics['epoch'].between(1, 100).all():
        raise ValueError('Deze notebook verwacht uitsluitend referentieepochs 1 tot en met 100.')
    if not (metrics['n'] > 0).all() or not (metrics['activation_bits'] == 32).all():
        raise ValueError('Ongeldig aantal voorbeelden of afwijkend activatiebeleid.')
    if not (metrics['weight_bits'] == metrics['weight_format'].map(BITS)).all():
        raise ValueError('Formaatnaam en elementbitbreedte komen niet overeen.')
    if not ((metrics['correct1'] >= 0) & (metrics['correct1'] <= metrics['correct5']) &
            (metrics['correct5'] <= metrics['n'])).all():
        raise ValueError('Ongeldige correct1/correct5-aantallen.')
    metrics['accuracy_from_counts'] = 100.0 * metrics['correct1'] / metrics['n']
    top5_from_counts = 100.0 * metrics['correct5'] / metrics['n']
    if not ((metrics['accuracy_from_counts'] - metrics['top1_percent']).abs() < 1e-4).all():
        raise ValueError('Top-1-accuracy wijkt af van de tellingen.')
    if not ((top5_from_counts - metrics['top5_percent']).abs() < 1e-4).all():
        raise ValueError('Top-5-accuracy wijkt af van de tellingen.')
    expected_n = {'train': int(config.get('train_examples', 73257)),
                  'test': int(config.get('test_examples', 26032))}
    metrics['full_split'] = metrics.apply(lambda row: row['n'] == expected_n[row['phase']], axis=1)
    if not metrics['full_split'].all():
        validation_notes.append('Er zijn rijen zonder de volledige afgesproken split; die worden niet als volledige epochmeting gebruikt.')
    for (epoch, phase), group in metrics.groupby(['epoch', 'phase']):
        if set(group['weight_format']) != set(FORMATS):
            validation_notes.append(f'Epoch {epoch}, {phase}: niet alle vijf formaten aanwezig.')
    print(f'{len(metrics)} unieke metricrijen; aantallen en accuracy consistent.')
    for note in validation_notes:
        print('LET OP:', note)
        '''),
        _cell('markdown', r'''
## Gezamenlijke trainingsdoelen

| Formaat | Trainingsdoel | Lossgewicht |
|---|---|---:|
| E4M3 | Harde labels | 1 |
| E5M2 | Harde labels | 1 |
| E2M3 | Losgekoppelde E4M3-kansverdeling | 1 |
| E3M2 | Losgekoppelde E5M2-kansverdeling | 1 |
| E2M1 | ½ E2M3-kansverdeling + ½ E3M2-kansverdeling, beide losgekoppeld | 1 |

Temperatuur 1. Vijf forward/backward-passes en één Adam-update per minibatch. De vijf losses worden opgeteld. De combinatie van twee teachers geeft E2M1 één lossbijdrage. De kansverdelingen worden na softmax gemiddeld.

De distillatie van voorspellingen staat los van de gewichtsconversie: elk formaat wordt rechtstreeks uit de actuele `G(W)` gemaakt. Alle vijf modi gebruiken hun eigen BatchNorm-toestand. De twee 8-bitteachers zijn modi van hetzelfde netwerk en leren vanaf nul mee.
        '''),
        _cell('markdown', r'''
## Primair eindpunt: epoch 100

Alle vijf scores hieronder horen bij hetzelfde bevestigde epoch-100-checkpoint. Naast volledige metingen moet `report.json` aangeven dat honderd epochs volledig zijn opgeslagen (`completed_epochs == 100`). Bij een opslagfout kunnen er al epoch-100-metingen bestaan zonder een voltooid checkpoint; die worden niet in deze primaire tabel getoond. Een eerdere epoch wordt niet als eindresultaat vervangen. De beschikbare epochmetingen blijven verderop zichtbaar.
        '''),
        _cell('code', r'''
endpoint = pd.DataFrame()
complete_test = pd.DataFrame()
if not metrics.empty:
    complete_test = metrics[(metrics['phase'] == 'test') & metrics['full_split']].copy()
    candidate = complete_test[complete_test['epoch'] == 100]
    if (report.get('completed_epochs') == 100 and len(candidate) == 5
            and set(candidate['weight_format']) == set(FORMATS)):
        endpoint = candidate.set_index('weight_format').loc[FORMATS].reset_index()
        table = endpoint[['weight_format', 'weight_bits', 'correct1', 'n',
                          'accuracy_from_counts', 'hard_ce']].copy()
        table.columns = ['Formaat', 'Elementbits', 'Correct', 'Testbeelden', 'Accuracy (%)', 'Harde test-CE']
        table['Accuracy (%)'] = table['Accuracy (%)'].map(lambda value: f'{value:.6f}')
        table['Harde test-CE'] = table['Harde test-CE'].map(lambda value: f'{value:.6f}')
        display(table)
    else:
        print('Geen bevestigde opgeslagen epoch 100 met volledige metingen voor alle vijf formaten; het primaire eindpunt ontbreekt.')
        if len(candidate) == 5 and report.get('completed_epochs') != 100:
            print('Er zijn wel epoch-100-testmetingen, maar report.json bevestigt geen voltooid epoch-100-checkpoint. De curves mogen deze beschikbare metingen tonen.')
else:
    print('Geen SVHN-eindresultaat beschikbaar.')
        '''),
        _cell('markdown', r'''
## Accuracyverloop per formaat

Elk formaat heeft een eigen paneel, met dezelfde asschaal. De blauwe curve toont de volledige testmetingen. Procentlabels staan bij epoch 1, 25, 50, 75, 100 en de laatste beschikbare epoch; zo blijft een honderd-epochcurve leesbaar. Een grijze lijn toont de volledige trainingsmetingen. Training en test verschillen in dataset en modelmodus; het verschil is geen zuivere schatting van generalisatieverlies.
        '''),
        _cell('code', r'''
fig_accuracy = None
if not metrics.empty:
    complete = metrics[metrics['full_split']].copy()
    if complete.empty:
        print('Geen volledige epochmetingen voor een accuracygrafiek.')
    else:
        fig_accuracy, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        axes = axes.ravel()
        low = max(0, math.floor(float(complete['accuracy_from_counts'].min()) - 3))
        high = min(100, math.ceil(float(complete['accuracy_from_counts'].max()) + 3))
        if high <= low:
            high = min(100, low + 5)
            low = max(0, high - 5)
        for ax, fmt in zip(axes, FORMATS):
            for phase, color, style, label in [('train', '#8a8f98', '--', 'Train'),
                                               ('test', '#1768ac', '-', 'Test')]:
                frame = complete[(complete['weight_format'] == fmt) & (complete['phase'] == phase)].sort_values('epoch')
                if frame.empty:
                    continue
                ax.plot(frame['epoch'], frame['accuracy_from_counts'], linestyle=style,
                        marker='o', markevery=max(1, len(frame) // 12), markersize=3, color=color, linewidth=1.4, label=label)
                if phase == 'test':
                    annotated = frame[frame['epoch'].isin([1, 25, 50, 75, 100, int(frame['epoch'].max())])]
                    for row in annotated.itertuples():
                        ax.annotate(f'{row.accuracy_from_counts:.2f}%',
                                    (row.epoch, row.accuracy_from_counts),
                                    xytext=(0, 7), textcoords='offset points',
                                    ha='center', fontsize=8, color=color)
            ax.set_title(f'{fmt} · W{BITS[fmt]}/A32')
            ax.set_xticks([1, 25, 50, 75, 100])
            ax.set_xlim(0, 103)
            ax.set_ylim(low, high)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(alpha=0.22)
            if ax.lines:
                ax.legend(frameon=False, fontsize=8)
        axes[-1].axis('off')
        fig_accuracy.suptitle('SVHN · CNN-8 · vijf directe MX-formaten · B32/A32\n'
                              'Adam start-lr 0,001 · mijlpalen 50/75/90 · weight decay 0 · batch 128 · T=1', fontsize=12)
        fig_accuracy.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
else:
    print('Geen accuracycurves: technische check of training zonder voltooide metingen.')
        '''),
        _cell('markdown', r'''
## Losses en volledige meettabel

`objective_loss` is het daadwerkelijk geoptimaliseerde trainingsdoel. Voor E2M3, E3M2 en E2M1 is dit een zachte distillatieloss; voor de twee 8-bitmodi is dit harde cross-entropy. `hard_ce` vergelijkt elke voorspelling met de echte labels en is daardoor beter geschikt om het lossverloop van verschillende modi naast elkaar te bekijken. Bij test zijn beide losses harde cross-entropy.

De absolute hoogte van zachte en harde cross-entropy is niet rechtstreeks vergelijkbaar. Ook de totale loss van vijf modi is niet rechtstreeks vergelijkbaar met de oude som van drie modi.
        '''),
        _cell('code', r'''
fig_loss = None
if not metrics.empty:
    complete = metrics[metrics['full_split']].copy()
    fig_loss, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()
    for ax, fmt in zip(axes, FORMATS):
        for phase, color, style, label in [('train', '#8a8f98', '--', 'Harde train-CE'),
                                           ('test', '#1768ac', '-', 'Harde test-CE')]:
            frame = complete[(complete['weight_format'] == fmt) & (complete['phase'] == phase)].sort_values('epoch')
            if not frame.empty:
                ax.plot(frame['epoch'], frame['hard_ce'], linestyle=style,
                        marker='o', markevery=max(1, len(frame) // 12), markersize=3, color=color, label=label)
        ax.set_title(fmt)
        ax.set_xticks([1, 25, 50, 75, 100])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Harde cross-entropy')
        ax.grid(alpha=0.22)
        if ax.lines:
            ax.legend(frameon=False, fontsize=8)
    axes[-1].axis('off')
    fig_loss.suptitle('Harde cross-entropy per gewichtsformaat', fontsize=12)
    fig_loss.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    table = metrics[['epoch', 'phase', 'weight_format', 'n', 'correct1',
                     'accuracy_from_counts', 'objective_loss', 'hard_ce', 'objective_type', 'lr']]
    display(table.sort_values(['epoch', 'phase', 'weight_format']).reset_index(drop=True))
else:
    print('Geen epochlosses beschikbaar.')
        '''),
        _cell('markdown', r'''
## Looptijd en technische rapportage

Bij een voltooide run zijn de honderd train- en testfasetijden daadwerkelijk gemeten. Een eventuele raming tijdens een onvoltooide run is herkenbaar als raming. Datavoorbereiding, controles, checkpointopslag en exports vallen buiten de som van de fasetijden. De sessieduur bevat ook die bijkomende werkzaamheden en betreft bij hervatten alleen de huidige sessie; de opgesomde epochtijden bevatten alle voltooide epochs.
        '''),
        _cell('code', r'''
timing = report.get('timing', {})
if timing:
    print('Gemeten timing en eventuele raming uit report.json:')
    print(json.dumps(timing, indent=2, ensure_ascii=False))
else:
    timing_keys = [key for key in report if any(token in key.lower() for token in ['seconds', 'time', 'timing'])]
    if timing_keys:
        print(json.dumps({key: report[key] for key in timing_keys}, indent=2, ensure_ascii=False))
    else:
        print('Geen timingvelden in dit rapport.')
for key in ['environment', 'execution_settings', 'validation', 'checks', 'warnings', 'error']:
    if key in report:
        print('\n' + key + ':')
        print(json.dumps(report[key], indent=2, ensure_ascii=False))
        '''),
        _cell('markdown', r'''
## Optioneel: figuren bewaren

Zet `SAVE_FIGURES` hieronder op `True` om PNG- en PDF-bestanden te schrijven. Iedere uitvoering maakt een nieuwe analysemap. De bestaande meetbestanden en checkpoints blijven behouden.
        '''),
        _cell('code', r'''
SAVE_FIGURES = False
if SAVE_FIGURES:
    analysis_dir = RUN_DIR / ('analyse_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    analysis_dir.mkdir(exist_ok=False)
    for name, fig in [('accuracy_per_formaat', fig_accuracy), ('harde_ce_per_formaat', fig_loss),
                      ('drie_vs_vijf_epoch100', fig_comparison), ('drie_vs_vijf_curves', fig_comparison_curves)]:
        if fig is not None:
            fig.savefig(analysis_dir / f'{name}.png', dpi=200, bbox_inches='tight')
            fig.savefig(analysis_dir / f'{name}.pdf', bbox_inches='tight')
    print('Figuren opgeslagen in:', analysis_dir)
        '''),
        _cell('markdown', r'''
## Interpretatie en vervolg

- Deze run toetst de nieuwe gezamenlijke training. De distillatiestructuur is een vooraf gekozen ontwerp, geen bewezen optimum.
- Verschillen tussen de vijf modi zijn gemeten binnen één gezamenlijk model. Ze zijn geen vergelijking tussen vijf onafhankelijk geoptimaliseerde modellen.
- De historische directe drie-formaatrun geeft context voor de uitbreiding. De truncatie-, ×64- en MSB-proeven worden hier niet als referentie gebruikt.
- Deze honderd-epochreferentie begint vanaf nul, zonder het vijf-epochpilotcheckpoint. Voor een latere vergelijking met truncatie blijven lossgewichten, teachers, dataset en optimizerbeleid gelijk.
- Voor truncatie moeten we afzonderlijk beslissen hoe beide 8-bitformaten beschikbaar worden, de schaalcorrectie van elke nieuwe route numeriek afleiden en één canonieke E2M1-afleiding vastleggen. De bestaande ×64-regel wordt niet automatisch op nieuwe routes toegepast.

Bronnen voor deze proef: `PROTOCOL.md`, `README.md`, de meegeleverde bron- en pakketmanifesten; nieuwe metingen uit `report.json` en `metrics.csv`; historische metingen uit de geverifieerde bestanden in `reference_data/`.
        '''),
    ]
    insertion = next(i for i, cell in enumerate(cells) if '## Looptijd en technische rapportage' in ''.join(cell['source']))
    cells[insertion:insertion] = comparison_cells()
    return {'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python (.venv-train)', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.13'},
                         'mx_experiment': {'package': 'mx_weights_five_direct_reference100_v1',
                                           'endpoint_epoch': 100, 'unexecuted_template': True}},
            'nbformat': 4, 'nbformat_minor': 5}


def comparison_cells() -> list[dict]:
    return [
        _cell('markdown', r'''
## Historische referentie: drie directe MX-gewichtsformaten

De referentie bestaat uit de eerdere gezamenlijke **directe** E4M3/E2M3/E2M1-training. De bestanden zijn afzonderlijk opgenomen in `reference_data/`. Het metadata-bestand identificeert de oorspronkelijke feedbackbundel en bevat de hash van de meettabel. De notebook controleert die hash, het type referentie, de voorbeeldtellingen en de accuracyberekening voordat hij een vergelijking maakt.

Deze vergelijking gaat over twee volledige trainingsopzetten. De nieuwe run heeft vijf formaatpasses, twee harde-labelteachers en voor E2M1 een gemiddelde van twee 6-bitteachers. De oude run heeft drie passes, één harde-labelteacher en de keten E4M3 → E2M3 → E2M1. Er zijn ook vijf in plaats van drie actieve BatchNorm-banken en andere opeenvolgende dropouttrekkingen. Gelijke seeds maken de gradiënten en RNG-trajecten daarom niet identiek.

**Interpretatie:** dit is een beschrijvende historische vergelijking, geen geïsoleerde causale meting van alleen het toevoegen van E5M2/E3M2. Er is geen variatie over meerdere seeds gemeten. De oude truncatie-, ×64- en MSB-proeven worden hier niet als directe referentie gebruikt.
        '''),
        _cell('code', r'''
COMMON_FORMATS = ['E4M3', 'E2M3', 'E2M1']
reference_metrics = pd.DataFrame()
reference_info = {}
reference_compatible = False
reference_notes = []
ref_dir = RUN_DIR / 'reference_data'
ref_csv = ref_dir / 'triple_direct_metrics.csv'
ref_json = ref_dir / 'triple_direct_reference.json'
if not ref_csv.is_file() or not ref_json.is_file():
    print('De historische directe drie-formaatreferentie ontbreekt. Er worden geen vergelijkingswaarden ingevuld.')
else:
    reference_info = json.loads(ref_json.read_text(encoding='utf-8'))
    actual_hash = hashlib.sha256(ref_csv.read_bytes()).hexdigest()
    if actual_hash != reference_info.get('metrics_sha256'):
        raise ValueError('De historische meettabel wijkt af van haar opgeslagen hash.')
    if reference_info.get('reference_kind') != 'three_format_joint_direct':
        raise ValueError('De opgenomen referentie is niet de afgesproken directe drie-formaatrun.')
    if reference_info.get('status') != 'PASS' or reference_info.get('completed_epochs') != 100:
        raise ValueError('De historische referentie bevestigt geen voltooide honderd-epochrun.')
    ref_config = reference_info.get('config', {})
    if ref_config.get('truncation') is not False or ref_config.get('activation_bits') != 32:
        raise ValueError('De referentie heeft een ander truncatie- of activatiebeleid.')
    reference_metrics = pd.read_csv(ref_csv)
    ref_required = {'epoch', 'phase', 'weight_format', 'n', 'correct1', 'top1_percent'}
    if not ref_required.issubset(reference_metrics.columns):
        raise ValueError('Vereiste kolommen ontbreken in de historische referentie.')
    reference_metrics['weight_format'] = reference_metrics['weight_format'].astype(str).map(lambda value: value.rsplit('_', 1)[-1])
    if set(reference_metrics['weight_format']) != set(COMMON_FORMATS):
        raise ValueError('De historische meettabel moet precies E4M3, E2M3 en E2M1 bevatten.')
    if not set(reference_metrics['phase']).issubset({'train', 'test'}):
        raise ValueError('Onbekende meetfase in historische referentie.')
    if reference_metrics.duplicated(['epoch', 'phase', 'weight_format']).any():
        raise ValueError('Dubbele rijen in de historische meettabel.')
    for field in ['epoch', 'n', 'correct1', 'top1_percent']:
        reference_metrics[field] = pd.to_numeric(reference_metrics[field], errors='raise')
        if not reference_metrics[field].map(math.isfinite).all():
            raise ValueError('Niet-eindige historische waarden: ' + field)
    for field in ['epoch', 'n', 'correct1']:
        if not (reference_metrics[field] == reference_metrics[field].round()).all():
            raise ValueError('Niet-gehele historische telling: ' + field)
        reference_metrics[field] = reference_metrics[field].astype('int64')
    if not reference_metrics['epoch'].between(1, 100).all():
        raise ValueError('Ongeldige historische epoch.')
    if not ((reference_metrics['n'] > 0) & (reference_metrics['correct1'] >= 0) &
            (reference_metrics['correct1'] <= reference_metrics['n'])).all():
        raise ValueError('Ongeldige historische voorbeeldtellingen.')
    reference_metrics['accuracy_from_counts'] = 100.0 * reference_metrics['correct1'] / reference_metrics['n']
    if not ((reference_metrics['accuracy_from_counts'] - reference_metrics['top1_percent']).abs() < 1e-4).all():
        raise ValueError('Historische accuracy wijkt af van de tellingen.')
    ref_expected = {'train': ref_config.get('train_examples'), 'test': ref_config.get('test_examples')}
    if any(ref_expected[phase] is None for phase in set(reference_metrics['phase'])):
        raise ValueError('Historische splitgrootte ontbreekt in de bronmetadata.')
    if not reference_metrics.apply(lambda row: row['n'] == ref_expected[row['phase']], axis=1).all():
        raise ValueError('De historische referentie bevat onvolledige splitmetingen.')
    shared_settings = ['dataset', 'model', 'expand', 'activation_bits', 'mx_block_size',
                       'train_examples', 'test_examples', 'batch_size', 'optimizer',
                       'lr', 'weight_decay', 'scheduler_milestones']
    for key in shared_settings:
        if key not in ref_config or key not in config:
            reference_notes.append(f'{key}: niet in beide protocollen bekend.')
        elif ref_config[key] != config[key]:
            reference_notes.append(f'{key}: oud={ref_config[key]!r}, nieuw={config[key]!r}.')
    reference_compatible = not reference_notes
    print('Historische bron:', reference_info.get('source_filename', 'zie metadata'))
    print('SHA-256 oorspronkelijke bron:', reference_info.get('source_sha256', 'niet opgenomen'))
    print('Historische metingen:', len(reference_metrics), 'rijen; bronhash en tellingen gecontroleerd.')
    if reference_notes:
        print('De vergelijking wordt geblokkeerd door protocolverschillen of ontbrekende metadata:')
        for note in reference_notes:
            print('-', note)
    else:
        print('Dataset, model, A32, B32 en opgegeven optimizerinstellingen stemmen overeen.')
        print('De verschillende gezamenlijke distillatieopzetten blijven een beperking van de interpretatie.')
        '''),
        _cell('markdown', r'''
## Accuracyvergelijking op epoch 100

De tabel en figuur vergelijken **epoch 100 met epoch 100**, voor de drie gedeelde formaten. Het oorspronkelijke beste E4M3-checkpoint wordt niet als vervanging gebruikt. De twee methoden staan op een categorische as; beide gebruiken dezelfde ronde stippen en een percentage bij elk punt. Ieder formaat heeft een eigen paneel.

Een positief verschil in procentpunten betekent hier een hogere gemeten score voor de vijf-formaatrun. Het is geen significantietoets. E5M2 en E3M2 hebben in de oude drie-formaatrun geen score en worden daarom niet in deze vergelijking opgenomen.
        '''),
        _cell('code', r'''
endpoint_comparison = pd.DataFrame()
fig_comparison = None
if not reference_metrics.empty and reference_compatible and not endpoint.empty:
    old_endpoint = reference_metrics[(reference_metrics['phase'] == 'test') & (reference_metrics['epoch'] == 100)]
    if len(old_endpoint) != 3 or set(old_endpoint['weight_format']) != set(COMMON_FORMATS):
        print('De historische meettabel bevat geen volledige epoch 100 voor alle drie gedeelde formaten.')
    else:
        old = old_endpoint.set_index('weight_format')
        new = endpoint.set_index('weight_format')
        rows = []
        for fmt in COMMON_FORMATS:
            if int(old.loc[fmt, 'n']) != int(new.loc[fmt, 'n']):
                raise ValueError('Verschillende testsetgroottes bij de eindpuntvergelijking.')
            rows.append({'Formaat': fmt, 'Epoch': 100,
                         '3 formaten, direct (%)': float(old.loc[fmt, 'accuracy_from_counts']),
                         '5 formaten, direct (%)': float(new.loc[fmt, 'accuracy_from_counts']),
                         'Verschil (procentpunt)': float(new.loc[fmt, 'accuracy_from_counts'] - old.loc[fmt, 'accuracy_from_counts']),
                         'Verschil correcte beelden': int(new.loc[fmt, 'correct1'] - old.loc[fmt, 'correct1'])})
        endpoint_comparison = pd.DataFrame(rows)
        display(endpoint_comparison.round(6))
        fig_comparison, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
        values = endpoint_comparison[['3 formaten, direct (%)', '5 formaten, direct (%)']].to_numpy().ravel()
        span = float(values.max() - values.min())
        padding = max(0.15, span * 0.2)
        for ax, row in zip(axes, endpoint_comparison.to_dict('records')):
            scores = [row['3 formaten, direct (%)'], row['5 formaten, direct (%)']]
            ax.scatter(scores, [0, 1], s=70, marker='o', color=['#777d86', '#1768ac'], zorder=3)
            for score, position in zip(scores, [0, 1]):
                ax.annotate(f'{score:.3f}%', (score, position), xytext=(0, 12),
                            textcoords='offset points', ha='center', fontsize=10)
            ax.set_title(f"{row['Formaat']} · Δ {row['Verschil (procentpunt)']:+.3f} pp")
            ax.set_yticks([0, 1], ['3 formaten, direct', '5 formaten, direct'])
            ax.set_ylim(-0.45, 1.55)
            ax.set_xlim(max(0, float(values.min()) - padding), min(100, float(values.max()) + padding))
            ax.set_xlabel('Testaccuracy (%)')
            ax.grid(axis='x', alpha=0.22)
        fig_comparison.suptitle('SVHN · CNN-8 · A32/B32 · epoch 100 in beide trainingen\n'
                               'Historische vergelijking van drie en vijf directe MX-gewichtsformaten', fontsize=12)
        fig_comparison.tight_layout(rect=[0, 0, 1, 0.88])
        plt.show()
else:
    print('Geen primaire historische vergelijking: vereist een bevestigde nieuwe epoch 100 en een geldige directe drie-formaatreferentie.')
        '''),
        _cell('markdown', r'''
## Vergelijking van het leerverloop

Deze drie panelen gebruiken uitsluitend epochs waarvoor beide trainingen een volledige testmeting hebben. Bij een nog onvoltooide nieuwe run wordt de oude curve dus beperkt tot dezelfde beschikbare epochs. De eindlabels tonen de laatste gezamenlijke epoch, niet noodzakelijk epoch 100. Ontbrekende waarden worden niet geïnterpoleerd of aangevuld.
        '''),
        _cell('code', r'''
fig_comparison_curves = None
matched_curves = {}
if not reference_metrics.empty and reference_compatible and not complete_test.empty:
    for fmt in COMMON_FORMATS:
        old = reference_metrics[(reference_metrics['phase'] == 'test') & (reference_metrics['weight_format'] == fmt)]
        new = complete_test[complete_test['weight_format'] == fmt]
        merged = old[['epoch', 'accuracy_from_counts', 'n']].merge(
            new[['epoch', 'accuracy_from_counts', 'n']], on='epoch', how='inner', suffixes=('_old', '_new'))
        if not (merged['n_old'] == merged['n_new']).all():
            raise ValueError('Verschillende voorbeeldtellingen in gematchte curve-epochs.')
        if not merged.empty:
            matched_curves[fmt] = merged.sort_values('epoch')
    if matched_curves:
        fig_comparison_curves, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
        maximum_epoch = max(int(frame['epoch'].max()) for frame in matched_curves.values())
        all_scores = pd.concat([frame[['accuracy_from_counts_old', 'accuracy_from_counts_new']]
                                for frame in matched_curves.values()]).to_numpy().ravel()
        low, high = max(0, math.floor(float(all_scores.min()) - 2)), min(100, math.ceil(float(all_scores.max()) + 2))
        for ax, fmt in zip(axes, COMMON_FORMATS):
            frame = matched_curves.get(fmt)
            if frame is None:
                ax.text(0.5, 0.5, 'Geen gezamenlijke epochs', transform=ax.transAxes, ha='center')
            else:
                for field, color, label, offset in [
                    ('accuracy_from_counts_old', '#777d86', '3 formaten, direct', -16),
                    ('accuracy_from_counts_new', '#1768ac', '5 formaten, direct', 10)]:
                    ax.plot(frame['epoch'], frame[field], color=color, marker='o', markersize=3,
                            markevery=max(1, len(frame) // 12), linewidth=1.4, label=label)
                    last = frame.iloc[-1]
                    ax.annotate(f"{last[field]:.2f}%", (last['epoch'], last[field]),
                                xytext=(-3, offset), textcoords='offset points', ha='right', color=color, fontsize=9)
                ax.legend(frameon=False, fontsize=8)
            ax.set_title(fmt)
            ticks = sorted(set([value for value in [1, 25, 50, 75, 100] if value <= maximum_epoch] + [maximum_epoch]))
            ax.set_xticks(ticks)
            ax.set_xlim(0, max(2, maximum_epoch * 1.04))
            ax.set_ylim(low, max(low + 1, high))
            ax.set_xlabel('Epoch in beide trainingen')
            ax.set_ylabel('Testaccuracy (%)')
            ax.grid(alpha=0.22)
        fig_comparison_curves.suptitle('Gedeelde MX-formaten · dezelfde epochs · directe kwantisatie\n'
                                      'Verschillende gezamenlijke trainingsopzetten; beschrijvende vergelijking', fontsize=12)
        fig_comparison_curves.tight_layout(rect=[0, 0, 1, 0.88])
        plt.show()
else:
    print('Geen historische curvenvergelijking beschikbaar.')
        '''),
    ]


def write_notebook(out: Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / NOTEBOOK_NAME
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(build_notebook(out), handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    reference_source = ROOT / 'reference_data'
    reference_names = ('triple_direct_metrics.csv', 'triple_direct_reference.json')
    present = [(reference_source / name).is_file() for name in reference_names]
    if any(present) and not all(present):
        raise ValueError('De historische referentie moet zowel metadata als meettabel bevatten.')
    if all(present):
        metadata = json.loads((reference_source / reference_names[1]).read_text(encoding='utf-8'))
        digest = hashlib.sha256((reference_source / reference_names[0]).read_bytes()).hexdigest()
        if digest != metadata.get('metrics_sha256'):
            raise ValueError('De historische referentietabel wijkt af van haar metadata.')
        destination = out / 'reference_data'
        # Include the original source evidence as well as normalized plotting data.
        # A new run owns this new directory; no earlier research files are replaced.
        shutil.copytree(reference_source, destination)
    return path


if __name__ == '__main__':
    # Creates the unexecuted package template only, never a training result.
    target = Path(__file__).resolve().parent / NOTEBOOK_NAME
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(build_notebook(), handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print(target)
