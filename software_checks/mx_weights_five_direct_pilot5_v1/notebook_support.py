"""Standalone, unexecuted result notebook for the five-format direct pilot.

No training results are embedded. Only report.json and metrics.csv are read;
figure exports use a new analysis directory and never overwrite raw results.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

NOTEBOOK_NAME = 'MX_five_direct_pilot5_resultaten.ipynb'
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
# Vijf directe MX-gewichtsformaten — vijf-epochpilot

**Onderzoeksvraag:** leert één gezamenlijk netwerk met E4M3, E5M2, E2M3, E3M2 en E2M1 correct wanneer ieder formaat rechtstreeks uit dezelfde FP32-mastergewichten wordt gekwantiseerd?

SVHN · CNN-8 (`svhnq`, expand=2) · A32 · MX-blokken van 32 · Adam, lr 0,001, weight decay 0 · batch 128 · seed 20261009 · nieuwe initialisatie.

**Primaire vergelijking:** alle vijf formaten op hetzelfde checkpoint na epoch 5. Er wordt geen beste epoch op de testset geselecteerd. Een onvoltooide run wordt niet als een voltooide vijf-epochproef gepresenteerd.

Dit is een verkennende pilot voor numerieke correctheid, leerverloop en looptijd. Een technische PASS bewijst geen accuracyverbetering. Deze notebook start geen training en wijzigt geen ruwe onderzoeksdata.
        '''),
        _cell('code', r'''
from pathlib import Path
from datetime import datetime
import json
import math
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
print('Deze notebook bevat uitsluitend waarden uit deze run.')
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
    if not metrics['epoch'].between(1, 5).all():
        raise ValueError('Deze notebook verwacht uitsluitend pilotepochs 1 tot en met 5.')
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
## Primair eindpunt: epoch 5

Alle vijf scores hieronder horen bij hetzelfde bevestigde epoch-5-checkpoint. Naast volledige metingen moet `report.json` aangeven dat vijf epochs volledig zijn opgeslagen (`completed_epochs == 5`). Bij een opslagfout kunnen er al epoch-5-metingen bestaan zonder een voltooid checkpoint; die worden niet in deze primaire tabel getoond. Een eerdere epoch wordt niet als eindresultaat vervangen. De beschikbare epochmetingen blijven verderop zichtbaar.
        '''),
        _cell('code', r'''
endpoint = pd.DataFrame()
complete_test = pd.DataFrame()
if not metrics.empty:
    complete_test = metrics[(metrics['phase'] == 'test') & metrics['full_split']].copy()
    candidate = complete_test[complete_test['epoch'] == 5]
    if (report.get('completed_epochs') == 5 and len(candidate) == 5
            and set(candidate['weight_format']) == set(FORMATS)):
        endpoint = candidate.set_index('weight_format').loc[FORMATS].reset_index()
        table = endpoint[['weight_format', 'weight_bits', 'correct1', 'n',
                          'accuracy_from_counts', 'hard_ce']].copy()
        table.columns = ['Formaat', 'Elementbits', 'Correct', 'Testbeelden', 'Accuracy (%)', 'Harde test-CE']
        table['Accuracy (%)'] = table['Accuracy (%)'].map(lambda value: f'{value:.6f}')
        table['Harde test-CE'] = table['Harde test-CE'].map(lambda value: f'{value:.6f}')
        display(table)
    else:
        print('Geen bevestigde opgeslagen epoch 5 met volledige metingen voor alle vijf formaten; het primaire eindpunt ontbreekt.')
        if len(candidate) == 5 and report.get('completed_epochs') != 5:
            print('Er zijn wel epoch-5-testmetingen, maar report.json bevestigt geen voltooid epoch-5-checkpoint. De curves mogen deze beschikbare metingen tonen.')
else:
    print('Geen SVHN-eindresultaat beschikbaar.')
        '''),
        _cell('markdown', r'''
## Accuracyverloop per formaat

Elk formaat heeft een eigen paneel, met dezelfde asschaal. De blauwe stippen met percentages zijn de volledige testmetingen. Een grijze lijn toont de volledige trainingsmetingen. Training en test verschillen in dataset en modelmodus; het verschil is geen zuivere schatting van generalisatieverlies.
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
                        marker='o', markersize=4, color=color, linewidth=1.6, label=label)
                if phase == 'test':
                    for row in frame.itertuples():
                        ax.annotate(f'{row.accuracy_from_counts:.2f}%',
                                    (row.epoch, row.accuracy_from_counts),
                                    xytext=(0, 7), textcoords='offset points',
                                    ha='center', fontsize=8, color=color)
            ax.set_title(f'{fmt} · W{BITS[fmt]}/A32')
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlim(0.65, 5.35)
            ax.set_ylim(low, high)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.grid(alpha=0.22)
            if ax.lines:
                ax.legend(frameon=False, fontsize=8)
        axes[-1].axis('off')
        fig_accuracy.suptitle('SVHN · CNN-8 · vijf directe MX-formaten · B32/A32\n'
                              'Adam 0,001 · weight decay 0 · batch 128 · distillatie T=1', fontsize=12)
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
                        marker='o', markersize=4, color=color, label=label)
        ax.set_title(fmt)
        ax.set_xticks([1, 2, 3, 4, 5])
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

De projectie naar 100 epochs is een raming op basis van deze pilot. Ze is geen gemeten honderd-epochduur en zegt niets over de uiteindelijke accuracy. De basis voor deze raming is de som van de train- en testfasetijden. Datavoorbereiding, controles, checkpointopslag en exports vallen buiten die basis. De werkelijke sessieduur bevat ook die bijkomende werkzaamheden. Laptopbelasting en opwarming kunnen de projectie beïnvloeden.
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
    for name, fig in [('accuracy_per_formaat', fig_accuracy), ('harde_ce_per_formaat', fig_loss)]:
        if fig is not None:
            fig.savefig(analysis_dir / f'{name}.png', dpi=200, bbox_inches='tight')
            fig.savefig(analysis_dir / f'{name}.pdf', bbox_inches='tight')
    print('Figuren opgeslagen in:', analysis_dir)
        '''),
        _cell('markdown', r'''
## Interpretatie en vervolg

- Deze run toetst de nieuwe gezamenlijke training. De distillatiestructuur is een vooraf gekozen ontwerp, geen bewezen optimum.
- Verschillen tussen de vijf modi zijn gemeten binnen één gezamenlijk model. Ze zijn geen vergelijking tussen vijf onafhankelijk geoptimaliseerde modellen.
- De oude drie-formaattrainingen en de afgeronde MSB-proef vormen geen gematchte controle voor deze vijf-formaatpilot.
- De volledige vijf-formaatreferentie wordt pas na beoordeling van deze pilot vanaf nul getraind. Voor een latere vergelijking met truncatie blijven lossgewichten, teachers, dataset en optimizerbeleid gelijk.
- Voor truncatie moeten we afzonderlijk beslissen hoe beide 8-bitformaten beschikbaar worden, de schaalcorrectie van elke nieuwe route numeriek afleiden en één canonieke E2M1-afleiding vastleggen. De bestaande ×64-regel wordt niet automatisch op nieuwe routes toegepast.

Bronnen voor deze proef: `PROTOCOL.md`, `README.md`, de meegeleverde bron- en pakketmanifesten; metingen uitsluitend uit `report.json` en `metrics.csv` van de hierboven aangegeven run.
        '''),
    ]
    return {'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python (.venv-train)', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.13'},
                         'mx_experiment': {'package': 'mx_weights_five_direct_pilot5_v1',
                                           'endpoint_epoch': 5, 'unexecuted_template': True}},
            'nbformat': 4, 'nbformat_minor': 5}


def write_notebook(out: Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / NOTEBOOK_NAME
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(build_notebook(out), handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    return path


if __name__ == '__main__':
    # Creates the unexecuted package template only, never a training result.
    target = Path(__file__).resolve().parent / NOTEBOOK_NAME
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(build_notebook(), handle, indent=1, ensure_ascii=False)
        handle.write('\n')
    print(target)
