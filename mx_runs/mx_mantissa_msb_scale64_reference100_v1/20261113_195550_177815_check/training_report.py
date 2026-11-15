"""Read-only report generation for a fresh 100-epoch MSB x64 training run.

Only the Python standard library is required. Matplotlib is imported lazily for
optional figures. This helper never loads a model, checkpoint or dataset and
never changes the recorded CSV/JSON evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'mx-msb-scale64-reference100-v1'
NOTEBOOK_NAME = 'MX_MSB_scale64_reference100_resultaten.ipynb'
EPOCHS = 100
BITS = (4, 6, 8)
FORMATS = {4: 'E2M1', 6: 'E2M3', 8: 'E4M3'}
NEW_METHOD = 'msb_bestexp'
REFERENCE_METHOD = 'lsb_trunc'
LABELS = {REFERENCE_METHOD: 'LSB ×64 (eerder gemeten)',
          NEW_METHOD: 'MSB ×64 (nieuwe run)'}
COLORS = {REFERENCE_METHOD: '#73818B', NEW_METHOD: '#245A81'}
COMPARISON_KEYS = ('initialization_hash_matches', 'runtime_matches_reference',
                   'source_matches_reference', 'dataset_matches_reference')
# Computed from all 300 test rows in the unchanged historical epoch_metrics.csv
# (file SHA256 cb6c9ddcff6c263bf24c3cde3b02bab8ead694ae21e278b58d1632bf67ce60ab).
# Canonical records: [epoch, weight_bits, n, correct1, correct5], sorted by epoch
# and weight_bits; json.dumps(records, separators=(',', ':')).encode('utf-8').
REFERENCE_TEST_COUNTS_SHA256 = '56c544c8abe9a325714f45c008ff3ef451db4f679dd5f553a4e1c99c56d5da52'
STYLE = {'axes.spines.top': False, 'axes.spines.right': False,
         'axes.titleweight': 'bold', 'font.size': 10, 'figure.facecolor': 'white',
         'axes.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .16,
         'savefig.facecolor': 'white'}


class OptionalPlotWarning(str):
    """A displayable message for an unavailable optional plot dependency."""


def _integer(value, name):
    number = float(value)
    if not math.isfinite(number) or int(number) != number:
        raise ValueError('Ongeldig geheel getal: ' + name)
    return int(number)


def _finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Niet-eindige waarde: ' + name)
    return number


def _pct(value, signed=False):
    return (f'{value:+.3f}' if signed else f'{value:.3f}').replace('.', ',')


def expected_lr(epoch):
    """Learning rate used during the epoch, before the boundary scheduler step."""
    if epoch <= 50:
        return .001
    if epoch <= 75:
        return .0001
    if epoch <= 90:
        return .00001
    return .000001


def reference_counts_digest(rows):
    records = [[row[key] for key in ('epoch', 'weight_bits', 'n', 'correct1', 'correct5')]
               for row in sorted(rows, key=lambda row: (row['epoch'], row['weight_bits']))]
    return hashlib.sha256(json.dumps(records, separators=(',', ':')).encode('utf-8')).hexdigest()


def _read_metrics(path, method):
    rows, seen = [], set()
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'method', 'epoch', 'phase', 'weight_bits', 'weight_format', 'n',
                    'correct1', 'correct5', 'top1_percent', 'top5_percent', 'loss', 'lr'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(Path(path).name + ' mist vereiste kolommen.')
        for raw in reader:
            if raw['phase'] != 'test' or raw['method'] != method:
                raise ValueError(Path(path).name + ': onjuiste fase of methode.')
            row = dict(raw)
            for key in ('epoch', 'weight_bits', 'n', 'correct1', 'correct5'):
                row[key] = _integer(raw[key], key)
            for key in ('top1_percent', 'top5_percent', 'loss', 'lr'):
                row[key] = _finite(raw[key], key)
            epoch, bit = row['epoch'], row['weight_bits']
            if not 1 <= epoch <= EPOCHS or bit not in BITS:
                raise ValueError('Verwacht uitsluitend epochs 1–100 en W4/W6/W8.')
            if row['weight_format'] not in (FORMATS[bit], f'MXFP{bit}_{FORMATS[bit]}'):
                raise ValueError('Gewichtsformaat past niet bij de precisie.')
            if row['n'] != 26032 or not 0 <= row['correct1'] <= row['correct5'] <= row['n']:
                raise ValueError('Onjuiste testdekking of aantallen correcte voorspellingen.')
            if row['loss'] < 0 or not math.isclose(row['lr'], expected_lr(epoch), rel_tol=1e-12, abs_tol=0.):
                raise ValueError('Loss of LR wijkt af van het oorspronkelijke 100-epochschema op epoch ' + str(epoch) + '.')
            for top in (1, 5):
                expected = 100 * row[f'correct{top}'] / row['n']
                if abs(row[f'top{top}_percent'] - expected) >= 1e-4:
                    raise ValueError('Accuracy strookt niet met het aantal correcte voorspellingen.')
                row[f'top{top}_percent'] = expected
            key = (epoch, bit)
            if key in seen:
                raise ValueError('Dubbele epoch/precisie in ' + Path(path).name)
            seen.add(key)
            rows.append(row)
    epochs = sorted({r['epoch'] for r in rows})
    if epochs and epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError('De meetreeks moet zonder ontbrekende epochs bij epoch 1 beginnen.')
    for epoch in epochs:
        if {(epoch, bit) for bit in BITS} - seen:
            raise ValueError('Een epoch mist een van de drie precisies.')
    if method == REFERENCE_METHOD:
        if len(rows) != EPOCHS * len(BITS):
            raise ValueError('De historische referentie moet alle 300 testmetingen bevatten.')
        if reference_counts_digest(rows) != REFERENCE_TEST_COUNTS_SHA256:
            raise ValueError('Historische aantallen verschillen van de vastgelegde LSB ×64-referentie.')
    return sorted(rows, key=lambda r: (r['epoch'], r['weight_bits']))


def load_results(out):
    """Validate actual and historical observations as two distinct data sources."""
    out = Path(out).expanduser().resolve()
    result = {'path': str(out), 'report': {}, 'new_rows': [], 'reference_rows': [],
              'errors': [], 'notices': [], 'completed_epochs': 0,
              'display_new_scores': False, 'training_complete': False,
              'comparable': False, 'endpoint_ready': False,
              'reason': 'De nieuwe MSB-training is nog niet uitgevoerd of gerapporteerd.'}
    reference = out / 'reference_lsb.csv'
    if reference.is_file():
        try:
            result['reference_rows'] = _read_metrics(reference, REFERENCE_METHOD)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            result['errors'].append('Historische referentie: ' + str(exc))
    else:
        result['notices'].append('reference_lsb.csv is nog niet beschikbaar in deze map.')

    report_path = out / 'run_report.json'
    if not report_path.is_file():
        return result
    try:
        report = json.loads(report_path.read_text(encoding='utf-8-sig'))
        if not isinstance(report, dict):
            raise ValueError('run_report.json moet een JSON-object bevatten.')
        result['report'] = report
        if report.get('schema') != SCHEMA:
            raise ValueError('Dit is niet het rapport van de MSB ×64-training vanaf nul.')
        protocol = report.get('protocol', {})
        if not isinstance(protocol, dict) or _integer(protocol.get('planned_epochs', EPOCHS), 'planned_epochs') != EPOCHS:
            raise ValueError('Deze notebook verwacht een nieuwe volledige run van 100 epochs.')
        if (_integer(protocol.get('reference_end_epoch', 100), 'reference_end_epoch') != 100
                or not math.isclose(_finite(protocol.get('lr', .001), 'lr'), .001, rel_tol=1e-12)):
            raise ValueError('Het oorspronkelijke 100-epochschema met begin-LR 0,001 is niet bevestigd.')
        completed = _integer(report.get('completed_epochs', 0), 'completed_epochs')
        if not 0 <= completed <= EPOCHS:
            raise ValueError('completed_epochs moet tussen 0 en 100 liggen.')
        result['completed_epochs'] = completed
        comparison = report.get('reference_comparison', {})
        if not isinstance(comparison, dict):
            raise ValueError('reference_comparison moet een JSON-object bevatten.')
        result['comparable'] = all(comparison.get(k) is True for k in COMPARISON_KEYS)

        if report.get('synthetic_only') is True or report.get('synthetic_fixture_only') is True:
            result['reason'] = 'Synthetische softwarecontrole; er zijn geen nieuwe onderzoeksaccuracies.'
            return result
        if report.get('command') == 'check':
            result['reason'] = 'Controlemodus; de nieuwe SVHN-training is hiermee niet uitgevoerd.'
            return result
        if report.get('status') not in ('PASS', 'STOPPED', 'RUNNING'):
            result['reason'] = 'De uitvoering is niet succesvol afgerond. Nieuwe scores worden niet als onderzoeksresultaat weergegeven.'
            return result
        if report.get('network_evaluated') is not True:
            return result
        metrics = out / 'accuracy.csv'
        if not metrics.is_file():
            if completed or report.get('training_completed') is True:
                raise ValueError('Training wordt als afgerond gemeld, maar accuracy.csv ontbreekt.')
            result['reason'] = 'Er zijn nog geen opgeslagen testmetingen van de nieuwe MSB-run.'
            return result
        rows = _read_metrics(metrics, NEW_METHOD)
        observed = max((r['epoch'] for r in rows), default=0)
        if observed != completed:
            raise ValueError('CSV-voortgang verschilt van completed_epochs in het rapport.')
        full_rows = len(rows) == EPOCHS * len(BITS) and completed == EPOCHS
        if report.get('training_completed') is True and not full_rows:
            raise ValueError('Voltooide training is geclaimd zonder alle 300 nieuwe testmetingen.')
        if report.get('status') == 'PASS' and not (full_rows and report.get('training_completed') is True):
            raise ValueError('Training-PASS vereist 100 epochs en training_completed=True.')
        result.update(new_rows=rows, display_new_scores=bool(rows),
                      training_complete=(report.get('status') == 'PASS' and full_rows
                                         and report.get('training_completed') is True))
        result['endpoint_ready'] = (result['training_complete'] and result['comparable']
                                    and len(result['reference_rows']) == EPOCHS * len(BITS))
        if rows:
            result['reason'] = ''
        if rows and not result['comparable']:
            result['notices'].append('Initialisatie, runtime, bron en dataset zijn niet alle vier als gelijk aan de referentie bevestigd; geen eindpuntvergelijking.')
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        result['errors'].append('Nieuwe MSB-run: ' + str(exc))
        result.update(new_rows=[], display_new_scores=False, training_complete=False, endpoint_ready=False,
                      reason='Resultaatcontrole mislukt; er worden geen nieuwe accuracies ingevuld of getoond.')
    return result


def warning_lines(report):
    """Show one line per warning category/message, without double-counting.

    New warning_counts are authoritative when a corresponding warning string
    is also present. For older repeated warnings, count the saved entries.
    """
    grouped, covered_messages = {}, set()
    counts = report.get('warning_counts', [])
    if isinstance(counts, list):
        for item in counts:
            if not isinstance(item, dict) or not item.get('message'):
                continue
            message = str(item['message'])
            category = str(item.get('category') or '')
            try:
                count = _integer(item.get('count'), 'warning_count')
                if count <= 0:
                    count = None
            except (ValueError, TypeError, OverflowError):
                count = None
            key = (category, message)
            if key not in grouped:
                grouped[key] = count
            else:
                prior = grouped[key]
                grouped[key] = prior + count if prior is not None and count is not None else None
            covered_messages.add(message)
    warnings = report.get('warnings', [])
    if isinstance(warnings, (str, dict)):
        warnings = [warnings]
    if isinstance(warnings, list):
        for item in warnings:
            category = str(item.get('category') or '') if isinstance(item, dict) else ''
            message = str(item.get('message', item)) if isinstance(item, dict) else str(item)
            if not message or message in covered_messages:
                continue
            key = (category, message)
            grouped[key] = grouped.get(key, 0) + 1
    lines = []
    for (category, message), count in grouped.items():
        prefix = category + ': ' if category else ''
        frequency = f'{count}× gemeld' if count is not None else 'aantal niet geldig geregistreerd'
        lines.append(prefix + message + ' (' + frequency + ')')
    return lines


def summary_markdown(result, include_warnings=True):
    report = result['report']
    text = f'## Status: {report.get("status", "NOG NIET UITGEVOERD")}\n\n'
    if result['endpoint_ready']:
        text += ('**De nieuwe MSB-run heeft alle 100 epochs vanaf nul voltooid.** '
                 'Het eindpunt wordt vergeleken met de eerder gemeten LSB ×64-run op dezelfde epoch 100.\n\n')
    elif result['display_new_scores']:
        text += ('**De geplande eindpuntvergelijking is nog niet beschikbaar.** '
                 'De nieuwe curves tonen alleen daadwerkelijk opgeslagen meetpunten. '
                 'Ontbrekende epochs worden niet aangevuld.\n\n')
    else:
        text += result['reason'] + '\n\n'
    text += f'**Nieuwe MSB-epochs afgerond:** {result["completed_epochs"]}/100.\n\n'
    if result['reference_rows']:
        text += ('De LSB ×64-cijfers zijn **historische metingen**, overgenomen uit de vastgelegde referentierun. '
                 'In deze uitvoering is geen nieuwe LSB- of RNE-training gedaan.\n\n')
    text += ('Deze volledige run begint opnieuw vanaf een verse initialisatie, seed 20261009. '
             'Het eerdere vijf-epochpilotcheckpoint en zijn metingen worden niet in deze nieuwe run opgenomen. '
             'Er is geen warmstart of epoch-0-baselinemeting. De vaste schaalverhouding is '
             '`S_E2 = 64 × S_E4`; beide schalen kunnen tijdens training veranderen. Er zijn geen geleerde `phi`-parameters.\n\n'
             '**Grens van de historische vergelijking:** overeenkomst van historische begin-RNG en samplevolgorde '
             'is niet geverifieerd, omdat die historische bestanden ontbreken. Gelijke seed en modelinitialisatie '
             'zijn geen bewijs van bitexacte CUDA-uitvoering. Eén nieuwe en één historische run leveren geen onzekerheid over meerdere seeds.\n\n')
    comparison = report.get('reference_comparison', {})
    if isinstance(comparison, dict) and comparison:
        text += '| Vergelijkingscontrole | Gerapporteerd |\n|---|---|\n'
        names = {'initialization_hash_matches': 'Modelinitialisatie gelijk',
                 'runtime_matches_reference': 'Runtime overeenkomstig',
                 'source_matches_reference': 'Bron overeenkomstig',
                 'dataset_matches_reference': 'Dataset en preprocessing overeenkomstig'}
        for key in COMPARISON_KEYS:
            value = comparison.get(key)
            label = 'Bevestigd' if value is True else ('Afwijking' if value is False else 'Niet bevestigd')
            text += f'| {names[key]} | {label} |\n'
        text += '\n'
    text += f'**Gelezen resultatenmap:** `{result["path"]}`\n'
    messages = list(result['errors']) + list(result['notices'])
    if report.get('error'):
        messages.append(str(report['error']))
    if messages:
        text += '\n### Meldingen\n\n' + '\n'.join('- ' + value for value in dict.fromkeys(messages)) + '\n'
    warnings = warning_lines(report) if include_warnings else []
    if warnings:
        text += '\n### Waarschuwingen\n\n' + '\n'.join('- ' + value for value in warnings) + '\n'
    return text


def endpoint_markdown(result):
    if not result['endpoint_ready']:
        return ('## Eindpunt op epoch 100\n\nNog geen volledige gecontroleerde vergelijking. '
                'Deze tabel verschijnt na 100 nieuwe MSB-epochs, met geldige historische metingen '
                'en bevestigde overeenkomst van initialisatie, runtime, bron en dataset.')
    new = {r['weight_bits']: r for r in result['new_rows'] if r['epoch'] == EPOCHS}
    old = {r['weight_bits']: r for r in result['reference_rows'] if r['epoch'] == EPOCHS}
    text = ('## Eindpunt op epoch 100\n\n'
            '| Precisie | LSB eerder: correct / n | LSB eerder [%] | MSB nieuw: correct / n | MSB nieuw [%] | Nieuw − eerder [pp] |\n'
            '|---|---:|---:|---:|---:|---:|\n')
    for bit in BITS:
        a, b = old[bit], new[bit]
        text += (f'| W{bit}/A32 · {FORMATS[bit]} | {a["correct1"]} / {a["n"]} | {_pct(a["top1_percent"])} | '
                 f'{b["correct1"]} / {b["n"]} | {_pct(b["top1_percent"])} | '
                 f'{_pct(b["top1_percent"]-a["top1_percent"], True)} |\n')
    return text + ('\npp = procentpunt. De primaire vergelijking gebruikt voor beide methoden epoch 100. '
                   'Er wordt geen beste testepoch gekozen; de historische selectie op epoch 42 wordt niet gebruikt.\n')


def _limits(values, right_room=False):
    low, high = min(values), max(values)
    width = max(high - low, .8)
    return max(0., low - .18 * width), min(100., high + (.55 if right_room else .22) * width)


def create_figures(result):
    figures, errors = {}, []
    if not (result['new_rows'] or result['reference_rows']):
        return figures, errors
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return figures, [OptionalPlotWarning('Optionele figuren niet beschikbaar: '
                                            + type(exc).__name__ + ': ' + str(exc))]
    try:
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
            for ax, bit in zip(axes, BITS):
                values = []
                for method, data in ((REFERENCE_METHOD, result['reference_rows']), (NEW_METHOD, result['new_rows'])):
                    rows = [r for r in data if r['weight_bits'] == bit]
                    if rows:
                        ys = [r['top1_percent'] for r in rows]
                        values.extend(ys)
                        ax.plot([r['epoch'] for r in rows], ys, color=COLORS[method],
                                marker='o', markersize=3.5, linewidth=1.6,
                                markevery=sorted(set(range(0, len(rows), 10)) | {len(rows) - 1}),
                                linestyle='--' if method == REFERENCE_METHOD else '-', label=LABELS[method])
                ax.set(title=f'W{bit}/A32 · {FORMATS[bit]}', xlabel='Trainingsepoch vanaf initialisatie',
                       ylabel='Testaccuracy [%]', xticks=(1, 25, 50, 75, 100), xlim=(0, 101))
                if values:
                    ax.set_ylim(*_limits(values))
                for boundary in (50.5, 75.5, 90.5):
                    ax.axvline(boundary, color='#B4BCC2', linewidth=.8, linestyle=':', zorder=0)
            if result['endpoint_ready']:
                title = '100 epochs: nieuwe MSB-run tegenover historische LSB ×64-referentie'
            elif not result['new_rows']:
                title = 'ALLEEN HISTORISCHE REFERENTIE · geen nieuwe MSB-scores weergegeven'
            else:
                title = 'ONVOLLEDIGE VERGELIJKING · uitsluitend daadwerkelijk gemeten punten'
            fig.suptitle(title + '\nEpochs 1–100 · LR-verlagingen vanaf epochs 51, 76 en 91',
                         fontsize=12.5, fontweight='bold')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
            fig.subplots_adjust(top=.77, bottom=.20, wspace=.30)
            figures['accuracy_epochs1_to100'] = fig
            if result['endpoint_ready']:
                fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
                for ax, bit in zip(axes, BITS):
                    values = []
                    for y, (method, data) in enumerate(((REFERENCE_METHOD, result['reference_rows']), (NEW_METHOD, result['new_rows']))):
                        row = next(r for r in data if r['weight_bits'] == bit and r['epoch'] == EPOCHS)
                        value = row['top1_percent']
                        values.append(value)
                        ax.scatter([value], [y], color=COLORS[method], s=65, marker='o', zorder=3)
                        ax.annotate(_pct(value) + '%', (value, y), xytext=(7, 0),
                                    textcoords='offset points', va='center', fontsize=10)
                    ax.set(title=f'W{bit}/A32 · {FORMATS[bit]}', xlabel='Testaccuracy op epoch 100 [%]',
                           yticks=range(2), yticklabels=[LABELS[REFERENCE_METHOD], LABELS[NEW_METHOD]], ylim=(1.6, -.6))
                    ax.set_xlim(*_limits(values, right_room=True))
                    ax.grid(axis='y', visible=False)
                fig.suptitle('Epoch 100 vanaf nul · MSB nieuw tegenover LSB eerder gemeten\n'
                             'Eén run per methode; historische samplevolgorde niet geverifieerd',
                             fontsize=13, fontweight='bold')
                fig.subplots_adjust(top=.76, bottom=.15, left=.13, right=.98, wspace=.90)
                figures['accuracy_epoch100'] = fig
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        errors.append('Figuren niet beschikbaar: ' + type(exc).__name__ + ': ' + str(exc))
    return figures, errors


def _notebook(out, result):
    def md(text):
        return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}
    def code(text, parameters=False):
        return {'cell_type': 'code', 'metadata': {'tags': ['parameters']} if parameters else {},
                'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)}
    cells = [
        md('# MSB-mantissaconversie met vaste schaalverhouding ×64\n\n'
           'Eén nieuwe training vanaf nul: 100 epochs met MSB + optimale exponentkeuze. '
           'De eerdere vijf-epochpilot wordt niet hervat of aan deze meetreeks toegevoegd. '
           'De LSB ×64-cijfers zijn eerder gemeten referentiedata. Deze notebook start geen training.'),
        md('*Status bij aanmaak van deze notebook; voer de cellen uit om de bestanden opnieuw te lezen.*\n\n'
           + summary_markdown(result, include_warnings=False)
           + '\nEventuele waarschuwingen verschijnen hieronder één keer per soort, met hun geregistreerde aantal, '
           'wanneer de actuele statuscel wordt uitgevoerd.'),
        md('## Resultatenmap\n\nAls de map is verplaatst, pas `RUN_DIR` aan naar de map met '
           '`training_report.py`, `run_report.json`, `accuracy.csv` en `reference_lsb.csv`.'),
        code('from pathlib import Path\nimport importlib.util\n'
             f'RUN_DIR = Path({str(out.resolve())!r})\n'
             'if not RUN_DIR.is_dir():\n    RUN_DIR = Path.cwd()\n'
             'helper = RUN_DIR / "training_report.py"\n'
             'if not helper.is_file():\n'
             '    raise FileNotFoundError("Kies bij RUN_DIR de resultatenmap met training_report.py.")\n'
             'spec = importlib.util.spec_from_file_location("msb_scale64_report", helper)\n'
             'tr = importlib.util.module_from_spec(spec)\n'
             'spec.loader.exec_module(tr)\n'
             'from IPython.display import Markdown, display\n', True),
        code('result = tr.load_results(RUN_DIR)\ndisplay(Markdown(tr.summary_markdown(result)))\n'),
        md('## Vastgelegde proef\n\n'
           'Verse initialisatie met seed 20261009; Adam met begin-LR 0,001 en weight decay 0. '
           'De oorspronkelijke milestones 50/75/90 blijven ingesteld. De gebruikte LR is 0,001 in epochs 1–50, '
           '0,0001 in 51–75, 0,00001 in 76–90 en 0,000001 in 91–100. '
           'Geen onafhankelijke of bevroren absolute E2-schalen: steeds `S_E2 = 64 × S_E4`. '
           'E4M3, E2M3 en E2M1 delen de trainbare FP32-masters binnen de nieuwe MSB-run. '
           'De drie BN-banken leren mee; A32, B32 en gezamenlijke recursieve distillatie blijven behouden.'),
        code('display(Markdown(tr.endpoint_markdown(result)))\n'),
        code('figures, plot_errors = tr.create_figures(result)\n'
             'for name, figure in figures.items():\n    display(figure)\n'
             'if figures:\n    import matplotlib.pyplot as plt\n'
             '    for figure in figures.values():\n        plt.close(figure)\n'
             'for message in plot_errors:\n    display(Markdown(message))\n'
             'if not figures and not plot_errors:\n'
             '    display(Markdown("Nog geen gecontroleerde meetbestanden om te plotten."))\n'),
        md('## Interpretatiegrenzen\n\n'
           '- De nieuwe MSB-run en de historische LSB-run zijn afzonderlijke uitvoeringen met één seed.\n'
           '- Hetzelfde beginmodel en protocol bewijzen geen gelijke historische RNG of samplevolgorde.\n'
           '- Honderd epochs met één vaste configuratie bewijzen geen optimale instelling of algemene superioriteit.\n'
           '- Door gezamenlijke updates kunnen ook E4M3 en E2M3 tussen de methoden verschillen.\n'
           '- Het eindpunt is vooraf gekozen op epoch 100; er wordt geen beste testcheckpoint geselecteerd.\n'
           '- De eerdere MSB-pilot en haar checkpoint maken geen deel uit van de nieuwe trainingsreeks.\n'
           '- De eerdere post-hocproef gebruikte een andere schaaltoestand en een getraind model. '
           'Die accuracies zijn geen baseline voor deze training vanaf nul.\n'
           '- FP32-emulatie levert geen bewijs van hardwarelatency, energieverbruik of algemene superioriteit.'),
    ]
    for index, cell in enumerate(cells):
        cell['id'] = 'msb64full-' + str(index)
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
            'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


def write_outputs(out: Path) -> dict:
    """Write a readable notebook and optional PNGs beside existing evidence."""
    out = Path(out).expanduser().resolve()
    result = load_results(out)
    response = {'notebook_path': None, 'figures': [], 'errors': list(result['errors']), 'warnings': []}
    try:
        out.mkdir(parents=True, exist_ok=True)
        target = out / NOTEBOOK_NAME
        target.write_text(json.dumps(_notebook(out, result), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        response['notebook_path'] = str(target)
    except OSError as exc:
        response['errors'].append('Notebook niet beschikbaar: ' + str(exc))
    figures, errors = create_figures(result)
    for item in errors:
        response['warnings' if isinstance(item, OptionalPlotWarning) else 'errors'].append(str(item))
    if figures:
        import matplotlib.pyplot as plt
        for name, figure in figures.items():
            try:
                directory = out / 'figures'
                directory.mkdir(exist_ok=True)
                target = directory / (name + '.png')
                figure.savefig(target, dpi=160, bbox_inches='tight')
                response['figures'].append(str(target))
            except (OSError, RuntimeError, ValueError) as exc:
                response['errors'].append('Figuur niet beschikbaar: ' + name + ': ' + str(exc))
            finally:
                plt.close(figure)
    return response
