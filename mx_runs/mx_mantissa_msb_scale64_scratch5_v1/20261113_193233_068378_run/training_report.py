"""Read-only report generation for a fresh five-epoch MSB x64 training run.

Only the Python standard library is required. Matplotlib is imported lazily for
optional figures. This helper never loads a model, checkpoint or dataset and
never changes the recorded CSV/JSON evidence.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SCHEMA = 'mx-msb-scale64-scratch5-v1'
NOTEBOOK_NAME = 'MX_MSB_scale64_scratch5_resultaten.ipynb'
BITS = (4, 6, 8)
FORMATS = {4: 'E2M1', 6: 'E2M3', 8: 'E4M3'}
NEW_METHOD = 'msb_bestexp'
REFERENCE_METHOD = 'lsb_trunc'
LABELS = {REFERENCE_METHOD: 'LSB ×64 (eerder gemeten)',
          NEW_METHOD: 'MSB ×64 (nieuwe run)'}
COLORS = {REFERENCE_METHOD: '#73818B', NEW_METHOD: '#245A81'}
COMPARISON_KEYS = ('initialization_hash_matches', 'runtime_matches_reference',
                   'source_matches_reference')
# Exact integer counts from the pinned historical fixed_scale64_reference.
# REFERENCE.json SHA256: a323762894db83b15af10e79a8aff0128c50f7946805a04517e17a9f627377db
REFERENCE_COUNTS = {
    (1, 4): (22634, 25661), (1, 6): (22591, 25663), (1, 8): (22953, 25733),
    (2, 4): (23317, 25790), (2, 6): (23317, 25771), (2, 8): (23740, 25806),
    (3, 4): (23607, 25778), (3, 6): (23409, 25758), (3, 8): (23914, 25803),
    (4, 4): (23661, 25795), (4, 6): (23722, 25797), (4, 8): (24123, 25827),
    (5, 4): (23661, 25818), (5, 6): (23767, 25833), (5, 8): (24212, 25856),
}
STYLE = {'axes.spines.top': False, 'axes.spines.right': False,
         'axes.titleweight': 'bold', 'font.size': 10, 'figure.facecolor': 'white',
         'axes.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .16,
         'savefig.facecolor': 'white'}


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
            if not 1 <= epoch <= 5 or bit not in BITS:
                raise ValueError('Verwacht uitsluitend epochs 1–5 en W4/W6/W8.')
            if row['weight_format'] not in (FORMATS[bit], f'MXFP{bit}_{FORMATS[bit]}'):
                raise ValueError('Gewichtsformaat past niet bij de precisie.')
            if row['n'] != 26032 or not 0 <= row['correct1'] <= row['correct5'] <= row['n']:
                raise ValueError('Onjuiste testdekking of aantallen correcte voorspellingen.')
            if row['loss'] < 0 or not math.isclose(row['lr'], .001, rel_tol=1e-12, abs_tol=0.):
                raise ValueError('Verwacht een niet-negatieve loss en LR 0,001 in epochs 1–5.')
            for top in (1, 5):
                expected = 100 * row[f'correct{top}'] / row['n']
                if abs(row[f'top{top}_percent'] - expected) >= 1e-4:
                    raise ValueError('Accuracy strookt niet met het aantal correcte voorspellingen.')
                row[f'top{top}_percent'] = expected
            key = (epoch, bit)
            if key in seen:
                raise ValueError('Dubbele epoch/precisie in ' + Path(path).name)
            if method == REFERENCE_METHOD and (row['correct1'], row['correct5']) != REFERENCE_COUNTS[key]:
                raise ValueError('Historische aantallen verschillen van de vastgelegde LSB ×64-referentie.')
            seen.add(key)
            rows.append(row)
    epochs = sorted({r['epoch'] for r in rows})
    if epochs and epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError('De meetreeks moet zonder ontbrekende epochs bij epoch 1 beginnen.')
    for epoch in epochs:
        if {(epoch, bit) for bit in BITS} - seen:
            raise ValueError('Een epoch mist een van de drie precisies.')
    if method == REFERENCE_METHOD and len(rows) != 15:
        raise ValueError('De historische referentie moet alle vijftien testmetingen bevatten.')
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
        if not isinstance(protocol, dict) or _integer(protocol.get('planned_epochs', 5), 'planned_epochs') != 5:
            raise ValueError('Deze notebook verwacht een budget van vijf epochs.')
        if (_integer(protocol.get('reference_end_epoch', 100), 'reference_end_epoch') != 100
                or not math.isclose(_finite(protocol.get('lr', .001), 'lr'), .001, rel_tol=1e-12)):
            raise ValueError('Het oorspronkelijke 100-epochschema met begin-LR 0,001 is niet bevestigd.')
        completed = _integer(report.get('completed_epochs', 0), 'completed_epochs')
        if not 0 <= completed <= 5:
            raise ValueError('completed_epochs moet tussen 0 en 5 liggen.')
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
        full_rows = len(rows) == 15 and completed == 5
        if report.get('training_completed') is True and not full_rows:
            raise ValueError('Voltooide training is geclaimd zonder alle vijftien nieuwe testmetingen.')
        if report.get('status') == 'PASS' and not (full_rows and report.get('training_completed') is True):
            raise ValueError('Training-PASS vereist vijf epochs en training_completed=True.')
        result.update(new_rows=rows, display_new_scores=bool(rows),
                      training_complete=(report.get('status') == 'PASS' and full_rows
                                         and report.get('training_completed') is True))
        result['endpoint_ready'] = (result['training_complete'] and result['comparable']
                                    and len(result['reference_rows']) == 15)
        if rows:
            result['reason'] = ''
        if rows and not result['comparable']:
            result['notices'].append('Initialisatie, runtime en bron zijn niet alle drie als gelijk aan de referentie bevestigd; geen eindpuntvergelijking.')
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        result['errors'].append('Nieuwe MSB-run: ' + str(exc))
        result.update(new_rows=[], display_new_scores=False, training_complete=False, endpoint_ready=False,
                      reason='Resultaatcontrole mislukt; er worden geen nieuwe accuracies ingevuld of getoond.')
    return result


def summary_markdown(result):
    report = result['report']
    text = f'## Status: {report.get("status", "NOG NIET UITGEVOERD")}\n\n'
    if result['endpoint_ready']:
        text += ('**De nieuwe MSB-run heeft alle vijf epochs vanaf nul voltooid.** '
                 'Het eindpunt wordt vergeleken met de eerder gemeten LSB ×64-run op dezelfde epoch 5.\n\n')
    elif result['display_new_scores']:
        text += ('**De geplande eindpuntvergelijking is nog niet beschikbaar.** '
                 'De nieuwe curves tonen alleen daadwerkelijk opgeslagen meetpunten. '
                 'Ontbrekende epochs worden niet aangevuld.\n\n')
    else:
        text += result['reason'] + '\n\n'
    text += f'**Nieuwe MSB-epochs afgerond:** {result["completed_epochs"]}/5.\n\n'
    if result['reference_rows']:
        text += ('De LSB ×64-cijfers zijn **historische metingen**, overgenomen uit de vastgelegde referentierun. '
                 'In deze uitvoering is geen nieuwe LSB- of RNE-training gedaan.\n\n')
    text += ('Dit is training vanaf een verse initialisatie, seed 20261009. Er is geen warmstart, '
             'epoch-100-oudermodel of epoch-0-baselinemeting. De vaste schaalverhouding is '
             '`S_E2 = 64 × S_E4`; beide schalen kunnen tijdens training veranderen. Er zijn geen geleerde `phi`-parameters.\n\n'
             '**Grens van de historische vergelijking:** overeenkomst van historische begin-RNG en samplevolgorde '
             'is niet geverifieerd, omdat die historische bestanden ontbreken. Gelijke seed en modelinitialisatie '
             'zijn geen bewijs van bitexacte CUDA-uitvoering. Eén nieuwe en één historische run leveren geen onzekerheid over meerdere seeds.\n\n')
    comparison = report.get('reference_comparison', {})
    if isinstance(comparison, dict) and comparison:
        text += '| Vergelijkingscontrole | Gerapporteerd |\n|---|---|\n'
        names = {'initialization_hash_matches': 'Modelinitialisatie gelijk',
                 'runtime_matches_reference': 'Runtime overeenkomstig',
                 'source_matches_reference': 'Bron overeenkomstig'}
        for key in COMPARISON_KEYS:
            value = comparison.get(key)
            label = 'Bevestigd' if value is True else ('Afwijking' if value is False else 'Niet bevestigd')
            text += f'| {names[key]} | {label} |\n'
        text += '\n'
    text += f'**Gelezen resultatenmap:** `{result["path"]}`\n'
    messages = list(result['errors']) + list(result['notices'])
    if report.get('error'):
        messages.append(str(report['error']))
    for warning in report.get('warnings', []):
        messages.append(str(warning.get('message', warning)) if isinstance(warning, dict) else str(warning))
    if messages:
        text += '\n### Meldingen\n\n' + '\n'.join('- ' + value for value in messages) + '\n'
    return text


def endpoint_markdown(result):
    if not result['endpoint_ready']:
        return ('## Eindpunt op epoch 5\n\nNog geen volledige gecontroleerde vergelijking. '
                'Deze tabel verschijnt na vijf nieuwe MSB-epochs, met geldige historische metingen '
                'en bevestigde overeenkomst van initialisatie, runtime en bron.')
    new = {r['weight_bits']: r for r in result['new_rows'] if r['epoch'] == 5}
    old = {r['weight_bits']: r for r in result['reference_rows'] if r['epoch'] == 5}
    text = ('## Eindpunt op epoch 5\n\n'
            '| Precisie | LSB eerder: correct / n | LSB eerder [%] | MSB nieuw: correct / n | MSB nieuw [%] | Nieuw − eerder [pp] |\n'
            '|---|---:|---:|---:|---:|---:|\n')
    for bit in BITS:
        a, b = old[bit], new[bit]
        text += (f'| W{bit}/A32 · {FORMATS[bit]} | {a["correct1"]} / {a["n"]} | {_pct(a["top1_percent"])} | '
                 f'{b["correct1"]} / {b["n"]} | {_pct(b["top1_percent"])} | '
                 f'{_pct(b["top1_percent"]-a["top1_percent"], True)} |\n')
    return text + ('\npp = procentpunt. Dit is het vooraf gekozen laatste pilotmeetpunt, '
                   'geen selectie van de beste testepoch. De historische epochs 42 en 100 worden hiervoor niet gebruikt.\n')


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
                                marker='o', markersize=4.5, linewidth=1.8,
                                linestyle='--' if method == REFERENCE_METHOD else '-', label=LABELS[method])
                ax.set(title=f'W{bit}/A32 · {FORMATS[bit]}', xlabel='Trainingsepoch vanaf initialisatie',
                       ylabel='Testaccuracy [%]', xticks=range(1, 6), xlim=(.85, 5.15))
                if values:
                    ax.set_ylim(*_limits(values))
            if result['endpoint_ready']:
                title = 'Vijf epochs: nieuwe MSB-run tegenover historische LSB ×64-referentie'
            elif not result['new_rows']:
                title = 'ALLEEN HISTORISCHE REFERENTIE · geen nieuwe MSB-scores weergegeven'
            else:
                title = 'ONVOLLEDIGE VERGELIJKING · uitsluitend daadwerkelijk gemeten punten'
            fig.suptitle(title + '\nAfzonderlijke precisies · dezelfde geplande epochs 1–5',
                         fontsize=12.5, fontweight='bold')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
            fig.subplots_adjust(top=.77, bottom=.20, wspace=.30)
            figures['accuracy_epochs1_to5'] = fig
            if result['endpoint_ready']:
                fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
                for ax, bit in zip(axes, BITS):
                    values = []
                    for y, (method, data) in enumerate(((REFERENCE_METHOD, result['reference_rows']), (NEW_METHOD, result['new_rows']))):
                        row = next(r for r in data if r['weight_bits'] == bit and r['epoch'] == 5)
                        value = row['top1_percent']
                        values.append(value)
                        ax.scatter([value], [y], color=COLORS[method], s=65, marker='o', zorder=3)
                        ax.annotate(_pct(value) + '%', (value, y), xytext=(7, 0),
                                    textcoords='offset points', va='center', fontsize=10)
                    ax.set(title=f'W{bit}/A32 · {FORMATS[bit]}', xlabel='Testaccuracy op epoch 5 [%]',
                           yticks=range(2), yticklabels=[LABELS[REFERENCE_METHOD], LABELS[NEW_METHOD]], ylim=(1.6, -.6))
                    ax.set_xlim(*_limits(values, right_room=True))
                    ax.grid(axis='y', visible=False)
                fig.suptitle('Epoch 5 vanaf nul · MSB nieuw tegenover LSB eerder gemeten\n'
                             'Eén run per methode; historische samplevolgorde niet geverifieerd',
                             fontsize=13, fontweight='bold')
                fig.subplots_adjust(top=.76, bottom=.15, left=.13, right=.98, wspace=.90)
                figures['accuracy_epoch5'] = fig
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
           'Eén nieuwe training vanaf nul: vijf epochs met MSB + optimale exponentkeuze. '
           'De LSB ×64-cijfers zijn eerder gemeten referentiedata. Deze notebook start geen training.'),
        md('*Status bij aanmaak van deze notebook; voer de cellen uit om de bestanden opnieuw te lezen.*\n\n'
           + summary_markdown(result)),
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
           'De oorspronkelijke milestones 50/75/90 blijven ingesteld en worden binnen vijf epochs niet bereikt. '
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
           '- Vijf epochs onderzoeken vroege training, niet convergentie of optimale instellingen.\n'
           '- Door gezamenlijke updates kunnen ook E4M3 en E2M3 tussen de methoden verschillen.\n'
           '- Het eindpunt is vooraf gekozen op epoch 5; er wordt geen beste testcheckpoint geselecteerd.\n'
           '- De eerdere post-hocproef gebruikte een andere schaaltoestand en een getraind model. '
           'Die accuracies zijn geen baseline voor deze training vanaf nul.\n'
           '- FP32-emulatie levert geen bewijs van hardwarelatency, energieverbruik of algemene superioriteit.'),
    ]
    for index, cell in enumerate(cells):
        cell['id'] = 'msb64-' + str(index)
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
            'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


def write_outputs(out: Path) -> dict:
    """Write a readable notebook and optional PNGs beside existing evidence."""
    out = Path(out).expanduser().resolve()
    result = load_results(out)
    response = {'notebook_path': None, 'figures': [], 'errors': list(result['errors'])}
    try:
        out.mkdir(parents=True, exist_ok=True)
        target = out / NOTEBOOK_NAME
        target.write_text(json.dumps(_notebook(out, result), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        response['notebook_path'] = str(target)
    except OSError as exc:
        response['errors'].append('Notebook niet beschikbaar: ' + str(exc))
    figures, errors = create_figures(result)
    response['errors'].extend(errors)
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
