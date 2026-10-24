# Hergebruik en wijzigingen

## Afzonderlijk nieuw evaluatiepakket

`mx_triple_scale64_trained_posthoc_eval_v1` voegt de afgesproken A/B/C-proef
toe voor de voltooide schaal ×64-training. De bestaande trainingspakketten,
eerdere evaluatiepakketten en resultaten worden niet aangepast.

De basis is `mx_triple_scale64_posthoc_eval_v1`, dat dezelfde A/B/C-mapping
eerder op de directe en naïeve training heeft geëvalueerd. De numerieke
functies `shifted_scale_codes` en `transform_codes` zijn qua Python-AST
ongewijzigd. Dat wordt door de lokale validatie bevestigd.

## Ongewijzigde trainings- en codecbron

De onderstaande twaalf modules zijn bytegelijk aan het daadwerkelijk gebruikte
`mx_triple_scale64_reference100_v1`-trainingspakket:

- `mx_format.py`, `mx_e2m1.py`, `mx_e4m3.py`;
- `mx_adapter.py`, `joint_adapter.py`, `joint_support.py`;
- `svhn_support.py`, `smoke_support.py`, `resume_core.py`;
- `trunc_adapter.py`, `trunc_support.py`, `posthoc_truncation.py`.

`REUSE_MANIFEST.json` vermeldt de exacte hashes. Het originele upstreamarchief
en de bestaande compatibiliteitsaanpassingen blijven behouden. Er wordt geen
codec, trainingsquantizer, STE, optimizer of schaalkeuze gewijzigd.

## Nieuwe ondersteuning voor de B-getrainde exports

`scale64.py` accepteert uitsluitend de trainingssoort `scale64`. De validatie
gebruikt de eigen bron- en afleidingsschema's van de schaal ×64-training.
De opgeslagen lagere codes en schalen moeten exact overeenkomen met route B
uit dezelfde E4M3-bron. Dit vervangt de oude A-reconstructiecontrole die bij de
naïeve training hoorde.

Alle metadata over bronschalen verwijst in dit nieuwe pakket consequent naar
de opgeslagen E4M3-bron. Ook de opgeslagen getrainde E2M3/E2M1-modi hebben dus
`source_bits=8`, `scale_exponent_offset=6` en schaalanker `source8`. Hun
vergelijking met zichzelf blijft een afzonderlijke nulverschilcontrole.

`evaluation.py` bewaart de bestaande negen routes en gepaarde evaluatie. Het
voegt een controle toe op de exacte modelstaat van B versus de opgeslagen
lagere modi, plus gerapporteerde logit-, CE- en voorspellingverschillen.

## Runner en referentiebeleid

`run_eval.py` krijgt één bronargument: `--scale64-run`. De opdrachten zijn
uitsluitend `check` en `evaluate`. Het pakket maakt nieuwe resultaatmappen en
ondersteunt geen training, nieuwe selectie, datasetdownload of BN-herkalibratie.

`EXPECTED_REFERENCE.json` pint de werkelijke bronrapporthash, selectie en
compacte exporthashes. `SCALE64_REFERENCE_PROTOCOL.json` bevat het exacte
oorspronkelijke trainingsprotocol. De runner controleert daarnaast alle
epochmetingen, exportmetadata, bronhashes, dataset en oorspronkelijke runtime.

`prior_posthoc` bevat ongewijzigde eerdere A/B/C-resultaten. `REFERENCE.json`
legt hun hashes en bronbundel vast. De runner controleert dezelfde beginstaat,
data en uitvoeringscondities. Historische tabellen worden naar de nieuwe
resultaatmap gekopieerd, zodat de notebook zelfstandig alle drie trainingen
kan vergelijken. De oorspronkelijke modellen worden daarbij niet herhaald.

## Notebook

De nieuwe notebook heet `MX_triple_scale64_trained_posthoc_resultaten.ipynb`.
Hij wordt leeg meegeleverd en krijgt bij `evaluate` de nieuwe runmap ingevuld.
`notebook_support.py` leest alleen opgeslagen resultaten. De bestaande
integriteitscontroles zijn aangepast aan één nieuwe trainingssoort en twee
checkpointrollen, met aanvullende B-reconstructie- en historische controles.

De accuracyfiguur bevat epoch 100 en E4M3-best42 naast elkaar. Twee extra
overzichtsfiguren vergelijken A/B/C over de directe, naïeve en schaal ×64-training.
Daarbij blijven de eerdere eigen best54-checkpoints herkenbaar.

CSV-floats worden met `float_precision='round_trip'` ingelezen. Dat voorkomt
een eerder vastgestelde parserafwijking van één ULP bij een zeer grote eindige
CE-waarde. De oorspronkelijke CSV's worden niet veranderd.

## Validatie

`local_validation` bevat de controleprogramma's en hun feitelijke rapporten:
de onafhankelijke code-oracles, bevroren modelreconstructie, echte runnerflow
op synthetische beelden, foutpaden en notebookuitvoering zonder modelimports.
Alle synthetische uitvoer blijft als zodanig gemarkeerd. Er wordt geen nieuwe
SVHN-accuracy gesuggereerd. Zie `local_validation/VALIDATION.md` voor de scope.
