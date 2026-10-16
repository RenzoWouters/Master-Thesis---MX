# CNN-8 / SVHN: proefrun van vijf volledige epochs

Dit pakket gebruikt de werkelijk aangeleverde Any-Precision-code. Het voert
alleen de eerste vijf epochs van het oorspronkelijke SVHN-recept uit. Het
100-epoch-learning-rateschema wordt NIET naar vijf epochs geschaald.
Er is geen MX, geen truncatie vanuit opgeslagen codes, geen nieuwe architectuur,
geen validatiesplit en geen trainingssubset.

## Plaatsing en starten op Windows

Pak de inhoud van deze ZIP uit in een NIEUWE map:

    C:\Users\renzo\master-thesis-mx\software_checks\anyprecision_svhn_pilot5_v1

Controleer dat `run_svhn.py` direct in die map staat. De ZIP bevat geen extra
buitenste map. Bestaande software_checks, experimenten en ResNet-resultaten
worden niet aangepast. Gebruik in PowerShell:

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_pilot5_v1\run_svhn.py" check
```

`check` gebruikt geen SVHN-data en voert geen optimizerstap uit. Het controleert
het pakket, PyTorch 2.10.0, torchvision 0.25.0, Python 3.13, SciPy, CUDA, het
model en uitvoervormen bij alle vijf precisies. Het draait in een afzonderlijk
proces: het initialiseert de latere trainingsrun niet vooraf.

### Alleen wanneer SciPy ontbreekt

De oorspronkelijke `datasets/svhn.py` heeft `scipy.io.loadmat` nodig. Laat een
reeds aanwezige, werkende SciPy-installatie ongemoeid. Alleen als `check` meldt
dat SciPy ontbreekt:

```powershell
& ".\.venv-train\Scripts\python.exe" -m pip install --only-binary=:all: --no-deps "scipy==1.17.0"
& ".\.venv-train\Scripts\python.exe" -m pip check
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_pilot5_v1\run_svhn.py" check
```

`--no-deps` voorkomt dat deze opdracht NumPy, PyTorch of andere bestaande
pakketten wijzigt. Bij een versieconflict stoppen: niet op eigen initiatief
upgraden of downgraden. De installatie van SciPy is een compatibiliteitsstap,
geen onderzoekswijziging. De exacte omgeving wordt per run opgeslagen.
De SciPy 1.17.0-release ondersteunt Python 3.13; pip check controleert de
geinstalleerde afhankelijkheidsmetadata. Zie COMPATIBILITY.md.

### Vijf volledige epochs

Start uitsluitend wanneer `check` afsluit met `EINDSTATUS: PASS`:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_pilot5_v1\run_svhn.py" run --download --stop-after-epoch 5
```

De eerste keer worden de drie originele SVHN-bestanden opgehaald naar
`data\svhn`. Bestaande bestanden worden gecontroleerd en niet overschreven.
De download en het inladen tellen niet mee als trainingsepochtijd.
De oorspronkelijke loader laadt de beelden in het systeemgeheugen; dit is
iets anders dan GPU-geheugen. Het inladen van `extra` kan merkbaar duren.

De training gebruikt 604.388 voorbeelden per epoch en 4.722 minibatches bij
batchgrootte 128. Vijf epochs zijn 23.610 optimizerstappen. Iedere stap gebruikt
vijf forward/backward-paden, niet vijf afzonderlijke optimizerupdates.
De regels met `UPSTREAM epoch 0` horen bij de eerste epoch: de auteurscode
telt vanaf nul, terwijl onze samenvattingen epochs 1 tot en met 5 gebruiken.
Na iedere epoch worden alle 26.032 testbeelden bij alle vijf precisies
beoordeeld. De laatste onvolledige minibatch wordt niet weggelaten.

Er is geen automatische start van de volledige 100-epochtraining. Deze versie
weigert een stopgrens boven vijf. De latere volledige referentie begint in een
nieuwe run vanaf initialisatie, niet automatisch vanuit deze proef.

## Resultaten

Per commando ontstaat een nieuwe map:

    pilot_runs\anyprecision_svhn_pilot5\<tijdstempel>_run

De exacte map wordt in de terminal getoond. Belangrijkste bestanden:

| Bestand | Betekenis |
|---|---|
| `run_report.json` | Status, omgeving, protocol, waarschuwingen, tijdschatting |
| `checkpoints\epoch_00001.pt` ... | Een onveranderlijk checkpoint per volledige epoch |
| `best_checkpoint.pt` | Kopie van het checkpoint met beste W32/A32-testaccuracy |
| `last_checkpoint.pt` | Kopie van de laatste afgeronde epoch; bij volledige proef epoch 5 |
| `checkpoint_selection.json` | Welke epochs geselecteerd zijn en volgens welke regel |
| `selected_metrics.csv` | Alle vijf precisies voor beide checkpointrollen |
| `epoch_metrics.csv` | Alle trainings- en testmetrics per epoch en precisie |
| `epoch_times.csv`, `timing_estimate.json` | Gemeten epochtijden en extrapolatie naar 100 epochs |
| `batch_metrics\*.jsonl` | Opgeslagen ruwe minibatchmetrics |
| `sample_order\*.npy` | Volgorde en volledige dekking van trainings- en testvoorbeelden |
| `SVHN_pilot5_resultaten.ipynb` | Notebook dat deze runbestanden inleest |
| `feedback_bundle.zip` | Kleine bundel om te delen; geen dataset of gewichten |

Open het notebook IN DE RESULTATENMAP, selecteer `.venv-train` als kernel en
gebruik Run All. Het notebookpad naar de betreffende run is reeds ingevuld.
De notebooktemplate in de softwaremap heeft nog geen onderzoeksresultaten.
Pandas, Matplotlib en ipykernel zijn alleen nodig voor de notebookweergave.
Het notebook traint niet en schrijft geen ruwe gegevens of checkpoints.

`best` wordt uitsluitend op W32/A32 bepaald. Bij exact gelijke scores blijft de
eerdere epoch geselecteerd, overeenkomstig `>` in de broncode. Alle vijf
precisies komen uit hetzelfde geselecteerde checkpoint. `last` blijft een
aparte gerapporteerde rol, ook wanneer beide rollen dezelfde epoch bevatten.

## Tijdschatting

Vanaf twee volledig gemeten epochs verschijnt een eerste extrapolatie.
Voor een ononderbroken vijf-epochproef is de hoofdschatting:

    T100 = T(epoch 1) + 99 * mediaan[T(epoch 2), ..., T(epoch 5)]

Iedere T omvat training, volledige test, controles en checkpointopslag.
Download, datasetinladen en afsluitende rapportexport/checks zijn afzonderlijk.
De projecties met de snelste en traagste warme epoch zijn GEEN statistisch
betrouwbaarheidsinterval. Thermische veranderingen en achtergrondbelasting
in een urenlange run zijn daarmee niet bewezen afgedekt.
Een eerste epoch na hervatten wordt opnieuw als koude processtart behandeld.

## Stoppen en hervatten als beveiliging

De bedoeling is vijf epochs in een doorlopende run. Toch blijft hervatten
mogelijk. Een eerste Ctrl+C TIJDENS de epochloop vraagt een nette stop na de
lopende volledige epoch. Een tweede Ctrl+C stopt direct; dan is alleen het
vorige volledige checkpoint hervatbaar. Tijdens downloaden of inladen stopt
Ctrl+C direct. Bestanden worden niet verwijderd.

Na een storing: gebruik een werkelijk aanwezig `epoch_0000N.pt` met de
bijbehorende `.sha256` en geef het volledige pad op:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_pilot5_v1\run_svhn.py" resume --checkpoint "C:\volledig\pad\naar\epoch_00002.pt" --stop-after-epoch 5
```

Dit is een padvoorbeeld, niet een bestaand checkpoint. Hervatten maakt een
NIEUWE sessiemap. Model, Adam, scheduler, BatchNorm en alle toevalstoestanden
worden hersteld. Er is geen mid-epochherstel of garantie over andere apparaten.
Met de oorspronkelijke `cudnn.benchmark=True` is bitexacte CUDA-herhaling niet
gegarandeerd. De oude checkpointbestanden blijven ongewijzigd.

## Validatiestatus van dit pakket

De meegeleverde offline tests zijn uitgevoerd op CPU, met PyTorch 2.10.0 en
torchvision 0.25.0, op kleine synthetische gegevens. Zij controleren exacte
gelijkheid met de oorspronkelijke, niet-geinstrumenteerde `forward()`:
model, optimizer, scheduler, RNG en epochmetrics. Ook dropout/hervatten,
selectie, checksumbeveiliging, datasetinladen met kunstmatige .mat-bestanden,
LR-grenzen en notebookuitvoering zijn gecontroleerd.

Er zijn hier GEEN echte SVHN-epochs, GEEN echte datasetdownload en GEEN
RTX-5070-GPU-tests uitgevoerd. De eerste echte metingen komen uit jouw laptop.
Zie `local_validation/validation_report.json` en `local_validation/VALIDATION.md`.
