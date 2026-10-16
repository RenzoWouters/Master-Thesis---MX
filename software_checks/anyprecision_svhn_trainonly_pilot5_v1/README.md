# Any-Precision CNN-8 / SVHN train-only: proefrun van vijf epochs

Dit is een NIEUW pakket. De inhoud komt in:

    C:\Users\renzo\master-thesis-mx\software_checks\anyprecision_svhn_trainonly_pilot5_v1

`run_svhn.py` moet direct in die map staan. Bestaande code, experimenten en
train+extra-/ResNet-resultaten worden niet aangepast. Deze ZIP bevat de eerder
gebruikte auteurscode; je hoeft niets van GitHub opnieuw te downloaden.

## Protocol

De expliciet gekozen onderzoekswijziging is `train_extra` -> `train`.
De pilotstop na vijf volledige epochs blijft behouden. Verder blijven model,
preprocessing, vijf precisies, recursieve distillatie, Adam, batchgrootte,
LR-mijlpalen, testset en checkpointselectie zoals in het vorige werkende pakket.

- CNN-8: svhnq, expand=2, 132.954 parameters inclusief vijf BN-banken.
- Alle 73.257 gewone trainbeelden, zonder subset en zonder aparte validatiesplit.
- Geen extra-beelden; extra_32x32.mat wordt niet geladen, gehasht of verwijderd.
- Alle 26.032 testbeelden na iedere epoch, bij alle vijf precisies.
- Batch 128, drop_last=False: 573 trainminibatches per epoch en 2.865 updates na vijf epochs.
- Adam, lr=0.001, weight_decay=0; oorspronkelijke LR-mijlpalen 50/75/90 blijven.
- Tijdens deze pilot blijft lr=0.001. Geen versneld vijf-epoch-LR-schema.
- Oorspronkelijke W/A-precisies 1/2/4/8/32 en recursieve distillatie.
- Geen MX, geen nieuwe kwantisator, geen AMP, geen torch.compile.
- workers=0 en pin_memory=True, cudnn.benchmark=True, zoals de vorige SVHN-run.
- Dezelfde expliciete lokale seed 20261009; historische auteursseed is onbekend.
- Run begint met nieuwe initialisatie; geen train+extra-checkpoint inladen.

Dit is GEEN paperreproductie: de paper gebruikte train+extra en honderd epochs.
Zelfs honderd train-only-epochs geven minder data en minder optimizerupdates.
Zie PROTOCOL.md en trainonly.patch voor de precieze afbakening.

## 1. Bestaande omgeving controleren (geen training of download)

```powershell
Set-Location "C:\Users\renzo\master-thesis-mx"

& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_trainonly_pilot5_v1\run_svhn.py" check
```

Ga alleen verder na `EINDSTATUS: PASS`. De bestaande .venv-train wordt gebruikt;
geen nieuwe omgeving en geen automatische installatie of upgrade. In jouw vorige
feedbackbundel waren SciPy, PyTorch en torchvision al werkend aanwezig. Bij een
conflict eerst het rapport bekijken; niet automatisch pakketten vervangen.

## 2. Vijf volledige train-only-epochs in een doorlopende run

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_trainonly_pilot5_v1\run_svhn.py" run --stop-after-epoch 5
```

Bestaande bestanden uit `data\svhn` worden gebruikt. `train_32x32.mat` en
`test_32x32.mat` zijn voldoende. `extra_32x32.mat` mag blijven staan: het pakket
opent dat bestand niet en gebruikt het niet. Er is bewust geen --download nodig
wanneer de bestanden uit je vorige run nog aanwezig zijn.

Alleen als train of test ontbreekt, kan dezelfde run-opdracht met `--download`
worden gestart. Dat downloadt uitsluitend ontbrekende train/test-bestanden.
Een bestaand bestand met een verkeerde checksum wordt nooit vervangen.

Verwachte controles na het inladen:

    DATA PASS: train=73257, test=26032; extra NIET gebruikt; ...
    MINIBATCHES per epoch: train=573; test=204; batch=128, drop_last=False.
    NIEUWE initialisatie. Geen ResNet- of proefcheckpoint ingeladen.

Daarna epochs 1..5, telkens met LR=0.001. De oude UPSTREAM-log telt vanaf nul:
UPSTREAM epoch 0 is onze eerste epoch. De volledige referentie start NIET vanzelf.

## 3. Resultaten en notebook

Nieuwe sessiemappen staan onder:

    C:\Users\renzo\master-thesis-mx\pilot_runs\anyprecision_svhn_trainonly_pilot5

Elke aanroep krijgt een eigen tijdstempel. Het exacte pad staat bovenaan de terminal.
Open na de run `SVHN_trainonly_pilot5_resultaten.ipynb` UIT DE RESULTATENMAP.
Selecteer de bestaande .venv-train-kernel en voer Run All uit. Het notebook leest
alleen JSON/CSV-bestanden en kan niet trainen of data downloaden.

Bewaar `best_checkpoint.pt` EN `last_checkpoint.pt`, plus de afzonderlijke
`checkpoints\epoch_00001.pt` ... `epoch_00005.pt`. Beste betekent de eerste strikt
hoogste W32/A32-testaccuracy, alle precisies van DATZELFDE checkpoint. Laatste is
epoch 5. Beide rollen worden gerapporteerd, ook als ze samenvallen.

De tijdschatting voor honderd TRAIN-ONLY-epochs gebruikt epoch 1 + 99 maal de
mediaan van epochs die niet de eerste in een proces zijn. Ze omvat training,
volledige testevaluatie en checkpointopslag, maar geen download/data-inladen.
Vijf-epochaccuracies zijn geen voorspellingen voor honderd epochs. De paperwaarden
blijven zichtbaar als literatuurcontext met een expliciete datasplitwaarschuwing.

`feedback_bundle.zip` bevat de kleine rapporten en logs, geen dataset of gewichten.
Dat is het bestand om terug te sturen na de proef.

## Stoppen en hervatten (alleen als nodig)

De eerste Ctrl+C vraagt stoppen na de huidige volledige epoch. Een tweede Ctrl+C
breekt direct af; alleen eerder volledig opgeslagen checkpoints zijn hervatbaar.
Een leeg bestand STOP_AFTER_EPOCH in de sessiemap heeft hetzelfde zachte-stopdoel.

Hervatten maakt een NIEUWE sessiemap; het oude checkpoint wordt niet aangepast.
Gebruik alleen een checkpoint uit DIT train-only-pakket, bijvoorbeeld:

```powershell
& ".\.venv-train\Scripts\python.exe" ".\software_checks\anyprecision_svhn_trainonly_pilot5_v1\run_svhn.py" resume --checkpoint "VOLLEDIG_PAD_NAAR_TRAINONLY_EPOCHCHECKPOINT.pt" --stop-after-epoch 5
```

Schema/config/bron/data/backend-contracten moeten exact overeenkomen. Een oud
train+extra-checkpoint wordt afgewezen. Met cuDNN-benchmarking wordt geen algemene
bitexacte CUDA-herhaling beloofd. CPU-synthetische hervattests zijn afzonderlijk
uitgevoerd; zie local_validation/VALIDATION.md. Voor de latere honderd-epochreferentie
beginnen we opnieuw met initialisatie, niet automatisch vanuit deze proef.
